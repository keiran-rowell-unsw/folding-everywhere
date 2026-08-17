//! ESMFold 1 + 2 tab.
//!
//! The job logic here is lifted unchanged from the `fold_gui` app of
//! `folding-everywhere` — same weight URLs, same HF-Mirror fallback, same curl
//! flags, same progress callbacks — so a fold run through this tab is the same
//! computation the standalone app performed. Only the plumbing around it (state
//! ownership, the `/api/esmfold/*` route prefix) is new.
//!
//! ESMFold2 is a stochastic diffusion model, so it exposes a **seed** (same seed
//! -> same structure, bit-exact to a PyTorch fp32 run pinned at that seed);
//! ESMFold1 is deterministic and ignores the seed. On first use each model's
//! weights are auto-downloaded (ESMFold1 ~8.4 GB, ESMFold2 ~30 GB) via the OS
//! `curl`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---- ESMFold1 (deterministic) ----
use esmfold::constants::Constants;
use esmfold::pdb::{mean_plddt as v1_mean_plddt, to_pdb as v1_to_pdb};
use esmfold::pipeline::fold_cb;
use esmfold::weights::Weights as V1Weights;

// ---- ESMFold2 (diffusion, seedable) ----
use esmfold2::standalone as v2;
use esmfold2::weights::Weights as V2Weights;

use crate::{home, jstr};

const V1_URL: &str = "https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin";
const V1_BYTES: u64 = 8_442_062_570;
const V2_REPO: &str = "https://huggingface.co/biohub/ESMC-6B/resolve/main";
const V2_HEAD_URL: &str = "https://huggingface.co/biohub/ESMFold2/resolve/main/model.safetensors";
const V2_TOTAL_BYTES: f32 = 30.0e9; // approx, for the progress bar
const VALID_AA: &str = "ACDEFGHIKLMNPQRSTVWYXBUZOacdefghiklmnpqrstvwyxbuzo";
const MAX_LEN: usize = 500;
const LOG_CAP: usize = 5000;

pub const EXAMPLE_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

#[derive(Default, Clone)]
pub struct State {
    pub busy: bool,
    phase: String,
    progress: f32,
    log: Vec<String>,
    error: Option<String>,
    done: bool,
    plddt: f32,
    ptm: f32,
    pdb_ready: bool,
    model_path: String,
}

fn v1_dir() -> PathBuf {
    std::env::var("ESMFOLD_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".esmfold"))
}
fn v2_dir() -> PathBuf {
    std::env::var("ESMFOLD2_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".esmfold2"))
}
pub fn out_pdb() -> PathBuf { home().join(".fold_gui").join("prediction.pdb") }

fn set<F: FnOnce(&mut State)>(st: &Arc<Mutex<State>>, f: F) { f(&mut st.lock().unwrap()); }
fn logmsg(st: &Arc<Mutex<State>>, m: &str) {
    let mut s = st.lock().unwrap();
    s.log.push(m.to_string());
    if s.log.len() > LOG_CAP { s.log.remove(0); }
}

// HuggingFace is blocked in some regions; mirror to hf-mirror.com (same path layout).
const HF: &str = "https://huggingface.co";
const HF_MIRROR: &str = "https://hf-mirror.com";

/// Download `url`, but: (1) honor an `HF_ENDPOINT` override (the standard hf-mirror
/// convention), and (2) if huggingface.co is unreachable, automatically retry via
/// hf-mirror.com. So users in restricted regions need no configuration.
fn curl_to(st: &Arc<Mutex<State>>, url: &str, dest: &PathBuf, label: &str, base_done: u64, total: f32) -> Result<(), String> {
    if dest.exists() { return Ok(()); }
    let primary = match std::env::var("HF_ENDPOINT") {
        Ok(ep) if !ep.trim().is_empty() => url.replace(HF, ep.trim().trim_end_matches('/')),
        _ => url.to_string(),
    };
    let mirror = primary.replace(HF, HF_MIRROR);
    set(st, |s| s.phase = format!("Connecting to {label} source… (auto-falls back to HF-Mirror if blocked)"));
    logmsg(st, &format!("Fetching {label} from {}…", primary.split("/resolve/").next().unwrap_or(&primary)));
    match curl_one(st, &primary, dest, label, base_done, total) {
        Ok(()) => Ok(()),
        Err(e1) => {
            if mirror != primary {
                logmsg(st, &format!("huggingface.co unreachable — switching to HF-Mirror (hf-mirror.com)… [{e1}]"));
                set(st, |s| s.phase = format!("Retrying {label} via HF-Mirror…"));
                curl_one(st, &mirror, dest, label, base_done, total)
                    .map_err(|e2| format!("both huggingface.co and hf-mirror.com failed ({e1}; {e2})"))
            } else {
                Err(e1)
            }
        }
    }
}

fn curl_one(st: &Arc<Mutex<State>>, url: &str, dest: &PathBuf, label: &str, base_done: u64, total: f32) -> Result<(), String> {
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    if dest.exists() { return Ok(()); }
    let part = dest.with_extension("part");
    let _ = std::fs::remove_file(&part);
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    // Timeouts are essential: in regions where huggingface.co is *blocked* the
    // connection (or its CDN redirect) HANGS rather than failing, so without these
    // curl would sit at 0 bytes forever and the HF-Mirror fallback would never fire.
    //   --connect-timeout: give up connecting (incl. DNS) after 25 s,
    //   --speed-limit/--speed-time: abort if < 1 KB/s for 30 s (a stalled transfer).
    // A genuinely working download easily clears 1 KB/s, so these never abort it.
    let mut child = Command::new(curl)
        .args([
            "-L", "--fail", "--silent", "--show-error",
            "--connect-timeout", "25",
            "--speed-limit", "1024", "--speed-time", "30",
            "-o",
        ])
        .arg(&part).arg(url)
        .spawn().map_err(|e| format!("could not start curl ({e}). curl ships with Windows 10+/macOS/Linux."))?;
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                if !status.success() { return Err(format!("download failed for {label} (curl {:?})", status.code())); }
                break;
            }
            None => {
                if let Ok(md) = std::fs::metadata(&part) {
                    let frac = ((base_done + md.len()) as f32 / total).min(0.999);
                    set(st, |s| { s.progress = frac; s.phase = format!("Downloading {label}… {:.1} GB", (base_done + md.len()) as f32 / 1e9); });
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    Ok(())
}

fn sanitize(seq: &str) -> Result<String, String> {
    let s: String = seq.chars().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase()).collect();
    if s.is_empty() { return Err("Please enter a protein sequence (one-letter amino-acid codes).".into()); }
    if s.len() > MAX_LEN { return Err(format!("Sequence is {} aa; this build caps at {} aa for reasonable CPU runtime.", s.len(), MAX_LEN)); }
    if let Some(bad) = s.chars().find(|c| !VALID_AA.contains(*c)) { return Err(format!("Invalid amino-acid letter '{bad}'.")); }
    Ok(s)
}

// ---------------------------------------------------------------- ESMFold1 path
fn run_v1(st: &Arc<Mutex<State>>, seq: &str) -> Result<(f32, f32), String> {
    let wp = v1_dir().join("pytorch_model.bin");
    if !wp.exists() {
        logmsg(st, "ESMFold1 weights not found — downloading (~8.4 GB, one time)…");
        curl_to(st, V1_URL, &wp, "ESMFold1 weights", 0, V1_BYTES as f32)?;
    }
    set(st, |s| { s.phase = "Loading ESMFold1 weights…".into(); s.model_path = wp.display().to_string(); s.progress = 0.0; });
    let w = V1Weights::open(wp.to_str().unwrap()).map_err(|e| format!("open weights: {e}"))?;
    let consts = Constants::embedded();
    let stc = st.clone();
    let mut last = String::new();
    let out = fold_cb(&w, &consts, seq, &mut |msg, frac| {
        let mut s = stc.lock().unwrap();
        s.phase = msg.to_string(); s.progress = frac;
        if msg != last { s.log.push(msg.to_string()); if s.log.len() > LOG_CAP { s.log.remove(0); } }
        drop(s); last = msg.to_string();
    });
    // Masked by atom existence and on the 0..100 scale, so this is the same number
    // upstream `esm.pretrained.esmfold_v1()` reports as `output["mean_plddt"]`.
    let plddt = v1_mean_plddt(&out.plddt.data, &out.aatype, &consts, out.l);
    let pdb = v1_to_pdb(&out.atom37.data, &out.plddt.data, &out.aatype, &consts, out.l);
    std::fs::create_dir_all(out_pdb().parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(out_pdb(), pdb).map_err(|e| e.to_string())?;
    Ok((plddt, out.ptm))
}

// ---------------------------------------------------------------- ESMFold2 path
fn run_v2(st: &Arc<Mutex<State>>, seq: &str, seed: u64, num_loops: usize, num_sampling_steps: usize) -> Result<(f32, f32), String> {
    let dir = v2_dir();
    let esmc_dir = dir.join("esmc");
    let index = esmc_dir.join("model.safetensors.index.json");
    let head = dir.join("esmfold2_head.safetensors");

    if !index.exists() || !head.exists() {
        logmsg(st, "ESMFold2 weights not found — downloading (~30 GB, one time)…");
        // 1) ESM-C shard index, then every shard it references
        curl_to(st, &format!("{V2_REPO}/model.safetensors.index.json"), &index, "ESM-C index", 0, V2_TOTAL_BYTES)?;
        let txt = std::fs::read_to_string(&index).map_err(|e| e.to_string())?;
        let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        let mut shards: Vec<String> = j["weight_map"].as_object().ok_or("bad index.json")?
            .values().map(|v| v.as_str().unwrap().to_string()).collect();
        shards.sort(); shards.dedup();
        let mut done: u64 = 0;
        for (i, sh) in shards.iter().enumerate() {
            logmsg(st, &format!("Downloading ESM-C shard {}/{} ({sh})", i + 1, shards.len()));
            curl_to(st, &format!("{V2_REPO}/{sh}"), &esmc_dir.join(sh), &format!("ESM-C {}/{}", i + 1, shards.len()), done, V2_TOTAL_BYTES)?;
            if let Ok(md) = std::fs::metadata(esmc_dir.join(sh)) { done += md.len(); }
        }
        // 2) ESMFold2 head
        logmsg(st, "Downloading ESMFold2 head…");
        curl_to(st, V2_HEAD_URL, &head, "ESMFold2 head", done, V2_TOTAL_BYTES)?;
    }

    set(st, |s| { s.phase = "Loading ESMFold2 weights (ESM-C 6B, memory-mapped)…".into(); s.model_path = dir.display().to_string(); s.progress = 0.0; });
    logmsg(st, "Loading ESM-C 6B + ESMFold2 head…");
    let w_esmc = V2Weights::open_sharded(index.to_str().unwrap()).map_err(|e| format!("open ESM-C: {e}"))?;
    let w = V2Weights::open(head.to_str().unwrap()).map_err(|e| format!("open head: {e}"))?;
    set(st, |s| { s.phase = format!("Folding with ESMFold2 (seed {seed})…"); s.progress = 0.0; });
    logmsg(st, &format!("Folding {} residues with ESMFold2, seed {seed} (num_loops={num_loops}, num_sampling_steps={num_sampling_steps})…", seq.len()));
    let stc = st.clone();
    let mut last = String::new();
    let o = v2::fold_cb(seq, seed, &w_esmc, &w, num_loops, num_sampling_steps, &mut |msg, frac| {
        let mut s = stc.lock().unwrap();
        s.phase = msg.to_string(); s.progress = frac;
        if msg != last { s.log.push(msg.to_string()); if s.log.len() > LOG_CAP { s.log.remove(0); } }
        drop(s); last = msg.to_string();
    });
    std::fs::create_dir_all(out_pdb().parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(out_pdb(), &o.pdb).map_err(|e| e.to_string())?;
    // ESMFold2's plddt_mean is a per-residue mean in 0..1 (already correct — its
    // pLDDT is [L], not [L,37]); rescale to 0..100 so both tabs report one unit.
    Ok((o.plddt_mean * 100.0, o.ptm))
}

pub struct Job {
    pub model: u8,
    pub seed: u64,
    pub loops: usize,
    pub steps: usize,
    pub seq: String,
}

/// Parse the `model\nseed\nloops\nsteps\nseq` body the page posts. Defaults are
/// the official ESMFold2 release depth (20 loops, 68 sampling steps), clamped to
/// a sane CPU range.
pub fn parse_job(body: &str) -> Job {
    let mut it = body.splitn(5, '\n');
    Job {
        model: it.next().unwrap_or("1").trim().parse().unwrap_or(1),
        seed: it.next().unwrap_or("0").trim().parse().unwrap_or(0),
        loops: it.next().unwrap_or("20").trim().parse().unwrap_or(20).clamp(1, 64),
        steps: it.next().unwrap_or("68").trim().parse().unwrap_or(68).clamp(1, 200),
        seq: it.next().unwrap_or("").to_string(),
    }
}

pub fn run_fold(st: Arc<Mutex<State>>, job: Job) {
    let model = job.model;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(f32, f32), String> {
        let seq = sanitize(&job.seq)?;
        logmsg(&st, &format!("ESMFold{model}: folding sequence of length {}", seq.len()));
        if model == 2 { run_v2(&st, &seq, job.seed, job.loops, job.steps) } else { run_v1(&st, &seq) }
    }));
    let mut s = st.lock().unwrap();
    s.busy = false;
    match result {
        Ok(Ok((plddt, ptm))) => {
            s.done = true; s.progress = 1.0; s.phase = "Complete".into();
            s.plddt = plddt; s.ptm = ptm; s.pdb_ready = true;
            s.log.push(format!("Done. mean pLDDT {:.2}, pTM {:.3}", plddt, ptm));
        }
        Ok(Err(e)) => { s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
        Err(_) => { let e = "internal error during folding (panic)".to_string(); s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
    }
}

pub fn starting() -> State { State { busy: true, phase: "Starting…".into(), ..Default::default() } }

pub fn idle() -> State { State { phase: "Idle".into(), ..Default::default() } }

/// Reject a start without disturbing the tab's state (used by the global run lock).
pub fn refuse(st: &Arc<Mutex<State>>, msg: &str) {
    let mut s = st.lock().unwrap();
    s.busy = false;
    s.error = Some(msg.to_string());
    s.phase = "Blocked".into();
}

pub fn status_json(s: &State) -> String {
    let log = s.log.iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",");
    format!(
        "{{\"busy\":{},\"phase\":{},\"progress\":{:.4},\"done\":{},\"plddt\":{:.2},\"ptm\":{:.4},\"error\":{},\"pdb\":{},\"model\":{},\"log\":[{}]}}",
        s.busy, jstr(&s.phase), s.progress, s.done, s.plddt, s.ptm,
        s.error.as_deref().map(jstr).unwrap_or_else(|| "null".into()),
        if s.pdb_ready { "\"prediction.pdb\"" } else { "null" }.to_string(),
        jstr(&s.model_path), log
    )
}
