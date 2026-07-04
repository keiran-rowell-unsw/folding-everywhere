//! Unified ESMFold1 / ESMFold2 local-web GUI (pure Rust, no Python/GPU).
//!
//! Double-click the exe: it starts a local server and opens your browser. Pick a
//! model (ESMFold1 or ESMFold2), paste a sequence, click Fold. ESMFold2 is a
//! stochastic diffusion model, so it exposes a **seed** (same seed -> same
//! structure, bit-exact to a PyTorch run pinned at that seed); ESMFold1 is
//! deterministic and ignores the seed. On first use each model's weights are
//! auto-downloaded (ESMFold1 ~8.4 GB, ESMFold2 ~30 GB) via the OS `curl`.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tiny_http::{Header, Method, Response, Server};

// ---- ESMFold1 (deterministic) ----
use esmfold::constants::Constants;
use esmfold::pdb::to_pdb as v1_to_pdb;
use esmfold::pipeline::fold_cb;
use esmfold::weights::Weights as V1Weights;

// ---- ESMFold2 (diffusion, seedable) ----
use esmfold2::standalone as v2;
use esmfold2::weights::Weights as V2Weights;

const V1_URL: &str = "https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin";
const V1_BYTES: u64 = 8_442_062_570;
const V2_REPO: &str = "https://huggingface.co/biohub/ESMC-6B/resolve/main";
const V2_HEAD_URL: &str = "https://huggingface.co/biohub/ESMFold2/resolve/main/model.safetensors";
const V2_TOTAL_BYTES: f32 = 30.0e9; // approx, for the progress bar
const VALID_AA: &str = "ACDEFGHIKLMNPQRSTVWYXBUZOacdefghiklmnpqrstvwyxbuzo";
const MAX_LEN: usize = 500;

#[derive(Default, Clone)]
struct State {
    busy: bool,
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

fn home() -> PathBuf {
    let base = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| ".".into());
    PathBuf::from(base)
}
fn v1_dir() -> PathBuf { std::env::var("ESMFOLD_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".esmfold")) }
fn v2_dir() -> PathBuf { std::env::var("ESMFOLD2_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".esmfold2")) }
fn out_pdb() -> PathBuf { home().join(".fold_gui").join("prediction.pdb") }

fn set<F: FnOnce(&mut State)>(st: &Arc<Mutex<State>>, f: F) { f(&mut st.lock().unwrap()); }
const LOG_CAP: usize = 5000;
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
    let plddt = out.plddt.data.iter().sum::<f32>() / out.plddt.data.len() as f32;
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
    Ok((o.plddt_mean, o.ptm))
}

fn run_fold(st: Arc<Mutex<State>>, model: u8, seed: u64, loops: usize, steps: usize, seq: String) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(f32, f32), String> {
        let seq = sanitize(&seq)?;
        logmsg(&st, &format!("ESMFold{model}: folding sequence of length {}", seq.len()));
        if model == 2 { run_v2(&st, &seq, seed, loops, steps) } else { run_v1(&st, &seq) }
    }));
    let mut s = st.lock().unwrap();
    s.busy = false;
    match result {
        Ok(Ok((plddt, ptm))) => {
            s.done = true; s.progress = 1.0; s.phase = "Complete".into();
            s.plddt = plddt; s.ptm = ptm; s.pdb_ready = true;
            s.log.push(format!("Done. mean pLDDT {:.3}, pTM {:.3}", plddt, ptm));
        }
        Ok(Err(e)) => { s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
        Err(_) => { let e = "internal error during folding (panic)".to_string(); s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"), c if (c as u32) < 0x20 => o.push(' '), c => o.push(c),
        }
    }
    o.push('"'); o
}
fn status_json(s: &State) -> String {
    let log = s.log.iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",");
    format!(
        "{{\"busy\":{},\"phase\":{},\"progress\":{:.4},\"done\":{},\"plddt\":{:.4},\"ptm\":{:.4},\"error\":{},\"pdb\":{},\"model\":{},\"log\":[{}]}}",
        s.busy, jstr(&s.phase), s.progress, s.done, s.plddt, s.ptm,
        s.error.as_deref().map(jstr).unwrap_or_else(|| "null".into()),
        if s.pdb_ready { "\"prediction.pdb\"" } else { "null" }.to_string(),
        jstr(&s.model_path), log
    )
}

fn main() {
    let state = Arc::new(Mutex::new(State::default()));
    set(&state, |s| s.phase = "Idle".into());
    let server = Server::http("127.0.0.1:0").expect("could not start local server");
    let url = format!("http://{}/", server.server_addr());
    println!("ESMFold (1+2) GUI running at {url}");
    let _ = if cfg!(windows) { Command::new("cmd").args(["/C", "start", "", &url]).spawn() }
        else if cfg!(target_os = "macos") { Command::new("open").arg(&url).spawn() }
        else { Command::new("xdg-open").arg(&url).spawn() };

    for mut req in server.incoming_requests() {
        let u = req.url().to_string();
        let method = req.method().clone();
        if u == "/" {
            let html = INDEX_HTML.replace("__EXAMPLE__", EXAMPLE_SEQ);
            let r = Response::from_string(html).with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap());
            let _ = req.respond(r);
        } else if u == "/api/status" {
            let body = status_json(&state.lock().unwrap());
            let _ = req.respond(Response::from_string(body).with_header(Header::from_bytes("Content-Type", "application/json").unwrap()));
        } else if u == "/api/fold" && method == Method::Post {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            // body = "model\nseed\nloops\nsteps\nseq"  (loops/steps used only for ESMFold2)
            let mut it = body.splitn(5, '\n');
            let model: u8 = it.next().unwrap_or("1").trim().parse().unwrap_or(1);
            let seed: u64 = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
            // Defaults are the official ESMFold2 release depth (20 loops, 68 sampling steps);
            // clamped to a sane CPU range.
            let loops: usize = it.next().unwrap_or("20").trim().parse().unwrap_or(20).clamp(1, 64);
            let steps: usize = it.next().unwrap_or("68").trim().parse().unwrap_or(68).clamp(1, 200);
            let seq = it.next().unwrap_or("").to_string();
            let mut s = state.lock().unwrap();
            if s.busy {
                drop(s);
                let _ = req.respond(Response::from_string("{\"ok\":false,\"error\":\"already running\"}"));
            } else {
                *s = State { busy: true, phase: "Starting…".into(), ..Default::default() };
                drop(s);
                let stt = state.clone();
                std::thread::spawn(move || run_fold(stt, model, seed, loops, steps, seq));
                let _ = req.respond(Response::from_string("{\"ok\":true}"));
            }
        } else if u == "/api/pdb" {
            match std::fs::read(out_pdb()) {
                Ok(data) => {
                    let r = Response::from_data(data)
                        .with_header(Header::from_bytes("Content-Type", "chemical/x-pdb").unwrap())
                        .with_header(Header::from_bytes("Content-Disposition", "attachment; filename=\"prediction.pdb\"").unwrap());
                    let _ = req.respond(r);
                }
                Err(_) => { let _ = req.respond(Response::from_string("not found").with_status_code(404)); }
            }
        } else {
            let _ = req.respond(Response::from_string("not found").with_status_code(404));
        }
    }
}

const EXAMPLE_SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESMFold 1 + 2 — pure-Rust fp32</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--muted:#94a3b8;--accent:#e67e22;--accent2:#2e86c1;--ok:#22c55e;--err:#ef4444}
*{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:820px;margin:0 auto;padding:32px 20px}
.head{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:22px;margin:0} .sub{color:var(--muted);font-size:13px;margin:6px 0 22px}
.links{display:flex;gap:8px}
a.pill{display:inline-block;text-decoration:none;font-size:12px;font-weight:600;color:#dbeafe;background:#1d3a5f;border:1px solid #2e5a8a;border-radius:999px;padding:5px 12px}
.card{background:var(--card);border:1px solid #334155;border-radius:12px;padding:18px;margin-bottom:16px}
label{font-size:13px;color:var(--muted)} .lab{display:flex;justify-content:space-between;align-items:center}
.example{font-size:12px;color:#7dd3fc;cursor:pointer;text-decoration:underline}
textarea{width:100%;height:120px;margin-top:6px;background:#0b1220;color:var(--ink);border:1px solid #334155;border-radius:8px;padding:10px;font-family:ui-monospace,Consolas,monospace;font-size:13px;resize:vertical}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{background:var(--accent);color:#111;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer;font-size:14px}
button:disabled{opacity:.5;cursor:default}
.muted{color:var(--muted);font-size:12px}
input[type=number]{background:#0b1220;color:var(--ink);border:1px solid #334155;border-radius:6px;padding:6px 8px;width:100px}
.model{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--muted);margin:6px 0 2px;word-break:break-all}
.barwrap{height:10px;background:#0b1220;border-radius:6px;overflow:hidden;margin:10px 0}
.bar{height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .3s}
.phase{font-size:14px;margin-bottom:2px}
.log{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--muted);white-space:pre-wrap;height:230px;overflow-y:scroll;background:#0b1220;border:1px solid #334155;border-radius:8px;padding:10px;margin-top:10px}
.err{color:var(--err);font-size:13px;margin-top:8px}
.ok{color:var(--ok)} a.btn{display:inline-block;background:var(--ok);color:#04210f;text-decoration:none;border-radius:8px;padding:9px 16px;font-weight:600}
.stats{display:flex;gap:24px;margin-top:8px;font-size:14px}
fieldset{border:1px solid #334155;border-radius:8px;margin-top:12px;padding:10px}
legend{font-size:12px;color:var(--muted);padding:0 6px}
</style></head><body><div class="wrap">
<div class="head">
  <h1>ESMFold <span class="muted">1 + 2 — pure-Rust fp32</span></h1>
  <div class="links">
    <a class="pill" href="https://github.com/lingxusb/folding-everywhere" target="_blank" rel="noopener">Project ↗</a>
    <a class="pill" href="https://lingxusb.github.io" target="_blank" rel="noopener">Author ↗</a>
  </div>
</div>
<div class="sub">Fold a protein sequence to 3D all-atom coordinates. No Python, no GPU — pure Rust on the CPU.</div>

<div class="card">
  <fieldset>
    <legend>Model</legend>
    <div class="row">
      <label><input type="radio" name="model" value="1" checked> <b>ESMFold1</b> — ESM-2 3B, deterministic (~8 GB, faster)</label>
    </div>
    <div class="row">
      <label><input type="radio" name="model" value="2"> <b>ESMFold2</b> — ESM-C 6B + diffusion (~30 GB, slower)</label>
    </div>
    <div class="row" id="seedrow" style="display:none">
      <label>Seed <input id="seed" type="number" value="0" min="0"></label>
      <label>Loops <input id="loops" type="number" value="20" min="1" max="64"></label>
      <label>Sampling steps <input id="steps" type="number" value="68" min="1" max="200"></label>
    </div>
    <div class="row" id="e2note" style="display:none">
      <span class="muted">ESMFold2 is stochastic: same seed → identical structure (bit-exact to a PyTorch fp32 run at that seed). ESMFold1 has no seed. <b>Loops</b> (trunk refinement) and <b>sampling steps</b> (diffusion) default to the official release depth (20 / 68); lower them for a faster, lower-quality fold. Our benchmarks use 3 / 14 to make the bit-exact fp32 check fast. (A single diffusion sample is produced.)</span>
    </div>
  </fieldset>
  <div class="lab" style="margin-top:14px"><label for="seq">Protein sequence (one-letter codes)</label>
    <span class="example" id="ex">Load example (ubiquitin, 76 aa)</span></div>
  <textarea id="seq" placeholder="Paste a protein sequence, or click &quot;Load example&quot; above."></textarea>
  <div class="row">
    <button id="go">Fold protein</button>
    <span class="muted" id="hint">First run downloads the selected model (one time).</span>
  </div>
</div>

<div class="card" id="progcard" style="display:none">
  <div class="phase" id="phase">Starting…</div>
  <div class="barwrap"><div class="bar" id="bar"></div></div>
  <div class="muted" id="pct">0%</div>
  <div class="model" id="model"></div>
  <div class="log" id="log"></div>
  <div class="err" id="err"></div>
</div>

<div class="card" id="result" style="display:none">
  <div class="ok" style="font-weight:600;margin-bottom:6px">✓ Prediction complete</div>
  <div class="stats"><div>mean pLDDT: <b id="plddt">–</b></div><div>pTM: <b id="ptm">–</b></div></div>
  <div class="row"><a class="btn" id="dl" href="/api/pdb">Download PDB</a></div>
</div>

<script>
const $=id=>document.getElementById(id);
const EXAMPLE="__EXAMPLE__";
$('ex').onclick=()=>{$('seq').value=EXAMPLE;};
function model(){return document.querySelector('input[name=model]:checked').value;}
function syncSeed(){const v2 = model()==='2';
  $('seedrow').style.display = v2 ? 'flex':'none';
  $('e2note').style.display = v2 ? 'flex':'none';
  $('hint').textContent = v2 ? 'First run downloads ESMFold2 (~30 GB, one time); folding takes a few minutes (longer at the default 20/68 quality).' : 'First run downloads ESMFold1 (~8.4 GB, one time).';}
document.querySelectorAll('input[name=model]').forEach(r=>r.onchange=syncSeed); syncSeed();
$('go').onclick=async()=>{
  const seq=$('seq').value.trim();
  if(!seq){alert('Enter a sequence, or click "Load example".');return;}
  const seed=$('seed')?$('seed').value||'0':'0';
  const loops=$('loops')?$('loops').value||'20':'20';
  const steps=$('steps')?$('steps').value||'68':'68';
  $('go').disabled=true; $('result').style.display='none'; $('err').textContent=''; $('model').textContent='';
  $('progcard').style.display='block';
  await fetch('/api/fold',{method:'POST',body:model()+'\n'+seed+'\n'+loops+'\n'+steps+'\n'+seq});
  poll();
};
function poll(){
  fetch('/api/status').then(r=>r.json()).then(s=>{
    $('phase').textContent=s.phase;
    $('bar').style.width=(s.progress*100).toFixed(1)+'%';
    $('pct').textContent=(s.progress*100).toFixed(0)+'%';
    if(s.model) $('model').textContent='Model: '+s.model;
    const el=$('log'); const nearBottom=(el.scrollHeight-el.scrollTop-el.clientHeight)<40;
    el.textContent=s.log.join('\n'); if(nearBottom) el.scrollTop=el.scrollHeight;
    if(s.error){$('err').textContent='Error: '+s.error; $('go').disabled=false; return;}
    if(s.done){$('plddt').textContent=s.plddt.toFixed(3); $('ptm').textContent=s.ptm.toFixed(3);
      $('result').style.display='block'; $('go').disabled=false; return;}
    setTimeout(poll,700);
  }).catch(()=>setTimeout(poll,1000));
}
</script>
</div></body></html>"#;
