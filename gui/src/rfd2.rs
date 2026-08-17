//! RFdiffusion2 tab.
//!
//! The job logic here is lifted unchanged from the `rfd2_gui` app of
//! `rfdiffusion2-rs`: same checkpoint URL and resumable download, same EMA
//! state-dict selection, same IGSO(3) tables, same ligand-topology handling, so
//! a design run through this tab is byte-identical to the standalone app's.
//! Only the plumbing (state ownership, the `/api/rfd2/*` route prefix) is new.
//!
//! Everything the model needs except the checkpoint is compiled in: the chemical
//! database, the AF2 frame tables, the SE(3) Clebsch-Gordan bases (in the `rfd2`
//! crate), the IGSO(3) noise tables, and the ligand-topology library.
//!
//! **The one real limitation**, stated in the UI rather than hidden: ligand bond
//! orders and aromaticity are *perceived by OpenBabel from 3D coordinates* in the
//! reference pipeline (3 of 4 demo inputs carry no CONECT records), so they are
//! an input to this port, not something it computes. A PDB whose ligands are not
//! in the bundled set therefore cannot be run here without one Python step.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use rfd2::design::{run_design_with, DesignConfig};
use rfd2::ligand::LigandSet;
use rfd2::model::rf::{Arch, RoseTTAFold};
use rfd2::nn::Params;
use rfd2::noiser::Igso3;
use rfd2::weights::Weights;

use crate::home;

// ---- compiled-in data -------------------------------------------------------
static IGSO3: &[u8] = include_bytes!("../data/igso3.safetensors");
/// Every ligand from the 41 benchmark inputs, one entry per ligand rather than
/// per input file. `atom_frames` are deliberately absent: they are a function of
/// the whole ligand block's bond matrix in the USER's atom order, so they are
/// recomputed in Rust (`ligand::recompute_frames`, verified on 1178 atoms).
static LIGAND_LIBRARY: &[u8] = include_bytes!("../data/ligand_library.safetensors");
pub static LIBRARY_INDEX: &str = include_str!("../data/ligand_library.json");
pub static EXAMPLE_PDB: &str = include_str!("../data/example_M0584_1ldm.pdb");

const CKPT_URL: &str = "https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt";
const CKPT_BYTES: u64 = 1_338_843_322;

#[derive(Default, Clone)]
pub struct State {
    pub busy: bool,
    phase: String,
    progress: f32,
    log: Vec<String>,
    pub error: Option<String>,
    done: bool,
    n_done: usize,
    n_total: usize,
    length: usize,
    secs: f32,
    pdb_ready: bool,
}

fn app_dir() -> PathBuf {
    std::env::var("RFD2_HOME").map(PathBuf::from).unwrap_or_else(|_| home().join(".rfdiffusion2"))
}
fn ckpt_path() -> PathBuf { app_dir().join("RFD_173.pt") }
pub fn out_dir() -> PathBuf { app_dir().join("out") }

fn mmss(secs: f32) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 { format!("{}h{:02}m", s / 3600, (s % 3600) / 60) }
    else { format!("{}m{:02}s", s / 60, s % 60) }
}

fn set<F: FnOnce(&mut State)>(st: &Arc<Mutex<State>>, f: F) { f(&mut st.lock().unwrap()); }
fn logmsg(st: &Arc<Mutex<State>>, m: &str) {
    // No cap and no trimming: every row must survive for the whole run. The
    // client fetches only rows it has not seen (`/api/rfd2/status?from=N`), so an
    // unbounded log costs nothing per poll.
    st.lock().unwrap().log.push(m.to_string());
}

/// Download the checkpoint with a live progress bar, resuming a partial file.
fn fetch_ckpt(st: &Arc<Mutex<State>>) -> Result<PathBuf, String> {
    let dest = ckpt_path();
    if dest.exists() && std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) == CKPT_BYTES {
        logmsg(st, "Checkpoint already downloaded.");
        return Ok(dest);
    }
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = dest.with_extension("part");
    set(st, |s| s.phase = "Downloading RFdiffusion2 weights (1.34 GB, one time)…".into());
    logmsg(st, &format!("Fetching {CKPT_URL}"));
    logmsg(st, &format!("Caching in {}", dest.display()));

    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    // `-C -` resumes a partial file, so a dropped connection costs only what was
    // left. Timeouts matter: a blocked route HANGS rather than failing, and
    // without them the bar would sit at 0 % forever with no error.
    let mut child = Command::new(curl)
        .args(["-L", "--fail", "--silent", "--show-error", "-C", "-",
               "--connect-timeout", "25", "--speed-limit", "1024", "--speed-time", "60", "-o"])
        .arg(&part).arg(CKPT_URL)
        .spawn()
        .map_err(|e| format!("could not start curl ({e}). curl ships with Windows 10+, macOS and Linux."))?;
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                if !status.success() {
                    return Err(format!("download failed (curl exit {:?}). \
                        Partial file kept at {} — press Design again to resume.",
                        status.code(), part.display()));
                }
                break;
            }
            None => {
                let got = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                let f = got as f32 / CKPT_BYTES as f32;
                set(st, |s| {
                    s.progress = f * 0.55;
                    s.phase = format!("Downloading weights… {:.2} / 1.34 GB ({:.0} %)",
                                      got as f64 / 1e9, f * 100.0);
                });
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
    logmsg(st, "Download complete.");
    Ok(dest)
}

pub struct Job {
    pdb_text: String,
    ligands: String,
    contig: String,
    length: String,
    big_t: usize,
    num_designs: usize,
    seed: u64,
    self_cond: bool,
    custom_sidecar: String,
}

pub fn parse_job(v: &serde_json::Value) -> Job {
    let g = |k: &str| v[k].as_str().unwrap_or("").to_string();
    Job {
        pdb_text: g("pdb"),
        ligands: g("ligands"),
        contig: g("contig"),
        length: g("length"),
        big_t: v["T"].as_u64().unwrap_or(100) as usize,
        num_designs: v["n"].as_u64().unwrap_or(1).max(1) as usize,
        seed: v["seed"].as_u64().unwrap_or(0),
        self_cond: v["self_cond"].as_bool().unwrap_or(false),
        custom_sidecar: g("custom_sidecar"),
    }
}

pub fn run(st: &Arc<Mutex<State>>, job: Job) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let ckpt = fetch_ckpt(st)?;

    set(st, |s| { s.phase = "Loading model…".into(); s.progress = 0.58; });
    logmsg(st, "Reading the checkpoint (7208 tensors, 82.9 M parameters)…");
    let w = Weights::open(&ckpt.to_string_lossy()).map_err(|e| format!("read checkpoint: {e}"))?;
    // The official .pt holds BOTH state dicts (7208 EMA + 7208 final, 14419 names
    // in all), so its keys are prefixed. `inference.state_dict_to_load` is
    // `model_state_dict` — the EMA weights — which is what the reference loads.
    // A converted safetensors export has the bare names, so accept either.
    let root = if w.has("model_state_dict.model.latent_emb.emb.weight") {
        logmsg(st, "Checkpoint holds both state dicts; using the EMA `model_state_dict`.");
        "model_state_dict.model"
    } else {
        "model"
    };
    let model = RoseTTAFold::load(&Params::root(&w, root), Arch::rfd173());

    // IGSO(3) tables: the last reachable CDF row is sigma = 1.5, which is what
    // `_corrupt_rotmats_multi_t` hard-codes.
    let nf = Weights::from_static(IGSO3).map_err(|e| format!("igso3: {e}"))?;
    let cdf = nf.get("igso3.cdf");
    let n = cdf.shape[1];
    let igso3 = Igso3::new(nf.get("igso3.omega_grid").data,
                           cdf.data[(cdf.shape[0] - 1) * n..].to_vec());

    // Ligand topology is an INPUT, not a computation — see the module header.
    let names: Vec<String> = job.ligands.split(',')
        .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    // A user-supplied sidecar wins; otherwise the built-in 56-ligand library.
    let src = if !job.custom_sidecar.trim().is_empty() {
        let p = job.custom_sidecar.trim().to_string();
        if !std::path::Path::new(&p).exists() {
            return Err(format!("sidecar file not found: {p}"));
        }
        logmsg(st, &format!("Using your ligand topology: {p}"));
        p
    } else {
        let tmp = std::env::temp_dir().join(format!("rfd2_lib_{}.safetensors", std::process::id()));
        std::fs::write(&tmp, LIGAND_LIBRARY).map_err(|e| e.to_string())?;
        logmsg(st, &format!("Ligand topology from the built-in library: {}", job.ligands));
        tmp.to_string_lossy().to_string()
    };
    let mut topo = LigandSet::load(&src, &names).map_err(|e| format!(
        "ligand topology: {e}\n\nBond orders and aromaticity are perceived by OpenBabel from 3D \
         coordinates in the reference pipeline, so they are an INPUT here, not something this port \
         computes. Choose the ligand from the built-in library, or build a topology for it once \
         with  rfdiffusion2/python/gen_ligand_bonds.py <your.pdb> <LIG,LIG>  and give that file's path."))?;
    // A sidecar is consumed POSITIONALLY, so if it was not built from this very
    // file its atom order may differ. Align by atom name (a no-op when the order
    // already matches); refuses on a different atom set rather than guessing.
    match topo.align_to_pdb(&job.pdb_text) {
        Ok(unnamed) if !unnamed.is_empty() => logmsg(st, &format!(
            "note: topology for {unnamed:?} has no atom names; assuming this file's order")),
        Ok(_) => {}
        Err(e) => return Err(format!("{e}")),
    }

    let cfg = DesignConfig {
        input_pdb: String::new(),
        ligands: names,
        contigs: job.contig.clone(),
        big_t: job.big_t,
        final_step: 1,
        seed_offset: job.seed,
        deterministic: true,
        rots_exp_rate: 10,
        str_self_cond: job.self_cond,
        partial_t: None,
        length: if job.length.trim().is_empty() { None } else { Some(job.length.trim().into()) },
    };

    std::fs::create_dir_all(out_dir()).map_err(|e| e.to_string())?;
    set(st, |s| { s.n_total = job.num_designs; s.phase = "Designing…".into(); });
    for i in 0..job.num_designs {
        // 1-based everywhere the user can see it; `i` stays 0-based internally
        // because it is the seed offset and must match the reference.
        let d = i + 1;
        logmsg(st, &format!("── Design {d} of {} — {} denoising steps, seed {} ──",
                            job.num_designs, job.big_t, job.seed + i as u64));
        let t_des = std::time::Instant::now();
        let st_cb = st.clone();
        let (nd, ndes) = (d, job.num_designs);
        let out = run_design_with(&model, &cfg, &job.pdb_text, &topo, &igso3, i,
            move |it, t, total| {
                let el = t_des.elapsed().as_secs_f32();
                let per = el / it as f32;
                let left = per * (total - it) as f32;
                logmsg(&st_cb, &format!(
                    "  step {it:>3}/{total}   t = {t:<4}  {per:5.1} s/step   elapsed {}   remaining ~{}", mmss(el), mmss(left)));
                set(&st_cb, |s| {
                    // designs share the 0.60-1.00 band; steps fill each design's slice
                    let base = (nd - 1) as f32 / ndes as f32;
                    let frac = base + (it as f32 / total as f32) / ndes as f32;
                    s.progress = 0.6 + 0.4 * frac;
                    s.phase = format!("Design {nd} of {ndes} — step {it}/{total} (t = {t}),                                        ~{} remaining", mmss(left));
                });
            })
            .map_err(|e| format!("{e}"))?;
        let path = out_dir().join(format!("design_{d}.pdb"));
        std::fs::write(&path, out.pdb.as_bytes()).map_err(|e| e.to_string())?;
        let l = out.indep.len();
        set(st, |s| {
            s.n_done = d;
            s.length = l;
            s.progress = 0.6 + 0.4 * (d as f32 / ndes as f32);
            s.pdb_ready = true;
        });
        logmsg(st, &format!("  ✓ design {d} complete in {} — wrote {}",
                            mmss(t_des.elapsed().as_secs_f32()), path.display()));
    }
    set(st, |s| {
        s.done = true;
        s.phase = "Complete".into();
        s.progress = 1.0;
        s.secs = t0.elapsed().as_secs_f32();
    });
    Ok(())
}

/// Run in a worker thread with the panic guard the standalone app used: a panic
/// deep in the model (a missing weight key, say) would otherwise kill the thread
/// silently and leave the page showing "Loading model…" forever.
pub fn run_guarded(st: Arc<Mutex<State>>, job: Job) {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&st, job)));
    let msg = match r {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(p) => Some(format!("internal error: {}",
            p.downcast_ref::<String>().cloned().unwrap_or_else(||
                p.downcast_ref::<&str>().map(|s| s.to_string())
                 .unwrap_or_else(|| "panic".into())))),
    };
    if let Some(e) = msg {
        set(&st, |s| { s.error = Some(e.clone()); s.phase = "Failed".into(); });
        logmsg(&st, &format!("ERROR: {e}"));
    }
    set(&st, |s| s.busy = false);
}

pub fn json(st: &State, from: usize) -> String {
    let esc = |s: &str| serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    // Only the rows after `from`. Nothing is ever dropped server-side; the client
    // appends, so the window keeps every line of the run.
    let from = from.min(st.log.len());
    let log: Vec<String> = st.log[from..].iter().map(|l| esc(l)).collect();
    format!(
        "{{\"busy\":{},\"phase\":{},\"progress\":{:.4},\"log\":[{}],\"log_from\":{},\
         \"log_total\":{},\"error\":{},\"done\":{},\
         \"n_done\":{},\"n_total\":{},\"length\":{},\"secs\":{:.1},\"pdb_ready\":{}}}",
        st.busy, esc(&st.phase), st.progress, log.join(","), from, st.log.len(),
        st.error.as_deref().map(esc).unwrap_or_else(|| "null".into()),
        st.done, st.n_done, st.n_total, st.length, st.secs, st.pdb_ready)
}

pub fn starting() -> State { State { busy: true, phase: "Starting…".into(), ..Default::default() } }

pub fn idle() -> State { State { phase: "Idle".into(), ..Default::default() } }

pub fn refuse(st: &Arc<Mutex<State>>, msg: &str) {
    let mut s = st.lock().unwrap();
    s.busy = false;
    s.error = Some(msg.to_string());
    s.phase = "Blocked".into();
}
