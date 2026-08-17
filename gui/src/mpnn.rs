//! ProteinMPNN tab.
//!
//! The job logic here is lifted unchanged from the `mpnn_gui` app of
//! `proteinmpnn-rs`: same featurisation, same MT19937 draw accounting, same
//! decoding-order construction, so the same seed still yields the same sequences
//! as `protein_mpnn_run.py`. Only the plumbing (state ownership, the
//! `/api/mpnn/*` route prefix) is new.
//!
//! Unlike the other two tabs there is **no download step at all**: all four
//! published ProteinMPNN checkpoints are compiled into this executable.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use proteinmpnn::featurize::featurize;
use proteinmpnn::model::{score, ProteinMpnn};
use proteinmpnn::pdb::parse_pdb;
use proteinmpnn::rng::{randn, Mt19937};
use proteinmpnn::weights::Weights;
use proteinmpnn::{embedded, idx_to_aa, ALPHABET};

/// The example backbone the tab's "Load example" button loads: PDB 6EKB, 62
/// residues, one of the 20 proteins in `proteinmpnn/results/metrics.csv`. Its
/// native-sequence score there is 1.8975 for both PyTorch and this port, so the
/// example doubles as a check that the tab is wired up correctly.
pub static EXAMPLE_PDB: &str = include_str!("../data/example_6EKB.pdb");

const MAX_LEN: usize = 2000;
const MAX_SEQS: usize = 64;

#[derive(Default, Clone)]
pub struct Design {
    seq: String,
    score: f32,
    recovery: f32,
    temp: f64,
}

#[derive(Default)]
pub struct State {
    pub busy: bool,
    done: bool,
    phase: String,
    progress: f32,
    error: Option<String>,
    log: Vec<String>,
    native_seq: String,
    native_score: f32,
    chains: String,
    length: usize,
    results: Vec<Design>,
    elapsed: f64,
}

fn set<F: FnOnce(&mut State)>(st: &Arc<Mutex<State>>, f: F) {
    f(&mut st.lock().unwrap());
}

fn logmsg(st: &Arc<Mutex<State>>, m: &str) {
    let mut s = st.lock().unwrap();
    s.log.push(m.to_string());
    if s.log.len() > 400 {
        s.log.remove(0);
    }
}

// ---------------------------------------------------------------------------
// the design job
// ---------------------------------------------------------------------------

pub struct Job {
    pdb_text: String,
    model: String,
    temps: Vec<f64>,
    num_seq: usize,
    seed: u64,
    chains: Vec<char>,
    omit: String,
}

pub fn parse_job(v: &serde_json::Value) -> Job {
    let temps: Vec<f64> = v["temps"]
        .as_str()
        .unwrap_or("0.1")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .filter(|t: &f64| *t > 0.0 && *t <= 5.0)
        .collect();
    Job {
        pdb_text: v["pdb"].as_str().unwrap_or("").to_string(),
        model: v["model"].as_str().unwrap_or(embedded::DEFAULT_MODEL).to_string(),
        temps: if temps.is_empty() { vec![0.1] } else { temps },
        num_seq: (v["num_seq"].as_u64().unwrap_or(4) as usize).clamp(1, MAX_SEQS),
        seed: v["seed"].as_u64().unwrap_or(37),
        chains: v["chains"]
            .as_str()
            .unwrap_or("")
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect(),
        omit: v["omit"].as_str().unwrap_or("X").to_string(),
    }
}

pub fn job_has_pdb(job: &Job) -> bool { !job.pdb_text.trim().is_empty() }

pub fn run_job(st: Arc<Mutex<State>>, job: Job, cancel: Arc<AtomicBool>) {
    let t0 = std::time::Instant::now();
    let res = (|| -> Result<(), String> {
        // The parser is file-based; stage the upload in a temp file.
        let tmp = std::env::temp_dir().join(format!("mpnn_gui_{}.pdb", std::process::id()));
        std::fs::write(&tmp, job.pdb_text.as_bytes()).map_err(|e| e.to_string())?;
        set(&st, |s| {
            s.phase = "Parsing backbone…".into();
            s.progress = 0.05;
        });
        let structure = parse_pdb(&tmp.to_string_lossy()).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&tmp);

        let all: Vec<char> = structure.chain_ids();
        if all.is_empty() {
            return Err("No protein chains with N/CA/C/O backbone atoms found in that file.".into());
        }
        let designed: Vec<char> = if job.chains.is_empty() {
            all.clone()
        } else {
            job.chains.iter().copied().filter(|c| all.contains(c)).collect()
        };
        if designed.is_empty() {
            return Err(format!(
                "None of the requested chains are in this structure (it has {}).",
                all.iter().collect::<String>()
            ));
        }
        let fixed: Vec<char> = all.iter().copied().filter(|c| !designed.contains(c)).collect();
        let b = featurize(&structure, &designed, &fixed);
        if b.l == 0 {
            return Err("Structure has no usable residues.".into());
        }
        if b.l > MAX_LEN {
            return Err(format!("Structure is {} residues; this build caps at {MAX_LEN}.", b.l));
        }
        logmsg(&st, &format!(
            "{}: {} residues, chains [{}], designing [{}]",
            structure.name, b.l,
            all.iter().collect::<String>(),
            designed.iter().collect::<String>()
        ));

        let bytes = embedded::by_name(&job.model)
            .ok_or_else(|| format!("unknown model {}", job.model))?;
        let w = Weights::from_static(bytes).map_err(|e| e.to_string())?;
        let model = ProteinMpnn::load(&w, 48, 3, 3);

        set(&st, |s| {
            s.phase = "Encoding backbone…".into();
            s.progress = 0.12;
            s.length = b.l;
            s.chains = all.iter().collect();
        });

        let mut gen = Mt19937::new(job.seed);
        gen.skip(proteinmpnn::model::torch_init_draws(&w));

        let enc = model.encode(&b);
        let r1 = randn(&mut gen, b.l);
        let order1 = ProteinMpnn::decoding_order(&r1, &b.design_mask());
        let lp = model.forward_with(&enc, &b, &b.s, &order1);
        let native_score = score(&b.s, &lp, &b.mask);
        set(&st, |s| {
            s.native_seq = b.seq.clone();
            s.native_score = native_score;
            s.progress = 0.2;
        });
        logmsg(&st, &format!("native sequence score {native_score:.4}"));

        let mut omit = [0.0f64; 21];
        for c in job.omit.bytes() {
            if let Some(i) = ALPHABET.bytes().position(|x| x == c.to_ascii_uppercase()) {
                omit[i] = 1.0;
            }
        }
        let bias = [0.0f64; 21];
        let denom = b.mask.iter().filter(|&&m| m > 0.0).count().max(1);

        let total = job.temps.len() * job.num_seq;
        let mut n = 0usize;
        for &temp in &job.temps {
            for _ in 0..job.num_seq {
                if cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let r2 = randn(&mut gen, b.l);
                let order2 = ProteinMpnn::decoding_order(&r2, &b.design_mask());
                let out = model.sample_with(&enc, &b, &mut gen, &order2, temp, &omit, &bias);
                let lp = model.forward_with(&enc, &b, &out.s, &order2);
                let sc = score(&out.s, &lp, &b.mask);
                let seq: String = out.s.iter().map(|&i| idx_to_aa(i as usize)).collect();
                let rec = (0..b.l).filter(|&i| b.mask[i] > 0.0 && b.s[i] == out.s[i]).count();
                n += 1;
                let d = Design {
                    seq,
                    score: sc,
                    recovery: rec as f32 / denom as f32,
                    temp,
                };
                set(&st, |s| {
                    s.results.push(d);
                    s.progress = 0.2 + 0.8 * (n as f32 / total as f32);
                    s.phase = format!("Designing sequence {n} of {total}…");
                });
            }
        }
        Ok(())
    })();

    let dt = t0.elapsed().as_secs_f64();
    set(&st, |s| {
        s.busy = false;
        s.elapsed = dt;
        match res {
            Ok(()) => {
                s.done = true;
                s.progress = 1.0;
                s.phase = format!("Done — {} sequences in {:.1}s", s.results.len(), dt);
            }
            Err(e) => {
                s.error = Some(e);
                s.phase = "Failed".into();
            }
        }
    });
}

// ---------------------------------------------------------------------------
// wire format
// ---------------------------------------------------------------------------

pub fn state_json(s: &State) -> String {
    let results: Vec<serde_json::Value> = s
        .results
        .iter()
        .enumerate()
        .map(|(i, d)| {
            serde_json::json!({
                "n": i + 1, "seq": d.seq, "score": d.score,
                "recovery": d.recovery, "temp": d.temp
            })
        })
        .collect();
    serde_json::json!({
        "busy": s.busy, "done": s.done, "phase": s.phase, "progress": s.progress,
        "error": s.error, "log": s.log, "nativeSeq": s.native_seq,
        "nativeScore": s.native_score, "chains": s.chains, "length": s.length,
        "elapsed": s.elapsed, "results": results,
    })
    .to_string()
}

pub fn fasta_of(s: &State) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        ">native, score={:.4}, length={}\n{}\n",
        s.native_score, s.length, s.native_seq
    ));
    for (i, d) in s.results.iter().enumerate() {
        out.push_str(&format!(
            ">T={}, sample={}, score={:.4}, seq_recovery={:.4}\n{}\n",
            d.temp, i + 1, d.score, d.recovery, d.seq
        ));
    }
    out
}

pub fn starting() -> State { State { busy: true, phase: "Starting…".into(), ..Default::default() } }

pub fn refuse(st: &Arc<Mutex<State>>, msg: &str) {
    let mut s = st.lock().unwrap();
    s.busy = false;
    s.error = Some(msg.to_string());
    s.phase = "Blocked".into();
}

/// The models compiled into this executable, for the page's model picker.
pub fn model_names() -> Vec<&'static str> { embedded::names() }
