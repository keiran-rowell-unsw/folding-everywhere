//! ESMFold v1 (pure-Rust fp32) CLI.
//!
//! Usage:
//!   fold --seq MSID... [-o out.pdb] [--name NAME]
//!   fold --fasta in.fasta [-o out.pdb]
//! Weights:  --weights PATH | $ESMFOLD_WEIGHTS | HF cache
//! Constants: --constants PATH | $ESMFOLD_CONSTANTS

use esmfold::constants::Constants;
use esmfold::pdb::to_pdb;
use esmfold::pipeline::fold;
use esmfold::weights::Weights;
use std::time::Instant;

fn find_weights(arg: Option<String>) -> Weights {
    let path = arg
        .or_else(|| std::env::var("ESMFOLD_WEIGHTS").ok())
        .or_else(|| {
            let base = format!("{}/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots", std::env::var("HOME").unwrap());
            std::fs::read_dir(&base).ok().and_then(|rd| {
                rd.flatten().map(|e| e.path().join("model.safetensors")).find(|p| p.exists()).map(|p| p.to_string_lossy().into_owned())
            })
        })
        .expect("weights not found: pass --weights or set ESMFOLD_WEIGHTS");
    eprintln!("weights: {path}");
    Weights::open(&path).expect("open weights")
}

fn read_fasta(path: &str) -> Vec<(String, String)> {
    let txt = std::fs::read_to_string(path).expect("read fasta");
    let mut out = Vec::new();
    let mut name = String::new();
    let mut seq = String::new();
    for line in txt.lines() {
        if let Some(h) = line.strip_prefix('>') {
            if !name.is_empty() {
                out.push((name.clone(), seq.clone()));
            }
            name = h.split_whitespace().next().unwrap_or("seq").to_string();
            seq.clear();
        } else {
            seq.push_str(line.trim());
        }
    }
    if !name.is_empty() {
        out.push((name, seq));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seq = None;
    let mut fasta = None;
    let mut out = None;
    let mut name = "protein".to_string();
    let mut weights_arg = None;
    let mut consts_arg = None;
    let mut dump = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seq" => { seq = Some(args[i + 1].clone()); i += 2; }
            "--fasta" => { fasta = Some(args[i + 1].clone()); i += 2; }
            "-o" | "--out" => { out = Some(args[i + 1].clone()); i += 2; }
            "--name" => { name = args[i + 1].clone(); i += 2; }
            "--weights" => { weights_arg = Some(args[i + 1].clone()); i += 2; }
            "--constants" => { consts_arg = Some(args[i + 1].clone()); i += 2; }
            "--dump" => { dump = Some(args[i + 1].clone()); i += 2; }
            other => { eprintln!("unknown arg {other}"); i += 1; }
        }
    }

    let w = find_weights(weights_arg);
    let consts = match consts_arg.or_else(|| std::env::var("ESMFOLD_CONSTANTS").ok()) {
        Some(p) => Constants::load(&p),
        None => Constants::embedded(), // self-contained: no external constants file
    };

    let jobs: Vec<(String, String)> = if let Some(s) = seq {
        vec![(name.clone(), s)]
    } else if let Some(f) = fasta {
        read_fasta(&f)
    } else {
        eprintln!("provide --seq or --fasta");
        std::process::exit(1);
    };

    for (nm, s) in jobs {
        let t0 = Instant::now();
        let o = fold(&w, &consts, &s);
        let dt = t0.elapsed().as_secs_f64();
        let plddt_mean = o.plddt.data.iter().sum::<f32>() / o.plddt.data.len() as f32;
        eprintln!("{nm}: L={} time={dt:.1}s plddt_mean={plddt_mean:.3} ptm={:.3}", o.l, o.ptm);
        let pdb = to_pdb(&o.atom37.data, &o.plddt.data, &o.aatype, &consts, o.l);
        let path = out.clone().unwrap_or_else(|| format!("{nm}.pdb"));
        std::fs::write(&path, pdb).expect("write pdb");
        eprintln!("wrote {path}");
        if let Some(d) = &dump {
            // raw little-endian f32 atom37 [L*37*3] + meta json
            let mut bytes = Vec::with_capacity(o.atom37.data.len() * 4);
            for &v in &o.atom37.data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(format!("{d}.atom37.f32"), bytes).expect("write dump");
            let meta = format!(
                "{{\"name\":\"{nm}\",\"L\":{},\"ptm\":{:.6},\"plddt_mean\":{:.6},\"time_s\":{:.3}}}",
                o.l, o.ptm, plddt_mean, dt
            );
            std::fs::write(format!("{d}.meta.json"), meta).expect("write meta");
        }
    }
}
