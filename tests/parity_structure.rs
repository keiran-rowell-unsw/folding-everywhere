//! P5: structure-module parity. Feeds the reference structure-module inputs
//! (trunk2sm outputs) and compares per-iteration atom14 positions, states, frames.

use esmfold::constants::Constants;
use esmfold::parity::compare;
use esmfold::structure::structure_module;
use esmfold::weights::Weights;

fn model_weights() -> Option<Weights> {
    if let Ok(p) = std::env::var("ESMFOLD_WEIGHTS") {
        return Weights::open(&p).ok();
    }
    let base = format!("{}/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots", std::env::var("HOME").unwrap());
    for e in std::fs::read_dir(&base).ok()?.flatten() {
        let p = e.path().join("model.safetensors");
        if p.exists() {
            return Weights::open(p.to_str().unwrap()).ok();
        }
    }
    None
}
fn fx(name: &str) -> Weights {
    let p = format!("{}/fixtures/trunk/flgM/{}.safetensors", env!("CARGO_MANIFEST_DIR"), name);
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}"))
}

#[test]
fn structure_module_parity() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: weights not found");
            return;
        }
    };
    let consts = Constants::load(&format!("{}/fixtures/constants/residue_constants.safetensors", env!("CARGO_MANIFEST_DIR")));
    let smin = fx("sm_inputs");
    let inputs = fx("inputs");
    let single = smin.get("single"); // [L,384]
    let pair = smin.get("pair"); // [L,L,128]
    let n = single.shape[0];
    let aatype: Vec<usize> = inputs.get("aatype").data.iter().map(|&x| x.round() as usize).collect();
    println!("L={n}");

    let out = structure_module(&single, &pair, &aatype, &w, &consts, n);
    assert_eq!(out.len(), 8);

    let refs = fx("structure");
    let frames_ref = refs.get("frames"); // [8,L,7]
    let pos_ref = refs.get("positions"); // [8,L,14,3]
    let states_ref = refs.get("states"); // [8,L,384]

    for it in 0..8 {
        let ps = compare(&out[it].positions, &pos_ref.data[it * n * 14 * 3..(it + 1) * n * 14 * 3]);
        if it == 0 || it == 7 {
            let fs = compare(&out[it].frames7, &frames_ref.data[it * n * 7..(it + 1) * n * 7]);
            let st = compare(&out[it].states, &states_ref.data[it * n * C_S..(it + 1) * n * C_S]);
            println!("iter{it} pos[{}]  frames[{}]  states[{}]", ps.summary(), fs.summary(), st.summary());
        } else {
            println!("iter{it} pos[{}]", ps.summary());
        }
        assert!(!ps.any_nan, "iter{it} NaN");
    }
    let pf = compare(&out[7].positions, &pos_ref.data[7 * n * 14 * 3..8 * n * 14 * 3]);
    let sf = compare(&out[7].states, &states_ref.data[7 * n * C_S..8 * n * C_S]);
    println!("FINAL positions {}", pf.summary());
    assert!(pf.cosine > 1.0 - 1e-5 && pf.max_abs < 1e-2, "final positions: {}", pf.summary());
    assert!(sf.cosine > 1.0 - 1e-5, "final states: {}", sf.summary());
}

const C_S: usize = 384;
