//! P4: folding-trunk block-stack parity (isolated from recycling/structure).
//! Feeds the captured final-recycle block-0 input through all 48 blocks and
//! compares each block output to the reference.

use esmfold::parity::compare;
use esmfold::trunk;
use esmfold::weights::Weights;

fn model_weights() -> Option<Weights> {
    if let Ok(p) = std::env::var("ESMFOLD_WEIGHTS") {
        return Weights::open(&p).ok();
    }
    let base = format!(
        "{}/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots",
        std::env::var("HOME").unwrap()
    );
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
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e} (run python/ref_trunk.py)"))
}

#[test]
fn trunk_relative_position() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: weights not found");
            return;
        }
    };
    let rp = fx("relpos_final").get("relpos");
    let l = rp.shape[0];
    let got = trunk::relative_position(l, &w);
    let s = compare(&got.data, &rp.data);
    println!("relative_position {}", s.summary());
    assert!(s.max_abs < 1e-6, "relpos {}", s.summary());
}

#[test]
fn trunk_block_stack() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: weights not found");
            return;
        }
    };
    let inp = fx("blk0_input_final");
    let blocks = fx("blocks_final_recycle");
    let mut s = inp.get("s_in");
    let mut z = inp.get("z_in");
    let l = s.shape[0];
    println!("L={l}");

    let mut worst_s = (0.0f32, 1.0f64);
    let mut worst_z = (0.0f32, 1.0f64);
    for idx in 0..trunk::NUM_BLOCKS {
        let (ns, nz) = trunk::block(&s, &z, &w, idx, l);
        s = ns;
        z = nz;
        let ws = s.amax().max(1e-6);
        let wz = z.amax().max(1e-6);
        let ss = compare(&s.data, &blocks.get(&format!("s_{idx:02}")).data);
        let sz = compare(&z.data, &blocks.get(&format!("z_{idx:02}")).data);
        if idx % 8 == 0 || idx == trunk::NUM_BLOCKS - 1 {
            println!("block {idx:02} s[{}] rel={:.2e}  z[{}] rel={:.2e}", ss.summary(), ss.max_abs / ws, sz.summary(), sz.max_abs / wz);
        }
        assert!(!ss.any_nan && !sz.any_nan, "block {idx} NaN");
        worst_s.0 = worst_s.0.max(ss.max_abs / ws);
        worst_s.1 = worst_s.1.min(ss.cosine);
        worst_z.0 = worst_z.0.max(sz.max_abs / wz);
        worst_z.1 = worst_z.1.min(sz.cosine);
    }
    println!("worst s: rel={:.3e} cos={:.10}; worst z: rel={:.3e} cos={:.10}", worst_s.0, worst_s.1, worst_z.0, worst_z.1);
    assert!(worst_s.1 > 1.0 - 1e-5 && worst_z.1 > 1.0 - 1e-5, "trunk cosine too low");
    assert!(worst_s.0 < 1e-3 && worst_z.0 < 1e-3, "trunk relative error too large");
}
