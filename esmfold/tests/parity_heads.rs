//! P3 + P6 heads: validate LM->trunk glue (s_s_0) and all output heads against
//! reference fixtures, using reference intermediate tensors (fast, no full run).

use esmfold::constants::Constants;
use esmfold::heads;
use esmfold::parity::compare;
use esmfold::pipeline::lm_to_trunk;
use esmfold::tensor::Tensor;
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
fn tfx(name: &str) -> Weights {
    Weights::open(&format!("{}/fixtures/trunk/flgM/{}.safetensors", env!("CARGO_MANIFEST_DIR"), name)).unwrap()
}

#[test]
fn lm_to_trunk_and_heads() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: weights not found");
            return;
        }
    };
    let consts = Constants::load(&format!("{}/fixtures/constants/residue_constants.safetensors", env!("CARGO_MANIFEST_DIR")));

    // --- P3: s_s_0 from the 37 LM states ---
    let lm = Weights::open(&format!("{}/fixtures/lm/flgM/esm_states.safetensors", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let states: Vec<Tensor> = (0..37).map(|i| lm.get(&format!("state_{i:02}"))).collect();
    let inputs = tfx("inputs");
    let aatype: Vec<usize> = inputs.get("aatype").data.iter().map(|&x| x.round() as usize).collect();
    let l = aatype.len();
    let s_s0 = lm_to_trunk(&states, &aatype, &w);
    let ss = compare(&s_s0.data, &inputs.get("s_s_0").data);
    println!("s_s_0 {}", ss.summary());
    // post-ReLU: a few sign-flips on tiny elements; cosine + mean are the honest metrics
    assert!(ss.cosine > 1.0 - 1e-6 && ss.mean_abs < 1e-4, "s_s_0 {}", ss.summary());

    // --- P6 heads, using reference final s_z / states / positions ---
    let final_fx = tfx("final");
    let struct_fx = tfx("structure");
    let heads_fx = tfx("heads");
    let s_z = final_fx.get("s_z");
    let states_final = Tensor::new(struct_fx.get("states").data[7 * l * 384..8 * l * 384].to_vec(), vec![l, 384]);
    let positions_final = struct_fx.get("positions").data[7 * l * 14 * 3..8 * l * 14 * 3].to_vec();

    let disto = heads::distogram(&s_z, &w);
    let d = compare(&disto.data, &heads_fx.get("distogram_logits").data);
    println!("distogram {}", d.summary());
    assert!(d.cosine > 1.0 - 1e-6, "distogram {}", d.summary());

    let pl = heads::plddt(&states_final, &w);
    let p = compare(&pl.data, &heads_fx.get("plddt").data);
    println!("plddt {}  (mean {:.4})", p.summary(), pl.data.iter().sum::<f32>() / pl.data.len() as f32);
    assert!(p.cosine > 1.0 - 1e-6 && p.max_abs < 1e-3, "plddt {}", p.summary());

    let ptm_logits = heads::ptm_logits(&s_z, &w);
    let ptm = heads::compute_ptm(&ptm_logits, l);
    let ptm_ref = heads_fx.get("ptm").data[0];
    println!("ptm {:.6} vs {:.6} (|d|={:.2e})", ptm, ptm_ref, (ptm - ptm_ref).abs());
    assert!((ptm - ptm_ref).abs() < 1e-4, "ptm {ptm} vs {ptm_ref}");

    let pae = heads::compute_pae(&ptm_logits, l);
    let pa = compare(&pae.data, &heads_fx.get("predicted_aligned_error").data);
    println!("pae {}", pa.summary());
    assert!(pa.max_abs < 1e-3, "pae {}", pa.summary());

    let atom37 = heads::atom14_to_atom37(&positions_final, &aatype, &consts, l);
    let a = compare(&atom37.data, &heads_fx.get("atom37").data);
    println!("atom37 {}", a.summary());
    assert!(a.max_abs < 1e-3, "atom37 {}", a.summary());
}
