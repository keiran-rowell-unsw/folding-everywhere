//! P2: ESM-2 3B per-layer parity vs the decomposed fp32 reference.
//! Requires the esmfold_v1 weights (env ESMFOLD_WEIGHTS or HF cache) and the
//! fixtures from `python/ref_lm.py` (fixtures/lm/flgM/esm_states.safetensors).

use esmfold::esm2::esm2_states;
use esmfold::parity::compare;
use esmfold::weights::Weights;

fn model_weights() -> Option<Weights> {
    if let Ok(p) = std::env::var("ESMFOLD_WEIGHTS") {
        return Weights::open(&p).ok();
    }
    let base = format!(
        "{}/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots",
        std::env::var("HOME").unwrap()
    );
    let snaps = std::fs::read_dir(&base).ok()?;
    for e in snaps.flatten() {
        let p = e.path().join("model.safetensors");
        if p.exists() {
            return Weights::open(p.to_str().unwrap()).ok();
        }
    }
    None
}

#[test]
fn esm2_per_layer_flgm() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: esmfold_v1 weights not found");
            return;
        }
    };
    let fx_path = format!(
        "{}/fixtures/lm/flgM/esm_states.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx = Weights::open(&fx_path).expect("lm fixtures (run python/ref_lm.py)");
    let ids: Vec<i64> = fx.get("input_ids").data.iter().map(|&x| x as i64).collect();
    println!("L = {}", ids.len());

    let states = esm2_states(&w, &ids);
    assert_eq!(states.len(), 37);

    // Raw max_abs scales with activation magnitude (ESM-2 trunk activations reach
    // ~thousands), so the honest metrics are cosine and magnitude-RELATIVE error.
    let mut worst_rel = 0.0f32;
    let mut worst_cos = 1.0f64;
    for i in 0..37 {
        let want = fx.get(&format!("state_{i:02}"));
        let s = compare(&states[i].data, &want.data);
        let amax = want.amax().max(1e-6);
        let rel = s.max_abs / amax;
        println!("state_{i:02}  {}  amax={:.2e}  rel_max_abs={:.2e}", s.summary(), amax, rel);
        assert!(!s.any_nan, "state {i} NaN");
        assert!(s.cosine > 1.0 - 1e-6, "state {i} cosine {:.10}", s.cosine);
        if rel > worst_rel {
            worst_rel = rel;
        }
        if s.cosine < worst_cos {
            worst_cos = s.cosine;
        }
    }
    println!("worst rel_max_abs = {worst_rel:.3e}, worst cosine = {worst_cos:.10}");
    // fp32 accumulation-order noise: relative error stays at fp32 epsilon (~1e-5).
    assert!(worst_rel < 5e-5, "ESM-2 relative drift too large: {worst_rel:.3e}");
}
