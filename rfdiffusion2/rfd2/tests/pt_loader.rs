//! SOP §4 rung 3 — weight loading. **Tolerance: exactly 0**, every tensor,
//! every value.
//!
//! The Rust side reads `RFD_173.pt` directly (ZIP + pickle). The fixture is the
//! same state dict written out by `torch.load` -> safetensors. If these agree
//! on all 82 911 693 values, the loader is not a source of error for any rung
//! above it.
//!
//! Skipped with a message (not failed) when the 1.34 GB checkpoint or the
//! 332 MB fixture is absent, so a fresh clone still runs the rest of the suite.

use rfd2::weights::Weights;
use std::path::Path;

fn ckpt_path() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{root}/../../ref_RFdiffusion2/rf_diffusion/model_weights/RFD_173.pt")
}

fn fixture_path() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{root}/../fixtures/weights/model_state_dict.safetensors")
}

/// The counts measured in `python/inventory_checkpoint.py`. These are the test.
const N_TENSORS: usize = 7208;
const N_PARAMS: usize = 82_911_693;

#[test]
fn pt_reader_matches_torch_load_exactly() {
    let (ck, fx) = (ckpt_path(), fixture_path());
    if !Path::new(&ck).exists() || !Path::new(&fx).exists() {
        eprintln!("SKIP: need {ck} and {fx}\n  (python/gen_weight_fixture.py)");
        return;
    }

    let pt = Weights::open(&ck).expect("open .pt");
    let want = Weights::open(&fx).expect("open fixture");

    // The fixture keys are the state-dict names; the .pt reader records every
    // (str -> tensor) pair it walks, so the same names must be present.
    let want_names = want.names();
    assert_eq!(want_names.len(), N_TENSORS, "fixture tensor count");

    let mut n_values = 0usize;
    let mut n_missing = Vec::new();
    for name in &want_names {
        // The .pt reader qualifies names by their containing dict, so the EMA
        // weights are reached explicitly rather than by whichever state dict
        // happened to be walked last.
        let qualified = format!("model_state_dict.{name}");
        if !pt.has(&qualified) {
            n_missing.push(qualified);
            continue;
        }
        let a = pt.get(&qualified);
        let b = want.get(name);
        assert_eq!(a.shape, b.shape, "{name}: shape");
        for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}[{i}]: .pt {x:e} != torch.load {y:e}"
            );
        }
        n_values += a.data.len();
    }

    assert!(
        n_missing.is_empty(),
        "{} tensors missing from the .pt reader, e.g. {:?}",
        n_missing.len(),
        &n_missing[..n_missing.len().min(5)]
    );
    assert_eq!(n_values, N_PARAMS, "total parameter count");
    println!(
        "pt loader: {}/{} tensors, {} parameters bit-identical to torch.load",
        want_names.len(),
        N_TENSORS,
        n_values
    );
}

/// The EMA / final distinction is load-bearing: only 570 of 7 208 tensors are
/// identical between the two state dicts, so a loader that reached for the
/// wrong one would produce a different model while looking plausible.
#[test]
fn ema_and_final_state_dicts_are_distinguished() {
    let ck = ckpt_path();
    if !Path::new(&ck).exists() {
        eprintln!("SKIP: need {ck}");
        return;
    }
    let pt = Weights::open(&ck).expect("open .pt");
    let names = pt.names();

    let ema: Vec<&String> = names
        .iter()
        .filter(|n| n.starts_with("model_state_dict."))
        .collect();
    let fin: Vec<&String> = names
        .iter()
        .filter(|n| n.starts_with("final_state_dict."))
        .collect();
    println!(
        "pt reader exposes {} names: {} EMA, {} final",
        names.len(),
        ema.len(),
        fin.len()
    );
    assert_eq!(ema.len(), N_TENSORS, "EMA tensors");
    assert_eq!(fin.len(), N_TENSORS, "final tensors");

    // ...and they must actually differ, or the qualification is a no-op.
    let mut n_equal = 0usize;
    for n in &ema {
        let bare = n.strip_prefix("model_state_dict.").unwrap();
        let other = format!("final_state_dict.{bare}");
        let a = pt.get(n);
        let b = pt.get(&other);
        if a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()) {
            n_equal += 1;
        }
    }
    println!("EMA vs final: {n_equal}/{N_TENSORS} tensors bit-identical");
    assert_eq!(
        n_equal, 570,
        "inventory_checkpoint.py measured 570 identical tensors"
    );
}
