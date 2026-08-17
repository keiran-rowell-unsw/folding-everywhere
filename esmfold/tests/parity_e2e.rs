//! P6 end-to-end: full fold(flgM) including the 4-recycle loop, validated against
//! the PyTorch reference final outputs.

use esmfold::constants::Constants;
use esmfold::parity::compare;
use esmfold::pipeline::fold;
use esmfold::weights::Weights;
use std::time::Instant;

const FLGM: &str = "MSIDRTSPLKPVSTVQTRETSDTPVQKTRQEKTSAATSASVTLSDAQAKLMQPGVSDINMERVEALKTAIRNGELKMDTGKIADSLIREAQSYLQSK";

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
fn e2e_flgm() {
    let w = match model_weights() {
        Some(w) => w,
        None => {
            eprintln!("SKIP: weights not found");
            return;
        }
    };
    let consts = Constants::load(&format!("{}/fixtures/constants/residue_constants.safetensors", env!("CARGO_MANIFEST_DIR")));

    let t0 = Instant::now();
    let out = fold(&w, &consts, FLGM);
    let dt = t0.elapsed().as_secs_f64();
    let l = out.l;
    println!("fold flgM L={l} took {dt:.1}s  ptm={:.4}  plddt_mean={:.4}", out.ptm, esmfold::pdb::mean_plddt(&out.plddt.data, &out.aatype, &consts, l));

    let heads_fx = tfx("heads");
    let struct_fx = tfx("structure");

    let ptm_ref = heads_fx.get("ptm").data[0];
    println!("ptm {:.5} vs ref {:.5} (|d|={:.2e})", out.ptm, ptm_ref, (out.ptm - ptm_ref).abs());

    let p = compare(&out.plddt.data, &heads_fx.get("plddt").data);
    println!("plddt {}", p.summary());

    let a = compare(&out.atom37.data, &heads_fx.get("atom37").data);
    println!("atom37 {}", a.summary());

    // backbone RMSD over atom37 (existing atoms only), no superposition (same frame)
    let pos_ref = &heads_fx.get("atom37").data;
    let exists = &heads_fx.get("atom37_atom_exists").data;
    let mut sse = 0.0f64;
    let mut cnt = 0usize;
    for idx in 0..l * 37 {
        if exists[idx] > 0.5 {
            for xyz in 0..3 {
                let d = (out.atom37.data[idx * 3 + xyz] - pos_ref[idx * 3 + xyz]) as f64;
                sse += d * d;
            }
            cnt += 1;
        }
    }
    let rmsd = (sse / cnt as f64).sqrt();
    println!("all-atom RMSD (no superposition) over {cnt} atoms = {rmsd:.6} A");

    // final atom14 positions vs structure fixture
    let pf = compare(&out.structure.last().unwrap().positions, &struct_fx.get("positions").data[7 * l * 14 * 3..8 * l * 14 * 3]);
    println!("final atom14 {}", pf.summary());

    assert!((out.ptm - ptm_ref).abs() < 1e-2, "ptm drift");
    assert!(rmsd < 0.05, "RMSD too large: {rmsd}");
}
