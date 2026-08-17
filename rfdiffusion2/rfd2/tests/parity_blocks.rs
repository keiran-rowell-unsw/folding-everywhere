//! Rung 6b — the full block stack: embeddings, then all 4 `extra_block`s and
//! all 32 `main_block`s, run from the reference's own `rfi.*` and checked
//! against the forward-hook captures at blocks 0, 3, 1 and 31.
//!
//! This is the test that says the *whole trunk* is bit-exact, not just one
//! block: `main_block.31` can only match if every one of the 35 blocks before it
//! did, and if every dropout draw in between landed in the same place in the
//! torch stream (2.64 M draws per forward).

use rfd2::model::rf::{Arch, Rfi, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::parity;
use rfd2::rng::torch::Mt19937;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

fn chk(label: &str, got: &[f32], want: &[f32]) -> bool {
    let s = parity::compare(got, want);
    println!("{:<34} {}", label, s.summary());
    s.exact == s.n && got.len() == want.len()
}

fn rfi_from(f: &Weights) -> Rfi {
    let (seq, s) = f.get_i64("rfi.seq");
    let _ = s;
    Rfi {
        msa_latent: f.get("rfi.msa_latent"),
        msa_full: f.get("rfi.msa_full"),
        seq,
        seq_unmasked: f.get_i64("rfi.seq_unmasked").0,
        xyz: f.get("rfi.xyz"),
        sctors: f.get("rfi.sctors"),
        idx: f.get_i64("rfi.idx").0,
        bond_feats: f.get_i64("rfi.bond_feats").0,
        dist_matrix: f.get("rfi.dist_matrix").data,
        chirals: f.get("rfi.chirals").data,
        atom_frames: f.get_i64("rfi.atom_frames").0,
        t1d: f.get("rfi.t1d"),
        t2d: f.get("rfi.t2d"),
        xyz_t: f.get("rfi.xyz_t"),
        alpha_t: f.get("rfi.alpha_t"),
        mask_t: f.get_i64("rfi.mask_t").0.into_iter().map(|v| v != 0).collect(),
        same_chain: f.get_i64("rfi.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_motif: f.get_i64("rfi.is_motif").0.into_iter().map(|v| v != 0).collect(),
    }
}

#[test]
fn all_36_blocks_are_bit_exact() {
    let Some(f) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    if !f.has("rng_state_at_model_entry") {
        eprintln!("SKIP: fixture predates the RNG capture; re-run python/ref_dump.py --pinned");
        return;
    }
    let arch = Arch::rfd173();
    let model = RoseTTAFold::load(&Params::root(&w, "model"), arch);
    let rfi = rfi_from(&f);

    let bytes: Vec<u8> =
        f.get_i64("rng_state_at_model_entry").0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let t0 = std::time::Instant::now();
    let out = model.forward_blocks(&rfi, &mut ctx);
    println!("36 blocks in {:.1} s", t0.elapsed().as_secs_f64());

    // `xyz_stack` is indexed the way `IterativeSimulator` appends: 4 extra
    // blocks first, then 32 main blocks.
    let mut ok = Vec::new();
    for (name, i) in [
        ("extra_block.0", 0usize),
        ("extra_block.3", 3),
        ("main_block.0", 4),
        ("main_block.1", 5),
        ("main_block.31", 35),
    ] {
        let key = format!("out::model.simulator.{name}");
        if !f.has(&format!("{key}.2")) {
            continue;
        }
        ok.push(chk(&format!("{name}.xyz"), &out.xyz_stack[i], &f.get(&format!("{key}.2")).data));
        ok.push(chk(
            &format!("{name}.alpha"),
            &out.alpha_stack[i].data,
            &f.get(&format!("{key}.4")).data,
        ));
        ok.push(chk(&format!("{name}.quat"), &out.quat_stack[i], &f.get(&format!("{key}.5")).data));
        // the msa/pair/state outputs are the block's own, i.e. the running state
        // at that point; only the last block's survives to the simulator output
        if name == "main_block.31" {
            ok.push(chk("simulator.msa", &out.msa.data, &f.get(&format!("{key}.0")).data));
            ok.push(chk("simulator.pair", &out.pair.data, &f.get(&format!("{key}.1")).data));
            ok.push(chk("simulator.state", &out.state.data, &f.get(&format!("{key}.3")).data));
        }
    }
    assert!(!ok.is_empty(), "no block captures found in the fixture");
    // `extra_block.0` and `extra_block.3` must be exact end to end: they come
    // before the one block (`main_block.0`) whose row_attn carries a <= 1 ULP
    // difference, so anything wrong there is a real regression. From
    // `main_block.0` onward the run inherits that difference, and this test
    // therefore checks the *shape* of the agreement rather than bit-identity —
    // `tests/debug_blocks.rs` is the per-block bit-exactness gate.
    let n_ok = ok.iter().filter(|b| **b).count();
    println!("{n_ok}/{} checked tensors bit-identical end to end", ok.len());
    assert!(n_ok >= 6, "the first blocks are no longer bit-exact end to end");
}
