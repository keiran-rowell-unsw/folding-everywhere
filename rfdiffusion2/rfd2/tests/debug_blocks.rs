//! Per-block bisection over the whole trunk: for every one of the 36 blocks,
//! run it from the reference's **own** captured inputs and RNG state and compare
//! its six outputs.
//!
//! Run this way, a failure is localised to a single block and cannot be blamed
//! on drift from an earlier one — which is exactly the question when block 0 is
//! bit-exact and block 3 is not.

use rfd2::model::iterblock::{BlockCfg, BlockInputs, IterBlock, TrackState};
use rfd2::model::rf::Arch;
use rfd2::nn::{Ctx, Params};
use rfd2::parity;
use rfd2::rng::torch::Mt19937;
use rfd2::tensor::Tensor;
use rfd2::{chemical_gen, geom};
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

fn frac(got: &[f32], want: &[f32]) -> (f64, f32) {
    let s = parity::compare(got, want);
    (s.exact_frac() * 100.0, s.max_abs)
}

fn cfg(arch: &Arch, extra: bool) -> BlockCfg {
    BlockCfg {
        n_head_msa: arch.n_head_msa,
        d_hidden_msa: if extra { arch.d_hidden_msa_extra } else { arch.d_hidden },
        n_head_pair: arch.n_head_pair,
        d_hidden: arch.d_hidden,
        use_global_attn: extra,
        enable_same_chain: arch.enable_same_chain,
        p_drop: arch.p_drop,
        se3_num_layers: arch.se3_layers,
        l0_in: arch.d_state,
        l1_in: 6,
        num_channels: arch.num_channels,
        num_degrees: arch.num_degrees,
        l0_out: arch.d_state,
        l1_out: 2,
        n_heads: arch.n_heads,
        div: arch.div,
        top_k: -1,
        n_extra_l1: 3,
    }
}

#[test]
fn every_block_from_reference_inputs() {
    let Some(io) = open("fixtures/blocks_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let arch = Arch::rfd173();
    let root = Params::root(&w, "model").sub("simulator");

    let dist_matrix = step.get("rfi.dist_matrix").data;
    let chirals = step.get("rfi.chirals").data;
    let atom_frames = step.get_i64("rfi.atom_frames").0;
    let is_motif: Vec<bool> =
        step.get_i64("rfi.is_motif").0.into_iter().map(|v| v != 0).collect();

    let mut first_bad: Option<String> = None;
    let mut n_blocks = 0usize;
    let mut n_bad = 0usize;
    let mut worst = 0.0f32;
    for (kind, n) in [("extra_block", arch.n_extra_block), ("main_block", arch.n_main_block)] {
        for i in 0..n {
            let b = format!("model.simulator.{kind}.{i}");
            if !io.has(&format!("in::{b}.0")) {
                continue;
            }
            let msa = io.get(&format!("in::{b}.0"));
            let pair = io.get(&format!("in::{b}.1"));
            let xyz = io.get(&format!("in::{b}.2"));
            let state = io.get(&format!("in::{b}.3"));
            let (seq_unmasked, s) = io.get_i64(&format!("in::{b}.4"));
            let l = s[s.len() - 1];
            let idx = io.get_i64(&format!("in::{b}.5")).0;
            let bond_feats = io.get_i64(&format!("in::{b}.6")).0;
            let same_chain: Vec<bool> =
                io.get_i64(&format!("in::{b}.7")).0.into_iter().map(|v| v != 0).collect();
            let rotation_mask: Vec<bool> =
                seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();

            let bytes: Vec<u8> =
                io.get_i64(&format!("rng::{b}")).0.into_iter().map(|v| v as u8).collect();
            let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

            let blk = IterBlock::load(&root.sub(kind).idx(i), cfg(&arch, kind == "extra_block"));
            let mut st = TrackState {
                msa: msa.clone(),
                pair: pair.clone(),
                xyz: xyz.data.clone(),
                state: state.clone(),
                alpha: Tensor::zeros(&[1, l, chemical_gen::NTOTALDOFS, 2]),
                quat: vec![0.0; l * 4],
            };
            let inp = BlockInputs {
                seq_unmasked: &seq_unmasked,
                idx: &idx,
                bond_feats: &bond_feats,
                dist_matrix: &dist_matrix,
                same_chain: &same_chain,
                chirals: &chirals,
                atom_frames: &atom_frames,
                is_motif: &is_motif,
                rotation_mask: &rotation_mask,
            };
            blk.forward(&mut st, &inp, &mut ctx);

            let (fm, _) = frac(&st.msa.data, &io.get(&format!("out::{b}.0")).data);
            let want_pair = io.get(&format!("out::{b}.1")).data;
            let (fp, mp) = frac(&st.pair.data, &want_pair);
            if fp < 100.0 && first_bad.is_none() {
                let bad: Vec<usize> = st
                    .pair
                    .data
                    .iter()
                    .zip(&want_pair)
                    .enumerate()
                    .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
                    .map(|(i, _)| i)
                    .collect();
                let cells: std::collections::BTreeSet<(usize, usize)> =
                    bad.iter().map(|i| ((i / 192) / l, (i / 192) % l)).collect();
                println!(
                    "    pair differs at {} values in cells {:?}; max|d| {:.3e}",
                    bad.len(),
                    cells.iter().take(8).collect::<Vec<_>>(),
                    mp
                );
            }
            let (fx, mx) = frac(&st.xyz, &io.get(&format!("out::{b}.2")).data);
            let want_state = io.get(&format!("out::{b}.3")).data;
            let (fs, ms) = frac(&st.state.data, &want_state);
            if fs < 100.0 && first_bad.is_none() {
                let bad: Vec<usize> = st
                    .state
                    .data
                    .iter()
                    .zip(&want_state)
                    .enumerate()
                    .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
                    .map(|(i, _)| i)
                    .collect();
                let nodes: std::collections::BTreeSet<usize> =
                    bad.iter().map(|i| i / 64).collect();
                println!(
                    "    state differs at {} values across nodes {:?}; max|d| {:.3e}; \
                     first: got {:e} ({:#010x}) want {:e} ({:#010x})",
                    bad.len(), nodes, ms,
                    st.state.data[bad[0]], st.state.data[bad[0]].to_bits(),
                    want_state[bad[0]], want_state[bad[0]].to_bits()
                );
            }
            let all = fm == 100.0 && fp == 100.0 && fx == 100.0 && fs == 100.0;
            println!(
                "{kind}.{i:<2} msa {fm:6.2}%  pair {fp:6.2}%  xyz {fx:6.2}%  state {fs:6.2}%  max|dxyz| {mx:.3e}  draws {}",
                ctx.rng.draws()
            );
            n_blocks += 1;
            if !all {
                n_bad += 1;
                if first_bad.is_none() {
                    first_bad = Some(format!("{kind}.{i}"));
                }
            }
            worst = worst.max(mx).max(ms).max(mp);
        }
    }
    // Assert the *measured* status rather than an aspiration. Re-measured
    // 2026-08-11 against a regenerated fixture (the previous one predated the
    // noiser change and reported a different pair of numbers): 34 of the 36
    // trunk blocks are bit-identical from the reference's own inputs, and the
    // two that are not have been bisected one level down, in
    // `tests/debug_pair2pair.rs`, to a single child module each —
    //   main_block.2  -> tri_mul_out   (69 values, 0.6 ULP at the worst site)
    //   main_block.23 -> row_attn    (1859 values, 0.7 ULP at the worst site)
    // Sub-ULP where they originate, and unchanged under a ~106-bit accumulator,
    // so the port is correctly rounded and the reference takes the other side of
    // the tie. `worst` is measured on `pair`, whose RMS is ~214, so 3.052e-5 is
    // ~1.4e-7 of the tensor's own scale.
    let n_exact = n_blocks - n_bad;
    println!("{n_exact}/{n_blocks} blocks bit-identical; worst max|d| {worst:.3e}");
    assert!(
        n_bad <= 2,
        "{n_bad} blocks are not bit-exact (measured 2: main_block.2, main_block.23); \
         first {:?}",
        first_bad
    );
    assert!(
        worst < 5e-5,
        "a block differs by {worst:.3e}, far more than the measured 3.052e-5 tie-straddle"
    );
}
