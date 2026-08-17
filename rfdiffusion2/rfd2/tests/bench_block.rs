//! Where a block's time actually goes. A stopwatch, not a parity test — but it
//! runs the real weights on the real captured inputs, so the split is the one
//! the model has.
use rfd2::model::iterblock::{BlockCfg, BlockInputs, IterBlock, TrackState};
use rfd2::model::rf::Arch;
use rfd2::nn::{Ctx, Params};
use rfd2::rng::torch::Mt19937;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use rfd2::{chemical_gen, geom};
use std::path::Path;
use std::time::Instant;

fn open(rel: &str) -> Option<Weights> {
    let p = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() { eprintln!("SKIP {p}"); return None; }
    Some(Weights::open(&p).expect("open"))
}

#[test]
fn block_time_split() {
    let Some(io) = open("fixtures/blocks_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let arch = Arch::rfd173();
    let b = "model.simulator.main_block.1";
    if !io.has(&format!("in::{b}.0")) { eprintln!("SKIP: no capture"); return; }

    let msa = io.get(&format!("in::{b}.0"));
    let pair = io.get(&format!("in::{b}.1"));
    let xyz = io.get(&format!("in::{b}.2"));
    let state = io.get(&format!("in::{b}.3"));
    let (seq_unmasked, s) = io.get_i64(&format!("in::{b}.4"));
    let l = s[s.len() - 1];
    let idx = io.get_i64(&format!("in::{b}.5")).0;
    let bond_feats = io.get_i64(&format!("in::{b}.6")).0;
    let same_chain: Vec<bool> = io.get_i64(&format!("in::{b}.7")).0.into_iter().map(|v| v != 0).collect();
    let rotation_mask: Vec<bool> = seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();
    let dist_matrix = step.get("rfi.dist_matrix").data;
    let chirals = step.get("rfi.chirals").data;
    let atom_frames = step.get_i64("rfi.atom_frames").0;
    let is_motif: Vec<bool> = step.get_i64("rfi.is_motif").0.into_iter().map(|v| v != 0).collect();
    let bytes: Vec<u8> = io.get_i64(&format!("rng::{b}")).0.into_iter().map(|v| v as u8).collect();

    let cfg = BlockCfg {
        n_head_msa: arch.n_head_msa, d_hidden_msa: arch.d_hidden, n_head_pair: arch.n_head_pair,
        d_hidden: arch.d_hidden, use_global_attn: false, enable_same_chain: arch.enable_same_chain,
        p_drop: arch.p_drop, se3_num_layers: arch.se3_layers, l0_in: arch.d_state, l1_in: 6,
        num_channels: arch.num_channels, num_degrees: arch.num_degrees, l0_out: arch.d_state,
        l1_out: 2, n_heads: arch.n_heads, div: arch.div, top_k: -1, n_extra_l1: 3,
    };
    let blk = IterBlock::load(&Params::root(&w, "model").sub("simulator").sub("main_block").sub("1"), cfg);
    let inp = BlockInputs {
        seq_unmasked: &seq_unmasked, idx: &idx, bond_feats: &bond_feats, dist_matrix: &dist_matrix,
        same_chain: &same_chain, chirals: &chirals, atom_frames: &atom_frames, is_motif: &is_motif,
        rotation_mask: &rotation_mask,
    };
    let mk = || TrackState {
        msa: msa.clone(), pair: pair.clone(), xyz: xyz.data.clone(), state: state.clone(),
        alpha: Tensor::zeros(&[1, l, chemical_gen::NTOTALDOFS, 2]), quat: vec![0.0; l * 4],
    };

    let t0 = Instant::now();
    let mut st = mk();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
    blk.forward(&mut st, &inp, &mut ctx);
    let whole = t0.elapsed().as_secs_f64();

    // str2str alone, from the same inputs
    let mut st2 = mk();
    let mut ctx2 = Ctx::new(Mt19937::from_torch_state(&bytes));
    let extra_l1 = rfd2::model::str2str::chiral_extra_l1(&st2.xyz, l, 3, &chirals);
    let t1 = Instant::now();
    let _ = blk.str2str.forward(
        &st2.msa, &st2.pair, &st2.xyz, 3, &st2.state, &idx, &rotation_mask, &bond_feats,
        &dist_matrix, &atom_frames, &is_motif, None, &extra_l1, 3, -1, &mut ctx2,
    );
    let se3 = t1.elapsed().as_secs_f64();

    // the track's own sub-stages, each from the same captured inputs
    let mut ctx3 = Ctx::new(Mt19937::from_torch_state(&bytes));
    let ca: Vec<f32> = (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let t = Instant::now();
    let mut rbf = geom::rbf_ca(&ca, l).reshape(&[1, l, l, geom::D_COUNT]);
    let pos = blk.pos.forward(&seq_unmasked, &idx, &bond_feats, &dist_matrix, &same_chain);
    for (i, v) in rbf.data.iter_mut().enumerate() { *v += pos.data[i]; }
    let t_rbf = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let m2 = blk.msa2msa.forward(&msa, &pair, &rbf, &state, &mut ctx3);
    let t_m2m = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let p2 = blk.msa2pair.forward(&m2, &pair);
    let t_m2p = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = blk.pair2pair.forward(&p2, &rbf, &state, &mut ctx3);
    let t_p2p = t.elapsed().as_secs_f64();

    // inside pair2pair
    let pp = &blk.pair2pair;
    println!("  [shapes] to_gate {}->{}  emb_rbf {}->{}",
             pp.to_gate.in_dim(), pp.to_gate.out_dim(), pp.emb_rbf.in_dim(), pp.emb_rbf.out_dim());
    let t = Instant::now();
    let rbf192 = pp.emb_rbf.forward(&rbf);
    let t_emb = t.elapsed().as_secs_f64();
    let gate_in = Tensor::zeros(&[1, l, l, pp.to_gate.in_dim()]);
    let t = Instant::now();
    let _ = pp.to_gate.forward(&gate_in);
    let t_gate = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = pp.tri_mul_out.forward(&p2);
    let t_tri = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = pp.row_attn.forward(&p2, &rbf192);
    let t_row = t.elapsed().as_secs_f64();
    let t_ff = 0.0;
    println!("  ..emb_rbf         {:.3} s", t_emb);
    println!("  ..to_gate         {:.3} s", t_gate);
    println!("  ..tri_mul (x2)    {:.3} s", t_tri * 2.0);
    println!("  ..axial attn (x2) {:.3} s", t_row * 2.0);
    let _ = t_ff;

    println!("main_block.1 total  {:.3} s", whole);
    println!("  rbf + pos         {:.3} s   {:.0} %", t_rbf, 100.0 * t_rbf / whole);
    println!("  msa2msa           {:.3} s   {:.0} %", t_m2m, 100.0 * t_m2m / whole);
    println!("  msa2pair          {:.3} s   {:.0} %", t_m2p, 100.0 * t_m2p / whole);
    println!("  pair2pair         {:.3} s   {:.0} %", t_p2p, 100.0 * t_p2p / whole);
    println!("  str2str (SE3)     {:.3} s   {:.0} %", se3, 100.0 * se3 / whole);
    println!("  track (msa/pair)  {:.3} s   {:.0} %", whole - se3, 100.0 * (whole - se3) / whole);
    println!("  projected 36-block trunk: {:.1} s", 36.0 * whole);
}
