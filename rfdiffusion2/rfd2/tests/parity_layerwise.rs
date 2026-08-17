//! The whole execution ladder for **one protein**, one row per module.
//!
//! Every other parity test answers one rung's question in its own format. This
//! one walks `M0584_1ldm` from the PDB file on disk to the written `.pdb`,
//! module by module, and prints a uniform row for each: how many values, how
//! many are bit-identical, the largest absolute difference and the largest ULP
//! gap. It writes `results/layerwise_M0584_1ldm.tsv` as well as printing.
//!
//! Two columns, because they answer different questions:
//!
//! * **standalone** — the Rust module is fed the *reference's own* inputs and
//!   the reference's RNG state at that module's entry. A row that is not exact
//!   here is a defect in that module.
//! * **cumulative** — the Rust module runs on its own upstream output, i.e. the
//!   real program. A row that is exact standalone but not cumulative is
//!   inherited drift, not a second bug.
//!
//! That distinction is the whole point: with only a cumulative column, one
//! 1-ULP disagreement in `main_block.0` looks like 36 broken blocks.
//!
//! Comparison is on `to_bits()`, never a tolerance. A tolerance cannot see the
//! `+0.0` vs `-0.0` class of defect (autograd *moves* the first gradient into an
//! input buffer rather than adding it to a zero), which once produced a stage
//! with `max|d| = 0`, `max_ulp = 0` and only 59.6 % bit-identical.
//!
//! Fixtures: `sample_init/stages`, `model_pinned/step0`, `blocks_io/io`,
//! `layerwise/io`, `score/step0`, `sampler/T2`, `weights`, `ligand`, `noiser`.
//! Regenerate `layerwise` with `python/gen_layerwise.py --pinned`.

use rfd2::indep::Indep;
use rfd2::ligand::LigandSet;
use rfd2::lj::{lj_forward, natoms, LjCfg, LjTables};
use rfd2::model::iterblock::{BlockCfg, BlockInputs, IterBlock, TrackState};
use rfd2::model::rf::{Arch, Rfi, RoseTTAFold};
use rfd2::model::xyzconv::XyzConverter;
use rfd2::nn::{Ctx, Params};
use rfd2::noiser::Igso3;
use rfd2::parity::{self, Stats};
use rfd2::prepro::{prepro, PreproOptions};
use rfd2::rng::torch::Mt19937;
use rfd2::sample_init::{Options, SampleInit};
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use rfd2::{chemical_gen, geom, score};
use std::path::Path;

const PDB: &str = "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb";
const CONTIGS: &str = "10,A106-106,10";
const BIG_T: usize = 2;
/// The step `gen_layerwise.py` / `ref_dump.py` capture.
const T_NOW: usize = 2;

fn root(rel: &str) -> String {
    format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn open(rel: &str) -> Option<Weights> {
    let p = root(rel);
    if !Path::new(&p).exists() {
        eprintln!("MISSING FIXTURE: {p}");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

// ---------------------------------------------------------------------------
// the table
// ---------------------------------------------------------------------------

struct Row {
    group: &'static str,
    stage: String,
    mode: &'static str, // "standalone" | "cumulative" | "e2e"
    s: Stats,
}

#[derive(Default)]
struct Table {
    rows: Vec<Row>,
}

impl Table {
    fn push(&mut self, group: &'static str, stage: impl Into<String>, mode: &'static str,
            got: &[f32], want: &[f32]) {
        let stage = stage.into();
        assert_eq!(
            got.len(),
            want.len(),
            "{group}/{stage} ({mode}): length {} vs reference {} — a shape \
             disagreement is a port bug, not a numeric one",
            got.len(),
            want.len()
        );
        self.rows.push(Row { group, stage, mode, s: parity::compare(got, want) });
    }

    /// Integer/boolean stages: bit-identity is the only meaningful metric, so
    /// they go through the same row shape via an exact-match count.
    fn push_i64(&mut self, group: &'static str, stage: impl Into<String>, mode: &'static str,
                got: &[i64], want: &[i64]) {
        let g: Vec<f32> = got.iter().map(|v| *v as f32).collect();
        let w: Vec<f32> = want.iter().map(|v| *v as f32).collect();
        self.push(group, stage, mode, &g, &w);
    }

    /// NaN-aware: which coordinate slots are NaN is load-bearing in `xyz`
    /// (`prepro` blanks every diffused row from slot 3), and `parity::compare`
    /// deliberately skips NaNs rather than counting them as agreement.
    fn push_nan(&mut self, group: &'static str, stage: impl Into<String>, mode: &'static str,
                got: &[f32], want: &[f32]) {
        let (g, w): (Vec<f32>, Vec<f32>) = got
            .iter()
            .zip(want)
            .map(|(a, b)| if a.is_nan() && b.is_nan() { (0.0, 0.0) } else { (*a, *b) })
            .unzip();
        self.push(group, stage, mode, &g, &w);
    }

    fn print(&self) {
        println!(
            "\n{:<10} {:<44} {:<11} {:>10} {:>10} {:>7} {:>11} {:>8}",
            "group", "stage", "mode", "values", "exact", "%", "max|d|", "max_ulp"
        );
        println!("{}", "-".repeat(116));
        for r in &self.rows {
            println!(
                "{:<10} {:<44} {:<11} {:>10} {:>10} {:>6.2}% {:>11.3e} {:>8}",
                r.group,
                r.stage,
                r.mode,
                r.s.n,
                r.s.exact,
                100.0 * r.s.exact_frac(),
                r.s.max_abs,
                r.s.max_ulp
            );
        }
    }

    fn write_tsv(&self, path: &str) {
        let mut out =
            String::from("group\tstage\tmode\tvalues\texact\tpct_bitexact\tmax_abs\tmax_ulp\n");
        for r in &self.rows {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6e}\t{}\n",
                r.group,
                r.stage,
                r.mode,
                r.s.n,
                r.s.exact,
                100.0 * r.s.exact_frac(),
                r.s.max_abs,
                r.s.max_ulp
            ));
        }
        if let Some(dir) = Path::new(path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(path, out).expect("write tsv");
        println!("\nwrote {path}");
    }

    fn inexact(&self, mode: &str) -> Vec<&Row> {
        self.rows.iter().filter(|r| r.mode == mode && r.s.exact != r.s.n).collect()
    }
}

// ---------------------------------------------------------------------------
// helpers shared with the other parity tests
// ---------------------------------------------------------------------------

fn rfi_from(f: &Weights) -> Rfi {
    Rfi {
        msa_latent: f.get("rfi.msa_latent"),
        msa_full: f.get("rfi.msa_full"),
        seq: f.get_i64("rfi.seq").0,
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

fn indep_from(f: &Weights) -> Indep {
    Indep {
        seq: f.get_i64("indep.seq").0,
        xyz: f.get("indep.xyz").data,
        idx: f.get_i64("indep.idx").0,
        bond_feats: f.get_i64("indep.bond_feats").0,
        chirals: f.get("indep.chirals").data,
        same_chain: f.get_i64("indep.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_gp: f.get_i64("indep.is_gp").0.into_iter().map(|v| v != 0).collect(),
        terminus_type: f.get("indep.terminus_type").data,
        is_sm: f.get_i64("indep.is_sm").0.into_iter().map(|v| v != 0).collect(),
    }
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

fn ctx_from(f: &Weights, key: &str) -> Ctx {
    let bytes: Vec<u8> = f.get_i64(key).0.into_iter().map(|v| v as u8).collect();
    Ctx::new(Mt19937::from_torch_state(&bytes))
}

/// `rbf(cdist(CA,CA)) + pos(...)`, exactly as `IterBlock::forward` builds it.
/// Returned separately so the `pos` half can be compared on its own.
fn rbf_plus_pos(blk: &IterBlock, st: &TrackState, inp: &BlockInputs, l: usize)
    -> (Tensor, Tensor) {
    let ca: Vec<f32> = (0..l).flat_map(|i| st.xyz[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let mut rbf = geom::rbf_ca(&ca, l).reshape(&[1, l, l, geom::D_COUNT]);
    let pos =
        blk.pos.forward(inp.seq_unmasked, inp.idx, inp.bond_feats, inp.dist_matrix, inp.same_chain);
    for (i, v) in rbf.data.iter_mut().enumerate() {
        *v += pos.data[i];
    }
    (rbf, pos)
}

/// One block, sub-module by sub-module, emitting a row after each. This is
/// `IterBlock::forward` unrolled — the library method is the single source of
/// truth for the ORDER, and this must be kept in step with it.
#[allow(clippy::too_many_arguments)]
fn walk_block(
    tbl: &mut Table,
    io: &Weights,
    key: &str,
    blk: &IterBlock,
    st: &mut TrackState,
    inp: &BlockInputs,
    ctx: &mut Ctx,
    mode: &'static str,
    short: &str,
) {
    let l = inp.idx.len();
    let (rbf, _pos) = rbf_plus_pos(blk, st, inp, l);

    st.msa = blk.msa2msa.forward(&st.msa, &st.pair, &rbf, &st.state, ctx);
    tbl.push("trunk", format!("{short}.msa2msa"), mode, &st.msa.data,
             &io.get(&format!("out::{key}.msa2msa")).data);

    st.pair = blk.msa2pair.forward(&st.msa, &st.pair);
    tbl.push("trunk", format!("{short}.msa2pair"), mode, &st.pair.data,
             &io.get(&format!("out::{key}.msa2pair")).data);

    st.pair = blk.pair2pair.forward(&st.pair, &rbf, &st.state, ctx);
    tbl.push("trunk", format!("{short}.pair2pair"), mode, &st.pair.data,
             &io.get(&format!("out::{key}.pair2pair")).data);

    // the chiral gradient is recomputed from the CURRENT coordinates each block
    let extra_l1 = rfd2::chiral::chiral_grads(&st.xyz, l, 3, inp.chirals);
    let out = blk.str2str.forward(
        &st.msa, &st.pair, &st.xyz, 3, &st.state, inp.idx, inp.rotation_mask, inp.bond_feats,
        inp.dist_matrix, inp.atom_frames, inp.is_motif, None, &extra_l1, blk.cfg.n_extra_l1,
        blk.cfg.top_k, ctx,
    );
    st.xyz = out.xyz;
    st.state = out.state;
    st.alpha = out.alpha;
    st.quat = out.quat;
    tbl.push("trunk", format!("{short}.str2str.xyz"), mode, &st.xyz,
             &io.get(&format!("out::{key}.str2str.0")).data);
    tbl.push("trunk", format!("{short}.str2str.state"), mode, &st.state.data,
             &io.get(&format!("out::{key}.str2str.1")).data);
    tbl.push("trunk", format!("{short}.str2str.alpha"), mode, &st.alpha.data,
             &io.get(&format!("out::{key}.str2str.2")).data);
    tbl.push("trunk", format!("{short}.str2str.quat"), mode, &st.quat,
             &io.get(&format!("out::{key}.str2str.3")).data);
}

// ---------------------------------------------------------------------------

#[test]
fn layerwise_ladder() {
    // Every fixture is REQUIRED. The other tests skip when one is absent, which
    // is right for a bisection harness and wrong here: a silently missing stage
    // in a completeness table reads as "covered" when it is not. `prepro` was
    // recorded green at rung 4b while `alpha_t` and `t2d` had never been built,
    // because the tests were reading both out of a fixture.
    let mut missing = Vec::new();
    let mut need = |rel: &'static str| -> Option<Weights> {
        let w = open(rel);
        if w.is_none() {
            missing.push(rel);
        }
        w
    };
    let step = need("fixtures/model_pinned/step0.safetensors");
    let lay = need("fixtures/layerwise/io.safetensors");
    let bio = need("fixtures/blocks_io/io.safetensors");
    let init = need("fixtures/sample_init/stages.safetensors");
    let nois = need("fixtures/noiser/stages.safetensors");
    let scof = need("fixtures/score/step0.safetensors");
    let wts = need("fixtures/weights/model_state_dict.safetensors");
    assert!(
        missing.is_empty(),
        "the ladder is incomplete without these fixtures: {missing:?}\n\
         regenerate with python/gen_layerwise.py --pinned, python/ref_dump.py --pinned, \
         python/dump_io.py, python/gen_sample_init.py, python/gen_score.py"
    );
    let (step, lay, bio, init, nois, scof, w) =
        (step.unwrap(), lay.unwrap(), bio.unwrap(), init.unwrap(), nois.unwrap(),
         scof.unwrap(), wts.unwrap());

    let arch = Arch::rfd173();
    let model = RoseTTAFold::load(&Params::root(&w, "model"), arch);
    let mut tbl = Table::default();
    let t_start = std::time::Instant::now();

    // =====================================================================
    // A. input — PDB on disk -> Indep -> noised structure
    // =====================================================================
    {
        let text = std::fs::read_to_string(root(PDB)).expect("read pdb");
        let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
        let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
            .expect("ligand sidecar");
        let omega = nois.get("igso3.omega_grid").data;
        let cdf = nois.get("igso3.cdf");
        let n = cdf.shape[1];
        let igso3 = Igso3::new(omega, cdf.data[(cdf.shape[0] - 1) * n..].to_vec());

        let mut ctx = ctx_from(&init, "rng.at_sample_init");
        let opt = Options { big_t: BIG_T, ..Options::default() };
        let out =
            SampleInit::run(&text, &names, &topo, CONTIGS, &opt, &igso3, &mut ctx, &mut None)
                .expect("sample_init");

        for (label, got, pre) in [
            ("indep_cond", &out.indep, "out_indep"),
            ("indep_uncond", &out.indep_uncond, "dtac_uncond"),
            ("indep_orig", &out.indep_orig, "out_indep_orig"),
        ] {
            tbl.push_i64("input", format!("{label}.seq"), "cumulative", &got.seq,
                         &init.get_i64(&format!("{pre}.seq")).0);
            tbl.push_i64("input", format!("{label}.idx"), "cumulative", &got.idx,
                         &init.get_i64(&format!("{pre}.idx")).0);
            tbl.push_i64("input", format!("{label}.bond_feats"), "cumulative", &got.bond_feats,
                         &init.get_i64(&format!("{pre}.bond_feats")).0);
            tbl.push_i64(
                "input", format!("{label}.same_chain"), "cumulative",
                &got.same_chain.iter().map(|b| *b as i64).collect::<Vec<_>>(),
                &init.get_i64(&format!("{pre}.same_chain")).0,
            );
            tbl.push("input", format!("{label}.chirals"), "cumulative", &got.chirals,
                     &init.get(&format!("{pre}.chirals")).data);
            tbl.push_nan("input", format!("{label}.xyz"), "cumulative", &got.xyz,
                         &init.get(&format!("{pre}.xyz")).data);
        }
        tbl.push_i64(
            "input", "is_diffused", "cumulative",
            &out.is_diffused.iter().map(|b| *b as i64).collect::<Vec<_>>(),
            &init.get_i64("out.is_diffused").0,
        );
        assert_eq!(out.t_step_input, init.get_i64("out.t_step_input").0[0] as usize);
    }

    // =====================================================================
    // B. features — prepro, Indep -> Rfi (the network's input)
    // =====================================================================
    {
        let mut indep = indep_from(&step);
        let is_diffused: Vec<bool> =
            step.get_i64("is_diffused").0.into_iter().map(|v| v != 0).collect();
        let atom_frames = step.get_i64("rfi.atom_frames").0;
        let opt = PreproOptions { big_t: BIG_T, ..PreproOptions::default() };
        let r = prepro(&mut indep, T_NOW, &is_diffused, &atom_frames, &opt);
        for (name, got) in [
            ("msa_latent", &r.msa_latent.data), ("msa_full", &r.msa_full.data),
            ("sctors", &r.sctors.data), ("dist_matrix", &r.dist_matrix),
            ("chirals", &r.chirals), ("t1d", &r.t1d.data), ("t2d", &r.t2d.data),
            ("xyz_t", &r.xyz_t.data), ("alpha_t", &r.alpha_t.data),
        ] {
            tbl.push("features", format!("rfi.{name}"), "cumulative", got,
                     &step.get(&format!("rfi.{name}")).data);
        }
        tbl.push_nan("features", "rfi.xyz", "cumulative", &r.xyz.data,
                     &step.get("rfi.xyz").data);
        tbl.push_i64("features", "rfi.seq", "cumulative", &r.seq,
                     &step.get_i64("rfi.seq").0);
        tbl.push_i64("features", "rfi.seq_unmasked", "cumulative", &r.seq_unmasked,
                     &step.get_i64("rfi.seq_unmasked").0);
        tbl.push_i64("features", "rfi.bond_feats", "cumulative", &r.bond_feats,
                     &step.get_i64("rfi.bond_feats").0);
    }

    // From here on the network is driven from the reference's own `rfi.*`, so
    // the trunk rows are not contaminated by anything upstream of it.
    let rfi = rfi_from(&step);
    let l = rfi.seq.len();
    let a = &arch;

    // =====================================================================
    // C. embeddings — each of the five, driven individually
    // =====================================================================
    let mut ctx = ctx_from(&step, "rng_state_at_model_entry");
    let (msa0, msa_full0, pair0, state0) = {
        let (mut msa, mut pair, mut state) = model.latent_emb.forward(
            &rfi.msa_latent, &rfi.seq, &rfi.idx, &rfi.bond_feats, &rfi.dist_matrix,
            &rfi.same_chain,
        );
        tbl.push("embed", "latent_emb.msa", "standalone", &msa.data,
                 &step.get("out::model.latent_emb.0").data);
        tbl.push("embed", "latent_emb.pair", "standalone", &pair.data,
                 &step.get("out::model.latent_emb.1").data);
        tbl.push("embed", "latent_emb.state", "standalone", &state.data,
                 &step.get("out::model.latent_emb.2").data);

        let msa_full = model.full_emb.forward(&rfi.msa_full, &rfi.seq);
        tbl.push("embed", "full_emb", "standalone", &msa_full.data,
                 &step.get("out::model.full_emb").data);

        let be = model.bond_emb.forward(&rfi.bond_feats, l);
        tbl.push("embed", "bond_emb", "standalone", &be.data,
                 &step.get("out::model.bond_emb").data);
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += be.data[i];
        }

        // recycling: every `*_prev` is None on this path, i.e. zeros
        let natom = rfi.xyz.shape[2];
        let ca: Vec<f32> = (0..l)
            .flat_map(|i| rfi.xyz.data[(i * natom + 1) * 3..(i * natom + 1) * 3 + 3].to_vec())
            .collect();
        let (mr, pr, sr) = model.recycle.forward(
            &Tensor::zeros(&[1, l, a.d_msa]), &Tensor::zeros(&[1, l, l, a.d_pair]), &ca,
            &Tensor::zeros(&[1, l, a.d_state]), &rfi.sctors, None,
        );
        tbl.push("embed", "recycle.msa", "standalone", &mr.data,
                 &step.get("out::model.recycle.0").data);
        tbl.push("embed", "recycle.pair", "standalone", &pr.data,
                 &step.get("out::model.recycle.1").data);
        tbl.push("embed", "recycle.state", "standalone", &sr.data,
                 &step.get("out::model.recycle.2").data);
        for i in 0..l * a.d_msa {
            msa.data[i] += mr.data[i];
        }
        for (i, v) in pair.data.iter_mut().enumerate() {
            *v += pr.data[i];
        }
        for (i, v) in state.data.iter_mut().enumerate() {
            *v += sr.data[i];
        }

        let (pair, state) = model.templ_emb.forward(
            &rfi.t1d, &rfi.t2d, &rfi.alpha_t, &rfi.xyz_t, &rfi.mask_t, &pair, &state, &mut ctx,
        );
        tbl.push("embed", "templ_emb.pair", "standalone", &pair.data,
                 &step.get("out::model.templ_emb.0").data);
        tbl.push("embed", "templ_emb.state", "standalone", &state.data,
                 &step.get("out::model.templ_emb.1").data);
        (msa, msa_full, pair, state)
    };

    // =====================================================================
    // D. trunk — 36 blocks, cumulative: each module on its own upstream output
    // =====================================================================
    let natom = rfi.xyz.shape[2];
    let xyz3: Vec<f32> =
        (0..l).flat_map(|i| rfi.xyz.data[i * natom * 3..i * natom * 3 + 9].to_vec()).collect();
    let rotation_mask: Vec<bool> = rfi.seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();
    let inp = BlockInputs {
        seq_unmasked: &rfi.seq_unmasked,
        idx: &rfi.idx,
        bond_feats: &rfi.bond_feats,
        dist_matrix: &rfi.dist_matrix,
        same_chain: &rfi.same_chain,
        chirals: &rfi.chirals,
        atom_frames: &rfi.atom_frames,
        is_motif: &rfi.is_motif,
        rotation_mask: &rotation_mask,
    };
    let sim = Params::root(&w, "model").sub("simulator");
    let extra: Vec<IterBlock> =
        (0..a.n_extra_block).map(|i| IterBlock::load(&sim.sub("extra_block").idx(i), cfg(a, true)))
            .collect();
    let main: Vec<IterBlock> =
        (0..a.n_main_block).map(|i| IterBlock::load(&sim.sub("main_block").idx(i), cfg(a, false)))
            .collect();

    let mut st = TrackState {
        msa: msa_full0.clone(),
        pair: pair0.clone(),
        xyz: xyz3.clone(),
        state: state0.clone(),
        alpha: Tensor::zeros(&[1, l, chemical_gen::NTOTALDOFS, 2]),
        quat: vec![0.0; l * 4],
    };
    // `pos` is identical for all 36 blocks (pure function of fixed inputs), so
    // the reference captured it once and it is checked once.
    {
        let (_, pos) = rbf_plus_pos(&extra[0], &st, &inp, l);
        tbl.push("trunk", "pos", "standalone", &pos.data,
                 &lay.get("out::model.simulator.pos").data);
    }
    for (i, blk) in extra.iter().enumerate() {
        walk_block(&mut tbl, &lay, &format!("model.simulator.extra_block.{i}"), blk, &mut st,
                   &inp, &mut ctx, "cumulative", &format!("extra_block.{i}"));
    }
    // the extra blocks carry `msa_full`; the main blocks carry `msa`
    st.msa = msa0.clone();
    for (i, blk) in main.iter().enumerate() {
        walk_block(&mut tbl, &lay, &format!("model.simulator.main_block.{i}"), blk, &mut st,
                   &inp, &mut ctx, "cumulative", &format!("main_block.{i}"));
    }

    // =====================================================================
    // E. refiner — 4 calls of str_refiner, each with its own LJ/chiral grads
    // =====================================================================
    {
        let conv = XyzConverter::new();
        let lj_tables = LjTables::new();
        let lj_cfg = LjCfg::default();
        for k in 0..a.n_ref_block {
            let mut extra_l1 = vec![0.0f32; l * 6 * 3];
            let mut extra_l0: Option<Vec<f32>> = None;
            if a.use_lj_l1 {
                let xyzaa = conv.compute_all_atom(inp.seq_unmasked, &st.xyz, 3, &st.alpha.data);
                let out = lj_forward(inp.seq_unmasked, &xyzaa, inp.bond_feats, inp.dist_matrix,
                                     &lj_tables, &lj_cfg);
                // `torch.autograd.grad(natoms * Elj, ...)`: the incoming gradient
                // on `xyzaa` is the atom count, not 1.
                let n = natoms(inp.seq_unmasked, &lj_tables, lj_cfg.use_h);
                let dxyzaa: Vec<f32> = out.dljedx.iter().map(|v| n * v).collect();
                let g = rfd2::xyzconv_bwd::backward(inp.seq_unmasked, &st.xyz, 3,
                                                    &st.alpha.data, &dxyzaa);
                for i in 0..l {
                    extra_l1[i * 18..i * 18 + 9].copy_from_slice(&g.dxyz[i * 9..i * 9 + 9]);
                }
                extra_l0 = Some(g.dalpha);
            }
            if a.use_chiral_l1 {
                let dch = rfd2::chiral::chiral_grads(&st.xyz, l, 3, inp.chirals);
                for i in 0..l {
                    extra_l1[i * 18 + 9..i * 18 + 18].copy_from_slice(&dch[i * 9..i * 9 + 9]);
                }
            }
            let out = model.simulator.str_refiner.forward(
                &st.msa, &st.pair, &st.xyz, 3, &st.state, inp.idx, inp.rotation_mask,
                inp.bond_feats, inp.dist_matrix, inp.atom_frames, inp.is_motif,
                extra_l0.as_deref(), &extra_l1, 6, a.refiner_topk, &mut ctx,
            );
            st.xyz = out.xyz;
            st.state = out.state;
            st.alpha = out.alpha;
            st.quat = out.quat;
            let key = format!("out::model.simulator.str_refiner#{k}");
            tbl.push("refiner", format!("str_refiner[{k}].xyz"), "cumulative", &st.xyz,
                     &lay.get(&format!("{key}.0")).data);
            tbl.push("refiner", format!("str_refiner[{k}].state"), "cumulative", &st.state.data,
                     &lay.get(&format!("{key}.1")).data);
            tbl.push("refiner", format!("str_refiner[{k}].alpha"), "cumulative", &st.alpha.data,
                     &lay.get(&format!("{key}.2")).data);
            tbl.push("refiner", format!("str_refiner[{k}].quat"), "cumulative", &st.quat,
                     &lay.get(&format!("{key}.3")).data);
        }
        let xyzaa = conv.compute_all_atom(inp.seq_unmasked, &st.xyz, 3, &st.alpha.data);
        tbl.push("refiner", "compute_all_atom", "cumulative", &xyzaa,
                 &step.get("out::model.simulator.4").data);
        tbl.push("refiner", "simulator.msa", "cumulative", &st.msa.data,
                 &step.get("out::model.simulator.0").data);
        tbl.push("refiner", "simulator.pair", "cumulative", &st.pair.data,
                 &step.get("out::model.simulator.1").data);
        tbl.push("refiner", "simulator.state", "cumulative", &st.state.data,
                 &step.get("out::model.simulator.5").data);
    }
    println!("cumulative pass done in {:.1} s", t_start.elapsed().as_secs_f64());

    // =====================================================================
    // D'. trunk — the same 36 blocks, each from the REFERENCE's own inputs
    // =====================================================================
    let t_sa = std::time::Instant::now();
    for (kind, n) in [("extra_block", a.n_extra_block), ("main_block", a.n_main_block)] {
        for i in 0..n {
            let key = format!("model.simulator.{kind}.{i}");
            assert!(
                bio.has(&format!("in::{key}.0")),
                "blocks_io has no input capture for {key}; regenerate with \
                 python/dump_io.py --pinned --match 'simulator\\.(extra|main)_block' --out blocks_io"
            );
            let (seq_unmasked, _) = bio.get_i64(&format!("in::{key}.4"));
            let idx = bio.get_i64(&format!("in::{key}.5")).0;
            let bond_feats = bio.get_i64(&format!("in::{key}.6")).0;
            let same_chain: Vec<bool> =
                bio.get_i64(&format!("in::{key}.7")).0.into_iter().map(|v| v != 0).collect();
            let rmask: Vec<bool> = seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();
            let binp = BlockInputs {
                seq_unmasked: &seq_unmasked,
                idx: &idx,
                bond_feats: &bond_feats,
                dist_matrix: &rfi.dist_matrix,
                same_chain: &same_chain,
                chirals: &rfi.chirals,
                atom_frames: &rfi.atom_frames,
                is_motif: &rfi.is_motif,
                rotation_mask: &rmask,
            };
            let mut bst = TrackState {
                msa: bio.get(&format!("in::{key}.0")),
                pair: bio.get(&format!("in::{key}.1")),
                xyz: bio.get(&format!("in::{key}.2")).data,
                state: bio.get(&format!("in::{key}.3")),
                alpha: Tensor::zeros(&[1, l, chemical_gen::NTOTALDOFS, 2]),
                quat: vec![0.0; l * 4],
            };
            let mut bctx = ctx_from(&bio, &format!("rng::{key}"));
            let blk = if kind == "extra_block" { &extra[i] } else { &main[i] };
            walk_block(&mut tbl, &lay, &key, blk, &mut bst, &binp, &mut bctx, "standalone",
                       &format!("{kind}.{i}"));
        }
    }
    println!("standalone pass done in {:.1} s", t_sa.elapsed().as_secs_f64());

    // =====================================================================
    // F. heads  +  G. the wrapper that actually produces px0
    // =====================================================================
    {
        let mut wctx = ctx_from(&step, "rng_state_at_model_entry");
        let t0 = std::time::Instant::now();
        let out = score::forward_from_rfi(&model, &rfi, &mut wctx);
        println!("forward_from_rfi in {:.1} s", t0.elapsed().as_secs_f64());
        // The five projection heads are still at their zero initialisation in
        // RFD_173 (`AuxiliaryPredictor.reset_parameter` zeros weight AND bias,
        // and RFdiffusion2 never trains them), so every one emits exactly 0.0
        // and `bind_pred` emits sigmoid(0) = 0.5. These rows check weight
        // loading, shape and the channel-first permutes — not arithmetic.
        let m = &out.model;
        tbl.push("heads", "c6d.dist [untrained]", "cumulative", &m.c6d.dist.data,
                 &step.get("out::model.c6d_pred.0").data);
        tbl.push("heads", "c6d.omega [untrained]", "cumulative", &m.c6d.omega.data,
                 &step.get("out::model.c6d_pred.1").data);
        tbl.push("heads", "c6d.theta [untrained]", "cumulative", &m.c6d.theta.data,
                 &step.get("out::model.c6d_pred.2").data);
        tbl.push("heads", "c6d.phi [untrained]", "cumulative", &m.c6d.phi.data,
                 &step.get("out::model.c6d_pred.3").data);
        tbl.push("heads", "aa_pred [untrained]", "cumulative", &m.logits_aa.data,
                 &step.get("out::model.aa_pred").data);
        tbl.push("heads", "lddt_pred [untrained]", "cumulative", &m.lddt.data,
                 &step.get("out::model.lddt_pred").data);
        tbl.push("heads", "pae_pred [untrained]", "cumulative", &m.logits_pae.data,
                 &step.get("out::model.pae_pred").data);
        tbl.push("heads", "pde_pred [untrained]", "cumulative", &m.logits_pde.data,
                 &step.get("out::model.pde_pred").data);
        tbl.push("heads", "bind_pred [untrained]", "cumulative", &[m.p_bind],
                 &step.get("out::model.bind_pred").data);

        // `atom37` is one backbone per refinement block; the reference captured
        // all 40 as [1, 40, L, 37, 3], so they are compared as one flat run.
        let atom37: Vec<f32> = out.atom37.concat();
        tbl.push("wrapper", "forward_from_rfi.atom37", "cumulative", &atom37,
                 &scof.get("ffr0.atom37").data);
        tbl.push("wrapper", "forward_from_rfi.px0", "cumulative", out.px0(),
                 &scof.get("step.px0").data);
    }

    // =====================================================================
    // H. output — the two run files, if the end-to-end runs were done
    // =====================================================================
    let ref_pdb = root("runs/M0584_1ldm_T2/ref/design_0-atomized-bb-False.pdb");
    let rs_pdb = root("runs/M0584_1ldm_T2/rs/design_0-atomized-bb-False.pdb");
    if Path::new(&ref_pdb).exists() && Path::new(&rs_pdb).exists() {
        let (ra, rb) = (std::fs::read_to_string(&ref_pdb).unwrap(),
                        std::fs::read_to_string(&rs_pdb).unwrap());
        let (la, lb): (Vec<&str>, Vec<&str>) = (ra.lines().collect(), rb.lines().collect());
        assert_eq!(la.len(), lb.len(), "the two output files differ in line count");
        let coord = |l: &str| -> Option<[f32; 3]> {
            if !(l.starts_with("ATOM") || l.starts_with("HETATM")) || l.len() < 54 {
                return None;
            }
            Some([
                l[30..38].trim().parse().ok()?,
                l[38..46].trim().parse().ok()?,
                l[46..54].trim().parse().ok()?,
            ])
        };
        let (mut g, mut wv) = (Vec::new(), Vec::new());
        for (x, y) in la.iter().zip(&lb) {
            if let (Some(p), Some(q)) = (coord(x), coord(y)) {
                g.extend_from_slice(&q);
                wv.extend_from_slice(&p);
            }
        }
        tbl.push("output", "written .pdb coordinates", "e2e", &g, &wv);
        let same = la.iter().zip(&lb).filter(|(x, y)| x == y).count();
        println!("\noutput file: {same} / {} lines byte-identical", la.len());
    } else {
        println!("\n(no end-to-end run in runs/M0584_1ldm_T2/ — output row skipped)");
    }

    // =====================================================================
    tbl.print();
    tbl.write_tsv(&root("results/layerwise_M0584_1ldm.tsv"));
    println!("total {:.1} s", t_start.elapsed().as_secs_f64());

    // ---- what is asserted -------------------------------------------------
    // STANDALONE is the real gate: fed the reference's own inputs, every module
    // must reproduce the reference bit for bit. Exactly ONE exception remains,
    // and only because it has been adjudicated in exact arithmetic.
    //
    // There were two. Both were single `layer_norm` elements whose exact value
    // lands a few 1e-9 of an fp32 ULP from a midpoint — i.e. ~1e-16 relative,
    // right at f64's resolution — so whichever side's f64 error points the wrong
    // way loses. They resolved in OPPOSITE directions:
    //
    //   main_block.2  tri_mul_out.norm  exact sits 3.12e-9 ULP BELOW the midpoint
    //                 -> the REFERENCE was right and the port was wrong, because
    //                    `layer_norm_f64` summed naively and lost one f64 ULP of
    //                    `var`. FIXED (ops::reduce::sum_compensated); this block
    //                    is now bit-identical.
    //   main_block.23 row_attn.norm_pair exact sits 5.57e-9 ULP ABOVE the midpoint
    //                 -> the PORT is correctly rounded and the reference is the
    //                    1-ULP-wrong side. Its mean and var already agree to
    //                    float80, so no summation change applies; "fixing" it
    //                    would mean reproducing ATen's rounding error on purpose.
    //
    // This asserts the MEASURED status, not an aspiration. A second block, a
    // different module, or a larger gap fails.
    const TIE_BLOCKS: [&str; 1] = ["main_block.23."];
    // The cap is on the rows the ladder prints — `pair2pair`'s and `str2str`'s
    // OUTPUTS, i.e. the tie after amplification through the block's residual
    // chain and sigmoid gate, not the 9.5e-7 at the origin. Measured worst:
    // 3.052e-5 on main_block.23.pair2pair, ~1.4e-7 of `pair`'s own RMS (~214).
    const TIE_MAX_ABS: f32 = 5e-5;

    let bad = tbl.inexact("standalone");
    for r in &bad {
        println!("standalone not exact: {:<40} {}", r.stage, r.s.summary());
    }
    let unexpected: Vec<&&Row> =
        bad.iter().filter(|r| !TIE_BLOCKS.iter().any(|p| r.stage.starts_with(p))).collect();
    assert!(
        unexpected.is_empty(),
        "{} standalone stages are not bit-exact outside the two bisected ties \
         in main_block.2 / main_block.23: {:?}",
        unexpected.len(),
        unexpected.iter().map(|r| &r.stage).collect::<Vec<_>>()
    );
    for r in &bad {
        assert!(
            r.s.max_abs < TIE_MAX_ABS,
            "{} differs by {:.3e}, far more than the bisected tie-straddle limit {:.0e} \
             — this is no longer the documented rounding tie",
            r.stage,
            r.s.max_abs,
            TIE_MAX_ABS
        );
    }

    // CUMULATIVE inherits that one tie and amplifies it through 40 blocks, so it
    // is judged against the tensor's own scale rather than at tolerance 0.
    let n_exact_cum = tbl.rows.iter().filter(|r| r.mode == "cumulative" && r.s.exact == r.s.n)
        .count();
    let n_cum = tbl.rows.iter().filter(|r| r.mode == "cumulative").count();
    println!("\ncumulative: {n_exact_cum}/{n_cum} stages bit-identical end to end");
    for r in tbl.rows.iter().filter(|r| r.mode == "cumulative") {
        assert!(!r.s.any_nan, "{} has NaNs", r.stage);
        // A bit-identical row needs no further check, and asking for one is
        // actively wrong on tensors that legitimately hold infinities:
        // `rfi.dist_matrix` is a graph distance with +inf for every unbonded
        // pair (3048 of 5041 here), so `inf - inf` makes `mean_abs` and
        // `cosine` NaN even when all 5041 values are byte-equal.
        if r.s.exact == r.s.n {
            continue;
        }
        assert!(
            r.s.cosine.is_finite() && r.s.cosine > 0.999_999_9,
            "{} diverged: {}",
            r.stage,
            r.s.summary()
        );
    }
}
