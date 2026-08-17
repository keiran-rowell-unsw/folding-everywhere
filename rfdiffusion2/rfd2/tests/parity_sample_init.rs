//! Rung 4e, end to end — `sample_init` from the PDB on disk.
//!
//! Every earlier rung-4e test starts from the reference's own captured input
//! for one stage. This one starts from nothing but the file and the seed, runs
//! the whole chain, and compares the three structures the sampler is handed.
//! It is therefore the test that would catch a stage boundary being wired up
//! wrong even though both stages are individually exact.
//!
//! `xyz` is compared NaN-aware: which slots are NaN is load-bearing, and
//! `parity::compare` deliberately skips them.

use rfd2::indep::Indep;
use rfd2::ligand::LigandSet;
use rfd2::nn::Ctx;
use rfd2::noiser::Igso3;
use rfd2::rng::torch::Mt19937;
use rfd2::sample_init::{Options, SampleInit};
use rfd2::weights::Weights;
use std::path::Path;

const PDB: &str = "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb";
const CONTIGS: &str = "10,A106-106,10";
/// `diffuser.T` of the captured run.
const BIG_T: usize = 2;

fn root(rel: &str) -> String {
    format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"))
}

fn open(rel: &str) -> Option<Weights> {
    let p = root(rel);
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing");
        return None;
    }
    Some(Weights::open(&p).expect("open"))
}

fn cmp_nan(got: &[f32], want: &[f32]) -> (usize, usize) {
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let e = got
        .iter()
        .zip(want)
        .filter(|(a, b)| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits())
        .count();
    (e, got.len())
}

/// The last reachable row of the 1000x1000 CDF — `_corrupt_rotmats_multi_t`
/// hard-codes `sigma = 1.5` and `bucketize` puts that at index 999.
fn igso3_from(f: &Weights) -> Igso3 {
    let omega = f.get("igso3.omega_grid").data;
    let cdf = f.get("igso3.cdf");
    let n = cdf.shape[1];
    let row = cdf.data[(cdf.shape[0] - 1) * n..].to_vec();
    Igso3::new(omega, row)
}

fn check(name: &str, got: &Indep, f: &Weights, prefix: &str, bad: &mut Vec<String>) {
    println!("{name}:");
    let mut chk_i64 = |field: &str, g: &[i64]| {
        let want = f.get_i64(&format!("{prefix}.{field}")).0;
        if g.len() != want.len() {
            println!("  {field:<14} LEN {} vs {}", g.len(), want.len());
            bad.push(format!("{name}.{field}"));
            return;
        }
        let diff = g.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("  {field:<14} {} / {} exact", g.len() - diff, want.len());
        if diff != 0 {
            bad.push(format!("{name}.{field}"));
        }
    };
    chk_i64("seq", &got.seq);
    chk_i64("idx", &got.idx);
    chk_i64("bond_feats", &got.bond_feats);
    chk_i64(
        "same_chain",
        &got.same_chain.iter().map(|b| *b as i64).collect::<Vec<_>>(),
    );
    chk_i64("is_gp", &got.is_gp.iter().map(|b| *b as i64).collect::<Vec<_>>());

    for (field, g) in [
        ("terminus_type", got.terminus_type.as_slice()),
        ("chirals", got.chirals.as_slice()),
        ("xyz", got.xyz.as_slice()),
    ] {
        let want = f.get(&format!("{prefix}.{field}"));
        if g.len() != want.data.len() {
            println!("  {field:<14} LEN {} vs {}", g.len(), want.data.len());
            bad.push(format!("{name}.{field}"));
            continue;
        }
        let (e, n) = cmp_nan(g, &want.data);
        println!("  {field:<14} {e} / {n} bit-identical (NaN-aware)");
        if e != n {
            bad.push(format!("{name}.{field}"));
        }
    }
}

#[test]
fn sample_init_matches() {
    let Some(f) = open("fixtures/sample_init/stages.safetensors") else {
        return;
    };
    let Some(nf) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let pdb_path = root(PDB);
    if !Path::new(&pdb_path).exists() {
        eprintln!("SKIP: {pdb_path} missing");
        return;
    }
    let text = std::fs::read_to_string(&pdb_path).expect("read pdb");
    let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
        .expect("ligand sidecar");
    let igso3 = igso3_from(&nf);

    // the generator exactly as `seed_all(seed_offset)` left it
    let bytes: Vec<u8> = f
        .get_i64("rng.at_sample_init")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));

    let opt = Options {
        big_t: BIG_T,
        ..Options::default()
    };
    let out = SampleInit::run(&text, &names, &topo, CONTIGS, &opt, &igso3, &mut ctx, &mut None)
        .expect("sample_init");

    println!(
        "L = {}, {} diffused, t_step_input = {}, {} torch draws",
        out.indep.len(),
        out.is_diffused.iter().filter(|d| **d).count(),
        out.t_step_input,
        ctx.rng.draws()
    );

    let mut bad = Vec::new();
    check("indep (cond)", &out.indep, &f, "out_indep", &mut bad);
    // `dtac_*` are captured inside `diffuse_then_add_conditional`, which
    // separates a wiring error in the motif copy-back from an arithmetic error
    // in the noiser: the unconditional structure is `diffuse` alone.
    check("indep_uncond", &out.indep_uncond, &f, "dtac_uncond", &mut bad);
    check("indep_orig", &out.indep_orig, &f, "out_indep_orig", &mut bad);

    let want_diff: Vec<bool> = f
        .get_i64("out.is_diffused")
        .0
        .into_iter()
        .map(|v| v != 0)
        .collect();
    let ok = out.is_diffused == want_diff;
    println!("is_diffused    {}", if ok { "match" } else { "DIFFER" });
    if !ok {
        bad.push("is_diffused".into());
    }
    let want_t = f.get_i64("out.t_step_input").0[0] as usize;
    println!("t_step_input   {} vs {want_t}", out.t_step_input);
    if out.t_step_input != want_t {
        bad.push("t_step_input".into());
    }

    // The generator must land exactly where the reference's did, or every draw
    // the sampler makes afterwards is shifted.
    let after: Vec<u8> = f
        .get_i64("rng.after_sample_init")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut want_next = Ctx::new(Mt19937::from_torch_state(&after));
    let a: Vec<f32> = (0..64).map(|_| ctx.rng.uniform_f32()).collect();
    let b: Vec<f32> = (0..64).map(|_| want_next.rng.uniform_f32()).collect();
    let (eg, ng) = cmp_nan(&a, &b);
    println!("following RNG draw  {eg} / {ng} bit-identical");
    if eg != ng {
        bad.push("rng position".into());
    }

    assert!(bad.is_empty(), "sample_init not bit-exact: {bad:?}");
}
