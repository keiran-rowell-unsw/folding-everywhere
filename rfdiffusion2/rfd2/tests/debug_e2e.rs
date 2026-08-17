//! Attribute the end-to-end PDB difference: run the port's own loop, then write
//! the output twice — once from the port's `px0` and once from the reference's
//! — and diff the two. Whatever differs is caused by `px0`; whatever is common
//! is the writer's.

use rfd2::design::save_outputs;
use rfd2::ligand::LigandSet;
use rfd2::model::rf::{Arch, RoseTTAFold};
use rfd2::nn::{Ctx, Params};
use rfd2::noiser::Igso3;
use rfd2::prepro::PreproOptions;
use rfd2::rng::torch::Mt19937;
use rfd2::sample_init::{Options as InitOptions, SampleInit};
use rfd2::sampler::{run_loop, SamplerOptions};
use rfd2::weights::Weights;
use std::path::Path;

fn root(rel: &str) -> String { format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR")) }
fn open(rel: &str) -> Option<Weights> {
    let p = root(rel);
    if !Path::new(&p).exists() { eprintln!("SKIP: {p}"); return None; }
    Some(Weights::open(&p).expect("open"))
}

#[test]
fn attribute_pdb_difference() {
    let Some(f) = open("fixtures/sampler/T2.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(nf) = open("fixtures/noiser/stages.safetensors") else { return };
    let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names).unwrap();
    let input = std::fs::read_to_string(root(
        "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb")).unwrap();
    let omega = nf.get("igso3.omega_grid").data;
    let cdf = nf.get("igso3.cdf");
    let n = cdf.shape[1];
    let igso3 = Igso3::new(omega, cdf.data[(cdf.shape[0] - 1) * n..].to_vec());

    let mut ctx = Ctx::new(Mt19937::new(0));
    let init = SampleInit::run(&input, &names, &topo, "10,A106-106,10",
        &InitOptions { big_t: 2, ..InitOptions::default() }, &igso3, &mut ctx, &mut None).unwrap();
    let mut indep = init.indep;
    let af = topo.atom_frames();
    let opt = SamplerOptions { big_t: 2, final_step: 1, rots_exp_rate: 10,
        prepro: PreproOptions { big_t: 2, ..PreproOptions::default() },
        ..SamplerOptions::default() };
    let model = RoseTTAFold::load(&Params::root(&w, "model"), Arch::rfd173());
    let traj = run_loop(&model, &mut indep, &init.is_diffused, &af, 2, &opt, &mut ctx, |_,_,_| {});
    let mine = traj.px0.last().unwrap().clone();

    let want_px0 = f.get("stack.px0");
    let l = indep.len();
    let theirs = want_px0.data[..l * 37 * 3].to_vec();
    let mut e = 0usize; let mut worst = 0.0f32;
    for (a, b) in mine.iter().zip(&theirs) {
        if a.to_bits() == b.to_bits() { e += 1 } else { worst = worst.max((a - b).abs()) }
    }
    println!("px0: {e} / {} bit-identical, max|d| {worst:.4e} A", mine.len());

    let mut lignames = vec![String::new(); l];
    let sm: Vec<usize> = (0..l).filter(|&i| indep.is_sm[i]).collect();
    let mut k = 0usize;
    for nm in topo.names() {
        for _ in 0..topo.get(nm).unwrap().n_atoms { lignames[sm[k]] = nm.clone(); k += 1; }
    }
    let a = save_outputs(&mine, &indep, &init.indep_orig, &init.is_diffused, &lignames, &input, &topo);
    let b = save_outputs(&theirs, &indep, &init.indep_orig, &init.is_diffused, &lignames, &input, &topo);
    let al: Vec<&str> = a.lines().collect();
    let bl: Vec<&str> = b.lines().collect();
    let diff = al.iter().zip(&bl).filter(|(x, y)| x != y).count();
    println!("PDB from my px0 vs PDB from reference px0: {diff} differing lines of {}", al.len());
    for (x, y) in al.iter().zip(&bl).filter(|(x, y)| x != y).take(4) {
        println!("  mine |{x}|");
        println!("  ref  |{y}|");
    }
}
