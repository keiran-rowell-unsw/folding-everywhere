//! Bisection for the output path: run it from the *reference's* `px0` so any
//! difference is the writer's, not the network's. Prints, never asserts.

use rfd2::design::save_outputs;
use rfd2::indep::Indep;
use rfd2::ligand::LigandSet;
use rfd2::weights::Weights;
use std::path::Path;

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

#[test]
fn output_from_reference_px0() {
    let Some(f) = open("fixtures/sampler/T2.safetensors") else { return };
    let Some(fx) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let ref_pdb = std::env::var("RFD2_REF_PDB").unwrap_or_default();
    if ref_pdb.is_empty() || !Path::new(&ref_pdb).exists() {
        eprintln!("SKIP: set RFD2_REF_PDB to the reference output .pdb");
        return;
    }
    let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
        .expect("sidecar");
    let input = std::fs::read_to_string(root(
        "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb",
    ))
    .expect("input pdb");

    // the reference's own trajectory end-point and structures
    let px0 = f.get("stack.px0"); // [n_steps, L, 37, 3], already flipped
    let l = fx.get_i64("indep.seq").0.len();
    let last = &px0.data[..l * 37 * 3];

    let mk = |xyz: Vec<f32>| Indep {
        seq: f.get_i64("out.seq").0,
        xyz,
        idx: fx.get_i64("indep.idx").0,
        bond_feats: fx.get_i64("indep.bond_feats").0,
        chirals: fx.get("indep.chirals").data,
        same_chain: fx.get_i64("indep.same_chain").0.into_iter().map(|v| v != 0).collect(),
        is_gp: fx.get_i64("indep.is_gp").0.into_iter().map(|v| v != 0).collect(),
        terminus_type: fx.get("indep.terminus_type").data,
        is_sm: fx.get_i64("indep.is_sm").0.into_iter().map(|v| v != 0).collect(),
    };
    let indep = mk(f.get("s1.x_t").data);
    let indep_orig = mk(f.get("out.indep_orig_xyz").data);
    let is_diffused: Vec<bool> =
        f.get_i64("out.is_diffused").0.into_iter().map(|v| v != 0).collect();

    let mut lignames = vec![String::new(); l];
    let mut k = 0usize;
    let sm: Vec<usize> = (0..l).filter(|&i| indep.is_sm[i]).collect();
    for n in topo.names() {
        let cnt = topo.get(n).map(|t| t.n_atoms).unwrap_or(0);
        for _ in 0..cnt {
            lignames[sm[k]] = n.clone();
            k += 1;
        }
    }

    let got = save_outputs(last, &indep, &indep_orig, &is_diffused, &lignames, &input, &topo);
    let want = std::fs::read_to_string(&ref_pdb).expect("ref pdb");
    let gl: Vec<&str> = got.lines().collect();
    let wl: Vec<&str> = want.lines().collect();
    println!("lines: got {} want {}", gl.len(), wl.len());
    let n = gl.len().min(wl.len());
    let diff = (0..n).filter(|&i| gl[i] != wl[i]).count();
    println!("differing lines: {diff} / {n}");
    for i in 0..n {
        if gl[i] != wl[i] {
            println!("  want |{}|", wl[i]);
            println!("  got  |{}|", gl[i]);
            break;
        }
    }
}
