//! Bisection harness for `sample_init`: where does the coordinate difference
//! enter? Not a gate — it prints and never asserts.

use rfd2::indep::make_indep;
use rfd2::ligand::LigandSet;
use rfd2::nn::Ctx;
use rfd2::noiser::{add_fake_frame_legs, forward_marginal, rigid_frames_from_atom_14, Igso3, Rigids};
use rfd2::openfold::atom37_from_rigid;
use rfd2::rng::torch::Mt19937;
use rfd2::weights::Weights;
use std::path::Path;

const PDB: &str = "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb";
const CONTIGS: &str = "10,A106-106,10";

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

fn report(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: len");
    let mut e = 0;
    let mut worst = 0.0f32;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits() {
            e += 1;
        } else {
            if first.is_none() {
                first = Some((i, *g, *w));
            }
            let d = (g - w).abs();
            if d.is_finite() && d > worst {
                worst = d;
            }
        }
    }
    print!("{label:<26} {e} / {} bit-identical  max|d| {worst:.4e}", got.len());
    if let Some((i, g, w)) = first {
        print!("   first diff [{i}] got {g:.9e} want {w:.9e}");
    }
    println!();
}

#[test]
fn bisect_diffuse() {
    let Some(f) = open("fixtures/sample_init/stages.safetensors") else {
        return;
    };
    let Some(nf) = open("fixtures/noiser/stages.safetensors") else {
        return;
    };
    let pdb_path = root(PDB);
    if !Path::new(&pdb_path).exists() {
        return;
    }
    let text = std::fs::read_to_string(&pdb_path).expect("read pdb");
    let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
        .expect("ligand sidecar");
    let feats = rfd2::pdb::parse_pdb_str(&text, true, true);
    let cmap = rfd2::contig::ContigMap::parse(&feats, CONTIGS).expect("contigs");
    let indep0 = make_indep(&feats, &names, &topo).expect("make_indep");
    let init_crds = rfd2::chemical::table_f32("INIT_CRDS");
    let (indep, _masks) =
        rfd2::insert::insert_contig_pre_atomization(&indep0, &cmap, &[true], &init_crds.data);
    let l = indep.len();

    report("indep.xyz vs d0.in_xyz", &indep.xyz, &nf.get("d0.in_xyz").data);

    let state: Vec<u8> = f
        .get_i64("rng.at_sample_init")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&state));

    let xyz = add_fake_frame_legs(&indep.xyz, l, &indep.is_sm, &mut ctx);
    report("after fake legs", &xyz, &nf.get("legs0.out").data);

    let (rots, trans) = rigid_frames_from_atom_14(&xyz, l, rfd2::chemical_gen::NTOTAL);
    report("rigids_0.rots", &rots, &nf.get("d0.rigids_0_rots").data);
    report("rigids_0.trans", &trans, &nf.get("d0.rigids_0_trans").data);

    let omega = nf.get("igso3.omega_grid").data;
    let cdf = nf.get("igso3.cdf");
    let n = cdf.shape[1];
    let ig = Igso3::new(omega, cdf.data[(cdf.shape[0] - 1) * n..].to_vec());

    let all = vec![true; l];
    let rt = forward_marginal(&Rigids { rots, trans }, 1.0, &all, false, &ig, &mut ctx);
    report("rigids_t.rots", &rt.rots, &nf.get("d0.rigids_t_rots").data);
    report("rigids_t.trans", &rt.trans, &nf.get("d0.rigids_t_trans").data);

    report("a37_0.in_rots", &rt.rots, &nf.get("a37_0.in_rots").data);
    report("a37_0.in_trans", &rt.trans, &nf.get("a37_0.in_trans").data);

    let xt = atom37_from_rigid(&rt, &mut ctx);
    report("atom37", &xt, &nf.get("a37_0.out").data);

    // narrowed to 36 slots, which is what `diffuse` assigns
    let nt = rfd2::chemical_gen::NTOTAL;
    let mut narrowed = vec![0.0f32; l * nt * 3];
    for i in 0..l {
        for a in 0..nt {
            for c in 0..3 {
                narrowed[(i * nt + a) * 3 + c] = xt[(i * 37 + a) * 3 + c];
            }
        }
    }
    report("d0.out_xyz", &narrowed, &nf.get("d0.out_xyz").data);
}
