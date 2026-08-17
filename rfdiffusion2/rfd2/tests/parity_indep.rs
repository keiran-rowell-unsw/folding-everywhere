//! Rung 4e, stage 1 — `aa_model.make_indep`.
//!
//! The first thing `sample_init` builds, and the structure every later stage
//! transforms. Compared field by field against `s1_make_indep.*` in
//! `fixtures/sample_init/stages.safetensors`, which was captured by hooking the
//! real function.
//!
//! `xyz` is compared with NaN treated as a *value*, not as a tolerance question:
//! which of the 36 atom slots are NaN is load-bearing (the network's masks are
//! derived from it), so a NaN in the wrong place has to fail.

use rfd2::indep::make_indep;
use rfd2::ligand::LigandSet;
use rfd2::parity;
use rfd2::pdb;
use rfd2::weights::Weights;
use std::path::Path;

const PDB: &str = "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb";
const LIGANDS: [&str; 2] = ["NAD", "OXM"];

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

/// Bitwise comparison that treats NaN as equal to NaN — the fixture and the port
/// must agree on *where* the NaNs are, and `parity::compare` deliberately skips
/// them.
fn cmp_with_nan(got: &[f32], want: &[f32]) -> (usize, usize, usize) {
    assert_eq!(got.len(), want.len(), "length {} vs {}", got.len(), want.len());
    let mut exact = 0;
    let mut nan_mismatch = 0;
    for (g, w) in got.iter().zip(want) {
        if g.is_nan() && w.is_nan() {
            exact += 1;
        } else if g.is_nan() != w.is_nan() {
            nan_mismatch += 1;
        } else if g.to_bits() == w.to_bits() {
            exact += 1;
        }
    }
    (exact, got.len(), nan_mismatch)
}

#[test]
fn make_indep_matches() {
    let Some(f) = open("fixtures/sample_init/stages.safetensors") else { return };
    let pdb_path = root(PDB);
    if !Path::new(&pdb_path).exists() {
        eprintln!("SKIP: {pdb_path} missing");
        return;
    }
    let text = std::fs::read_to_string(&pdb_path).expect("read pdb");
    let names: Vec<String> = LIGANDS.iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
        .expect("ligand sidecar");

    // `make_indep` removes the non-target HETATM records before parsing, so the
    // ligand rows are exactly the named ligands in the order given.
    let feats = pdb::parse_pdb_str(&text, true, true);
    let indep = make_indep(&feats, &names, &topo).expect("make_indep");

    let l = indep.len();
    println!("L = {l}  ({} polymer + {} ligand atoms)",
             indep.is_sm.iter().filter(|s| !**s).count(),
             indep.is_sm.iter().filter(|s| **s).count());

    let mut bad = Vec::new();
    let mut chk_i64 = |name: &str, got: &[i64]| {
        let want = f.get_i64(&format!("s1_make_indep.{name}")).0;
        let n = got.len().min(want.len());
        let same = got.len() == want.len() && got == want.as_slice();
        let diff = (0..n).filter(|&i| got[i] != want[i]).count();
        println!("  {name:<15} {} / {} exact{}", got.len() - diff, want.len(),
                 if got.len() != want.len() { format!(" [LEN {} vs {}]", got.len(), want.len()) } else { String::new() });
        if !same {
            bad.push(name.to_string());
        }
    };
    chk_i64("seq", &indep.seq);
    chk_i64("idx", &indep.idx);
    chk_i64("bond_feats", &indep.bond_feats);
    let sc: Vec<i64> = indep.same_chain.iter().map(|b| *b as i64).collect();
    chk_i64("same_chain", &sc);
    let gp: Vec<i64> = indep.is_gp.iter().map(|b| *b as i64).collect();
    chk_i64("is_gp", &gp);

    for (name, got) in [
        ("terminus_type", indep.terminus_type.as_slice()),
        ("chirals", indep.chirals.as_slice()),
        ("xyz", indep.xyz.as_slice()),
    ] {
        let want = f.get(&format!("s1_make_indep.{name}"));
        if got.len() != want.data.len() {
            println!("  {name:<15} LEN {} vs {}", got.len(), want.data.len());
            bad.push(name.to_string());
            continue;
        }
        let (exact, n, nanmis) = cmp_with_nan(got, &want.data);
        let s = parity::compare(got, &want.data);
        println!("  {name:<15} {exact} / {n} exact (NaN-aware){}  max|d| {:.3e}",
                 if nanmis > 0 { format!("  [{nanmis} NaN-placement mismatches]") } else { String::new() },
                 s.max_abs);
        if exact != n {
            bad.push(name.to_string());
        }
    }

    assert!(bad.is_empty(), "make_indep fields not bit-exact: {bad:?}");
}
