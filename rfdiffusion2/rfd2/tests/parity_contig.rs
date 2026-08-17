//! Rung 4e, stage 2 — `ContigMap` and `insert_contig_pre_atomization`.
//!
//! The stage where the row count changes: 4 protein residues + 50 ligand atoms
//! become 21 designed residues + the same 50 atoms. Everything downstream
//! indexes into that layout, so a single misplaced row would be invisible here
//! and catastrophic three stages later.

use rfd2::contig::ContigMap;
use rfd2::indep::make_indep;
use rfd2::insert::insert_contig_pre_atomization;
use rfd2::ligand::LigandSet;
use rfd2::parity;
use rfd2::pdb;
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

fn cmp_nan(got: &[f32], want: &[f32]) -> (usize, usize) {
    assert_eq!(got.len(), want.len());
    let mut e = 0;
    for (g, w) in got.iter().zip(want) {
        if (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits() {
            e += 1;
        }
    }
    (e, got.len())
}

#[test]
fn contig_insertion_matches() {
    let Some(f) = open("fixtures/sample_init/stages.safetensors") else { return };
    let pdb_path = root(PDB);
    if !Path::new(&pdb_path).exists() {
        eprintln!("SKIP: {pdb_path} missing");
        return;
    }
    let text = std::fs::read_to_string(&pdb_path).expect("read pdb");
    let names: Vec<String> = ["NAD", "OXM"].iter().map(|s| s.to_string()).collect();
    let topo = LigandSet::load(&root("fixtures/ligand/M0584_1ldm.safetensors"), &names)
        .expect("ligand sidecar");
    let feats = pdb::parse_pdb_str(&text, true, true);
    let indep0 = make_indep(&feats, &names, &topo).expect("make_indep");

    let cmap = ContigMap::parse(&feats, CONTIGS).expect("contig parse");
    println!(
        "contig: {} designed rows, {} chains, {} mapped from the reference",
        cmap.contig_length,
        cmap.n_inpaint_chains,
        cmap.hal_idx0.len()
    );
    // the contig index arrays, before the ligand rows are appended
    let want_hal = f.get_i64("out.hal_idx0").0;
    let want_ref = f.get_i64("out.ref_idx0").0;
    let n_contig = cmap.hal_idx0.len();
    let got_hal: Vec<i64> = cmap.hal_idx0.iter().map(|v| *v as i64).collect();
    let got_ref: Vec<i64> = cmap.ref_idx0.iter().map(|v| *v as i64).collect();
    println!(
        "  hal_idx0[..{n_contig}] {}   ref_idx0[..{n_contig}] {}",
        if got_hal == want_hal[..n_contig] { "match" } else { "DIFFER" },
        if got_ref == want_ref[..n_contig] { "match" } else { "DIFFER" }
    );

    let chemical = rfd2::chemical::table_f32("INIT_CRDS");
    let has_termini = [true];
    let (indep, masks) =
        insert_contig_pre_atomization(&indep0, &cmap, &has_termini, &chemical.data);
    println!("L after insertion = {}", indep.len());

    let mut bad = Vec::new();
    let mut chk_i64 = |name: &str, got: &[i64]| {
        let want = f.get_i64(&format!("s2_insert_contig.{name}")).0;
        if got.len() != want.len() {
            println!("  {name:<15} LEN {} vs {}", got.len(), want.len());
            bad.push(name.to_string());
            return;
        }
        let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("  {name:<15} {} / {} exact", got.len() - diff, want.len());
        if diff != 0 {
            bad.push(name.to_string());
        }
    };
    chk_i64("seq", &indep.seq);
    chk_i64("idx", &indep.idx);
    chk_i64("bond_feats", &indep.bond_feats);
    chk_i64("same_chain", &indep.same_chain.iter().map(|b| *b as i64).collect::<Vec<_>>());
    chk_i64("is_gp", &indep.is_gp.iter().map(|b| *b as i64).collect::<Vec<_>>());

    for (name, got) in [
        ("terminus_type", indep.terminus_type.as_slice()),
        ("chirals", indep.chirals.as_slice()),
        ("xyz", indep.xyz.as_slice()),
    ] {
        let want = f.get(&format!("s2_insert_contig.{name}"));
        if got.len() != want.data.len() {
            println!("  {name:<15} LEN {} vs {}", got.len(), want.data.len());
            bad.push(name.to_string());
            continue;
        }
        let (e, n) = cmp_nan(got, &want.data);
        let s = parity::compare(got, &want.data);
        println!("  {name:<15} {e} / {n} exact (NaN-aware)  max|d| {:.3e}", s.max_abs);
        if e != n {
            bad.push(name.to_string());
        }
    }

    // the 1-D masks, if the fixture carries them
    for (name, got) in [
        ("is_res_str_shown", &masks.is_res_str_shown),
        ("is_res_seq_shown", &masks.is_res_seq_shown),
    ] {
        let key = format!("s2_masks.{name}");
        if !f.has(&key) {
            continue;
        }
        let want = f.get_i64(&key).0;
        let g: Vec<i64> = got.iter().map(|b| *b as i64).collect();
        let ok = g == want;
        println!("  {name:<15} {}", if ok { "match" } else { "DIFFER" });
        if !ok {
            bad.push(name.to_string());
        }
    }

    assert!(bad.is_empty(), "contig insertion not bit-exact: {bad:?}");
}
