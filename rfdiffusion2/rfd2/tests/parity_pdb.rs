//! Rung 4c — PDB parsing. **Tolerance: exactly 0.**
//!
//! Three real inputs, including two ligand-bearing ones, so HETATM handling and
//! the 4-character atom-name columns are exercised rather than assumed.

use rfd2::pdb;
use rfd2::weights::Weights;
use std::path::Path;

fn fixture() -> Option<(Weights, serde_json::Value)> {
    let root = env!("CARGO_MANIFEST_DIR");
    let st = format!("{root}/../fixtures/parse/parse.safetensors");
    let js = format!("{root}/../fixtures/parse/parse_meta.json");
    if !Path::new(&st).exists() || !Path::new(&js).exists() {
        eprintln!("SKIP: run python/gen_parse_fixtures.py first");
        return None;
    }
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&js).unwrap()).unwrap();
    Some((Weights::open(&st).unwrap(), meta))
}

fn ref_root() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{root}/../../ref_RFdiffusion2")
}

#[test]
fn parse_matches_reference_on_real_inputs() {
    let Some((f, meta)) = fixture() else { return };
    let mut n_cases = 0usize;

    for (tag, m) in meta.as_object().unwrap() {
        let rel = m["path"].as_str().unwrap();
        let path = format!("{}/{}", ref_root(), rel);
        if !Path::new(&path).exists() {
            eprintln!("  (skip {tag}: {path} missing)");
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let got = pdb::parse_pdb_str(&text, true, true);

        // residue count
        let n_res = m["n_res"].as_u64().unwrap() as usize;
        assert_eq!(got.len(), n_res, "{tag}: residue count");

        // sequence tokens
        let (want_seq, _) = f.get_i64(&format!("{tag}.seq"));
        assert_eq!(got.seq(), want_seq, "{tag}: seq tokens");

        // residue numbering
        let (want_idx, _) = f.get_i64(&format!("{tag}.idx"));
        assert_eq!(got.idx(), want_idx, "{tag}: residue numbering");

        // (chain, resSeq) pairs
        let want_pdb_idx = m["pdb_idx"].as_array().unwrap();
        assert_eq!(want_pdb_idx.len(), got.len(), "{tag}: pdb_idx length");
        for (i, r) in got.residues.iter().enumerate() {
            let w = want_pdb_idx[i].as_array().unwrap();
            assert_eq!(r.chain, w[0].as_str().unwrap(), "{tag}: chain[{i}]");
            assert_eq!(r.res_seq, w[1].as_i64().unwrap(), "{tag}: resSeq[{i}]");
        }

        // atom mask
        let (want_mask, _) = f.get_i64(&format!("{tag}.mask"));
        assert_eq!(got.mask.len(), want_mask.len(), "{tag}: mask size");
        for (i, (g, w)) in got.mask.iter().zip(&want_mask).enumerate() {
            assert_eq!(*g, *w != 0, "{tag}: mask[{i}]");
        }

        // coordinates, bit for bit (missing atoms zeroed on both sides)
        let want_xyz = f.get(&format!("{tag}.xyz"));
        assert_eq!(got.xyz.len(), want_xyz.data.len(), "{tag}: xyz size");
        for (i, (g, w)) in got.xyz.iter().zip(&want_xyz.data).enumerate() {
            assert_eq!(
                g.to_bits(), w.to_bits(),
                "{tag}: xyz[{i}] got {g} want {w}"
            );
        }

        // heteroatoms
        let n_het = m["n_het"].as_u64().unwrap() as usize;
        assert_eq!(got.het.len(), n_het, "{tag}: hetatom count");
        let want_info = m["info_het"].as_array().unwrap();
        for (i, h) in got.het.iter().enumerate() {
            let w = &want_info[i];
            assert_eq!(h.idx, w["idx"].as_i64().unwrap(), "{tag}: het[{i}].idx");
            assert_eq!(h.atom_id, w["atom_id"].as_str().unwrap(), "{tag}: het[{i}].atom_id");
            assert_eq!(h.name, w["name"].as_str().unwrap(), "{tag}: het[{i}].name");
            assert_eq!(h.res_idx, w["res_idx"].as_i64().unwrap(), "{tag}: het[{i}].res_idx");
            assert_eq!(h.atom_type, w["atom_type"].as_str().unwrap(),
                       "{tag}: het[{i}].atom_type");
        }
        if got.het.len() > 0 {
            let want_het_xyz = f.get(&format!("{tag}.xyz_het"));
            for (i, h) in got.het.iter().enumerate() {
                for c in 0..3 {
                    let w = want_het_xyz.data[i * 3 + c];
                    assert_eq!(h.xyz[c].to_bits(), w.to_bits(),
                               "{tag}: het[{i}].xyz[{c}]");
                }
            }
        }

        let n_present: usize = got.mask.iter().filter(|m| **m).count();
        println!(
            "  {tag}: {} residues, {n_present} atoms placed, {} hetatoms — exact",
            got.len(), got.het.len()
        );
        n_cases += 1;
    }
    assert!(n_cases > 0, "no parse cases ran");
    println!("pdb parse: {n_cases} real inputs bit-identical to the reference");
}

/// The reference drops HETATM hydrogens by the **element column** (`l[77]`),
/// not by atom name. A parser that filtered on the name would keep an atom
/// named e.g. "HG" that is really mercury, or drop one named "H1" that the
/// reference keeps because its element column says something else.
#[test]
fn hetatom_hydrogen_filter_uses_the_element_column() {
    let line_h = b"HETATM   91  H1  LIG B 332       0.000   0.000   0.000  1.00  0.00           H  ";
    let line_hg = b"HETATM   92  HG  LIG B 332       1.000   0.000   0.000  1.00  0.00          HG  ";
    let lines: Vec<&[u8]> = vec![line_h, line_hg];

    let kept = pdb::parse_pdb_lines_target(&lines, true, true);
    assert_eq!(kept.het.len(), 1, "the element-H line must be dropped");
    assert_eq!(kept.het[0].atom_id.trim(), "HG");

    let all = pdb::parse_pdb_lines_target(&lines, true, false);
    assert_eq!(all.het.len(), 2, "ignore_het_h=false keeps both");
    println!("hetatom H filter: keyed on element column 78, not atom name");
}

/// A genuine asymmetry in the reference, verified against it directly:
///
/// * **residue identity** comes from the FIRST line for a `(chain, resSeq)`
///   pair, because `first_atom_iter` deduplicates before anything else;
/// * **coordinates** come from the LAST matching line, because the `break` in
///   the placement loop exits the scan over `aa2long` — not the loop over
///   lines — so a later duplicate simply overwrites the earlier value.
///
/// It is easy to assume both go the same way. They do not, and a port that
/// picked "first wins" for coordinates would differ only on files with
/// duplicated atoms (alternate locations, merged models) — i.e. rarely, and
/// silently.
#[test]
fn identity_takes_first_line_but_coordinates_take_last() {
    let l1 = b"ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N  ";
    let l2 = b"ATOM      2  CA  ALA A   1       2.000   0.000   0.000  1.00  0.00           C  ";
    let l3 = b"ATOM      3  CA  ALA A   1       9.000   0.000   0.000  1.00  0.00           C  ";
    let lines: Vec<&[u8]> = vec![l1, l2, l3];
    let tf = pdb::parse_pdb_lines_target(&lines, false, true);
    assert_eq!(tf.len(), 1, "one residue");
    let ca_x = tf.xyz[(0 * rfd2::chemical_gen::NHEAVY + 1) * 3];
    assert_eq!(ca_x, 9.0, "duplicate atom: the LAST line must win");

    let m1 = b"ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N  ";
    let m2 = b"ATOM      2  CA  GLY A   1       2.000   0.000   0.000  1.00  0.00           C  ";
    let lines2: Vec<&[u8]> = vec![m1, m2];
    let tf2 = pdb::parse_pdb_lines_target(&lines2, false, true);
    assert_eq!(tf2.seq(), vec![0], "identity: ALA (0) from the first line, not GLY (7)");

    println!("duplicate handling: identity = first line, coordinates = last line");
}
