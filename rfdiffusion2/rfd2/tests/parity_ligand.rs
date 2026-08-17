//! Ligand topology — the sidecar must reproduce the ligand block of the
//! reference's `indep.bond_feats` **exactly**, and must refuse unknown ligands.

use rfd2::ligand::{LigandError, LigandSet};
use rfd2::weights::Weights;
use std::path::Path;

fn sidecar() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{root}/../fixtures/ligand/M0584_1ldm.safetensors")
}

fn step0() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    format!("{root}/../fixtures/model_pinned/step0.safetensors")
}

#[test]
fn sidecar_reproduces_the_reference_ligand_bond_block() {
    if !Path::new(&sidecar()).exists() || !Path::new(&step0()).exists() {
        eprintln!("SKIP: run python/gen_ligand_bonds.py and python/ref_dump.py");
        return;
    }
    let names = vec!["NAD".to_string(), "OXM".to_string()];
    let set = LigandSet::load(&sidecar(), &names).expect("load sidecar");

    assert_eq!(set.total_atoms(), 50, "NAD(44) + OXM(6)");
    assert_eq!(set.get("NAD").unwrap().n_bonds(), 48);
    assert_eq!(set.get("OXM").unwrap().n_bonds(), 5);

    let (got, n) = set.block_diag_bond_feats();

    // the reference's full bond_feats for the captured run: L = 71, with the
    // ligand occupying the trailing 50 x 50 block
    let f = Weights::open(&step0()).unwrap();
    let (bf, shape) = f.get_i64("indep.bond_feats");
    let l = shape[0];
    let off = l - n;
    assert_eq!(l, 71, "captured run length");
    assert_eq!(n, 50, "ligand block size");

    let mut n_bonds = 0usize;
    for i in 0..n {
        for j in 0..n {
            let g = got[i * n + j];
            let w = bf[(off + i) * l + (off + j)];
            assert_eq!(
                g, w,
                "ligand bond block [{i}][{j}]: sidecar {g} != reference {w}"
            );
            if g > 0 && i < j {
                n_bonds += 1;
            }
        }
    }

    // and the cross terms must be zero: ligands are not bonded to each other or
    // to the protein in bond_feats (those relations live in other channels)
    for i in 0..n {
        for j in 0..off {
            assert_eq!(
                bf[(off + i) * l + j],
                0,
                "ligand atom {i} unexpectedly bonded to protein position {j}"
            );
        }
    }

    println!(
        "ligand sidecar: {n}x{n} bond block exact ({n_bonds} bonds), \
cross-terms to {off} protein positions all zero"
    );
}

/// Element tokens for the ligand tail of `indep.seq`.
#[test]
fn sidecar_elements_match_the_reference_sequence_tail() {
    if !Path::new(&sidecar()).exists() || !Path::new(&step0()).exists() {
        eprintln!("SKIP");
        return;
    }
    let names = vec!["NAD".to_string(), "OXM".to_string()];
    let set = LigandSet::load(&sidecar(), &names).unwrap();
    let elems = set.elements();

    let f = Weights::open(&step0()).unwrap();
    let (seq, _) = f.get_i64("indep.seq");
    let off = seq.len() - elems.len();
    for (i, e) in elems.iter().enumerate() {
        assert_eq!(*e, seq[off + i], "ligand element token [{i}]");
    }
    println!("ligand elements: {} tokens match indep.seq tail", elems.len());
}

/// The scope boundary must fail loudly. A port that silently guessed bond orders
/// would produce plausible-but-wrong topology that nothing downstream detects.
#[test]
fn unknown_ligand_is_refused_not_guessed() {
    if !Path::new(&sidecar()).exists() {
        eprintln!("SKIP");
        return;
    }
    let names = vec!["NAD".to_string(), "NOPE".to_string()];
    match LigandSet::load(&sidecar(), &names) {
        Err(LigandError::LigandNotCovered { name, available }) => {
            assert_eq!(name, "NOPE");
            assert!(available.contains(&"NAD".to_string()));
            println!("unknown ligand refused: {name:?} not in {available:?}");
        }
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("an uncovered ligand must be refused, not guessed"),
    }

    match LigandSet::load("/nonexistent/sidecar.safetensors", &["NAD".to_string()]) {
        Err(LigandError::SidecarMissing { .. }) => {
            println!("missing sidecar refused with an actionable message");
        }
        _ => panic!("a missing sidecar must be refused"),
    }
}

/// `atom_frames` — the frame chosen for every ligand atom.
///
/// This is the field whose *tie-breaking* depends on CPython set iteration
/// order (20 of 50 atoms tie on priority here), which is precisely why it is
/// carried in the sidecar rather than recomputed.
#[test]
fn sidecar_atom_frames_match_the_reference() {
    if !Path::new(&sidecar()).exists() || !Path::new(&step0()).exists() {
        eprintln!("SKIP");
        return;
    }
    let names = vec!["NAD".to_string(), "OXM".to_string()];
    let set = LigandSet::load(&sidecar(), &names).unwrap();
    let got = set.atom_frames();

    let f = Weights::open(&step0()).unwrap();
    let (want, shape) = f.get_i64("rfi.atom_frames"); // [1, 50, 3, 2]
    assert_eq!(shape[1], set.total_atoms(), "frame count == ligand atom count");
    assert_eq!(got.len(), want.len(), "atom_frames size");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "atom_frames[{i}]");
    }
    println!(
        "ligand atom_frames: [{}, 3, 2] = {} values exact",
        set.total_atoms(), got.len()
    );
}

/// `chirals` — chirality constraints, also carried in the sidecar because they
/// come from the same OpenBabel molecule as the bonds.
#[test]
fn sidecar_chirals_match_the_reference() {
    if !Path::new(&sidecar()).exists() || !Path::new(&step0()).exists() {
        eprintln!("SKIP");
        return;
    }
    let w = Weights::open(&sidecar()).unwrap();
    let f = Weights::open(&step0()).unwrap();
    let want = f.get("rfi.chirals"); // [1, n, 5]

    // Chiral indices are GLOBAL, not ligand-local. `parse_ligands` does
    //     chirals[:, :-1] += sum(Ls)
    // to shift past preceding ligands, and the whole ligand block then sits
    // after the protein — so the first four columns (the four atom indices;
    // the fifth is the target angle) carry an offset of
    // protein_length + this ligand's offset within the ligand block.
    let (seq, _) = f.get_i64("indep.seq");
    let names = vec!["NAD".to_string(), "OXM".to_string()];
    let set = LigandSet::load(&sidecar(), &names).unwrap();
    let protein_len = seq.len() - set.total_atoms();

    let mut got: Vec<f32> = Vec::new();
    let mut lig_off = 0usize;
    for lig in ["NAD", "OXM"] {
        let key = format!("{lig}.chirals");
        if w.has(&key) {
            let t = w.get(&key);
            let rows = t.data.len() / 5;
            for r in 0..rows {
                for c in 0..5 {
                    let v = t.data[r * 5 + c];
                    got.push(if c < 4 {
                        v + (protein_len + lig_off) as f32
                    } else {
                        v
                    });
                }
            }
        }
        lig_off += set.get(lig).map(|l| l.n_atoms).unwrap_or(0);
    }
    assert_eq!(got.len(), want.data.len(), "chirals size");
    for (i, (g, wv)) in got.iter().zip(&want.data).enumerate() {
        assert_eq!(g.to_bits(), wv.to_bits(), "chirals[{i}] got {g} want {wv}");
    }
    println!("ligand chirals: {} values exact ({} constraints)",
             got.len(), got.len() / 5);
}
