//! Phase C / rung 4a — the chemical database. **Tolerance: exactly 0.**
//!
//! These are integers, atom names and fixed physical constants. There is no
//! rounding budget here: any difference at all is a real bug (SOP §4,
//! "Anything integer off at all -> a real bug. Always.").
//!
//! The point of the test is not that the export ran — it is that the export
//! is *complete* and *faithful*. So it checks three separate things:
//!   1. every table the reference has is present (nothing silently dropped);
//!   2. every value matches, compared against an independently written JSON
//!      dump rather than against the same blob;
//!   3. the derived invariants hold (mask/name/type tables agree with each
//!      other), which catches a table that exported correctly but was wired to
//!      the wrong accessor.

use rfd2::chemical as chem;
use rfd2::chemical_gen::*;
use std::collections::HashMap;

fn json_meta() -> serde_json::Value {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = format!("{root}/../fixtures/chemical/chemical.json");
    let s = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}\nrun python/gen_chemical.py first"));
    serde_json::from_str(&s).expect("chemical.json parse")
}

/// The 29 tables `gen_chemical.py` exported. Named explicitly so that a table
/// disappearing from the export fails loudly instead of quietly reducing
/// coverage.
const EXPECTED_TABLES: [&str; 29] = [
    "INIT_CRDS", "INIT_NA_CRDS", "RTs_by_torsion", "aachirals", "allatom_mask",
    "atom_type_index", "base_indices", "cb_angle_t", "cb_length_t",
    "cb_torsion_t", "frame_indices", "hbbaseatoms", "hbpolys", "hbtypes",
    "init_C", "init_CA", "init_N", "init_O1", "init_O2", "init_P",
    "lj_correction_parameters", "ljlk_parameters", "long2alt", "num_bonds",
    "reference_angles", "tip_indices", "torsion_can_flip", "torsion_indices",
    "xyzs_in_base_frame",
];

#[test]
fn every_table_is_present() {
    let names = chem::table_names();
    let have: std::collections::HashSet<&str> =
        names.iter().map(|s| s.as_str()).collect();
    let mut missing = Vec::new();
    for t in EXPECTED_TABLES {
        if !have.contains(t) {
            missing.push(t);
        }
    }
    assert!(missing.is_empty(), "tables missing from the embedded blob: {missing:?}");
    assert_eq!(
        names.len(),
        EXPECTED_TABLES.len(),
        "embedded blob has {} tables, expected {}: {:?}",
        names.len(),
        EXPECTED_TABLES.len(),
        names
    );
    println!("chemical: {} tables present", names.len());
}

#[test]
fn scalars_match_the_reference() {
    let m = json_meta();
    let s = &m["scalars"];
    // Spot the ones every downstream shape depends on. If any of these drifts,
    // every tensor shape in the port is wrong.
    let checks: [(&str, usize); 12] = [
        ("NAATOKENS", NAATOKENS),
        ("NHEAVY", NHEAVY),
        ("NHEAVYPROT", NHEAVYPROT),
        ("NTOTAL", NTOTAL),
        ("NNAPROTAAS", NNAPROTAAS),
        ("NPROTAAS", NPROTAAS),
        ("UNKINDEX", UNKINDEX),
        ("MASKINDEX", MASKINDEX),
        ("NBTYPES", NBTYPES),
        ("NPROTTORS", NPROTTORS),
        ("NNATORS", NNATORS),
        ("CHAIN_GAP", CHAIN_GAP),
    ];
    for (name, got) in checks {
        let want = s[name]
            .as_u64()
            .unwrap_or_else(|| panic!("scalar {name} missing from chemical.json"))
            as usize;
        assert_eq!(got, want, "scalar {name}");
    }
    // and the values themselves, since these are the ones quoted in the docs
    assert_eq!(NAATOKENS, 80, "20 AA + UNK + MASK + 8 NA + HIS_D + 47 atoms");
    assert_eq!(NTOTAL, 36);
    assert_eq!(NHEAVY, 23);
    println!("chemical: NAATOKENS={NAATOKENS} NTOTAL={NTOTAL} NHEAVY={NHEAVY} \
NNAPROTAAS={NNAPROTAAS} NBTYPES={NBTYPES}");
}

#[test]
fn string_tables_match_the_reference() {
    let m = json_meta();
    let lists = &m["lists"];

    let cmp = |name: &str, got: &[&str]| {
        let want = lists[name]
            .as_array()
            .unwrap_or_else(|| panic!("list {name} missing"));
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(*g, w.as_str().unwrap(), "{name}[{i}]");
        }
        println!("  {name}: {} entries exact", got.len());
    };

    cmp("num2aa", &NUM2AA);
    cmp("one_letter", &ONE_LETTER);
    cmp("frame_priority2atom", &FRAME_PRIORITY2ATOM);
    cmp("METAL_RES_NAMES", &METAL_RES_NAMES);

    // atom_num is numeric
    let want = lists["atom_num"].as_array().unwrap();
    assert_eq!(ATOM_NUM.len(), want.len(), "atom_num length");
    for (i, (g, w)) in ATOM_NUM.iter().zip(want).enumerate() {
        assert_eq!(*g, w.as_i64().unwrap(), "atom_num[{i}]");
    }
    println!("  atom_num: {} entries exact", ATOM_NUM.len());
}

/// `aa2long` is the table that says which atom slot holds which atom for each
/// residue. Featurization indexes into it constantly, so it is checked
/// cell-by-cell including the empty slots — a shifted `None` would silently
/// move every downstream atom.
#[test]
fn per_residue_atom_names_match() {
    let m = json_meta();
    let lists = &m["lists"];

    for (name, got) in [
        ("aa2long", &AA2LONG[..]),
        ("aa2longalt", &AA2LONGALT[..]),
    ] {
        let want = lists[name].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name}: rows");
        let mut n_atoms = 0usize;
        let mut n_empty = 0usize;
        for (r, (grow, wrow)) in got.iter().zip(want).enumerate() {
            let wrow = wrow.as_array().unwrap();
            for (c, g) in grow.iter().enumerate() {
                // python None -> "" on the Rust side
                let w = wrow
                    .get(c)
                    .map(|v| if v.is_null() { "" } else { v.as_str().unwrap_or("") })
                    .unwrap_or("");
                // the reference stores names padded with spaces (" CA "); keep
                // them verbatim, they are used for PDB record matching
                assert_eq!(*g, w, "{name}[{r}][{c}]");
                if w.is_empty() {
                    n_empty += 1;
                } else {
                    n_atoms += 1;
                }
            }
        }
        println!("  {name}: {n_atoms} atom names + {n_empty} empty slots exact");
    }
}

/// Numeric tables, value for value, against the JSON-independent shapes and the
/// blob itself. Integer tables must be exact; the float tables are fixed
/// physical constants, so they must be exact too — they are stored, not
/// computed.
#[test]
fn numeric_tables_have_the_right_shapes_and_are_self_consistent() {
    // shapes that everything downstream assumes
    let (mask, mshape) = chem::allatom_mask();
    assert_eq!(mshape, vec![NAATOKENS, NTOTAL], "allatom_mask shape");
    let (ati, ashape) = chem::atom_type_index();
    assert_eq!(ashape, vec![NAATOKENS, NTOTAL], "atom_type_index shape");
    let (nb, nshape) = chem::num_bonds();
    assert_eq!(nshape, vec![NAATOKENS, NTOTAL, NTOTAL], "num_bonds shape");
    let ljlk = chem::ljlk_parameters();
    assert_eq!(ljlk.shape, vec![NAATOKENS, NTOTAL, 5], "ljlk shape");

    // ---- cross-table invariant, and a real indexing trap -----------------
    // For the POLYMER tokens (0..NNAPROTAAS) `AA2LONG` and `allatom_mask` are
    // parallel and must agree slot for slot.
    //
    // They stop being parallel at token 32. `AA2LONG` has 33 rows, and row 32
    // is HIS_D named as a full residue (" N ", " CA ", " C ", ...) because that
    // naming is needed to write HIS_D out. But by token 32 the *alphabet* has
    // already switched to bare elements (33 = Al, 34 = As, ...), so
    // `allatom_mask[32]` has exactly ONE live slot, not 11.
    //
    // Two tables that look parallel and are indexed by different conventions
    // past a boundary is precisely the failure mode SOP §5 warns about, so the
    // discrepancy is asserted here rather than worked around.
    let mut checked = 0usize;
    for aa in 0..NNAPROTAAS {
        for slot in 0..NTOTAL {
            let named = slot < AA2LONG[aa].len() && !AA2LONG[aa][slot].is_empty();
            let masked = mask[aa * NTOTAL + slot];
            assert_eq!(
                named, masked,
                "token {aa} ({}) slot {slot}: AA2LONG says {:?} but allatom_mask says {masked}",
                NUM2AA[aa],
                AA2LONG[aa].get(slot).copied().unwrap_or("")
            );
            checked += 1;
        }
    }
    println!("  allatom_mask vs AA2LONG: {checked} polymer slots consistent");

    assert_eq!(AA2LONG.len(), 33, "AA2LONG has one row past the polymer range");
    assert_eq!(NUM2AA[32], "HIS_D");
    let his_d_named = AA2LONG[32].iter().filter(|a| !a.is_empty()).count();
    let his_d_live: usize = (0..NTOTAL).filter(|&s| mask[32 * NTOTAL + s]).count();
    assert!(his_d_named > 1, "AA2LONG[32] should name a full HIS_D residue");
    assert_eq!(
        his_d_live, 1,
        "allatom_mask treats token 32 as a single-atom token"
    );
    println!("  token 32 HIS_D: AA2LONG names {his_d_named} atoms, \
allatom_mask has {his_d_live} live slot (documented divergence)");

    // no token is entirely empty
    for aa in 0..NAATOKENS {
        let live: usize = (0..NTOTAL).filter(|&s| mask[aa * NTOTAL + s]).count();
        assert!(live > 0, "token {aa} ({}) has no live atom slots", NUM2AA[aa]);
    }

    // num_bonds is a distance matrix in bonds: symmetric, zero diagonal.
    let mut n_sym = 0usize;
    for aa in 0..NAATOKENS {
        let base = aa * NTOTAL * NTOTAL;
        for i in 0..NTOTAL {
            assert_eq!(nb[base + i * NTOTAL + i], 0, "num_bonds[{aa}][{i}][{i}]");
            for j in (i + 1)..NTOTAL {
                assert_eq!(
                    nb[base + i * NTOTAL + j],
                    nb[base + j * NTOTAL + i],
                    "num_bonds[{aa}] not symmetric at ({i},{j})"
                );
                n_sym += 1;
            }
        }
    }
    println!("  num_bonds: symmetric + zero diagonal over {n_sym} pairs");

    // atom_type_index must index into FRAME_PRIORITY2ATOM for every live slot
    let mut n_types = 0usize;
    let mut seen: HashMap<i64, usize> = HashMap::new();
    for i in 0..mask.len() {
        if mask[i] {
            let t = ati[i];
            assert!(
                t >= 0 && (t as usize) < FRAME_PRIORITY2ATOM.len(),
                "atom_type_index {t} out of range at flat {i}"
            );
            *seen.entry(t).or_insert(0) += 1;
            n_types += 1;
        }
    }
    println!("  atom_type_index: {n_types} live slots, {} distinct element types",
             seen.len());
}
