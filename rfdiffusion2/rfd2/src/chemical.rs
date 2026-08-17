//! The chemical database — RFdiffusion2's `rf2aa/chemical.py`.
//!
//! That module is ~2 900 lines of mostly *data*: an 80-token alphabet
//! (20 amino acids + UNK + MASK + 8 nucleic acids + HIS_D + 47 element types),
//! per-residue atom naming, bond graphs, ideal internal coordinates, LJ
//! parameters and torsion frames. Every featurization decision keys off it, so
//! a single wrong element here surfaces much later as an unexplained geometry
//! mismatch.
//!
//! It is therefore **not retyped**. `python/gen_chemical.py` exports the tables
//! mechanically from the pinned reference:
//!
//! * numeric tables -> `data/chemical.safetensors`, embedded with
//!   `include_bytes!` so there is no companion file at runtime (SOP §7);
//! * scalars and string tables -> `chemical_gen.rs`.
//!
//! `tests/parity_chemical.rs` then asserts every table element-for-element
//! against the reference, at tolerance **exactly 0** — these are integers,
//! names and fixed constants, so anything else is a real bug (SOP §4).

use crate::tensor::Tensor;
use crate::weights::Weights;
use std::sync::OnceLock;

pub use crate::chemical_gen::*;

/// The exported numeric tables, ~1.6 MB, baked into the executable.
static BLOB: &[u8] = include_bytes!("../data/chemical.safetensors");

static STORE: OnceLock<Weights> = OnceLock::new();

/// The singleton table store, mirroring the reference's `ChemicalData`
/// singleton (`rf2aa/chemical.py:ChemicalData.__new__`).
pub fn chem() -> &'static Weights {
    STORE.get_or_init(|| {
        Weights::from_static(BLOB).expect("embedded chemical.safetensors is corrupt")
    })
}

/// Fetch a float table by its reference name (e.g. `"ljlk_parameters"`).
pub fn table_f32(name: &str) -> Tensor {
    chem().get(name)
}

/// Fetch an integer table by its reference name (e.g. `"num_bonds"`).
///
/// Booleans are exported as int64, because safetensors has no bool and because
/// an accidental float round-trip through a mask is exactly the sort of silent
/// corruption this module exists to prevent.
pub fn table_i64(name: &str) -> (Vec<i64>, Vec<usize>) {
    chem().get_i64(name)
}

pub fn table_bool(name: &str) -> (Vec<bool>, Vec<usize>) {
    let (v, shape) = chem().get_i64(name);
    (v.into_iter().map(|x| x != 0).collect(), shape)
}

/// Names of every exported table (sorted) — used by the parity test to prove
/// nothing was dropped in export.
pub fn table_names() -> Vec<String> {
    chem().names()
}

// ---------------------------------------------------------------------------
// Convenience accessors for the tables the featurizer reaches for constantly.
// ---------------------------------------------------------------------------

/// `[NAATOKENS, NTOTAL]` — which atom slots exist for each token.
pub fn allatom_mask() -> (Vec<bool>, Vec<usize>) {
    table_bool("allatom_mask")
}

/// `[NAATOKENS, NTOTAL]` — element type index per atom slot.
pub fn atom_type_index() -> (Vec<i64>, Vec<usize>) {
    table_i64("atom_type_index")
}

/// `[NAATOKENS, NTOTAL, NTOTAL]` — bonded-distance (in bonds) between atoms.
pub fn num_bonds() -> (Vec<i64>, Vec<usize>) {
    table_i64("num_bonds")
}

/// `[NAATOKENS, NTOTAL, 5]` — Lennard-Jones / LK parameters.
pub fn ljlk_parameters() -> Tensor {
    table_f32("ljlk_parameters")
}

/// `[NAATOKENS, NTOTAL, 4]` — the four LJ correction flags, as bools.
pub fn lj_correction_parameters() -> (Vec<bool>, Vec<usize>) {
    table_bool("lj_correction_parameters")
}

/// Token index -> residue / atom name.
pub fn num2aa(i: usize) -> &'static str {
    NUM2AA[i]
}

/// Residue / atom name -> token index (linear scan over 80 entries; this is
/// called during parsing, not in the inner loop).
pub fn aa2num(name: &str) -> Option<usize> {
    NUM2AA.iter().position(|&n| n == name)
}

/// Is this token one of the 20 canonical amino acids?
pub fn is_canonical_aa(tok: usize) -> bool {
    tok < 20
}

/// Protein / nucleic-acid tokens (i.e. not a bare element).
pub fn is_polymer(tok: usize) -> bool {
    tok < NNAPROTAAS
}

/// Bare-element ("small molecule") tokens.
pub fn is_atom_token(tok: usize) -> bool {
    tok >= NNAPROTAAS && tok < NAATOKENS
}
