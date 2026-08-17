//! Pure-Rust, fp32 reimplementation of **ProteinMPNN** (Dauparas et al., 2022).
//!
//! Input: a protein backbone (PDB). Output: designed sequences + per-sequence
//! scores, identical to the reference PyTorch implementation.
//!
//! See `docs/CODE_STRUCTURE.md` for the module-by-module walkthrough.

pub mod embedded;
pub mod features;
pub mod featurize;
pub mod layers;
pub mod model;
pub mod ops;
pub mod parity;
pub mod pdb;
pub mod pth;
pub mod rng;
pub mod tensor;
pub mod weights;

/// The 21-letter alphabet ProteinMPNN indexes into (index 20 = X / unknown).
pub const ALPHABET: &str = "ACDEFGHIKLMNPQRSTVWYX";

pub fn aa_to_idx(c: u8) -> usize {
    ALPHABET.as_bytes().iter().position(|&a| a == c).unwrap_or(20)
}

pub fn idx_to_aa(i: usize) -> char {
    ALPHABET.as_bytes()[i.min(20)] as char
}
