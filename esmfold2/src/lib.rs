//! Pure-Rust fp32 bit-exact reimplementation of ESMFold2 (ESM-C 6B PLM +
//! looped "parcae" trunk + diffusion structure module + confidence/distogram
//! heads), validated module-by-module against the PyTorch reference.

pub mod config;
pub mod featurize;
pub mod ops;
pub mod parity;
pub mod rng;
pub mod tensor;
pub mod weights;

pub mod atom;
pub mod confidence;
pub mod diffusion;
pub mod esmc;
pub mod msa;
pub mod parcae;
pub mod pdb;
pub mod pipeline;
pub mod standalone;
pub mod trunk;

pub use tensor::Tensor;
pub use weights::Weights;
