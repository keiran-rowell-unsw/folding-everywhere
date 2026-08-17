//! Pure-Rust, fp32 reimplementation of **RFdiffusion2** (Ahern et al., 2025).
//!
//! The numerical target — what is bit-exact and what is only bounded — is
//! stated in `README.md` §0, and the reconnaissance behind every design
//! decision is in `docs/RECON.md`.
//!
//! Bottom-up build order (SOP §3): `tensor` -> `ops` -> `weights`/`pth` ->
//! `rng` -> the model modules -> the CLI. Nothing above a rung is written until
//! the rung below is green.
//!
//! The `tensor`, `ops`, `pth`, `weights`, `parity` and `rng::torch` modules are
//! carried over from `proteinmpnn-rs`, where they are already validated against
//! PyTorch fp32; `rng::numpy` and `rng::pyrandom` are new, because RFdiffusion2
//! draws from three generators rather than one.

pub mod atom_frames;
pub mod chemical;
pub mod chiral;
pub mod contig;
pub mod design;
pub mod dropout;
pub mod featurize;
pub mod chemical_gen;
pub mod geom;
pub mod indep;
pub mod insert;
pub mod ligand;
pub mod lj;
pub mod model;
pub mod nn;
pub mod noiser;
pub mod openfold;
pub mod ops;
pub mod output;
pub mod parity;
pub mod pdb;
pub mod prepro;
pub mod sample_init;
pub mod sampler;
pub mod score;
pub mod pth;
pub mod rng;
pub mod t2d;
pub mod tensor;
pub mod torsions;
pub mod weights;
pub mod xyzconv_bwd;
