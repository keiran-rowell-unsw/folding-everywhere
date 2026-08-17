//! ESMFold2 "release" checkpoint dimensions (from biohub/ESMFold2 config.json
//! and biohub/ESMC-6B config.json).

// ---- ESM-C 6B (frozen PLM backbone) ----
pub const ESMC_VOCAB: usize = 64;
pub const ESMC_D_MODEL: usize = 2560;
pub const ESMC_N_HEADS: usize = 40;
pub const ESMC_HEAD_DIM: usize = ESMC_D_MODEL / ESMC_N_HEADS; // 64
pub const ESMC_N_LAYERS: usize = 80;
pub const ESMC_FFN_HIDDEN: usize = 6912; // ((8/3*2560)+255)//256*256
pub const ESMC_ROPE_BASE: f32 = 10000.0;
pub const ESMC_RESIDUE_SCALE: f64 = 80.0 / 36.0; // sqrt() applied at use site

// ---- ESMFold2 trunk / heads ----
pub const D_SINGLE: usize = 384;
pub const D_PAIR: usize = 256;
pub const D_INPUTS: usize = 451;
pub const N_RELATIVE_RESIDX_BINS: usize = 32;
pub const N_RELATIVE_CHAIN_BINS: usize = 2;
pub const NUM_LOOPS_DEFAULT: usize = 3;
pub const NUM_DIFFUSION_SAMPLES_DEFAULT: usize = 32;

pub const FOLDING_TRUNK_LAYERS: usize = 48; // checkpoint overrides dataclass default (24)
pub const FOLDING_TRUNK_HEADS: usize = 8; // NB: trunk has NO attention module; heads unused there
pub const LM_ENCODER_LAYERS: usize = 4;
pub const PARCAE_CODA_LAYERS: usize = 2;
pub const CONFIDENCE_TRUNK_LAYERS: usize = 4;

// ---- diffusion module ----
pub const DM_SIGMA_DATA: f32 = 16.0;
pub const DM_C_ATOM: usize = 128;
pub const DM_C_TOKEN: usize = 768;
pub const DM_C_Z: usize = 256;
pub const DM_C_S_INPUTS: usize = 451;
pub const DM_FOURIER_DIM: usize = 256;
pub const DM_TOKEN_BLOCKS: usize = 12;
pub const DM_TOKEN_HEADS: usize = 16;
pub const DM_ATOM_BLOCKS: usize = 3;
pub const DM_ATOM_HEADS: usize = 4;
pub const DM_TRANSITION_MULT: usize = 2;
pub const DISTOGRAM_BINS: usize = 128;

// atom encoder / SWA + 3D RoPE
pub const SWA_WINDOW: usize = 128;
pub const SPATIAL_ROPE_BASE: f32 = 20.0;
pub const N_SPATIAL_ROPE_PAIRS_PER_AXIS: usize = 2;
pub const N_UID_ROPE_PAIRS: usize = 10;
pub const UID_ROPE_BASE: f32 = 10000.0;

// ---- diffusion sampling (ODE/EDM) ----
pub const GAMMA_0: f32 = 0.605;
pub const GAMMA_MIN: f32 = 1.107;
pub const NOISE_SCALE: f32 = 0.0;
pub const STEP_SCALE: f32 = 1.0;
pub const INFERENCE_S_MAX: f32 = 160.0;
pub const INFERENCE_S_MIN: f32 = 4e-4;
pub const INFERENCE_P: f32 = 8.0;
pub const INFERENCE_NUM_STEPS: usize = 14;
pub const MAX_INFERENCE_SIGMA: f32 = 256.0;

// ---- confidence head ----
pub const NUM_PLDDT_BINS: usize = 50;
pub const NUM_PDE_BINS: usize = 64;
pub const NUM_PAE_BINS: usize = 64;
pub const CONF_MIN_DIST: f32 = 2.0;
pub const CONF_MAX_DIST: f32 = 52.0;
