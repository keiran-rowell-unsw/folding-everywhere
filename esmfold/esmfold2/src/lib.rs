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

// ---------------------------------------------------------------------------
// wasm-bindgen public API
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Initialise the wasm module: install a panic hook that forwards Rust panics
/// to `console.error` so they are visible in browser DevTools.
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Return the crate version string (from `Cargo.toml`).
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Load one safetensors shard from a `Uint8Array` passed from JS and return an
/// opaque handle. Pass multiple handles to `WasmWeights::merge` for sharded
/// checkpoints.
///
/// ```js
/// import init, { load_shard } from "./pkg/esmfold2.js";
/// await init();
/// const buf = await fetch("model.safetensors").then(r => r.arrayBuffer());
/// const weights = load_shard(new Uint8Array(buf));
/// ```
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn load_shard(data: &[u8]) -> WasmWeights {
    WasmWeights { inner: Weights::from_bytes(data.to_vec()) }
}

/// Opaque wrapper around [`Weights`] exposed to JS.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmWeights {
    inner: Weights,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmWeights {
    /// Append another shard's bytes into this weight store (for sharded checkpoints).
    pub fn add_shard(&mut self, data: &[u8]) {
        self.inner.add_shard(data.to_vec());
    }

    /// List all weight tensor names (sorted).
    pub fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    /// Check whether a tensor name exists.
    pub fn has(&self, name: &str) -> bool {
        self.inner.has(name)
    }
}
