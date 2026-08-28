//! Pure-Rust fp32 bit-exact reimplementation of ESMFold2 (ESM-C 6B PLM +
//! looped "parcae" trunk + diffusion structure module + confidence/distogram
//! heads), validated module-by-module against the PyTorch reference.

pub mod config;
pub mod featurize;
pub mod rng;
pub mod tensor;

#[cfg(not(target_arch = "wasm32"))]
pub mod ops;
#[cfg(not(target_arch = "wasm32"))]
pub mod parity;
#[cfg(not(target_arch = "wasm32"))]
pub mod weights;
#[cfg(not(target_arch = "wasm32"))]
pub mod atom;
#[cfg(not(target_arch = "wasm32"))]
pub mod confidence;
#[cfg(not(target_arch = "wasm32"))]
pub mod diffusion;
#[cfg(not(target_arch = "wasm32"))]
pub mod esmc;
#[cfg(not(target_arch = "wasm32"))]
pub mod msa;
#[cfg(not(target_arch = "wasm32"))]
pub mod parcae;
#[cfg(not(target_arch = "wasm32"))]
pub mod pdb;
#[cfg(not(target_arch = "wasm32"))]
pub mod pipeline;
#[cfg(not(target_arch = "wasm32"))]
pub mod standalone;
#[cfg(not(target_arch = "wasm32"))]
pub mod trunk;

#[cfg(not(target_arch = "wasm32"))]
pub use tensor::Tensor;
#[cfg(not(target_arch = "wasm32"))]
pub use weights::Weights;

// ─────────────────────────────────────────────────────────────────────────────
// Browser / wasm-bindgen API surface
// All items below are compiled only when targeting wasm32.
// Native (CLI / library) behaviour is completely unaffected.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Initialise the wasm module: install a panic hook so Rust panics surface as
/// readable error messages in the browser console instead of opaque traps.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
}

/// Return a human-readable version string that can be used by the browser page
/// to confirm the wasm module loaded correctly.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn version() -> String {
    "esmfold2 wasm api v1".to_string()
}

/// Main browser entry-point.
///
/// Accepts a JSON request object:
/// ```json
/// {
///   "fasta": ">header\nMSEQ...",   // FASTA text OR bare amino-acid sequence
///   "seed":  0,                     // optional u64, default 0
///   "output_format": "pdb"          // "pdb" (default) — "cif" planned
/// }
/// ```
///
/// Returns a JSON response object:
/// ```json
/// {
///   "format":    "pdb",
///   "structure": "ATOM  ...\nEND\n",
///   "plddt":     0.0,
///   "ptm":       0.0
/// }
/// ```
///
/// # Browser limitations
///
/// Full end-to-end inference requires loading ~12 GB of model weights.
/// In-browser weight delivery is not yet wired (weights use `memmap2` on
/// native; an `ArrayBuffer`-based loader is TODO).  Until that loader exists
/// this function returns a clearly-labelled stub so the UI round-trip can be
/// exercised without crashing.
///
/// TODO: replace stub body with real weight loading from `ArrayBuffer` and
///       call `crate::standalone::fold(...)` once browser weight delivery is
///       implemented.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn fold_json(req_json: &str) -> Result<String, JsValue> {
    use serde::{Deserialize, Serialize};

    #[derive(Deserialize)]
    struct FoldRequest {
        fasta: String,
        seed: Option<u64>,
        output_format: Option<String>,
    }

    #[derive(Serialize)]
    struct FoldResponse {
        format: String,
        structure: String,
        plddt: f32,
        ptm: f32,
    }

    let req: FoldRequest = serde_json::from_str(req_json)
        .map_err(|e| JsValue::from_str(&format!("invalid request JSON: {e}")))?;

    let seq = parse_fasta_or_sequence(&req.fasta)
        .map_err(|e| JsValue::from_str(&format!("invalid FASTA/sequence: {e}")))?;

    let _seed = req.seed.unwrap_or(0);
    let format = req.output_format.as_deref().unwrap_or("pdb").to_string();

    // TODO: Load model weights from an ArrayBuffer supplied by the page
    //       (e.g. via <input type="file"> or fetch()) and call:
    //
    //   let w_esmc = Weights::from_bytes(&esmc_bytes)
    //       .map_err(|e| JsValue::from_str(&e.to_string()))?;
    //   let w = Weights::from_bytes(&fold_bytes)
    //       .map_err(|e| JsValue::from_str(&e.to_string()))?;
    //   let out = crate::standalone::fold(&seq, _seed, &w_esmc, &w, 1, 10);
    //   let structure = out.pdb;
    //   let plddt    = out.plddt_mean;
    //   let ptm      = out.ptm;
    //
    // Until weight loading is implemented we return a clearly-labelled stub
    // so the browser UI can be exercised end-to-end without crashing.

    let structure = format!(
        "REMARK STUB — sequence accepted ({} residues), model weights not yet loaded in browser.\n\
         REMARK TODO: implement ArrayBuffer weight loading and call standalone::fold.\n\
         END\n",
        seq.len()
    );

    let resp = FoldResponse { format, structure, plddt: 0.0, ptm: 0.0 };

    serde_json::to_string(&resp)
        .map_err(|e| JsValue::from_str(&format!("response encode failed: {e}")))
}

/// Parse a FASTA string (possibly multi-line, with a `>header` line) or a raw
/// uppercase amino-acid sequence.  Returns the residue string.
///
/// Returns an error if multiple FASTA records are detected (multiple `>`
/// header lines), since ESMFold2 folds a single chain at a time.
///
/// This helper is compiled for **all** targets so it can be unit-tested natively.
pub fn parse_fasta_or_sequence(input: &str) -> Result<String, String> {
    let s = input.trim();
    if s.is_empty() {
        return Err("empty input".into());
    }
    let seq: String = if s.starts_with('>') {
        let header_count = s.lines().filter(|l| l.starts_with('>')).count();
        if header_count > 1 {
            return Err(format!(
                "multi-record FASTA with {header_count} sequences is not supported; \
                 please provide a single sequence"
            ));
        }
        s.lines()
            .filter(|l| !l.starts_with('>'))
            .collect::<Vec<_>>()
            .join("")
    } else {
        s.to_string()
    };
    let seq = seq.trim().to_uppercase();
    if seq.is_empty() {
        return Err("no residues found after FASTA header".into());
    }
    if !seq.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err("sequence contains non-letter characters".into());
    }
    Ok(seq)
}
