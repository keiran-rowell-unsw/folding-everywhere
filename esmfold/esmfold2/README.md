# esmfold2 – Browser WASM Demo

This directory contains a minimal browser demo that exposes a FASTA input field and calls into the `esmfold2` wasm module.

## Building the wasm module

```bash
# 1. Add the wasm32 target (one-time setup)
rustup target add wasm32-unknown-unknown

# 2. Install wasm-pack (one-time setup)
cargo install wasm-pack

# 3. Build the wasm package
cd esmfold/esmfold2
wasm-pack build --target web --release
```

This produces a `pkg/` directory alongside `index.html`.

## Serving the demo locally

```bash
# From the esmfold/esmfold2 directory (where index.html lives)
python3 -m http.server 8080
```

Then open: <http://127.0.0.1:8080/index.html>

## Known limitations

- **Model weights are not yet loaded in-browser.**  
  The current `fold_json` export is a functional stub: it validates and parses the FASTA input, then returns a clearly-labelled `REMARK STUB` PDB response.  
  Full end-to-end inference requires ~12 GB of model weights delivered as `ArrayBuffer`s to the wasm module.  A weight-loading layer is noted as TODO in `src/lib.rs`.

- **Memory / performance.**  
  Even once weights are available, running a 6B-parameter model in-browser is extremely demanding.  A server-side or Node.js deployment (loading weights via the native `memmap2` path) is recommended for real workloads.

- **Native-only modules gated.**  
  Modules that depend on `memmap2`, `rayon`, or `matrixmultiply` (weights, ops, trunk, pipeline, standalone, etc.) are excluded from the wasm build via `#[cfg(not(target_arch = "wasm32"))]`.  The native CLI (`fold_standalone`) is unaffected.

## Native CLI usage

```bash
cd esmfold/esmfold2
cargo run --release --bin fold_standalone -- --help
```
