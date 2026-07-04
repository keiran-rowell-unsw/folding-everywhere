# ESMFold 1 + 2 GUI — Windows (and macOS / Linux)

A single executable with a clean local web UI: choose a model (ESMFold1 or ESMFold2),
paste a protein sequence, click **Fold protein**, watch progress, download the resulting
PDB. No Python, no GPU, nothing to install.

## What you get / what you need
- **Windows:** **`fold_gui.exe`** — the app (a few MB), now with **both models**.
  Double-click it. (`fold.exe` = ESMFold1 CLI, `fold_standalone.exe` = ESMFold2 CLI.)
- **macOS:** **`macos/fold_gui`** — the same app as a single **universal binary**
  (runs natively on both Apple Silicon and Intel Macs). (`macos/fold` = ESMFold1 CLI,
  `macos/fold_standalone` = ESMFold2 CLI.) See the **macOS** section below for the
  one-time Gatekeeper step.
- **Choose a model in the UI:**
  - **ESMFold1** (ESM-2 3B, deterministic) — weights **~8.4 GB**, saved to
    `%USERPROFILE%\.esmfold\`. Fast (seconds–minutes), **~10 GB RAM**.
  - **ESMFold2** (ESM-C 6B + diffusion) — weights **~30 GB** (ESM-C shards + head),
    saved to `%USERPROFILE%\.esmfold2\`, **~25 GB RAM**. Stochastic: set a **seed** (same
    seed → identical, bit-exact-to-PyTorch-fp32 structure). Two quality/speed knobs —
    **loops** (trunk refinement) and **sampling steps** (diffusion) — default to the official
    release depth (**20 / 68**) and can be lowered for speed (**~1.5–6 min**/protein at 3 / 14,
    ~10–20 min at 20 / 68). A single diffusion sample is produced.
- **Model weights** download **automatically on first use** from Hugging Face
  (`facebook/esmfold_v1`, `biohub/ESMC-6B`, `biohub/ESMFold2`), read directly — no
  conversion. The seed box appears only for ESMFold2 (ESMFold1 has no seed).
- **Your machine:** Windows 10+ (ships `curl.exe`), an AVX2 CPU (any PC from ~2014 on).
  ESMFold2 needs **~30 GB disk + ~25 GB RAM**; ESMFold1 needs ~9 GB disk + ~10 GB RAM.

## How to use
1. Double-click **`fold_gui.exe`**. A console window opens and your **default browser**
   opens to `http://127.0.0.1:<port>/`.
2. Paste a protein sequence (one-letter amino-acid codes) and click **Fold protein**.
3. First run: it downloads the weights (progress bar). Subsequent runs skip this.
4. Watch progress (both models report live, per-layer/step):
   - **ESMFold1:** *ESM-2 layer i/36 → folding trunk recycle r/4 block b/48 →
     structure module → confidence*.
   - **ESMFold2:** *ESM-C 6B layer i/80 → folding trunk loop r/4 block b/48 →
     diffusion sampling step s/10 → confidence*.
5. When done it shows **mean pLDDT** and **pTM**; click **Download PDB**.
   (Also saved at `%USERPROFILE%\.esmfold\output\prediction.pdb`.)
6. Keep the console window open while using it; close it to quit the app.

## Error handling
- Invalid amino-acid letters or empty input → a clear red message; fix and retry.
- Sequences are capped at 500 aa (CPU runtime); longer inputs are rejected with a note.
- Download failures (no internet, curl missing) → message asking you to check the
  connection; partial downloads are discarded and retried cleanly.
- Any internal error is caught and shown rather than crashing the app.

## Region note (HF-Mirror fallback)
- Weights download from Hugging Face (`huggingface.co`). In regions where that is
  blocked, the app **automatically retries from HF-Mirror (`hf-mirror.com`)** — no
  setup needed. To force a specific endpoint, set the environment variable
  `HF_ENDPOINT` (e.g. `set HF_ENDPOINT=https://hf-mirror.com` on Windows) before
  launching; all downloads then use it.

## Notes / limits
- The download URL and size are fixed to `facebook/esmfold_v1`. To use a
  pre-downloaded file, set the environment variable `ESMFOLD_WEIGHTS` to its path
  (`pytorch_model.bin` **or** a `model.safetensors`) and the app skips downloading.
- The "GUI" is a local web page served only on `127.0.0.1` (not exposed to the
  network); your browser is just the front-end.
- This Windows build was cross-compiled and its logic fully tested on Linux; if your
  CPU predates AVX2 and you see an "illegal instruction" error, ask for an
  `x86-64-v2` build (slower but runs on older CPUs).

## macOS
A prebuilt **`macos/fold_gui`** ships here as a **universal2** binary (arm64 + x86_64),
so it runs natively on both Apple Silicon and Intel Macs — no build step needed.

1. Open **Terminal** and `cd` to this folder.
2. Because the binary is unsigned and downloaded, macOS Gatekeeper quarantines it. Clear
   that once:  `xattr -dr com.apple.quarantine macos/fold_gui`  (or right-click the file
   in Finder → **Open** → **Open**). Then make it runnable: `chmod +x macos/fold_gui`.
3. Run it: `./macos/fold_gui`. It opens your default browser to `http://127.0.0.1:<port>/`.
   Model weights auto-download to `~/.esmfold` / `~/.esmfold2` on first use (same as
   Windows). It uses the system `curl` to download and `open` to launch the browser.

CLIs work the same way: `./macos/fold --seq <SEQUENCE> -o out.pdb` (ESMFold1) and
`./macos/fold_standalone <SEQUENCE> [seed]` (ESMFold2).

Rebuild from source with `./build_macos.sh` (uses cargo-zigbuild; see that script).

## Linux
Build and run the same `fold_gui` natively: `cargo build --release --bin fold_gui`; it
uses `xdg-open` to launch the browser and the system `curl` to download.
