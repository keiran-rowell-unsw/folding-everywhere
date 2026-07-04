# ESMFold GUI — Windows (and macOS / Linux)

A single executable with a clean local web UI: paste a protein sequence, click
**Fold protein**, watch per-layer progress, download the resulting PDB. No Python,
no GPU, nothing to install.

## What you get / what you need
- **`fold_gui.exe`** — the app (a few MB). Double-click it.
- **Model weights** — downloaded **automatically on first use** (~8.4 GB, one time)
  from Hugging Face (`facebook/esmfold_v1`, the official `pytorch_model.bin`, read
  directly — no conversion). Saved to `%USERPROFILE%\.esmfold\`.
- **Your machine:** Windows 10+ (ships `curl.exe`, used for the download), an AVX2
  CPU (essentially any PC from ~2014 on), and **~10 GB free RAM** + ~9 GB disk.
  CPU-only → expect a few minutes per protein.

## How to use
1. Double-click **`fold_gui.exe`**. A console window opens and your **default browser**
   opens to `http://127.0.0.1:<port>/`.
2. Paste a protein sequence (one-letter amino-acid codes) and click **Fold protein**.
3. First run: it downloads the weights (progress bar). Subsequent runs skip this.
4. Watch progress: *ESM-2 layer i/36 → folding trunk recycle r/4 block b/48 →
   structure module → confidence*.
5. When done it shows **mean pLDDT** and **pTM**; click **Download PDB**.
   (Also saved at `%USERPROFILE%\.esmfold\output\prediction.pdb`.)
6. Keep the console window open while using it; close it to quit the app.

## Error handling
- Invalid amino-acid letters or empty input → a clear red message; fix and retry.
- Sequences are capped at 500 aa (CPU runtime); longer inputs are rejected with a note.
- Download failures (no internet, curl missing) → message asking you to check the
  connection; partial downloads are discarded and retried cleanly.
- Any internal error is caught and shown rather than crashing the app.

## Notes / limits
- The download URL and size are fixed to `facebook/esmfold_v1`. To use a
  pre-downloaded file, set the environment variable `ESMFOLD_WEIGHTS` to its path
  (`pytorch_model.bin` **or** a `model.safetensors`) and the app skips downloading.
- The "GUI" is a local web page served only on `127.0.0.1` (not exposed to the
  network); your browser is just the front-end.
- This Windows build was cross-compiled and its logic fully tested on Linux; if your
  CPU predates AVX2 and you see an "illegal instruction" error, ask for an
  `x86-64-v2` build (slower but runs on older CPUs).

## macOS / Linux
The same `fold_gui` builds and runs identically (`cargo build --release --features gui
--bin fold_gui`); it uses `open`/`xdg-open` to launch the browser and the system
`curl` to download. On Apple Silicon, build on the Mac (native arm64).
