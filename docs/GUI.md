# The app — `gui`

One executable, one local web page, three tabs. This document is the reference for how it
behaves; the per-model science is in each model's README.

```
Folding Everywhere  — pure-Rust fp32 protein design   [Project ↗] [Author ↗]
┌────────────┬──────────────┬───────────────┐
│  ESMFold   │ ProteinMPNN  │ RFdiffusion2  │
└────────────┴──────────────┴───────────────┘
```

## Starting it

Double-click `gui` (`gui.exe` on Windows), or run it from a terminal. It:

1. binds the first free TCP port in **8710–8759** on `127.0.0.1` (loopback only — the app is
   not reachable from the network);
2. prints the URL to the console;
3. opens your default browser at it (`cmd /C start` on Windows, `open` on macOS,
   `xdg-open` on Linux).

If the browser does not open — common on a headless or minimal Linux box — just visit the
printed URL. Closing the console window quits the app.

The tab you are on is kept in the URL hash (`…/#mpnn`), so a reload returns you to it.

## One job at a time

All three models are CPU-heavy and would only slow each other down if interleaved, so the
app holds a **single global run lock**. Starting a job while another tab is running is
refused immediately, with the message shown in the second tab's error line; nothing is
queued and the running job is untouched.

ProteinMPNN is the only tab with a **Cancel** button (it checks between sequences). ESMFold
and RFdiffusion2 runs finish or fail; to abort one, quit the app.

## Tabs

### ESMFold — sequence → structure

Radio buttons pick **ESMFold1** (ESM-2 3B, deterministic) or **ESMFold2** (ESM-C 6B +
diffusion). ESMFold2 additionally exposes **seed**, **loops** (trunk refinement) and
**sampling steps** (diffusion); those controls are hidden for ESMFold1, which has no seed.
Loops/steps default to the official release depth (20 / 68) and can be lowered for a faster,
lower-quality fold.

Input is capped at **500 aa** for reasonable CPU runtime; invalid one-letter codes are
rejected with a clear message. Progress reports per ESM layer, per trunk block, per
diffusion step and then the confidence heads. On completion the page shows **mean pLDDT**
and **pTM** and offers the PDB, which is also written to `~/.fold_gui/prediction.pdb`.

Mean pLDDT is reported on the **0–100** scale and masked by atom existence, i.e. the exact
statistic upstream ESMFold reports as `output["mean_plddt"]`, so a GUI number is directly
comparable to a pLDDT quoted from `esm.pretrained.esmfold_v1()` or the ESM Atlas. It equals
the mean of the B-factor column of the downloaded PDB. (ESMFold2's per-residue pLDDT is
rescaled to the same 0–100 units for display; the ESMFold2 benchmark tables in
`esmfold/results/esmfold2/` report it in 0–1, so they read 100× smaller.)

### ProteinMPNN — structure → sequence

Drop a `.pdb` backbone (or paste it, or click *Load example*), pick a checkpoint, a number
of sequences, one or more temperatures, a seed, which chains to design and which amino acids
to omit. A Cα trace of the input is drawn on the canvas — drag to rotate.

Results are a table of designs with score and sequence recovery, plus a FASTA download in
the reference implementation's layout. The same seed gives the same sequences as
`protein_mpnn_run.py`. Nothing is downloaded: all four checkpoints are inside the binary.

### RFdiffusion2 — motif → designed backbone

Drop a motif PDB (catalytic residues + ligands), or click *Load example* for 1LDM lactate
dehydrogenase. The ligand list is populated from the `HETATM` records of *your* file, and
only ligands the built-in 56-ligand library knows are offered — bond orders and aromaticity
are an **input** to this port, not something it computes (see
[`../rfdiffusion2/README.md`](../rfdiffusion2/README.md)). A ligand outside the library
needs one run of `rfdiffusion2/python/gen_ligand_bonds.py`, and its sidecar path goes in the
*Ligand topology file* box.

Set the contig, number of denoising steps `T`, number of designs, seed and whether to use
self-conditioning. Designs are written to `~/.rfdiffusion2/out/design_N.pdb` and offered as
downloads. **`T = 100` is the real setting and takes ~70 min** for the bundled example on 4
CPU cores; `T = 2` (~1.5 min) is a quick sanity check, not a usable design.

## Where things are cached

| What | Default location | Override |
|---|---|---|
| ESMFold1 weights (~8.4 GB) | `~/.esmfold/pytorch_model.bin` | `ESMFOLD_HOME` |
| ESMFold2 weights (~30 GB) | `~/.esmfold2/` (ESM-C shards + head) | `ESMFOLD2_HOME` |
| RFdiffusion2 checkpoint (1.34 GB) | `~/.rfdiffusion2/RFD_173.pt` | `RFD2_HOME` |
| ESMFold output | `~/.fold_gui/prediction.pdb` | — |
| RFdiffusion2 output | `~/.rfdiffusion2/out/design_N.pdb` | — |
| ProteinMPNN weights | *inside the executable* | — |

On Windows `~` is `%USERPROFILE%`. Downloads use the OS `curl`, which ships with Windows
10+, macOS and Linux.

**Region note.** ESMFold weights come from Hugging Face. Where `huggingface.co` is blocked
the app **automatically retries via HF-Mirror (`hf-mirror.com`)** with no configuration; to
force an endpoint, set `HF_ENDPOINT` before launching. The download has connect and
stall timeouts precisely so a blocked route fails over instead of hanging at 0 %.

A dropped RFdiffusion2 download **resumes** where it left off (`curl -C -`) — press *Design
backbone* again. A dropped ESMFold download is discarded and restarted cleanly.

## HTTP endpoints

Each tab keeps the JSON shape its standalone app used, so the page's polling logic is
unchanged from the single-model GUIs.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the page |
| `GET` | `/api/esmfold/status` | phase, progress, full log, pLDDT, pTM, error |
| `POST` | `/api/esmfold/fold` | body `model\nseed\nloops\nsteps\nsequence` |
| `GET` | `/api/esmfold/pdb` | the predicted structure |
| `GET` | `/api/mpnn/status` | phase, progress, log, native score, designs |
| `POST` | `/api/mpnn/run` | JSON `{pdb, model, num_seq, temps, seed, chains, omit}` |
| `POST` | `/api/mpnn/cancel` | stop after the current sequence |
| `GET` | `/api/mpnn/fasta` | all designs as FASTA |
| `GET` | `/api/rfd2/status?from=N` | phase, progress, **log rows after N**, error |
| `POST` | `/api/rfd2/design` | JSON `{pdb, ligands, contig, length, T, n, seed, self_cond, custom_sidecar}` |
| `GET` | `/api/rfd2/pdb?i=N` | design *N* (1-based) |

The RFdiffusion2 log is served incrementally (`from=`) because a `T = 100` run emits a line
per step and the client appends rather than re-fetching.

## What is not covered by the automated checks

Every release is smoke-tested end to end against a known-good value from the benchmark files
— see [`BUILD.md`](BUILD.md#release-checks). **ESMFold2 is the exception**: its weights are
~30 GB, more than the build machine has free, so its tab is exercised only up to the
download step. Its code path is carried over unchanged from the shipped v1 `fold_gui`, and
its parity fixtures still run under `esmfold/`, but it has not been run end to end through
this app.

## Errors

Every job runs inside a panic guard, so an internal fault becomes a red error line rather
than a hung page or a dead app. Bad input (empty sequence, invalid amino acid, a PDB with no
backbone atoms, a ligand outside the library, a contig that does not parse) is reported with
a message that says what to do about it.
