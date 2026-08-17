Folding Everywhere v2 — ESMFold + ProteinMPNN + RFdiffusion2
============================================================

Three protein models in one program. No Python, no PyTorch, no GPU, no installer.
Each file below is a complete application.

Every platform folder holds the app plus one command-line tool per model. Windows
names carry a .exe suffix; the macOS files are universal (Apple Silicon + Intel)
and therefore about twice the size.

  gui               the app — all three models behind one page
  fold              ESMFold1 CLI       sequence  -> structure
  fold_standalone   ESMFold2 CLI       sequence  -> structure (seeded)
  mpnn              ProteinMPNN CLI    backbone  -> sequences
  rfd2              RFdiffusion2 CLI   motif     -> backbone

  linux-x86_64/     gui (36 MB)  fold (1 MB)  fold_standalone (1 MB)  mpnn (27 MB)  rfd2 (3 MB)
  windows-x86_64/   the same five, as .exe
  macos-universal/  the same five, universal binaries


QUICK START
-----------
Double-click **gui** (gui.exe on Windows). It starts a local server and opens your
browser on a page with three tabs:

  ESMFold        paste a sequence          -> predicted 3D structure  (PDB)
  ProteinMPNN    drop a backbone PDB       -> designed sequences      (FASTA)
  RFdiffusion2   drop a motif + ligands    -> designed backbone       (PDB)

Every tab has a "Load example" link, so you can see the whole thing work before
supplying your own input.

On macOS/Linux, if double-click does nothing, run it from a terminal:
    ./gui

For scripting and batch work, use the CLIs instead — each prints its options when
run with no arguments (or --help):
    ./fold --seq MQIFVKTLTGKTITLEV... -o out.pdb
    ./mpnn --pdb backbone.pdb --num_seq_per_target 8
    ./rfd2 --input-pdb motif.pdb --contigs '10,A106-106,10' --ligand NAD,OXM ...
The CLIs read the same weight caches the app fills, so nothing downloads twice.
macOS may say the file is from an unidentified developer: right-click -> Open, or
    xattr -dr com.apple.quarantine gui && chmod +x gui

Nothing is uploaded anywhere. The server listens on 127.0.0.1 only, and your
browser is just the front-end. Keep the console window open; close it to quit.

Only one job runs at a time across the three tabs — these are heavyweight models
and interleaving them would only make both slower. Starting a second one is
refused with a message; the running job is untouched.


MODEL WEIGHTS
-------------
ProteinMPNN needs NO download: all four published checkpoints are compiled into
the executable. The other two fetch their weights once, on first use, with the
OS `curl` (ships with Windows 10+, macOS and Linux):

  ESMFold1        ~8.4 GB   huggingface.co/facebook/esmfold_v1   -> ~/.esmfold
  ESMFold2        ~30 GB    huggingface.co/biohub/{ESMC-6B,ESMFold2} -> ~/.esmfold2
  RFdiffusion2    1.34 GB   files.ipd.uw.edu (Institute for Protein Design)
                                                                 -> ~/.rfdiffusion2

(`~` is %USERPROFILE% on Windows. Override with ESMFOLD_HOME / ESMFOLD2_HOME /
RFD2_HOME.) A dropped RFdiffusion2 download resumes where it left off — just press
Design again. In regions where huggingface.co is blocked the app automatically
retries via HF-Mirror (hf-mirror.com); no configuration needed. To force an
endpoint, set HF_ENDPOINT before launching.


WHAT YOUR MACHINE NEEDS
-----------------------
  ESMFold1       ~10 GB RAM,  ~9 GB disk    a few minutes per protein
  ESMFold2       ~25 GB RAM,  ~30 GB disk   ~1.5-6 min per protein
  ProteinMPNN     ~2 GB RAM,   no disk      seconds
  RFdiffusion2    ~2 GB RAM,  ~1.4 GB disk  see below

An x86-64 CPU with AVX2 (any PC from ~2013 on) or an Apple Silicon Mac. If your
CPU predates AVX2 and you see an "illegal instruction" error, ask for an
x86-64-v2 build (slower, but runs on older CPUs).


HOW LONG RFDIFFUSION2 TAKES
---------------------------
The bundled example is 1LDM lactate dehydrogenase, contig 10,A106-106,10 —
21 designed residues + 50 ligand atoms = 71 tokens. Measured with the shipped
binary on 4 CPU cores:

    T = 2    (quick sanity check)             ~1.5 min
    T = 20                                    ~15 min
    T = 100  (the GUI default, real setting)  ~70 min

Cost is ~42 s per denoising step at this size plus ~9 s fixed, and scales roughly
as L^1.8. RFdiffusion2 is normally run on GPU; this is a CPU-only implementation.


THE ONE LIMITATION, STATED PLAINLY
----------------------------------
For RFdiffusion2, ligand bond orders and aromaticity are perceived from 3D
coordinates by OpenBabel inside the reference pipeline — 3 of 4 demo inputs carry
no CONECT records at all — so they are an INPUT to this port, not something it
computes. Ten ligand sets (56 ligands) ship inside the program and are matched to
your file BY ATOM NAME, so they work on any structure:

    M0584_1ldm  NAD,OXM        M0636_1uaq  ZN,DUC       M0710_1ra0  FE,FPY
    M0054_1qfe  DHS            M0097_1ctt  ZN,DHZ       M0375_4ts9  FMC,PO4
    M0179_1q3s  MG,ADP         M0365_1pfk  FBP,MG,ADP   M0315_1ey3  DAK
    M0093_1dqa  NAP,COA

A PDB whose ligands are not in that list needs one run of
rfdiffusion2/python/gen_ligand_bonds.py from the source repo first. The program
refuses rather than guessing, so it will never silently return a wrong answer.


ACCURACY
--------
Each model is an independent from-scratch port, validated against its PyTorch
reference on this same 4-core CPU:

  ESMFold1       deterministic; ~0.0001 A RMSD from PyTorch fp32 over 15 proteins
  ESMFold2       bit-exact to a PyTorch fp32 run pinned at the same seed
  ProteinMPNN    160/160 designed sequences identical to PyTorch, 20 PDB structures
  RFdiffusion2   22/29 benchmark cases byte-identical to pinned PyTorch;
                 29/29 ligand atoms and CONECT records exact; in the other 7 the
                 whole residual is one backbone carbonyl oxygen, at most 0.192 A
                 -- about a sixth of a C-C bond, far below any structural
                 significance and far below the model's own inference noise

This release was checked end to end against known-good values from those
benchmarks: ProteinMPNN native score 1.4091 on 1UXO, RFdiffusion2 design sha256
ec256c4636cc on the bundled example at T=2, and an ESMFold1 ubiquitin fold
byte-identical to the stored reference PDB (mean pLDDT 85.82, pTM 0.8288).
ESMFold2 was not run end to end here — its 30 GB of weights did not fit on the
build machine — but its code is unchanged from the v1 release.

NOTE (2026-08-16): mean pLDDT is now reported the way upstream ESMFold reports
it -- masked by atom existence, on the 0-100 scale, i.e. esm.pretrained's
output["mean_plddt"] -- so the ubiquitin example reads 85.82 where earlier builds
read 0.7735. Coordinates and the per-atom B-factors are unchanged; only the
summary statistic moved. All three binaries here were rebuilt for this.

Full detail: results/ and docs/ in each model's subtree of the repository.


BUILD
-----
    ./build_all.sh      (needs rustup targets, cargo-zigbuild, zig 0.11+)

Distribution builds use -C target-cpu=x86-64-v3 (AVX2/FMA; Haswell 2013+, Zen 1+).
Instruction selection cannot change the answer — Rust never contracts a*b+c into
an FMA on its own and never reassociates a float reduction — and the shipped
binary was verified to produce byte-identical output to a target-cpu=native
build. See docs/BUILD.md.

Project: https://github.com/lingxusb/folding-everywhere
Author:  https://lingxusb.github.io
