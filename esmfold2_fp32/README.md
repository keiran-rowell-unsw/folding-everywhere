# esmfold2_fp32 — the PyTorch fp32 reference for ESMFold2

This folder holds the **PyTorch side** of the ESMFold2 work: the official released model
architecture, plus the harness we use to run it in **fp32** and validate the pure-Rust port
against it. It is here so a reader can see *what is out there* (the real ESMFold2 in PyTorch)
next to our re-implementation, and exactly how the two were compared.

> **This is not the official ESMFold2, and running it in fp32 is a precision variant of the
> official model.** ESMFold2, ESM-C, and their weights are the remarkable work of Chan
> Zuckerberg Biohub / EvolutionaryScale. We reproduce a **fp32** configuration for exact,
> deterministic comparison; its numbers may not match the official **bf16** release. Please
> cite and defer to the original authors — see [`reference_model/NOTICE.md`](reference_model/NOTICE.md).

## Contents

- **`reference_model/`** — the official released **ESMFold2** and **ESM-C** PyTorch
  architecture (Chan Zuckerberg Biohub, MIT), reproduced **unmodified** for reference.
  See [`reference_model/NOTICE.md`](reference_model/NOTICE.md) for attribution and license.
- **`harness/`** — our Python scripts that load the reference in **fp32 on CPU** and:
  - `common.py` — `load_model(fp32=True)` / feature prep / fixture I/O (the shared entry point).
  - `ref_e2e.py`, `ref_esmc.py` — end-to-end and ESM-C fp32 reference folds.
  - `dump_*.py` — per-module fixture generators (trunk blocks, parcae, MSA, diffusion,
    sampler, confidence, …) used to validate the Rust port module-by-module.
  - `bench_pytorch.py`, `run_benchmark*.py`, `proteins10.py` — the 10-protein benchmark
    (time, memory, pLDDT, pTM) that produced [`../results/esmfold2/`](../results/esmfold2/README.md).
  - `compare_bf16_fp32.py` — measures how far the released **bf16** path sits from **fp32**.
  - `torch_rng_prototype.py`, `gen_op_fixtures.py` — the CPU-RNG and op fixtures our Rust
    RNG / kernels are checked against.
- **`NOTES_fp32_vs_bf16.md`** — a technical note on why the released bf16 atom-attention
  kernel is not matched bit-for-bit, and why running fp32 on both sides is the clean choice.
- **`VALIDATION.md`** — the module-by-module and end-to-end validation write-up (path to the
  sub-milliångström fp32 match).

## Running it

These scripts require the official `transformers` fork + `esm` packages and the ESMFold2 /
ESM-C weights (not shipped here — downloaded from Hugging Face under the sources' own terms).
With those installed, e.g.:

```bash
python harness/ref_e2e.py ubiquitin76        # fp32 reference fold, seed 0
python harness/compare_bf16_fp32.py          # bf16-vs-fp32 divergence
python harness/run_benchmark_standalone.py   # the 10-protein benchmark
```

The pure-Rust reproduction of this exact fp32 pipeline is in [`../esmfold2/`](../esmfold2/).
