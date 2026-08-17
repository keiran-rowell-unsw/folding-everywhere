# Benchmark results

Accuracy, speed, and memory for both models, each validated against its own
**PyTorch fp32** reference on the same CPU. The two models are independent
(ESMFold1 = ESM-2 3B → IPA, deterministic; ESMFold2 = ESM-C 6B → diffusion,
seeded), so each has its own self-contained folder.

| Folder | Model | Headline vs PyTorch fp32 |
|---|---|---|
| [`esmfold1/`](esmfold1/README.md) | ESMFold1 (ESM-2 3B → folding trunk → IPA) | 15 proteins, all-atom RMSD **0.00002–0.0023 Å** (fp32 round-off) |
| [`esmfold2/`](esmfold2/README.md) | ESMFold2 (ESM-C 6B → looped trunk → diffusion), seed 0 | 10 proteins, Cα-aligned RMSD **0.04–0.25 mÅ**; bf16 diverges 1–15 Å |
| [`esmfold2_config_sweep/`](esmfold2_config_sweep/README.md) | ESMFold2 across 6 settings (varying `num_loops` / `num_sampling_steps`) | Rust vs PyTorch fp32 **bit-exact** (~1e-4 Å) at every config |

## Layout

```
results/
├── esmfold1/            # ESMFold1 (deterministic) benchmark
│   ├── README.md        # writeup + results table
│   ├── metrics.csv      # per-protein time, RAM, pLDDT, pTM, RMSD, max dev
│                         #   (pLDDT is 0–100, atom-masked = upstream's mean_plddt;
│                         #    the esmfold2/ table below reports pLDDT in 0–1)
│   └── figures/         # fig1_structure_overlay … fig5_time_memory (.png)
├── esmfold2/            # ESMFold2 (diffusion, seeded) benchmark
│   ├── README.md        # writeup + results table
│   ├── metrics.csv      # per-protein time, RAM, pLDDT, pTM, complex_plddt (fp32/bf16)
│   ├── accuracy.csv     # Kabsch RMSD of each structure vs pt_fp32
│   └── plots/           # fig1_accuracy … fig4_memory (.png)
└── esmfold2_config_sweep/  # ESMFold2 bit-exactness across inference settings
    ├── README.md        # 6-config table (varying num_loops / num_sampling_steps)
    ├── sweep.csv        # Rust-vs-PyTorch-fp32 RMSD per config
    ├── coords/          # single-sample coords (pt_*.npy, rust_*.npy) per config
    └── *.py             # sweep_configs.py (PyTorch) + compare_sweep.py
```

See each subfolder's `README.md` for the full tables, key findings, and reproduce steps.
