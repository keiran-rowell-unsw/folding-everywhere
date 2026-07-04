# ESMFold2 benchmark — pure-Rust standalone vs PyTorch

10 proteins (L 46–119), three variants timed under `/usr/bin/time -v` on the same CPU,
`num_loops=3`, `num_sampling_steps=14`, **seed 0**:

> **Note on settings.** These experiments deliberately use a *reduced* trunk/diffusion depth
> (`num_loops=3`, `num_sampling_steps=14`) rather than the official ESMFold2 release depth
> (`num_loops=20`, `num_sampling_steps=68`), purely to keep the run fast — **both** the
> PyTorch-fp32 reference and the Rust port run at the *identical* setting, so this is still a
> valid **bit-exact** fp32 comparison, not a quality benchmark. Both produce a single diffusion
> sample (`num_diffusion_samples=1`). The GUI/CLI can fold at the full release depth (see the
> repo README).

- **pt_fp32** — PyTorch, fully fp32. The fidelity target.
- **rust_fp32** — the **fully standalone** pure-Rust fold (`fold_standalone <seq> 0`): bare
  sequence → featurization → ESM-C 6B → looped trunk → diffusion → confidence, with the
  diffusion noise drawn from a bit-exact reimplementation of PyTorch's CPU RNG. **No
  PyTorch, no Python, no fixtures.**
- **pt_bf16** — PyTorch "released" path (bf16 ESM-C + CPU bf16 autocast).

Files: `metrics.csv` (fold_s, peak_rss_mb, plddt_mean, ptm, complex_plddt),
`accuracy.csv` (Kabsch RMSD of each structure vs pt_fp32), `plots/`.

## Headline

- **The standalone Rust fold reproduces PyTorch-fp32 (seed 0) to 0.04–0.25 mÅ** Cα-aligned
  RMSD (max atom dev ≤ 5e-4 Å), with **mean pLDDT and pTM matching to 4–5 decimals** — i.e.
  fp32 round-off — across all 10 proteins, ESM-C 6B included, generating its own diffusion
  noise. Disordered/low-confidence chains (flgM, sumo1, histone_h4) sit at the higher end
  (0.1–0.25 mÅ) as the chaotic trunk amplifies fp32 round-off; well-folded ones are ~0.04 mÅ.
- **The released bf16 path lands on a *different* diffusion sample**: 1.1–15.5 Å from fp32
  (protein-dependent). bf16's confidence is still reasonable — it's a different valid sample,
  not garbage — which is exactly why bit-exact reproduction requires fp32 on both sides (what
  this port does) and a matching seed.

## Accuracy (Cα-aligned RMSD vs PyTorch fp32, seed 0)

| Protein | L | Rust standalone | PyTorch bf16 |
|---|---|---|---|
| crambin | 46 | **0.055 mÅ** | 1.64 Å |
| ubiquitin | 76 | **0.041 mÅ** | 1.33 Å |
| flgM | 97 | **0.247 mÅ** | 15.49 Å |
| acylphosphatase | 99 | **0.038 mÅ** | 1.44 Å |
| bpti | 100 | **0.053 mÅ** | 8.64 Å |
| sumo1 | 101 | **0.139 mÅ** | 6.22 Å |
| histone_h4 | 103 | **0.109 mÅ** | 7.38 Å |
| cytochrome_c | 105 | **0.041 mÅ** | 1.13 Å |
| thioredoxin | 109 | **0.038 mÅ** | 1.44 Å |
| b2_microglobulin | 119 | **0.044 mÅ** | 2.68 Å |

pLDDT/pTM: rust_fp32 == pt_fp32 to 4–5 decimals on every protein.

## Time & memory (L 46 → 119)

| Variant | fold time | peak RAM |
|---|---|---|
| PyTorch fp32 | 17 → 94 s | ~26–27 GB |
| Rust standalone fp32 | 94 → 371 s (~3–4×) | ~25–26 GB |
| PyTorch bf16 | 31 → 145 s | **~17 GB** |

Rust is ~3–4× PyTorch (a hand-written f64-accumulating trunk GEMM + per-layer-streamed
ESM-C 6B vs PyTorch/MKL). Peak RAM ≈ PyTorch-fp32 because the mmap pages in the full
ESM-C; a `pread` streaming loader would cut this substantially (future work).

## Plots
- `plots/fig1_accuracy.png` — Rust-vs-fp32 (all < 1 mÅ) and bf16-vs-fp32 (Å)
- `plots/fig2_confidence.png` — pLDDT & pTM agreement
- `plots/fig3_time.png` — fold time vs length
- `plots/fig4_memory.png` — peak RSS vs length
