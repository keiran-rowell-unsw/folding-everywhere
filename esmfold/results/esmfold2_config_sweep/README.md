# ESMFold2 config sweep — bit-exact across `num_loops` / `num_sampling_steps`

A focused check that the pure-Rust fp32 ESMFold2 port stays **bit-exact to the PyTorch fp32
reference** as the two inference-depth knobs change — not just at the single benchmark setting.

One protein (**crambin46**, 46 aa, 352 atoms), **seed 0**, a single diffusion sample
(`num_diffusion_samples=1` on both sides), 6 configurations that vary `num_loops` and
`num_sampling_steps`. For each config we run the PyTorch fp32 reference and the Rust
`fold_standalone` at the *same* settings and compare the predicted coordinates directly
(both share a frame at a fixed seed, so an unaligned all-atom deviation is the bit-exactness
metric).

## Results

| # | `num_loops` | `num_sampling_steps` | RMSD (Å) | max dev (Å) | pLDDT (pt / rust) | pTM (pt / rust) |
|---|---|---|---|---|---|---|
| 1 (baseline) | 3  | 14 | 7.3e-05  | 1.95e-04 | 0.626521 / 0.62652 | 0.447456 / 0.44746 |
| 2 (loops)    | 6  | 14 | 1.21e-04 | 8.17e-04 | 0.605712 / 0.60571 | 0.412384 / 0.41238 |
| 3 (loops)    | 10 | 14 | 9.65e-05 | 4.62e-04 | 0.600380 / 0.60038 | 0.412665 / 0.41267 |
| 4 (steps)    | 3  | 28 | 1.06e-04 | 7.46e-04 | 0.629200 / 0.62920 | 0.447903 / 0.44790 |
| 5 (steps)    | 3  | 42 | 8.46e-05 | 3.48e-04 | 0.626516 / 0.62652 | 0.446883 / 0.44688 |
| 6 (both)     | 6  | 28 | 1.11e-04 | 5.09e-04 | 0.601890 / 0.60189 | 0.412282 / 0.41228 |

**Every configuration is bit-exact to fp32 round-off** — all-atom RMSD **7e-5–1.2e-4 Å**, max
atom deviation **≤ 8.2e-4 Å**, with pLDDT and pTM matching to 4–5 decimals. Note the
confidences *change* across configs (pLDDT 0.600 → 0.629), so the knobs genuinely alter the
fold — and the Rust port tracks PyTorch at each one. This confirms the RNG reproduction is
correct not only for the benchmark setting but across the loop count (extra per-loop dropout
draws) and the step count (extra per-step diffusion-noise draws).

## Folder contents
- `sweep.csv` — the table above (machine-readable).
- `coords/pt_*.npy`, `coords/rust_*.npy` — predicted coordinates `[n_atoms, 3]` for each
  config (PyTorch fp32 and Rust fp32).
- `sweep_configs.py` — the PyTorch fp32 reference sweep (needs the `esmfold2_fp32` harness deps).
- `compare_sweep.py` — recomputes the table from the `coords/` npys.

## Reproduce
```bash
# PyTorch fp32 references (needs the ESMFold2 weights + transformers fork; see ../../esmfold2_fp32/):
python sweep_configs.py crambin46 coords
# Rust, same configs (from repo root, after `cargo build --release`):
for c in "3 14" "6 14" "10 14" "3 28" "3 42" "6 28"; do set -- $c; \
  ./target/release/fold_standalone TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN 0 \
    coords/rust_crambin46_l$1_s$2.npy $1 $2; done
python compare_sweep.py crambin46 coords sweep.csv
```
