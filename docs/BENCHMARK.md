# ESMFold v1 — PyTorch fp32 vs pure-Rust fp32 benchmark

| protein | L | PyTorch time | PyTorch peak RSS | Rust time | Rust peak RSS | atom RMSD | max dev | pLDDT (torch/rust) | pTM (torch/rust) |
|---|---|---|---|---|---|---|---|---|---|
| crambin | 46 | 135s | 15.1 GB | 133s | 8.3 GB | 0.0001 Å | 0.000 Å | 0.423/0.423 | 0.174/0.174 |
| ubiquitin | 76 | 177s | 15.1 GB | 282s | 8.3 GB | 0.0000 Å | 0.000 Å | 0.774/0.774 | 0.829/0.829 |
| flgM | 97 | 202s | 15.1 GB | 420s | 8.4 GB | 0.0001 Å | 0.000 Å | 0.541/0.541 | 0.256/0.256 |
| trxa | 109 | 229s | 15.1 GB | 517s | 8.4 GB | 0.0000 Å | 0.000 Å | 0.815/0.815 | 0.898/0.898 |
| lysozyme | 129 | 253s | 15.1 GB | 711s | 8.5 GB | 0.0001 Å | 0.000 Å | 0.836/0.836 | 0.907/0.907 |
