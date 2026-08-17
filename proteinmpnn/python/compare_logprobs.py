"""Full-precision comparison of the two implementations' log-probabilities.

The FASTA header only carries 4 decimals, which is not enough to characterise
numerical agreement. This runs both sides on the native sequence and compares
the raw `[L, 21]` fp32 log-probability matrices element by element.
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import torch

from common import REF, REPO, RESULTS, load_model
from protein_mpnn_utils import tied_featurize, parse_PDB  # noqa: E402

RUST_BIN = os.path.join(REPO, "target", "release", "mpnn")
WEIGHTS = os.path.join(REF, "vanilla_model_weights", "v_48_020.pt")


@torch.no_grad()
def torch_logprobs(model, pdb, seed, n_burn):
    pdb_dict_list = parse_PDB(pdb)
    chains = sorted(i[-1:] for i in list(pdb_dict_list[0]) if i[:9] == "seq_chain")
    chain_id_dict = {pdb_dict_list[0]["name"]: (chains, [])}
    out = tied_featurize(pdb_dict_list, "cpu", chain_id_dict, None, None, None, None, None, False)
    X, S, mask, _, chain_M, chain_enc = out[0], out[1], out[2], out[3], out[4], out[5]
    chain_M_pos, residue_idx = out[10], out[12]

    # Match protein_mpnn_run.py's stream position: seed, then the draws the
    # ProteinMPNN constructor burns, then randn_1.
    torch.manual_seed(seed)
    if n_burn:
        torch.empty(n_burn, dtype=torch.float32).uniform_(0, 1)
    randn_1 = torch.randn(chain_M.shape)
    lp = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_enc, randn_1)
    return lp[0].numpy(), mask[0].numpy(), S[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, _ = load_model()
    n_burn = sum(p.numel() for p in model.parameters() if p.dim() > 1)
    # Rebuild the constructor's total draw count the same way the Rust side does.
    lin_w = sum(p.numel() for n, p in model.named_parameters()
                if n.endswith(".weight") and p.dim() > 1 and "W_s" not in n)
    lin_b = sum(p.numel() for n, p in model.named_parameters()
                if n.endswith(".bias") and "norm" not in n)
    emb = model.W_s.weight.numel()
    n_burn = lin_w + lin_b + emb + n_burn

    pdb_dir = os.path.join(RESULTS, "pdb")
    pdbs = sorted(os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb"))
    if args.limit:
        pdbs = pdbs[: args.limit]

    tmp = os.path.join(RESULTS, "_runs", "logprobs")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for i, pdb in enumerate(pdbs, 1):
        name = os.path.splitext(os.path.basename(pdb))[0]
        lp_t, mask, S = torch_logprobs(model, pdb, args.seed, n_burn)

        dump = os.path.join(tmp, f"{name}.f32")
        r = subprocess.run(
            [RUST_BIN, "--pdb", pdb, "--weights", WEIGHTS, "--score_only",
             "--dump", dump, "--seed", str(args.seed), "--quiet", "--out", os.devnull],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"{name}: rust failed: {r.stderr[-500:]}")
            continue
        lp_r = np.fromfile(dump, dtype=np.float32).reshape(lp_t.shape)

        d = np.abs(lp_t - lp_r)
        num = float((lp_t.astype(np.float64) * lp_r.astype(np.float64)).sum())
        den = float(np.linalg.norm(lp_t.astype(np.float64)) * np.linalg.norm(lp_r.astype(np.float64)))
        cos = num / den if den else 1.0
        bitexact = float((lp_t.view(np.int32) == lp_r.view(np.int32)).mean())
        # argmax agreement: which residue the model would greedily pick
        argmax_agree = float((lp_t.argmax(-1) == lp_r.argmax(-1)).mean())
        sc_t = float(-(lp_t[np.arange(len(S)), S] * mask).sum() / mask.sum())
        sc_r = float(-(lp_r[np.arange(len(S)), S] * mask).sum() / mask.sum())

        rows.append(dict(
            name=name, L=int(lp_t.shape[0]),
            max_abs=float(d.max()), mean_abs=float(d.mean()),
            cosine=cos, bitexact_frac=bitexact, argmax_agree=argmax_agree,
            torch_score=sc_t, rust_score=sc_r, score_absdiff=abs(sc_t - sc_r),
        ))
        print(f"[{i}/{len(pdbs)}] {name:6s} L={lp_t.shape[0]:4d} "
              f"max_abs={d.max():.3e} cos={cos:.12f} "
              f"argmax_agree={argmax_agree:.4f} dscore={abs(sc_t - sc_r):.3e}")

    out = os.path.join(RESULTS, "logprob_accuracy.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmax over all proteins: max_abs={max(r['max_abs'] for r in rows):.3e}, "
          f"min cosine={min(r['cosine'] for r in rows):.12f}, "
          f"min argmax agreement={min(r['argmax_agree'] for r in rows):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
