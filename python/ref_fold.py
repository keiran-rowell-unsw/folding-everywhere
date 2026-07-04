"""Full PyTorch fp32 ESMFold reference for one sequence (decomposed in-process).

Runs ESM-2 3B (fp32) then the folding trunk+structure+heads, producing final
atom37 coords + pLDDT + pTM. Used by benchmark.py (wrapped in /usr/bin/time -v).
Usage: python ref_fold.py <name> <SEQUENCE>
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.esm.modeling_esm import EsmModel
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk,
    categorical_lddt,
    compute_tm,
    make_atom14_masks,
)
from transformers.models.esm.openfold_utils import feats
from transformers.models.esm.openfold_utils import residue_constants as rc

from common import FIX, save_fixture, weights_path

torch.set_num_threads(int(os.environ.get("REF_THREADS", "4")))

VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P",
         "K", "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
AA2IDX = {a: i for i, a in enumerate(VOCAB)}


def write_pdb(path, atom37, mask, aatype, plddt):
    """atom37 [L,37,3], mask [L,37], aatype [L], plddt [L,37] (0..1). Matches the
    Rust pdb.rs column layout so the two PDBs are directly comparable."""
    L = atom37.shape[0]
    lines = []
    serial = 1
    for i in range(L):
        a = int(aatype[i])
        resn = rc.restype_1to3.get(rc.restypes[a], "UNK") if a < 20 else "UNK"
        for j, nm in enumerate(rc.atom_types):
            if mask[i, j] < 0.5:
                continue
            x, y, z = atom37[i, j]
            b = float(plddt[i, j]) * 100.0
            atname = nm if len(nm) >= 4 else f" {nm:<3}"
            elem = nm[0]
            lines.append(f"ATOM  {serial:>5} {atname} {resn:>3} A{i+1:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{b:>6.2f}          {elem:>2}")
            serial += 1
    lines.append("TER")
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    name, seq = sys.argv[1], sys.argv[2]
    out_pdb = sys.argv[3] if len(sys.argv) > 3 else None
    cfg = AutoConfig.from_pretrained("facebook/esmfold_v1")
    L = len(seq)

    # ---- Stage 1: ESM-2 3B fp32 ----
    t0 = time.monotonic()
    esm = EsmModel(cfg, add_pooling_layer=False).eval()
    own = dict(esm.named_parameters())
    with safe_open(weights_path(), framework="pt") as f:
        for k in f.keys():
            if k.startswith("esm.") and k[4:] in own:
                own[k[4:]].data = f.get_tensor(k).float()
    ids = [AA2IDX["<cls>"]] + [AA2IDX.get(a, 3) for a in seq] + [AA2IDX["<eos>"]]
    input_ids = torch.tensor([ids])
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        hs = esm(input_ids=input_ids, attention_mask=attn, output_hidden_states=True).hidden_states
    esm_s = torch.stack(hs, dim=2)[:, 1:-1].detach()  # [1,L,37,2560]
    del esm, own
    t_lm = time.monotonic() - t0

    # ---- Stage 2: folding ----
    t1 = time.monotonic()
    ecfg, tcfg = cfg.esmfold_config, cfg.esmfold_config.trunk
    c_s, c_z = tcfg.sequence_state_dim, tcfg.pairwise_state_dim
    sm_dim, hid = tcfg.structure_module.sequence_dim, ecfg.lddt_head_hid_dim
    n_tok = rc.restype_num + 3
    esm_s_combine = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
    esm_s_mlp = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, c_s), nn.ReLU(), nn.Linear(c_s, c_s))
    embedding = nn.Embedding(n_tok, c_s, padding_idx=0)
    trunk = EsmFoldingTrunk(tcfg)
    distogram_head = nn.Linear(c_z, 64)
    ptm_head = nn.Linear(c_z, 64)
    lddt_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 37 * 50))

    def load(mod, pre, f):
        sd = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
        mod.load_state_dict(sd, strict=False)

    with safe_open(weights_path(), framework="pt") as f:
        esm_s_combine.data = f.get_tensor("esm_s_combine").float()
        load(esm_s_mlp, "esm_s_mlp.", f)
        embedding.load_state_dict({"weight": f.get_tensor("embedding.weight").float()})
        load(trunk, "trunk.", f)
        load(distogram_head, "distogram_head.", f)
        load(ptm_head, "ptm_head.", f)
        load(lddt_head, "lddt_head.", f)
    for m in [esm_s_mlp, trunk, distogram_head, ptm_head, lddt_head, embedding]:
        m.eval()

    aatype = torch.tensor([[rc.restype_order_with_x.get(a, 20) for a in seq]])
    with torch.no_grad():
        s = (esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = esm_s_mlp(s) + embedding(aatype)
        s_z_0 = s_s_0.new_zeros(1, L, L, c_z)
        structure = trunk(s_s_0, s_z_0, aatype, torch.arange(L).unsqueeze(0), torch.ones(1, L), no_recycles=None)
        structure["aatype"] = aatype
        make_atom14_masks(structure)
        lddt = lddt_head(structure["states"]).reshape(structure["states"].shape[0], 1, L, -1, 50)
        plddt = categorical_lddt(lddt[-1], bins=50)[0]
        ptm = compute_tm(ptm_head(structure["s_z"]), max_bin=31, no_bins=64)
        atom37 = feats.atom14_to_atom37(structure["positions"][-1], structure)[0]
    t_fold = time.monotonic() - t1

    save_fixture(f"bench/{name}", "ref", {
        "atom37": atom37,
        "atom37_atom_exists": structure["atom37_atom_exists"][0],
        "plddt": plddt,
        "ptm": ptm.reshape(1),
    })
    if out_pdb is not None:
        write_pdb(out_pdb, atom37.numpy(), structure["atom37_atom_exists"][0].numpy(), aatype[0].numpy(), plddt.numpy())

    meta = {"name": name, "L": L, "t_lm": t_lm, "t_fold": t_fold, "t_total": t_lm + t_fold,
            "plddt_mean": float(plddt.mean()), "ptm": float(ptm)}
    with open(os.path.join(FIX, f"bench/{name}", "ref_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
