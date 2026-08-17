"""Decomposed fp32 reference for the folding trunk + structure module + heads.

Builds the folding submodules standalone (no 11GB LM), loads their fp32 weights,
feeds the precomputed esm_s (from ref_lm.py), runs the REAL HF trunk forward, and
dumps: s_s_0/s_z_0 (P3), per-block s/z of the final recycle (P4), final s/z,
structure-module outputs (P5), and heads (P6).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk,
    categorical_lddt,
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
)
from transformers.models.esm.openfold_utils import feats
from transformers.models.esm.openfold_utils import residue_constants as rc

from common import FIX, Manifest, save_fixture, weights_path


def load_into(module, prefix, f, own_strip=True):
    sd = {}
    for k in f.keys():
        if k.startswith(prefix):
            name = k[len(prefix):] if own_strip else k
            sd[name] = f.get_tensor(k).float()
    missing, unexpected = module.load_state_dict(sd, strict=False)
    return len(sd), missing, unexpected


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "flgM"
    cfg = AutoConfig.from_pretrained("facebook/esmfold_v1")
    ecfg = cfg.esmfold_config
    tcfg = ecfg.trunk
    c_s, c_z = tcfg.sequence_state_dim, tcfg.pairwise_state_dim
    esm_feats = cfg.hidden_size
    sm_dim = tcfg.structure_module.sequence_dim
    hid = ecfg.lddt_head_hid_dim
    n_tok = rc.restype_num + 3
    print(f"embed_aa={ecfg.embed_aa} max_recycles={tcfg.max_recycles} num_blocks={tcfg.num_blocks} "
          f"c_s={c_s} c_z={c_z} sm_dim={sm_dim} lddt_hid={hid} n_tok={n_tok}")

    torch.manual_seed(0)
    esm_s_combine = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
    esm_s_mlp = nn.Sequential(nn.LayerNorm(esm_feats), nn.Linear(esm_feats, c_s), nn.ReLU(), nn.Linear(c_s, c_s))
    embedding = nn.Embedding(n_tok, c_s, padding_idx=0) if ecfg.embed_aa else None
    trunk = EsmFoldingTrunk(tcfg)
    distogram_head = nn.Linear(c_z, 64)
    ptm_head = nn.Linear(c_z, 64)
    lm_head = nn.Linear(c_s, n_tok)
    lddt_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, hid), nn.Linear(hid, hid), nn.Linear(hid, 37 * 50))

    with safe_open(weights_path(), framework="pt") as f:
        keys = set(f.keys())
        esm_s_combine.data = f.get_tensor("esm_s_combine").float()
        load_into(esm_s_mlp, "esm_s_mlp.", f)
        if embedding is not None and "embedding.weight" in keys:
            embedding.load_state_dict({"weight": f.get_tensor("embedding.weight").float()})
        n, miss, unexp = load_into(trunk, "trunk.", f)
        print(f"trunk: loaded {n} tensors, missing={len(miss)} unexpected={len(unexp)}")
        if miss:
            print("  first missing:", miss[:6])
        load_into(distogram_head, "distogram_head.", f)
        load_into(ptm_head, "ptm_head.", f)
        load_into(lm_head, "lm_head.", f)
        load_into(lddt_head, "lddt_head.", f)

    for m in [esm_s_mlp, trunk, distogram_head, ptm_head, lm_head, lddt_head] + ([embedding] if embedding else []):
        m.eval()

    # --- load precomputed esm_s (37 states, with cls/eos), slice to residues, stack ---
    lm_fx = os.path.join(FIX, f"lm/{name}", "esm_states.safetensors")
    with safe_open(lm_fx, framework="np") as f:
        ids = f.get_tensor("input_ids").astype(int).tolist()
        states = [torch.from_numpy(f.get_tensor(f"state_{i:02d}")) for i in range(37)]
    L = len(ids) - 2  # residues (drop cls/eos)
    # each state [L+2, 2560]; slice [1:-1] -> [L,2560]; stack along layer axis -> [1,L,37,2560]
    esm_s = torch.stack([s[1:-1] for s in states], dim=1).unsqueeze(0)  # [1, L, 37, C]
    print(f"esm_s {tuple(esm_s.shape)} L={L}")

    # aatype (AF2 restype order), position_ids, attention_mask
    seq_chars = []  # recover residues from esm ids is awkward; read seq from manifest
    with open(os.path.join(FIX, f"lm/{name}", "manifest.json")) as fh:
        import json
        seq = json.load(fh)[0]["seq"]
    aatype = torch.tensor([[rc.restype_order_with_x.get(a, 20) for a in seq]], dtype=torch.long)
    assert aatype.shape[1] == L, (aatype.shape, L)
    position_ids = torch.arange(L).unsqueeze(0)
    attn = torch.ones(1, L)

    # --- forward glue (lines 2108-2174) ---
    with torch.no_grad():
        esm_s = esm_s.to(esm_s_combine.dtype)
        esm_s_c = (esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)  # [1,L,C]
        s_s_0 = esm_s_mlp(esm_s_c)
        s_z_0 = s_s_0.new_zeros(1, L, L, c_z)
        if embedding is not None:
            s_s_0 = s_s_0 + embedding(aatype)
        save_fixture(f"trunk/{name}", "inputs", {"s_s_0": s_s_0[0], "s_z_0": s_z_0[0], "esm_s_combined": esm_s_c[0], "aatype": aatype[0].float()})

        # hooks: capture per-block (s,z) outputs (keep last recycle = last num_blocks calls)
        cap = []
        def mk_hook():
            def hook(_m, _i, out):
                cap.append((out[0].detach().clone(), out[1].detach().clone()))
            return hook
        handles = [b.register_forward_hook(mk_hook()) for b in trunk.blocks]
        # capture block-0 INPUT (s, z after relpos) each recycle -> keep last
        blk0_in = []
        h0 = trunk.blocks[0].register_forward_pre_hook(
            lambda m, inp: blk0_in.append((inp[0].detach().clone(), inp[1].detach().clone()))
        )
        # capture relpos output each recycle -> keep last
        relpos = []
        hr = trunk.pairwise_positional_embedding.register_forward_hook(
            lambda _m, _i, out: relpos.append(out.detach().clone())
        )

        structure = trunk(s_s_0, s_z_0, aatype, position_ids, attn, no_recycles=None)
        for h in handles:
            h.remove()
        h0.remove()
        hr.remove()

        # final-recycle block-0 input + relpos + structure-module inputs
        save_fixture(f"trunk/{name}", "blk0_input_final", {"s_in": blk0_in[-1][0][0], "z_in": blk0_in[-1][1][0]})
        save_fixture(f"trunk/{name}", "relpos_final", {"relpos": relpos[-1][0]})
        sm_s_in = trunk.trunk2sm_s(structure["s_s"])
        sm_z_in = trunk.trunk2sm_z(structure["s_z"])
        save_fixture(f"trunk/{name}", "sm_inputs", {"single": sm_s_in[0], "pair": sm_z_in[0]})

    nb = tcfg.num_blocks
    final_blocks = cap[-nb:]
    blk = {}
    for i, (s, z) in enumerate(final_blocks):
        blk[f"s_{i:02d}"] = s[0]
        blk[f"z_{i:02d}"] = z[0]
    save_fixture(f"trunk/{name}", "blocks_final_recycle", blk)

    # final trunk outputs + structure module
    save_fixture(f"trunk/{name}", "final", {"s_s": structure["s_s"][0], "s_z": structure["s_z"][0]})
    sm = {
        "frames": structure["frames"][:, 0],           # [n_block, L, 7]
        "positions": structure["positions"][:, 0],     # [n_block, L, 14, 3]
        "states": structure["states"][:, 0],           # [n_block, L, sm_dim]
        "angles": structure["angles"][:, 0],
        "unnormalized_angles": structure["unnormalized_angles"][:, 0],
        "sidechain_frames": structure["sidechain_frames"][:, 0],
    }
    save_fixture(f"trunk/{name}", "structure", sm)

    # heads
    with torch.no_grad():
        disto = distogram_head(structure["s_z"])
        disto = (disto + disto.transpose(1, 2)) / 2
        structure["aatype"] = aatype
        make_atom14_masks(structure)
        lddt = lddt_head(structure["states"]).reshape(structure["states"].shape[0], 1, L, -1, 50)
        plddt = categorical_lddt(lddt[-1], bins=50)
        ptm_logits = ptm_head(structure["s_z"])
        ptm = compute_tm(ptm_logits, max_bin=31, no_bins=64)
        pae = compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=64)
        # atom37 final positions
        atom37 = feats.atom14_to_atom37(structure["positions"][-1], structure)  # [1,L,37,3]
    save_fixture(f"trunk/{name}", "heads", {
        "distogram_logits": disto[0],
        "plddt": plddt[0],
        "ptm": ptm.reshape(1),
        "predicted_aligned_error": pae["predicted_aligned_error"][0],
        "atom37": atom37[0],
        "atom37_atom_exists": structure["atom37_atom_exists"][0],
    })

    man = Manifest(os.path.join(FIX, f"trunk/{name}", "manifest.json"))
    man.add(L=L, num_blocks=nb, seq=seq, plddt_mean=float(plddt.mean()), ptm=float(ptm))
    man.write()
    print(f"done. plddt_mean={float(plddt.mean()):.3f} ptm={float(ptm):.3f}")


if __name__ == "__main__":
    main()
