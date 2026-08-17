"""Dump every intermediate of an upstream ProteinMPNN run to safetensors.

The Rust parity tests replay these stage by stage: featurization -> graph ->
edge features -> each encoder layer -> each decoder layer -> log-probs -> the
sampled sequence.

Everything here calls the *unmodified* upstream modules, so the fixtures cannot
drift from the published model. The forward pass is re-expressed inline only so
the intermediate tensors can be captured.

Usage:  python ref_dump.py <pdb> [name] [--seed N]
"""
import argparse
import os

import numpy as np
import torch

from common import REF, load_model, save_fixture  # noqa: F401  (sets sys.path)
from protein_mpnn_utils import (  # noqa: E402
    ProteinMPNN,  # noqa: F401
    cat_neighbors_nodes,
    gather_nodes,
    parse_PDB,
    tied_featurize,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def build_inputs(pdb_path, designed=None):
    pdb_dict_list = parse_PDB(pdb_path)
    all_chain_list = sorted(
        item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == "seq_chain"
    )
    designed_chain_list = designed if designed else all_chain_list
    fixed_chain_list = [c for c in all_chain_list if c not in designed_chain_list]
    chain_id_dict = {pdb_dict_list[0]["name"]: (designed_chain_list, fixed_chain_list)}
    out = tied_featurize(pdb_dict_list, "cpu", chain_id_dict, None, None, None, None, None, False)
    (X, S, mask, lengths, chain_M, chain_encoding_all, letter_list, visible_list,
     masked_list, masked_chain_length_list, chain_M_pos, omit_AA_mask, residue_idx,
     dihedral_mask, tied_pos, pssm_coef, pssm_bias, pssm_log_odds, bias_by_res, tied_beta) = out
    return dict(
        X=X, S=S, mask=mask, chain_M=chain_M, chain_M_pos=chain_M_pos,
        residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
        bias_by_res=bias_by_res, name=pdb_dict_list[0]["name"],
        designed=designed_chain_list, fixed=fixed_chain_list,
        seq="".join(ALPHABET[i] for i in S[0].tolist()),
    )


def _edge_input(feat, X, mask, residue_idx, chain_labels, E_idx):
    """`torch.cat((E_positional, RBF_all), -1)` — everything ProteinFeatures
    computes before the 416->128 projection."""
    from protein_mpnn_utils import gather_edges

    b = X[:, :, 1, :] - X[:, :, 0, :]
    c = X[:, :, 2, :] - X[:, :, 1, :]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
    Ca, N, C, O = X[:, :, 1, :], X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :]
    D_neighbors, _ = feat._dist(Ca, mask)

    rbf = [feat._rbf(D_neighbors)]
    pairs = [(N, N), (C, C), (O, O), (Cb, Cb), (Ca, N), (Ca, C), (Ca, O), (Ca, Cb),
             (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C), (N, Ca), (C, Ca),
             (O, Ca), (Cb, Ca), (C, N), (O, N), (Cb, N), (C, Cb), (O, Cb), (C, O)]
    for A, B in pairs:
        rbf.append(feat._get_rbf(A, B, E_idx))
    RBF_all = torch.cat(tuple(rbf), dim=-1)

    offset = residue_idx[:, :, None] - residue_idx[:, None, :]
    offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
    d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
    E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
    E_positional = feat.embeddings(offset.long(), E_chains)
    return torch.cat((E_positional, RBF_all), -1)


def order_masks(model, decoding_order, E_idx, mask):
    """Reproduce the mask_bw / mask_fw construction from ProteinMPNN.forward."""
    mask_size = E_idx.shape[1]
    perm = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(mask_size, mask_size))),
        perm, perm,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    return mask_1D * mask_attend, mask_1D * (1.0 - mask_attend)


@torch.no_grad()
def dump(pdb_path, out_name, seed, temperature=0.1):
    model, ckpt = load_model()
    inp = build_inputs(pdb_path)
    X, S, mask = inp["X"], inp["S"], inp["mask"]
    chain_M, chain_M_pos = inp["chain_M"], inp["chain_M_pos"]
    residue_idx, chain_encoding_all = inp["residue_idx"], inp["chain_encoding_all"]
    L = int(mask.shape[1])

    fx = {
        "X": X, "S": S, "mask": mask, "chain_M": chain_M, "chain_M_pos": chain_M_pos,
        "residue_idx": residue_idx, "chain_encoding_all": chain_encoding_all,
    }

    # ---- features ---------------------------------------------------------
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    fx["E"] = E
    fx["E_idx"] = E_idx

    # Intermediate pieces of ProteinFeatures, for finer-grained tests.
    b = X[:, :, 1, :] - X[:, :, 0, :]
    c = X[:, :, 2, :] - X[:, :, 1, :]
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
    fx["Cb"] = Cb
    D_neighbors, _ = model.features._dist(X[:, :, 1, :], mask)
    fx["D_neighbors"] = D_neighbors
    fx["RBF_D_neighbors"] = model.features._rbf(D_neighbors)

    # The 416-wide pre-projection edge tensor (16 positional + 400 RBF). Purely
    # geometric, so the Rust side should reproduce it bit-for-bit.
    fx["E_input"] = _edge_input(model.features, X, mask, residue_idx, chain_encoding_all, E_idx)

    # ---- encoder ----------------------------------------------------------
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]))
    h_E = model.W_e(E)
    fx["h_E_init"] = h_E
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    fx["enc_mask_attend"] = mask_attend
    for i, layer in enumerate(model.encoder_layers):
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        fx[f"enc{i}_h_V"] = h_V
        fx[f"enc{i}_h_E"] = h_E

    # ---- decoder (teacher forced, native sequence) ------------------------
    torch.manual_seed(seed)
    randn_1 = torch.randn(chain_M.shape)
    fx["randn_1"] = randn_1
    chain_M_fwd = chain_M * chain_M_pos * mask
    decoding_order = torch.argsort((chain_M_fwd + 0.0001) * torch.abs(randn_1))
    fx["decoding_order_fwd"] = decoding_order

    h_S = model.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    mask_bw, mask_fw = order_masks(model, decoding_order, E_idx, mask)
    fx["mask_bw"] = mask_bw[:, :, :, 0]
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    h_Vd = h_V
    for i, layer in enumerate(model.decoder_layers):
        h_ESV = cat_neighbors_nodes(h_Vd, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_Vd = layer(h_Vd, h_ESV, mask)
        fx[f"dec{i}_h_V"] = h_Vd
    logits = model.W_out(h_Vd)
    fx["logits"] = logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    fx["log_probs"] = log_probs

    # Cross-check against the public API (must be identical).
    torch.manual_seed(seed)
    randn_ref = torch.randn(chain_M.shape)
    ref_lp = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_ref)
    assert torch.equal(ref_lp, log_probs), "inline forward diverged from model.forward"

    # ---- sampling ---------------------------------------------------------
    omit_AAs_np = np.array([aa in "X" for aa in ALPHABET]).astype(np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))
    torch.manual_seed(seed)
    randn_2 = torch.randn(chain_M.shape)
    fx["randn_2"] = randn_2
    sample_dict = model.sample(
        X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
        temperature=temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
        chain_M_pos=chain_M_pos, omit_AA_mask=None, pssm_coef=torch.zeros(1, L),
        pssm_bias=torch.zeros(1, L, 21), pssm_multi=0.0, pssm_log_odds_flag=False,
        pssm_log_odds_mask=torch.zeros(1, L, 21), pssm_bias_flag=False,
        bias_by_res=inp["bias_by_res"],
    )
    fx["sample_S"] = sample_dict["S"]
    fx["sample_probs"] = sample_dict["probs"]
    fx["sample_decoding_order"] = sample_dict["decoding_order"]

    # Score of the sampled sequence, the number the FASTA header reports.
    S_sample = sample_dict["S"]
    lp_sample = model(
        X, S_sample, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all,
        randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"],
    )
    fx["sample_log_probs"] = lp_sample

    path = save_fixture("model", out_name, fx)
    print(f"wrote {path}")
    print(f"  name={inp['name']} L={L} K={int(E_idx.shape[-1])} "
          f"designed={inp['designed']} fixed={inp['fixed']}")
    print(f"  native   : {inp['seq']}")
    print(f"  sampled  : {''.join(ALPHABET[i] for i in S_sample[0].tolist())}")
    return fx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb")
    ap.add_argument("name", nargs="?", default=None)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.pdb))[0]
    dump(args.pdb, name, args.seed, args.temperature)
