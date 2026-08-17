"""Dump ESM-C 6B per-layer hidden states for a test sequence (fp32 CPU).

Builds the LM input the way compute_lm_hidden_states does for a single chain:
[BOS=0] + per-residue input_ids + [EOS=2], sequence_id all 0. Saves input_ids,
seq_id and the stacked [81, T, 2560] hidden states as .npy fixtures.
"""
import os, sys
import numpy as np
import torch
import common

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ubiquitin76"
    seq = common.PROTEINS[name]
    FIX = common.FIX
    from transformers.models.esmfold2 import protein_utils as pu
    from transformers.models.esmc.modeling_esmc import ESMCModel

    feats = pu.prepare_protein_features(seq)
    res_ids = feats["input_ids"][0].tolist()       # per-residue ESM-C ids
    lm_ids = torch.tensor([[0] + res_ids + [2]], dtype=torch.long)  # BOS .. EOS
    seq_id = torch.zeros_like(lm_ids)              # single chain

    torch.set_num_threads(8)
    model = ESMCModel.from_pretrained("biohub/ESMC-6B", dtype=torch.float32,
                                      low_cpu_mem_usage=True).eval()
    with torch.inference_mode():
        out = model(input_ids=lm_ids, sequence_id=seq_id, output_hidden_states=True)
    hs = out.hidden_states          # [81, 1, T, 2560]
    last = out.last_hidden_state    # [1, T, 2560]
    print("hidden_states:", tuple(hs.shape), "last:", tuple(last.shape), flush=True)

    np.save(os.path.join(FIX, f"esmc_{name}_ids.npy"), lm_ids[0].numpy().astype(np.int64))
    np.save(os.path.join(FIX, f"esmc_{name}_seqid.npy"), seq_id[0].numpy().astype(np.int64))
    np.save(os.path.join(FIX, f"esmc_{name}_hs.npy"),
            hs[:, 0].to(torch.float32).numpy())  # [81, T, 2560]
    print("saved esmc fixtures for", name, flush=True)

if __name__ == "__main__":
    main()
