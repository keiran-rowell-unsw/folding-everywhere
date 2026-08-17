"""Decomposed fp32 reference for the ESM-2 3B backbone.

Loads EsmModel (the `esm.*` submodule of esmfold_v1) in fp32 and runs HF's exact
forward with output_hidden_states, dumping all 37 hidden states + last_hidden_state.
Memory-safe: builds the model, then swaps each param's .data in place from the
mmap'd safetensors (peak ~11 GB), keeping HF's computed buffers (rotary inv_freq).
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.esm.modeling_esm import EsmModel

from common import FIX, Manifest, save_fixture, weights_path

# ESM-2 33-token alphabet (fixed order); matches the esm tokenizer.
VOCAB = list("0")  # placeholder, replaced below
VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
    "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-",
    "<null_1>", "<mask>",
]
AA2IDX = {a: i for i, a in enumerate(VOCAB)}


def tokenize(seq: str):
    ids = [AA2IDX["<cls>"]] + [AA2IDX[a] for a in seq] + [AA2IDX["<eos>"]]
    return ids


def load_esm_fp32():
    cfg = AutoConfig.from_pretrained("facebook/esmfold_v1")
    print(f"esm cfg: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads} ffn={cfg.intermediate_size} vocab={cfg.vocab_size} "
          f"posemb={cfg.position_embedding_type} token_dropout={cfg.token_dropout} "
          f"eps={cfg.layer_norm_eps} emb_ln_before={cfg.emb_layer_norm_before}")
    print("allocating EsmModel (random init, ~11GB)...", flush=True)
    esm = EsmModel(cfg, add_pooling_layer=False)
    esm.eval()
    own = dict(esm.named_parameters())
    loaded = 0
    with safe_open(weights_path(), framework="pt") as f:
        for k in f.keys():
            if not k.startswith("esm."):
                continue
            name = k[4:]
            if name in own:
                own[name].data = f.get_tensor(k).float()
                loaded += 1
    print(f"loaded {loaded} esm params (fp32)", flush=True)
    return esm


def main():
    seq = sys.argv[1] if len(sys.argv) > 1 else (
        "MSIDRTSPLKPVSTVQTRETSDTPVQKTRQEKTSAATSASVTLSDAQAKLMQPGVSDINM"
        "ERVEALKTAIRNGELKMDTGKIADSLIREAQSYLQSK"
    )
    name = sys.argv[2] if len(sys.argv) > 2 else "flgM"
    print(f"seq {name} L={len(seq)}")

    ids = tokenize(seq)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn = torch.ones_like(input_ids)

    esm = load_esm_fp32()
    with torch.no_grad():
        out = esm(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    hs = out.hidden_states  # tuple length num_layers+1 = 37
    last = out.last_hidden_state
    print(f"num hidden states: {len(hs)}; shapes {tuple(hs[0].shape)}; last {tuple(last.shape)}")

    # resolve final-LN ambiguity: is hs[-1] already post-emb_layer_norm_after?
    diff_last = (hs[-1] - last).abs().max().item()
    print(f"max|hs[-1]-last_hidden_state| = {diff_last:.3e} "
          f"({'hs[-1] is POST-LN (==last)' if diff_last < 1e-4 else 'hs[-1] is PRE-LN'})")

    tensors = {f"state_{i:02d}": hs[i][0] for i in range(len(hs))}
    tensors["last_hidden"] = last[0]
    tensors["input_ids"] = torch.tensor(ids, dtype=torch.float32)
    save_fixture(f"lm/{name}", "esm_states", tensors)

    man = Manifest(os.path.join(FIX, f"lm/{name}", "manifest.json"))
    man.add(kind="esm_states", n_states=len(hs), L=len(seq), seq=seq,
            hs_last_is_postln=bool(diff_last < 1e-4))
    man.write()
    print("done.")


if __name__ == "__main__":
    main()
