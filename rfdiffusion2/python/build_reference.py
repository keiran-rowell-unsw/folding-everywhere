#!/usr/bin/env python3
"""Phase A — construct the reference model exactly as `Sampler.load_model` does.

This is the gate for rungs 4-8: no fixture above rung 3 can exist until the
reference runs here. It reproduces `rf_diffusion/inference/model_runners.py`
`Sampler.load_model` step for step, so the config merge (which decides the
architecture) is the upstream one and not a guess.

    .venv/bin/python python/build_reference.py
"""
import os
import sys

import common
import torch


def load_inference_conf(config_name="aa", overrides=None):
    """Compose `config/inference/<name>.yaml` the way hydra would."""
    from hydra import compose, initialize_config_dir
    ref = common.add_ref_to_path()
    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name=config_name, overrides=overrides or [])


def build(ckpt_path=None, config_name="aa", verbose=True):
    ref = common.add_ref_to_path()
    ckpt_path = ckpt_path or common.CKPT_173

    from omegaconf import OmegaConf
    from rf_diffusion.config import config_format
    from rf_diffusion import noisers
    from rf_diffusion.frame_diffusion.rf_score.model import RFScore

    conf = load_inference_conf(config_name)
    OmegaConf.set_struct(conf, False)
    conf.inference.ckpt_path = ckpt_path

    if verbose:
        print(f"loading {ckpt_path}")
    weights_pkl = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights_conf = config_format.translate_obsolete_weight_options(
        weights_pkl["conf"])

    base_training_fp = os.path.join(
        ref, "rf_diffusion", "config", "training", "base.yaml")
    base_training_conf = OmegaConf.load(base_training_fp)
    for c in (weights_conf, base_training_conf):
        OmegaConf.set_struct(c, False)

    merged = OmegaConf.merge(base_training_conf, weights_conf, conf)

    diffuser = noisers.get(merged.diffuser)
    model = RFScore(merged.rf.model, diffuser, torch.device("cpu"))

    sd_key = merged.inference.state_dict_to_load  # 'model_state_dict' (EMA)
    missing, unexpected = model.load_state_dict(weights_pkl[sd_key],
                                                strict=True), None
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"diffuser: {type(diffuser).__name__} "
              f"(type={merged.diffuser.get('type')}, T={merged.diffuser.T})")
        print(f"state dict loaded: {sd_key}")
        print(f"model parameters: {n_params}")
        print(f"n_main_block={merged.rf.model.n_main_block} "
              f"n_extra_block={merged.rf.model.n_extra_block} "
              f"d_pair={merged.rf.model.d_pair} "
              f"n_head_pair={merged.rf.model.n_head_pair}")
    return model, diffuser, merged, weights_pkl


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    model, diffuser, conf, _ = build()
    # The parameter count is the rung-3 number; if these disagree the config
    # merge selected a different architecture than the checkpoint was saved from.
    n = sum(p.numel() for p in model.parameters())
    assert n == 82_911_693, f"parameter count {n} != 82911693 from the inventory"
    print("\nOK: reference model constructed on CPU and matches the "
          "inventoried parameter count.")
