#!/usr/bin/env python3
"""Capture the SE(3) transformer's *internal* tensors for one block.

The pieces that decide whether the refiner is bit-exact — the spherical
harmonics, the Clebsch-Gordan bases, the fused basis views, the per-edge
convolution output, the attention weights — are plain functions, not modules, so
no forward hook sees them. This wraps them directly.

Everything is captured for the **first** call only (`extra_block.0.str2str.se3`);
later blocks reuse the same code paths.

    PYTHONPATH=<ref> RFD2_REF=<ref> .venv/bin/python python/dump_se3.py --pinned
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def patch_nvtx():
    import contextlib

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield

    torch.cuda.nvtx.range = _noop
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--out", default="se3_io")
    args = ap.parse_args()

    patch_nvtx()
    if args.pinned:
        import pinned
        print(f"PINNED: {len(pinned.enable())} entry points")
    ref = common.add_ref_to_path()

    cap = {}
    done = {"n": 0}

    import rf2aa.SE3Transformer.se3_transformer.model.basis as basis_mod
    import rf2aa.SE3Transformer.se3_transformer.model.layers.convolution as conv_mod
    import rf2aa.SE3Transformer.se3_transformer.model.layers.attention as attn_mod
    import rf2aa.SE3Transformer.se3_transformer.model.layers.norm as norm_mod
    import rf2aa.SE3Transformer.se3_transformer.model.transformer as tf_mod

    orig_sh = basis_mod.get_spherical_harmonics

    def sh_spy(rel_pos, max_degree):
        out = orig_sh(rel_pos, max_degree)
        if "sh.0" not in cap:
            cap["rel_pos"] = rel_pos.detach().cpu().clone()
            for i, t in enumerate(out):
                cap[f"sh.{i}"] = t.detach().cpu().float().clone()
        return out
    basis_mod.get_spherical_harmonics = sh_spy

    orig_upd = basis_mod.update_basis_with_fused

    def upd_spy(b, max_degree, use_pad_trick, fully_fused):
        out = orig_upd(b, max_degree, use_pad_trick, fully_fused)
        if "basis.in0_fused" not in cap:
            for k, v in out.items():
                cap[f"basis.{k}"] = v.detach().cpu().float().clone()
        return out
    basis_mod.update_basis_with_fused = upd_spy
    tf_mod.update_basis_with_fused = upd_spy
    tf_mod.get_basis = basis_mod.get_basis

    orig_vconv = conv_mod.VersatileConvSE3.forward

    def vconv_spy(self, features, invariant_edge_feats, basis):
        out = orig_vconv(self, features, invariant_edge_feats, basis)
        i = done["n"]
        if i < 6:
            cap[f"vconv{i}.features"] = features.detach().cpu().clone()
            cap[f"vconv{i}.edge"] = invariant_edge_feats.detach().cpu().clone()
            cap[f"vconv{i}.radial"] = (
                self.radial_func(invariant_edge_feats).detach().cpu().clone())
            cap[f"vconv{i}.out"] = out.detach().cpu().clone()
            done["n"] = i + 1
        return out
    conv_mod.VersatileConvSE3.forward = vconv_spy

    orig_attn = attn_mod.AttentionSE3.forward

    def attn_spy(self, value, key, query, graph):
        out = orig_attn(self, value, key, query, graph)
        if "attn.key" not in cap:
            cap["attn.key"] = (key if torch.is_tensor(key) else key["0"]).detach().cpu().clone()
            cap["attn.value"] = (
                value if torch.is_tensor(value) else value["0"]).detach().cpu().clone()
            for k, v in query.items():
                cap[f"attn.query.{k}"] = v.detach().cpu().clone()
            for k, v in out.items():
                cap[f"attn.out.{k}"] = v.detach().cpu().clone()
        return out
    attn_mod.AttentionSE3.forward = attn_spy

    orig_norm = norm_mod.NormSE3.forward

    def norm_spy(self, features, *a, **k):
        out = orig_norm(self, features, *a, **k)
        if "norm.in.0" not in cap:
            for kk, v in features.items():
                cap[f"norm.in.{kk}"] = v.detach().cpu().clone()
            for kk, v in out.items():
                cap[f"norm.out.{kk}"] = v.detach().cpu().clone()
        return out
    norm_mod.NormSE3.forward = norm_spy

    from hydra import compose, initialize_config_dir
    from rf_diffusion.inference import model_runners
    from rf_diffusion import run_inference as ri

    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb",
        "inference.ligand='NAD,OXM'",
        "contigmap.contigs=['10,A106-106,10']",
        "inference.contig_as_guidepost=False",
        "inference.num_designs=1",
        "inference.deterministic=True",
        "inference.idealize_sidechain_outputs=False",
        "inference.write_trb_indep=False",
        "diffuser.T=2",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        conf = compose(config_name="aa", overrides=overrides)
    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(conf.inference.seed_offset)

    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    import rf_diffusion.features as features
    mconf = sampler._conf
    fc = features.init_tXd_inference(
        indep, getattr(mconf, "extra_tXd", []), mconf.extra_tXd_params,
        mconf.inference.conditions)
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": torch.tensor(1)}
    sampler.sample_step(int(t_step_input), indep, None, extra, fc)

    print(f"captured {len(cap)} tensors")
    common.write_fixture(args.out, "io", cap,
                         {"tag": "pinned" if args.pinned else "stock"})


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
