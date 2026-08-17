#!/usr/bin/env python3
"""Capture the refinement loop: `str_refiner`'s IO plus the two gradient terms.

`calc_lj_grads` and `calc_chiral_grads` are plain functions, so no forward hook
sees them — but their outputs are two of the SE(3) transformer's degree-1 inputs
and (for LJ) 40 of its degree-0 inputs, so the port needs them fixtured before
the reverse passes can be written.

Also captures `compute_all_atom`'s inputs and outputs, since the LJ gradient has
to be back-propagated through it.

    PYTHONPATH=<ref> RFD2_REF=<ref> .venv/bin/python python/dump_refiner.py --pinned
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
    ap.add_argument("--out", default="refiner_io")
    args = ap.parse_args()

    patch_nvtx()
    if args.pinned:
        import pinned
        print(f"PINNED: {len(pinned.enable())} entry points")
    ref = common.add_ref_to_path()

    cap = {}
    n = {"lj": 0, "aa": 0}

    import rf2aa.loss.loss as loss_mod
    import rf2aa.model.Track_module as tm
    import rf2aa.util_module as um

    orig_lj = loss_mod.calc_lj_grads

    def lj_spy(seq, xyz, alpha, toaa, *a, **k):
        out = orig_lj(seq, xyz, alpha, toaa, *a, **k)
        i = n["lj"]
        if i < 2:
            cap[f"lj{i}.seq"] = seq.detach().cpu().clone()
            cap[f"lj{i}.xyz"] = xyz.detach().cpu().clone()
            cap[f"lj{i}.alpha"] = alpha.detach().cpu().clone()
            cap[f"lj{i}.dxyz"] = out[0].detach().cpu().clone()
            cap[f"lj{i}.dalpha"] = out[1].detach().cpu().clone()
        n["lj"] = i + 1
        return out

    loss_mod.calc_lj_grads = lj_spy
    tm.calc_lj_grads = lj_spy

    # `LJLoss.forward` stashes `dljEdx` for its backward; capturing it splits the
    # LJ gradient into two independently testable halves — the LJ energy's own
    # derivative, and the reverse pass through `compute_all_atom`.
    orig_ljf = loss_mod.LJLoss.forward.__func__ if hasattr(loss_mod.LJLoss.forward, "__func__") \
        else loss_mod.LJLoss.forward

    class _Ctx:
        def __init__(self):
            self.saved = None

        def save_for_backward(self, *t):
            self.saved = t

    def ljf_spy(ctx, xs, *a, **k):
        out = orig_ljf(ctx, xs, *a, **k)
        i = n.get("ljf", 0)
        if i < 1:
            cap["ljf.xs"] = xs.detach().cpu().clone()
            cap["ljf.E"] = torch.as_tensor(out).detach().cpu().clone()
            if getattr(ctx, "to_save", None):
                cap["ljf.dljEdx"] = ctx.to_save[0].detach().cpu().clone()
        n["ljf"] = i + 1
        return out

    loss_mod.LJLoss.forward = staticmethod(ljf_spy)

    orig_aa = um.XYZConverter.compute_all_atom

    def aa_spy(self, seq, xyz, alphas):
        out = orig_aa(self, seq, xyz, alphas)
        i = n["aa"]
        if i < 2:
            cap[f"aa{i}.seq"] = seq.detach().cpu().clone()
            cap[f"aa{i}.xyz"] = xyz.detach().cpu().clone()
            cap[f"aa{i}.alphas"] = alphas.detach().cpu().clone()
            cap[f"aa{i}.frames"] = out[0].detach().cpu().clone()
            cap[f"aa{i}.xyzaa"] = out[1].detach().cpu().clone()
        n["aa"] = i + 1
        return out

    um.XYZConverter.compute_all_atom = aa_spy

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

    # str_refiner IO + the RNG state at its entry
    named = dict(sampler.model.named_modules())
    handles = []
    ref_mod = named.get("model.simulator.str_refiner")
    if ref_mod is None:
        sys.exit("no str_refiner module")

    def pre(mod, inp):
        if "rng::str_refiner" not in cap:
            cap["rng::str_refiner"] = torch.get_rng_state().clone()
            for i, o in enumerate(inp):
                if torch.is_tensor(o):
                    cap[f"in::str_refiner.{i}"] = o.detach().cpu().clone()

    def post(mod, inp, outp):
        if "out::str_refiner.0" not in cap:
            for i, o in enumerate(outp):
                if torch.is_tensor(o):
                    cap[f"out::str_refiner.{i}"] = o.detach().cpu().clone()

    handles.append(ref_mod.register_forward_pre_hook(pre))
    handles.append(ref_mod.register_forward_hook(post))

    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    import rf_diffusion.features as features
    mconf = sampler._conf
    fc = features.init_tXd_inference(
        indep, getattr(mconf, "extra_tXd", []), mconf.extra_tXd_params,
        mconf.inference.conditions)
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": torch.tensor(1)}
    sampler.sample_step(int(t_step_input), indep, None, extra, fc)
    for h in handles:
        h.remove()

    print(f"calc_lj_grads calls: {n['lj']}   compute_all_atom calls: {n['aa']}")
    print(f"captured {len(cap)} tensors")
    common.write_fixture(args.out, "io", cap,
                         {"tag": "pinned" if args.pinned else "stock"})


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
