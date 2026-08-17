#!/usr/bin/env python3
"""Rung 7 fixtures: the whole denoising loop, step by step.

`inference.model_runner` for the RFD_173 demo is **`NRBStyleSelfCond`**, not
`FlowMatching` — so the reverse step is `diffuser.reverse` (an Euler step on the
SE(3) interpolant), not `FlowMatching`'s `get_dt`/`apply_grads` pair. Getting
that wrong would produce a plausible trajectory with the wrong step size.

Per step this captures the torch generator on both sides, the coordinates in and
out, the two frames the Euler step interpolates between, and the scalars the
schedule produces — `_rots_euler_step`'s `get_scaling(t) * dt` in particular,
because it is computed from a mix of an int64 tensor and an fp32 one and the
pinning treats those differently.

    PYTHONPATH=<ref> PYTORCH_JIT=0 .venv/bin/python python/gen_sampler.py --pinned
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", default=True)
    ap.add_argument("--stock", dest="pinned", action="store_false")
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--self-cond", action="store_true",
                    help="inference.str_self_cond=True")
    ap.add_argument("--partial-t", type=int, default=None,
                    help="diffuser.partial_T")
    ap.add_argument("--out", default="sampler")
    ap.add_argument("--name", default=None)
    ap.add_argument("--contigs", default="10,A106-106,10")
    ap.add_argument("--length", default=None, help="contigmap.length, e.g. 25-25")
    args = ap.parse_args()

    if args.pinned:
        import pinned
        print(f"PINNED: {len(pinned.enable())} entry points")
    ref = common.add_ref_to_path()

    import contextlib

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield
    torch.cuda.nvtx.range = _noop
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None

    from hydra import compose, initialize_config_dir
    from rf_diffusion.inference import model_runners
    from rf_diffusion import run_inference as ri
    from rf_diffusion import features
    from se3_flow_matching.data import interpolant as itp

    cap = {}
    meta = {}
    step = {"i": -1}

    def tag():
        return f"s{step['i']}"

    # ---- the Euler step, both halves --------------------------------------
    orig_te = itp.Interpolant._trans_euler_step

    def te_spy(self, d_t, t, trans_1, trans_t):
        out = orig_te(self, d_t, t, trans_1, trans_t)
        cap[f"{tag()}.te_t"] = torch.as_tensor(t).detach().cpu().clone().float()
        cap[f"{tag()}.te_dt"] = torch.as_tensor(d_t).detach().cpu().clone().float()
        cap[f"{tag()}.te_trans1"] = trans_1.detach().cpu().clone()
        cap[f"{tag()}.te_transt"] = trans_t.detach().cpu().clone()
        cap[f"{tag()}.te_out"] = out.detach().cpu().clone()
        return out
    itp.Interpolant._trans_euler_step = te_spy

    orig_re = itp.Interpolant._rots_euler_step

    def re_spy(self, d_t, t, rotmats_1, rotmats_t):
        scaling = self.get_scaling(t)
        cap[f"{tag()}.re_scaling"] = \
            torch.as_tensor(scaling).detach().cpu().clone().float()
        cap[f"{tag()}.re_scaled_dt"] = \
            torch.as_tensor(scaling * d_t).detach().cpu().clone().float()
        cap[f"{tag()}.re_rot1"] = rotmats_1.detach().cpu().clone()
        cap[f"{tag()}.re_rott"] = rotmats_t.detach().cpu().clone()
        out = orig_re(self, d_t, t, rotmats_1, rotmats_t)
        cap[f"{tag()}.re_out"] = out.detach().cpu().clone()
        return out
    itp.Interpolant._rots_euler_step = re_spy

    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    pdb = f"{ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb"
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={pdb}",
        "inference.ligand='NAD,OXM'",
        f"contigmap.contigs=['{args.contigs}']",
        "inference.contig_as_guidepost=False",
        "inference.num_designs=1",
        "inference.deterministic=True",
        "inference.idealize_sidechain_outputs=False",
        "inference.write_trb_indep=False",
        f"diffuser.T={args.T}",
    ]
    if args.self_cond:
        overrides.append("inference.str_self_cond=True")
    if args.partial_t is not None:
        overrides.append(f"diffuser.partial_T={args.partial_t}")
    if args.length is not None:
        overrides.append(f"contigmap.length={args.length}")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        conf = compose(config_name="aa", overrides=overrides)

    import random as _pyrandom
    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(conf.inference.seed_offset)
    py_before = _pyrandom.getstate()
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    # The CPython generator: only a variable-length contig touches it, and
    # `get_sampled_mask` retries until the length fits, so the number of draws
    # is not a function of the contig alone.
    meta["py_pos_before"] = str(py_before[1][-1])
    meta["py_pos_after"] = str(_pyrandom.getstate()[1][-1])
    meta["sampled_mask"] = ";".join(contig_map.sampled_mask)
    meta["contig_deterministic"] = str(bool(contig_map.deterministic))
    meta["contigs"] = args.contigs
    meta["length"] = str(args.length)
    cap["out.hal_idx0"] = torch.as_tensor(contig_map.hal_idx0).long()
    cap["out.ref_idx0"] = torch.as_tensor(contig_map.ref_idx0).long()

    extra_tXd_names = getattr(sampler._conf, "extra_tXd", [])
    features_cache = features.init_tXd_inference(
        indep, extra_tXd_names, sampler._conf.extra_tXd_params,
        sampler._conf.inference.conditions)

    ts = torch.arange(int(t_step_input), sampler.inf_conf.final_step - 1, -1)
    meta["ts"] = ",".join(str(int(t)) for t in ts)

    rfo = None
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": None}
    px0_stack, xt_stack = [], []
    for it, t in enumerate(ts):
        step["i"] = it
        cap[f"s{it}.rng_before"] = torch.get_rng_state().clone()
        cap[f"s{it}.in_xyz"] = indep.xyz.detach().cpu().clone()
        cap[f"s{it}.t"] = torch.tensor(int(t))
        extra["n_steps"] = 1
        px0, x_t, seq_t, rfo, extra = sampler.sample_step(
            int(t), indep, rfo, extra, features_cache)
        cap[f"s{it}.rng_after"] = torch.get_rng_state().clone()
        cap[f"s{it}.px0"] = px0.detach().cpu().clone()
        cap[f"s{it}.x_t"] = x_t.detach().cpu().clone()
        cap[f"s{it}.indep_xyz_after_prepro"] = indep.xyz.detach().cpu().clone()
        indep.xyz = x_t
        px0_stack.append(px0)
        xt_stack.append(x_t)

    cap["stack.px0"] = torch.flip(torch.stack(px0_stack), [0]).detach().cpu().clone()
    cap["stack.denoised"] = torch.flip(torch.stack(xt_stack), [0]).detach().cpu().clone()
    cap["out.is_diffused"] = sampler.is_diffused.detach().cpu().clone()
    cap["out.seq"] = indep.seq.detach().cpu().clone()
    cap["out.indep_orig_xyz"] = sampler.indep_orig.xyz.detach().cpu().clone()

    meta.update({
        "T": str(int(conf.diffuser.T)),
        "t_step_input": str(int(t_step_input)),
        "final_step": str(int(conf.inference.final_step)),
        "L": str(indep.length()),
        "model_runner": str(conf.inference.model_runner),
        "n_steps": str(len(ts)),
        "tag": "pinned" if args.pinned else "stock",
        "str_self_cond": str(bool(conf.inference.str_self_cond)),
        "partial_T": str(conf.diffuser.partial_T),
    })
    print(f"captured {len(cap)} tensors over {len(ts)} steps")
    name = args.name or f"T{args.T}"
    common.write_fixture(args.out, name, cap, meta)
    _ = (contig_map, atomizer, seq_t)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
