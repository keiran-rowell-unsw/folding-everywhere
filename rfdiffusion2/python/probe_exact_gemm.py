#!/usr/bin/env python3
"""Stage 0, the reference half: is the *pinned reference* correctly rounded?

`docs/BITEXACT.md` pins the reference by promoting fp32 to f64 and rounding
once, on the argument that an f64 rounding error (~1e-16 relative) is ~9 orders
below an fp32 ULP. That is a probability, not a guarantee, and the port and the
reference disagree at exactly the rate it predicts.

The Rust side of Stage 0 settled its half by measurement: a full forward under a
double-double accumulator (~106 significand bits) is **byte-identical** to the
default lane-split f64 path, so the port's answer is the correctly-rounded one.

This script settles the other half. Products of two fp32 values are *exact* in
f64 (24 + 24 = 48 <= 53), so a dot product over fp32 inputs is a sum of exact
f64 terms and `math.fsum` — which is exactly rounded by construction — gives the
true value. Comparing the pinned reference's `torch` result against that says
whether MKL's f64 GEMM is the inexact side, and at what rate.

Real weights and real activations, at the model's real shapes; a random sample
of outputs, because fsum in Python over the full 968k x 192 is not the point.

    PYTHONPATH=<ref> PYTORCH_JIT=0 .venv/bin/python python/probe_exact_gemm.py
"""
import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def exact_dot_f32(x_row, w_row):
    """The exactly-rounded fp32 dot product of two fp32 vectors.

    Each product is exact in f64; `math.fsum` sums exact f64 terms with a single
    correct rounding, so the f32 narrowing of the result is the correctly-
    rounded answer by construction, not by probability.
    """
    return math.fsum(float(a) * float(b) for a, b in zip(x_row, w_row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000,
                    help="how many output values to check exactly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture", action="store_true",
                    help="hook the real to_k during a reference forward instead of "
                         "reconstructing its input (the reconstruction is WRONG: "
                         "to_k's input is layer-normed inside the attention)")
    args = ap.parse_args()

    import pinned
    ops = pinned.enable()
    print(f"PINNED: {len(ops)} entry points")

    port = common.PORT_ROOT
    from safetensors.torch import load_file

    # A real activation and a real weight, both at the shapes where the
    # disagreement was originally bisected (`row_attn.to_k`, K = 192).
    blocks = os.path.join(port, "fixtures", "blocks_io", "io.safetensors")
    weights = os.path.join(port, "fixtures", "weights", "model_state_dict.safetensors")
    for p in (blocks, weights):
        if not os.path.exists(p):
            sys.exit(f"missing fixture: {p}")

    wts = load_file(weights)

    if args.capture:
        x, w, ref = capture_real_to_k()
        print(f"captured the real to_k: input {tuple(x.shape)} weight {tuple(w.shape)}")
        return report(x.numpy(), w.numpy(), ref.numpy())

    bio = load_file(blocks)

    # main_block.0's pair input is [1, L, L, 192]; flatten to rows of K = 192.
    pair = None
    for k in sorted(bio):
        if k.startswith("in::model.simulator.main_block.0.") and bio[k].ndim == 4 \
                and bio[k].shape[-1] == 192:
            pair = bio[k]
            break
    if pair is None:
        sys.exit("could not find main_block.0's pair input in the fixture")
    x = pair.reshape(-1, pair.shape[-1]).contiguous().float()

    # `docs/BITEXACT.md` bisected the disagreement to `row_attn.to_k[4427, 157]`
    # with K = 192 -- the *pair* track's axial attention (4427 < 71^2 = 5041),
    # not the MSA track's, whose to_k is 256 x 256. Match on the K the
    # activation actually has so the wrong layer cannot be picked silently.
    wname = next(k for k in sorted(wts)
                 if "main_block.0" in k and k.endswith("pair2pair.row_attn.to_k.weight"))
    assert wts[wname].shape[1] == x.shape[1], \
        f"K mismatch: activation {tuple(x.shape)} vs weight {tuple(wts[wname].shape)}"
    w = wts[wname].float()
    print(f"activation {tuple(x.shape)}   weight {wname} {tuple(w.shape)}")

    # The pinned reference's own answer: promote to f64, MKL GEMM, round once.
    #
    # NOTE: this reconstruction feeds `to_k` the block's *raw* pair input, but
    # the real one is layer-normed inside the attention -- the exact value below
    # is -52.06 where docs/BITEXACT.md records -1.95089882612228371350. So this
    # path measures MKL's f64 GEMM at the right *shape* on real data; it does
    # NOT reproduce the documented computation. Use --capture for that.
    with torch.no_grad():
        ref = torch.nn.functional.linear(x, w).float()
    return report(x.numpy(), w.numpy(), ref.numpy())


def capture_real_to_k():
    """Run one reference forward and grab `to_k`'s actual input and output."""
    common.add_ref_to_path()
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

    ref = common.REF_ROOT
    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    pdb = f"{ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb"
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={pdb}",
        "inference.ligand='NAD,OXM'",
        "contigmap.contigs=['10,A106-106,10']",
        "inference.contig_as_guidepost=False", "inference.num_designs=1",
        "inference.deterministic=True", "inference.idealize_sidechain_outputs=False",
        "inference.write_trb_indep=False", "diffuser.T=2",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        conf = compose(config_name="aa", overrides=overrides)
    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(conf.inference.seed_offset)
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)

    grab = {}
    target = sampler.model.model.simulator.main_block[0].pair2pair.row_attn.to_k

    def hook(mod, inp, out):
        if "x" not in grab:
            grab["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().clone()
            grab["out"] = out.detach().reshape(-1, out.shape[-1]).float().clone()
    h = target.register_forward_hook(hook)

    extra_tXd_names = getattr(sampler._conf, "extra_tXd", [])
    fc = features.init_tXd_inference(indep, extra_tXd_names,
                                     sampler._conf.extra_tXd_params,
                                     sampler._conf.inference.conditions)
    sampler.sample_step(int(t_step_input), indep, None,
                        {"rfo_uncond": None, "rfo_cond": None, "n_steps": 1}, fc)
    h.remove()
    _ = (contig_map, atomizer)
    return grab["x"], target.weight.detach().float(), grab["out"]


def report(xnp, wnp, refnp):
    rows, outs = refnp.shape
    # ---- 1. the element docs/BITEXACT.md bisected ------------------------
    R, O = 4427, 157
    if R >= rows or O >= outs:
        R, O = 0, 0
    exact = exact_dot_f32(xnp[R], wnp[O])
    e32 = np.float32(exact)
    g32 = refnp[R, O]
    print()
    print(f"the bisected element  to_k[{R}, {O}]")
    print(f"  exact (fsum)      {exact!r}")
    print(f"  -> correctly rounded f32  {e32!r}")
    print(f"  pinned reference gave     {g32!r}"
          f"   {'MATCH' if e32.tobytes() == g32.tobytes() else 'DIFFERS'}")

    # ---- 2. every output, via a higher-precision scan ---------------------
    # A full fsum over 968k x 192 is not worth the minutes; np.longdouble
    # carries 64 significand bits (11 more than f64), so anything that narrows
    # differently from the reference is a candidate, and the candidates are then
    # settled exactly with fsum. A tie that longdouble also misses would need the
    # exact value within 2^-64 of an fp32 midpoint -- 2^-40 of values, i.e. none
    # here.
    print()
    print("scanning all outputs against an np.longdouble matmul ...")
    xl = xnp.astype(np.longdouble)
    wl = wnp.astype(np.longdouble)
    hp = xl @ wl.T
    hp32 = hp.astype(np.float32)
    cand = np.argwhere(hp32.view(np.int32) != refnp.view(np.int32))
    print(f"  candidates where longdouble and the reference narrow differently:"
          f" {len(cand)} / {rows * outs}   ({len(cand) / (rows * outs):.3e})")

    wrong = 0
    worst_ulp = 0
    for r, o in cand[:200]:
        r, o = int(r), int(o)
        ex = np.float32(exact_dot_f32(xnp[r], wnp[o]))
        g = refnp[r, o]
        if ex.tobytes() != g.tobytes():
            wrong += 1
            ulp = abs(int(ex.view(np.int32)) - int(np.float32(g).view(np.int32)))
            worst_ulp = max(worst_ulp, ulp)
            if wrong <= 5:
                print(f"    [{r}, {o}]  exact -> {ex!r}   reference {g!r}   ({ulp} ULP)")

    print()
    print(f"confirmed by exact summation: {wrong} of {min(len(cand), 200)} candidates "
          f"are the reference being wrong, worst {worst_ulp} ULP")
    print()
    if wrong:
        print("=> the PINNED REFERENCE is not correctly rounded.")
        print("   The Rust side is: a full forward under a double-double")
        print("   accumulator was byte-identical to the default path over")
        print("   998 260 outputs. So where the two disagree, the port is right.")
    else:
        print("=> no confirmed disagreement; the reference matched exact summation")
        print("   everywhere checked.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
