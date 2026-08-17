#!/usr/bin/env python3
"""Capture the **sub-module** outputs of every trunk block in one reference run.

This is the last granularity the fixture set was missing. What already exists:

    fixtures/sample_init/stages   the input stages (PDB -> Indep -> noised)
    fixtures/model_pinned/step0   rfi.*, the five embeddings, the six heads
    fixtures/blocks_io/io         *whole-block* in/out + RNG for all 36 blocks
    fixtures/refiner_io/io        str_refiner's inputs and outputs
    fixtures/score/step0          forward_from_rfi, the wrapper that makes px0
    fixtures/sampler/T2           the denoising loop

What did not exist: the four modules *inside* a block. `blocks_io` can say that
block 7 disagrees; only this fixture can say whether it was `msa2msa`,
`msa2pair`, `pair2pair` or `str2str`. That is the difference between "a block is
wrong" and "module by module".

Captured per block, for all 36:

    out::<block>.msa2msa        [1, 1, L, d_msa]
    out::<block>.msa2pair       [1, L, L, d_pair]
    out::<block>.pair2pair      [1, L, L, d_pair]
    out::<block>.str2str.{0..3} xyz / state / alpha / quat
    rng::<block>.<sub>          torch generator state on ENTRY to that module

The RNG snapshot per sub-module is what separates the two failure modes that
look identical in a value diff: a module that computes the wrong number, and a
module that consumed the wrong *count* of dropout draws (RFdiffusion2 runs in
training mode at inference — 2.64 M draws per forward). If the entry states
match and the outputs do not, it is arithmetic; if the entry states diverge, an
earlier module drew a different number of times.

`pos` is captured once, not 36 times: it is a pure function of `seq_unmasked`,
`idx`, `bond_feats`, `dist_matrix` and `same_chain`, none of which a block
changes, so all 36 outputs are the same tensor and storing them costs 3.9 MB
each for nothing.

    PYTORCH_JIT=0 .venv/bin/python python/gen_layerwise.py --pinned

Notes
-----
* `--pinned` is not optional for anything compared at tolerance 0.
* `PYTORCH_JIT=0` is set here rather than left to the caller: without it the
  SE(3) transformer's 608 ScriptModules run their own compiled graph, ignore the
  Python-level pinning, and cannot be hooked at all.
* Any edit to `python/pinned.py` invalidates this fixture along with every other
  one. A stale fixture is worse than no fixture.
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402  (pins determinism before torch)
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The four modules a block runs, in the order `IterBlock.forward` runs them.
SUBS = ["msa2msa", "msa2pair", "pair2pair", "str2str"]


def patch_nvtx():
    """CPU-only torch has no NVTX; make the SE(3) profiling hooks inert."""
    import contextlib

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield

    torch.cuda.nvtx.range = _noop
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None


def flatten(name, obj, out, depth=0):
    if depth > 4:
        return
    if torch.is_tensor(obj):
        out[name] = obj.detach().cpu().clone()
    elif isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            flatten(f"{name}.{i}", o, out, depth + 1)
    elif isinstance(obj, dict):
        for k, o in obj.items():
            flatten(f"{name}.{k}", o, out, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", default=True)
    ap.add_argument("--stock", dest="pinned", action="store_false")
    ap.add_argument("--contigs", default="10,A106-106,10")
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="layerwise")
    ap.add_argument("--max-mb", type=float, default=1024.0,
                    help="abort rather than silently filling the disk")
    args = ap.parse_args()

    patch_nvtx()
    if args.pinned:
        import pinned
        ops = pinned.enable()
        print(f"PINNED: {len(ops)} entry points")
    ref = common.add_ref_to_path()

    from hydra import compose, initialize_config_dir
    from rf_diffusion.inference import model_runners
    from rf_diffusion import run_inference as ri

    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={ref}/rf_diffusion/benchmark/input/"
        f"mcsa_41/M0584_1ldm.pdb",
        "inference.ligand='NAD,OXM'",
        f"contigmap.contigs=['{args.contigs}']",
        "inference.contig_as_guidepost=False",
        "inference.num_designs=1",
        "inference.deterministic=True",
        "inference.idealize_sidechain_outputs=False",
        "inference.write_trb_indep=False",
        f"diffuser.T={args.T}",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        conf = compose(config_name="aa", overrides=overrides)

    ri.seed_all(0)                                   # get_sampler()'s seeding
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(args.seed + conf.inference.seed_offset)   # per-design seeding

    named = dict(sampler.model.named_modules())
    captured = {}
    handles = []
    calls = {}

    def hook_module(dotted, key, want_rng=True):
        """Capture `dotted`'s output under `key`; snapshot the RNG on entry."""
        mod = named.get(dotted)
        if mod is None:
            print(f"  (no module {dotted})")
            return False

        def pre(m, inp, _k=key):
            # First call only. A module invoked more than once (str_refiner runs
            # 4x) would otherwise have its snapshot overwritten by a later call
            # and no longer match the tensors it is stored beside.
            if f"rng::{_k}" not in captured:
                captured[f"rng::{_k}"] = torch.get_rng_state().clone()

        def post(m, inp, outp, _k=key):
            n = calls.get(_k, 0)
            calls[_k] = n + 1
            if n == 0:
                flatten(f"out::{_k}", outp, captured)

        try:
            if want_rng:
                handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))
            return True
        except RuntimeError as e:
            # TorchScript ScriptModules refuse hooks. Recorded, never skipped
            # silently — a missing row must be visible in the table.
            print(f"  (cannot hook {dotted}: {e})")
            return False

    n_hooked = 0
    for kind, n in (("extra_block", 4), ("main_block", 32)):
        for i in range(n):
            blk = f"model.simulator.{kind}.{i}"
            for s in SUBS:
                n_hooked += hook_module(f"{blk}.{s}", f"{blk}.{s}")

    # `pos` is identical across all 36 blocks (see the module docstring), so it
    # is captured once, from block 0, and the Rust side checks it once.
    n_hooked += hook_module("model.simulator.extra_block.0.pos",
                            "model.simulator.pos")

    # str_refiner runs 4 times; each call is a separate row.
    refiner = named.get("model.simulator.str_refiner")
    if refiner is not None:
        box = {"n": 0}

        def ref_pre(m, inp):
            captured[f"rng::model.simulator.str_refiner#{box['n']}"] = \
                torch.get_rng_state().clone()

        def ref_post(m, inp, outp):
            flatten(f"out::model.simulator.str_refiner#{box['n']}", outp,
                    captured)
            box["n"] += 1

        handles.append(refiner.register_forward_pre_hook(ref_pre))
        handles.append(refiner.register_forward_hook(ref_post))
        n_hooked += 1
    print(f"hooked {n_hooked} modules")
    if not n_hooked:
        sys.exit("no module matched; the fixture would be unusable")

    # The generator position at model entry: every Rust stage has to start its
    # stream here or no dropout draw can land in the same place.
    def rng_pre(m, inp):
        if "rng_state_at_model_entry" not in captured:
            captured["rng_state_at_model_entry"] = torch.get_rng_state().clone()

    rf_net = named.get("model")
    if rf_net is None:
        sys.exit("no submodule named 'model' to anchor the RNG capture")
    handles.append(rf_net.register_forward_pre_hook(rng_pre))

    # ---- run one design's first sample_step -------------------------------
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    print(f"L = {indep.length()}  t_step_input = {t_step_input}")

    # NOTE: sampler._conf, not the composed inference conf — it is the
    # CHECKPOINT that supplies extra_tXd, and the pre-merge conf gives an empty
    # list, after which the featurizer cache is missing keys sample_step asks
    # for. (Same trap documented in ref_dump.py.)
    import rf_diffusion.features as features
    mconf = sampler._conf
    fc = features.init_tXd_inference(
        indep, getattr(mconf, "extra_tXd", []), mconf.extra_tXd_params,
        mconf.inference.conditions)

    t = int(t_step_input)
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": torch.tensor(1)}
    sampler.sample_step(t, indep, None, extra, fc)

    for h in handles:
        h.remove()

    if "rng_state_at_model_entry" not in captured:
        sys.exit("RNG state hook never fired — the fixture would be unusable")

    nbytes = sum(v.numel() * v.element_size() for v in captured.values())
    print(f"captured {len(captured)} tensors, {nbytes/1e6:.1f} MB")
    if nbytes / 1e6 > args.max_mb:
        sys.exit(f"refusing to write {nbytes/1e6:.0f} MB > --max-mb "
                 f"{args.max_mb}")

    # How many times each module ran, so the Rust side can assert it drove the
    # same number of invocations rather than assuming.
    meta = {
        "tag": "pinned" if args.pinned else "stock",
        "T": conf.diffuser.T, "t": t, "seed": args.seed,
        "contigs": args.contigs, "L": indep.length(),
        "torch": torch.__version__,
        "n_refiner_calls": box["n"] if refiner is not None else 0,
        "subs": ",".join(SUBS),
    }
    common.write_fixture(args.out, "io", captured, meta)

    if args.pinned:
        import pinned
        rep = pinned.report()
        zero = [k for k, v in rep.items() if v == 0]
        print(f"pinned ops that never fired: {len(zero)}")
        if zero:
            print(f"  {zero}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
