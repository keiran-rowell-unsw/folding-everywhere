#!/usr/bin/env python3
"""Capture the **inputs and outputs** of any set of modules in one reference run.

`ref_dump.py` records a fixed list of module *outputs*; that is enough to see
*that* a Rust block disagrees, never *where*. This script takes a regex and
captures both sides of every matching module, so a failing block can be bisected
in one reference run instead of one run per guess.

Tensors are produced by unmodified upstream code (forward hooks on the real
module objects), so there is no second implementation to drift.

    PYTHONPATH=<ref> RFD2_REF=<ref> .venv/bin/python python/dump_io.py \
        --pinned --match 'model\\.templ_emb($|\\.)' --out templ_io

Notes
-----
* `--pinned` must be given for anything compared at tolerance 0.
* `PYTORCH_JIT=0` is set here, not left to the caller: without it the SE(3)
  transformer's 608 ScriptModules run their own compiled graph, ignore the
  Python-level pinning, and cannot be hooked at all.
* Captures are keyed `in::<module>.<i>` / `out::<module>[.<path>]`, matching
  `ref_dump.py`'s naming so the Rust side can read either fixture.
"""
import argparse
import os
import re
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402  (pins determinism before torch)
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
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--match", action="append", required=True,
                    help="regex on the module's dotted name; repeatable")
    ap.add_argument("--out", required=True, help="fixtures/<out>/")
    ap.add_argument("--contigs", default="10,A106-106,10")
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-mb", type=float, default=4096.0,
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
        f"inference.input_pdb={ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb",
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

    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(args.seed + conf.inference.seed_offset)

    pats = [re.compile(p) for p in args.match]
    captured = {}
    handles = []
    named = dict(sampler.model.named_modules())
    hooked = []
    for name, mod in named.items():
        if not any(p.search(name) for p in pats):
            continue

        def mk_pre(nm):
            def hook(mod, inp):
                # First call only: later calls of the same module (e.g. a block
                # invoked once per recycle) would overwrite the earlier capture,
                # and the RNG snapshot has to match the tensors it accompanies.
                key = f"rng::{nm}"
                if key not in captured:
                    captured[key] = torch.get_rng_state().clone()
                for i, o in enumerate(inp):
                    flatten(f"in::{nm}.{i}", o, captured)
            return hook

        def mk_post(nm):
            def hook(mod, inp, outp):
                flatten(f"out::{nm}", outp, captured)
            return hook

        try:
            handles.append(mod.register_forward_pre_hook(mk_pre(name)))
            handles.append(mod.register_forward_hook(mk_post(name)))
            hooked.append(name)
        except RuntimeError as e:
            print(f"  (cannot hook {name}: {e})")
    print(f"hooked {len(hooked)} modules")
    if not hooked:
        sys.exit("no module matched; nothing would be written")

    # The network consumes the torch RNG *inside* the forward pass (dropout is
    # live at inference — nothing calls .eval()), so a Rust forward can only be
    # compared to this one if it starts from the same stream position. Snapshot
    # the generator state on entry to the top-level module.
    def rng_pre(mod, inp):
        if "rng_state_at_model_entry" not in captured:
            captured["rng_state_at_model_entry"] = torch.get_rng_state().clone()
    # `sampler.model` is the RFScore wrapper and is *not* invoked through
    # `__call__`, so a hook there never fires. The network itself is the
    # submodule named "model"; hooking it is also the right stream position,
    # since nothing between the two consumes the generator.
    rf_net = named.get("model")
    if rf_net is None:
        sys.exit("cannot find the LegacyRoseTTAFoldModule submodule 'model'")
    handles.append(rf_net.register_forward_pre_hook(rng_pre))

    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
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

    nbytes = sum(v.numel() * v.element_size() for v in captured.values())
    print(f"captured {len(captured)} tensors, {nbytes/1e6:.1f} MB")
    if nbytes / 1e6 > args.max_mb:
        sys.exit(f"refusing to write {nbytes/1e6:.0f} MB > --max-mb {args.max_mb}")

    meta = {"tag": "pinned" if args.pinned else "stock", "T": conf.diffuser.T,
            "t": t, "seed": args.seed, "contigs": args.contigs,
            "L": indep.length(), "torch": torch.__version__}
    if "rng_state_at_model_entry" not in captured:
        sys.exit("RNG state hook never fired — the fixture would be unusable")
    common.write_fixture(args.out, "io", captured, meta)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
