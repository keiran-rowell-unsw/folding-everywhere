#!/usr/bin/env python3
"""Rung 4e reconnaissance: what does `sample_init` ACTUALLY run?

`InferenceDataset` builds its pipeline from `conf.transforms.names`, so the
scope of the remaining featurization is a configuration question, not a
source-reading question. This runs the real thing and reports:

  * the transform list, in order, as applied;
  * how much of each RNG stream each transform consumes;
  * the shape/dtype/checksum of every field of `Indep` before and after;
  * which fields the diffusion step actually changes.

Nothing here is ported yet — this decides what has to be.

    PYTHONPATH=<ref> .venv/bin/python python/probe_featurize.py --pinned
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def rng_marks():
    """A cheap fingerprint of all three streams, to see who consumed what."""
    import random
    st = torch.get_rng_state()
    return {
        "torch": int(st.sum().item()),
        "numpy": int(np.random.get_state()[2]),
        "python": random.getstate()[1][-1],
    }


def describe(x):
    if torch.is_tensor(x):
        f = x.detach().cpu()
        chk = float(f.double().abs().sum()) if f.is_floating_point() else int(f.long().sum())
        return f"tensor{list(f.shape)} {str(f.dtype).replace('torch.','')} sum={chk}"
    if isinstance(x, np.ndarray):
        return f"ndarray{list(x.shape)} {x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    return f"{type(x).__name__} {x!r}"[:80]


def snapshot(indep):
    out = {}
    for k in sorted(vars(indep)):
        out[k] = describe(getattr(indep, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", default=True)
    ap.add_argument("--stock", dest="pinned", action="store_false")
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
    import rf_diffusion.inference.data_loader as idl

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

    print("\n=== transforms configured ===")
    up = getattr(conf, "upstream_inference_transforms", None)
    print("upstream:", list(up.names) if up else [])
    from omegaconf import OmegaConf
    print("transforms node:", OmegaConf.to_yaml(conf.transforms).strip()[:600])
    names = list(conf.transforms.get("names", []))
    print("main    :", names)
    import rf_diffusion.conditioning as conditioning
    print("ignored (legacy):", [n for n in names
                                if n in conditioning.LEGACY_TRANSFORMS_TO_IGNORE])
    print("diffuser:", conf.diffuser.type if hasattr(conf.diffuser, "type") else dict(conf.diffuser))

    # ---- instrument TransformedDataset so each transform is visible --------
    import rf_diffusion.data_loader as dl
    orig_getitem = dl.TransformedDataset.__getitem__

    def spy_getitem(self, i):
        # replicate, but log each step
        item = self.dataset[i]
        kwargs = item if isinstance(item, dict) else dict(item)
        prev = rng_marks()
        print("\n=== transform pipeline ===")
        for tf in self.transforms:
            name = getattr(tf, "__name__", type(tf).__name__)
            before = rng_marks()
            kwargs = tf(**kwargs)
            after = rng_marks()
            d = {k: after[k] != before[k] for k in after}
            used = ",".join(k for k, v in d.items() if v) or "-"
            keys = ",".join(sorted(kwargs.keys())) if isinstance(kwargs, dict) else "(tuple)"
            print(f"  {name:<42} rng[{used:<18}] -> keys: {keys[:90]}")
            prev = after
        _ = prev
        return kwargs

    dl.TransformedDataset.__getitem__ = spy_getitem

    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(conf.inference.seed_offset)
    print("\nRNG at sample_init entry:", rng_marks())
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    print("RNG after sample_init  :", rng_marks())

    print(f"\n=== Indep after sample_init (L = {indep.length()}) ===")
    for k, v in snapshot(indep).items():
        print(f"  {k:<20} {v}")

    print("\n=== ContigMap ===")
    for k in sorted(vars(contig_map)):
        v = getattr(contig_map, k)
        print(f"  {k:<20} {describe(v)}")

    print(f"\natomizer: {type(atomizer).__name__ if atomizer else None}")
    print(f"t_step_input: {t_step_input}")
    print(f"is_diffused: {int(sampler.is_diffused.sum())} of {len(sampler.is_diffused)} diffused")

    print("\n=== indep_orig vs indep_cond (what diffusion changed) ===")
    a, b = snapshot(sampler.indep_orig), snapshot(sampler.indep_cond)
    for k in sorted(a):
        tag = "SAME" if a[k] == b[k] else "CHANGED"
        if tag == "CHANGED":
            print(f"  {k:<20} {tag}\n      orig {a[k]}\n      cond {b[k]}")
    print("  unchanged:", ", ".join(k for k in sorted(a) if a[k] == b[k]))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
