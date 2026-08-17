#!/usr/bin/env python3
"""Run the **unmodified** upstream `run_inference.py`, with two shims.

The SOP forbids copying or editing upstream model code (§1.1), so everything
needed to make it run here is applied from the outside, in this wrapper:

1. **NVTX no-op.** `se3_transformer/model/layers/attention.py` does
   `from torch.cuda.nvtx import range as nvtx_range` at import time, and a
   CPU-only torch build raises `RuntimeError: NVTX functions not installed`.
   The binding happens at import, so the patch has to land before the import
   chain runs — which is why this is a wrapper and not a flag.

2. **Pinned mode** (`RFD2_PINNED=1`), the bit-exact convention from
   `docs/BITEXACT.md`: every fp32 op computed in f64 and rounded once.

Usage (identical CLI to upstream, plus the env var):

    PYTHONPATH=<ref> python run_reference.py --config-name=aa <overrides...>
    RFD2_PINNED=1 PYTHONPATH=<ref> python run_reference.py ...
"""
import contextlib
import os
import sys
import time

# ---- must precede any upstream import ------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402


@contextlib.contextmanager
def _noop_range(*args, **kwargs):
    yield


def patch_nvtx():
    """CPU-only torch has no NVTX; make the profiling hooks inert."""
    torch.cuda.nvtx.range = _noop_range
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None



def install_rfi_dump(ref):
    """Save `rfi.atom_frames` from the first prepro call to $RFD2_DUMP_RFI.

    Needed per protein, not once: `get_atom_frames` breaks priority ties by
    CPython set-iteration order, so the frames are a property of the pipeline
    run rather than something that can be recomputed (measured on M0584_1ldm:
    20 of 50 atoms tie, and recomputation disagreed on 1). The benchmark
    therefore runs the reference FIRST and builds each ligand sidecar from that
    run's own frames.
    """
    path = os.environ.get("RFD2_DUMP_RFI")
    if not path:
        return
    if ref not in sys.path:
        sys.path.insert(0, ref)
    import dataclasses
    import rf_diffusion.aa_model as aa_model
    from safetensors.torch import save_file
    orig = aa_model.Model.prepro
    done = {}

    def spy(self, indep_, t, is_diffused):
        rfi = orig(self, indep_, t, is_diffused)
        if not done:
            done["y"] = True
            d = {}
            for k, v in dataclasses.asdict(rfi).items():
                if torch.is_tensor(v):
                    v = v.detach().cpu().contiguous()
                    if v.dtype in (torch.int8, torch.int16, torch.int32,
                                   torch.bool, torch.uint8):
                        v = v.to(torch.int64)
                    d[f"rfi.{k}"] = v
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            save_file(d, path)
            print(f"[run_reference] wrote rfi dump -> {path} ({len(d)} tensors)",
                  file=sys.stderr)
        return rfi

    aa_model.Model.prepro = spy


def main():
    patch_nvtx()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    pinned_on = os.environ.get("RFD2_PINNED", "") not in ("", "0", "false")
    if pinned_on:
        import pinned
        ops = pinned.enable()
        print(f"[run_reference] PINNED mode: {len(ops)} entry points patched",
              file=sys.stderr)

    # Run upstream as a SCRIPT, not as an imported module: hydra resolves
    # `config_path='config/inference'` relative to the file when __name__ is
    # '__main__', but relative to the package (needing an __init__.py that does
    # not exist) when imported. runpy gives us the script semantics while still
    # letting the patches above land first.
    import runpy
    ref = os.environ.get("RFD2_REF") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(here))),
        "ref_RFdiffusion2")
    install_rfi_dump(ref)
    script = os.path.join(ref, "rf_diffusion", "run_inference.py")
    if not os.path.isfile(script):
        sys.exit(f"cannot find upstream run_inference.py at {script}; "
                 f"set RFD2_REF to the repo root")

    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        dt = time.time() - t0
        print(f"[run_reference] wall clock {dt:.1f} s", file=sys.stderr)
        if pinned_on:
            import pinned
            rep = pinned.report()
            print("[run_reference] pinned op fire counts (top 25):",
                  file=sys.stderr)
            for k, v in list(rep.items())[:25]:
                print(f"    {v:9d}  {k}", file=sys.stderr)
            zero = [k for k, v in rep.items() if v == 0]
            if zero:
                print(f"[run_reference] WARNING: {len(zero)} patched ops never "
                      f"fired: {zero}", file=sys.stderr)


if __name__ == "__main__":
    main()
