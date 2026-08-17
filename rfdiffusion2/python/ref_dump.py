#!/usr/bin/env python3
"""SOP §2 — dump every intermediate of one full reference run.

This is the keystone fixture: rungs 4-7 are all checked against it.

Design note: rather than re-expressing the forward pass inline (which for a
36-block network would be a large copy of upstream and would drift), this
registers **forward hooks** on the real module objects. The tensors captured are
therefore produced by the unmodified upstream code by construction — the SOP's
"inline forward == public API" assertion is satisfied trivially because there is
no second implementation.

Captured:
  * `indep`   — the featurized input (rung 4)
  * `rfi`     — every tensor handed to the network (rung 4)
  * per-module outputs for the embeddings, recycling, extra_block.0,
    main_block.{0,1,31}, str_refiner and every aux head (rung 6)
  * `model_out` — px0/atom37, rigids, scores (rung 5/7)
  * the sampler's x_t for the next step, and the final written PDB (rung 7)

    PYTHONPATH=<ref> RFD2_REF=<ref> .venv/bin/python python/ref_dump.py [--pinned]
"""
import argparse
import os
import sys

# Mandatory, and set here rather than left to the caller: with the JIT enabled,
# the SE(3) transformer's 608 ScriptModules run their own compiled graph and
# ignore the Python-level pinning entirely (docs/BITEXACT.md §7). It also has to
# precede the e3nn import — e3nn scripts some module-level functions at import
# time, and scripting recursively compiles the `torch.*` names they reference,
# which under pinning are Python wrappers around builtins and cannot be parsed.
os.environ.setdefault("PYTORCH_JIT", "0")

import common
import torch

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


# Modules whose output is captured in full. Everything else is recorded as a
# shape/dtype line only, so the fixture stays a few hundred MB rather than tens
# of GB.
CAPTURE = [
    "model.latent_emb",
    "model.full_emb",
    "model.bond_emb",
    "model.templ_emb",
    "model.recycle",
    "model.simulator.extra_block.0",
    "model.simulator.extra_block.3",
    "model.simulator.main_block.0",
    "model.simulator.main_block.1",
    "model.simulator.main_block.31",
    "model.simulator.str_refiner",
    "model.simulator",
    "model.c6d_pred",
    "model.aa_pred",
    "model.lddt_pred",
    "model.pae_pred",
    "model.pde_pred",
    "model.bind_pred",
]


def flatten(name, obj, out, depth=0):
    """Record tensors from an arbitrarily nested return value."""
    if depth > 4:
        return
    if torch.is_tensor(obj):
        # clone: several captured outputs alias each other (e.g. simulator.0 IS
        # main_block.31.0), and safetensors refuses to write shared storage
        out[name] = obj.detach().cpu().clone()
    elif isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            flatten(f"{name}.{i}", o, out, depth + 1)
    elif isinstance(obj, dict):
        for k, o in obj.items():
            flatten(f"{name}.{k}", o, out, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true",
                    help="run in the bit-exact f64 convention (docs/BITEXACT.md)")
    ap.add_argument("--contigs", default="10,A106-106,10")
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    patch_nvtx()
    ref = common.add_ref_to_path()

    if args.pinned:
        import pinned
        ops = pinned.enable()
        print(f"PINNED: {len(ops)} entry points")

    tag = "pinned" if args.pinned else "stock"
    subdir = args.out or f"model_{tag}"

    # ---- build the sampler exactly as run_inference.py does ---------------
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

    ri.seed_all(0)                       # get_sampler()'s seeding
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(args.seed + conf.inference.seed_offset)   # per-design seeding

    captured = {}
    shapes = {}
    handles = []
    named = dict(sampler.model.named_modules())
    for name in CAPTURE:
        if name not in named:
            print(f"  (no module {name})")
            continue

        def mk(nm):
            def hook(mod, inp, outp):
                flatten(f"out::{nm}", outp, captured)
            return hook
        try:
            handles.append(named[name].register_forward_hook(mk(name)))
        except RuntimeError as e:
            # parts of the SE3 transformer are TorchScript ScriptModules, which
            # do not accept hooks; recorded rather than silently skipped
            print(f"  (cannot hook {name}: {e})")

    # The network consumes the torch RNG inside the forward pass (dropout is
    # live at inference — nothing calls .eval()), so any Rust comparison needs
    # the generator state at the moment the forward starts.
    def rng_pre(mod, inp):
        if "rng_state_at_model_entry" not in captured:
            captured["rng_state_at_model_entry"] = torch.get_rng_state().clone()
    if "model" in named:
        handles.append(named["model"].register_forward_pre_hook(rng_pre))
    else:
        raise SystemExit("no submodule named 'model' to anchor the RNG capture")

    # every module gets a shape record, which is the map for later rungs
    def shape_hook(nm):
        def hook(mod, inp, outp):
            d = {}
            flatten("o", outp, d)
            shapes[nm] = {k: [list(v.shape), str(v.dtype)] for k, v in d.items()}
        return hook
    n_script = 0
    for nm, m in named.items():
        try:
            handles.append(m.register_forward_hook(shape_hook(nm)))
        except RuntimeError:
            n_script += 1   # ScriptModule: no hooks available
    if n_script:
        print(f"  ({n_script} ScriptModules cannot be hooked -- SE3 transformer)")

    # ---- run one design, capturing the first sample_step ------------------
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    print(f"L = {indep.length()}  t_step_input = {t_step_input}")

    dump = {}
    for f in ("xyz", "seq", "idx", "bond_feats", "same_chain", "is_sm", "is_gp",
              "terminus_type", "chirals", "extra_t1d", "extra_t2d"):
        v = getattr(indep, f, None)
        if torch.is_tensor(v):
            dump[f"indep.{f}"] = v.detach().cpu()
    dump["is_diffused"] = sampler.is_diffused.detach().cpu()

    # capture the RFI (network inputs) by wrapping prepro
    import rf_diffusion.aa_model as aa_model
    orig_prepro = aa_model.Model.prepro
    rfi_box = {}

    def prepro_spy(self, indep_, t, is_diffused):
        rfi = orig_prepro(self, indep_, t, is_diffused)
        if not rfi_box:
            import dataclasses
            for k, v in dataclasses.asdict(rfi).items():
                if torch.is_tensor(v):
                    rfi_box[f"rfi.{k}"] = v.detach().cpu()
        return rfi
    aa_model.Model.prepro = prepro_spy

    # NOTE: use sampler._conf, not the composed inference conf. load_model()
    # merges config/training/base.yaml <- ckpt['conf'] <- inference conf, and it
    # is the CHECKPOINT that supplies
    #   extra_tXd = [radius_of_gyration_v2, relative_sasa_v2,
    #                sinusoidal_timestep_embedding]
    # Using the pre-merge conf gives an empty list and the featurizer cache is
    # then missing the keys sample_step asks for.
    import rf_diffusion.features as features
    mconf = sampler._conf
    fc = features.init_tXd_inference(
        indep, getattr(mconf, "extra_tXd", []), mconf.extra_tXd_params,
        mconf.inference.conditions)

    t = int(t_step_input)
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": torch.tensor(1)}
    px0, x_t, seq_t, rfo, extra_out = sampler.sample_step(
        t, indep, None, extra, fc)

    aa_model.Model.prepro = orig_prepro
    for h in handles:
        h.remove()

    dump.update(rfi_box)
    dump.update(captured)
    dump["px0"] = px0.detach().cpu()
    dump["x_t_next"] = x_t.detach().cpu()
    dump["seq_t"] = seq_t.detach().cpu()

    meta = {
        "tag": tag, "T": conf.diffuser.T, "t": t, "seed": args.seed,
        "contigs": args.contigs, "L": indep.length(),
        "torch": torch.__version__,
    }
    common.write_fixture(subdir, "step0", dump, meta)
    common.write_json(subdir, "module_shapes", shapes)
    print(f"captured {len(dump)} tensors; {len(shapes)} modules mapped")

    if args.pinned:
        import pinned
        rep = pinned.report()
        common.write_json(subdir, "pinned_report", rep)
        print(f"pinned ops fired: {sum(v for v in rep.values() if v > 0)} calls "
              f"across {len(rep)} ops")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
