#!/usr/bin/env python3
"""Rung 4e fixtures: every stage of `sample_init`, from PDB to the model's input.

`python/probe_featurize.py` measured the pipeline for the target configuration:

    PDBLoaderDataset.getitem_inner
      process_target            (PDB parse)
      ContigMap                 (contig parse; no RNG for fixed-length contigs)
      aa_model.make_indep       (Indep + ligand)
      extract_centering_origin
      insert_contig_pre_atomization
    AddConditionalInputs        no RNG
    CenterPostTransform         no RNG (jitter = 0)
    update_inference_state      no RNG
    diffuse                     TORCH RNG ONLY

so the Rust side can be built and bisected one stage at a time. This captures
each stage's output, plus the torch RNG state on either side of `diffuse` —
the only randomness in the whole path for this configuration.

    PYTHONPATH=<ref> .venv/bin/python python/gen_sample_init.py --pinned
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

INDEP_FIELDS = ["seq", "xyz", "idx", "bond_feats", "chirals", "same_chain",
                "is_gp", "terminus_type"]


def add_indep(cap, prefix, indep):
    for f in INDEP_FIELDS:
        v = getattr(indep, f, None)
        if torch.is_tensor(v):
            cap[f"{prefix}.{f}"] = v.detach().cpu().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", default=True)
    ap.add_argument("--stock", dest="pinned", action="store_false")
    ap.add_argument("--out", default="sample_init")
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
    import rf_diffusion.aa_model as aa_model
    import rf_diffusion.inference.data_loader as idl
    import rf_diffusion.data_loader as dl

    cap = {}
    meta = {}

    # ---- stage spies -------------------------------------------------------
    orig_make_indep = aa_model.make_indep

    def make_indep_spy(*a, **k):
        out = orig_make_indep(*a, **k)
        indep = out[0] if isinstance(out, tuple) else out
        add_indep(cap, "s1_make_indep", indep)
        return out
    aa_model.make_indep = make_indep_spy

    orig_insert = aa_model.Model.insert_contig_pre_atomization

    def insert_spy(self, indep_orig, contig_map, metadata, *a, **k):
        out = orig_insert(self, indep_orig, contig_map, metadata, *a, **k)
        indep = out[0] if isinstance(out, tuple) else out
        add_indep(cap, "s2_insert_contig", indep)
        masks = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        if isinstance(masks, dict):
            for mk, mv in masks.items():
                if torch.is_tensor(mv):
                    cap[f"s2_masks.{mk}"] = mv.detach().cpu().clone()
        return out
    aa_model.Model.insert_contig_pre_atomization = insert_spy

    # `diffuse_then_add_conditional` returns both structures; the conditional
    # one is what the sampler runs on, but the unconditional one is what
    # `diffuse` alone produced, so capturing both separates a wiring error in
    # the motif copy-back from an arithmetic error in the noiser.
    orig_dtac = aa_model.diffuse_then_add_conditional

    def dtac_spy(conf, diffuser, indep, is_diffused, t):
        cap["dtac.in_xyz"] = indep.xyz.detach().cpu().clone()
        cap["dtac.rng_before"] = torch.get_rng_state().clone()
        uncond, cond = orig_dtac(conf, diffuser, indep, is_diffused, t)
        add_indep(cap, "dtac_uncond", uncond)
        add_indep(cap, "dtac_cond", cond)
        cap["dtac.rng_after"] = torch.get_rng_state().clone()
        return uncond, cond
    aa_model.diffuse_then_add_conditional = dtac_spy
    idl.aa_model.diffuse_then_add_conditional = dtac_spy

    # every transform, in order, with the torch RNG on either side
    orig_getitem = dl.TransformedDataset.__getitem__

    def getitem_spy(self, i):
        kwargs = self.dataset[i]
        for n, tf in enumerate(self.transforms):
            name = getattr(tf, "__name__", type(tf).__name__)
            cap[f"rng.before_{n}_{name}"] = torch.get_rng_state().clone()
            kwargs = tf(**kwargs)
            if isinstance(kwargs, dict) and "indep" in kwargs:
                add_indep(cap, f"s3_{n}_{name}", kwargs["indep"])
            meta[f"transform_{n}"] = name
        return kwargs
    dl.TransformedDataset.__getitem__ = getitem_spy

    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    pdb = f"{ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb"
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={pdb}",
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
    cap["rng.at_sample_init"] = torch.get_rng_state().clone()
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    cap["rng.after_sample_init"] = torch.get_rng_state().clone()

    add_indep(cap, "out_indep", indep)
    add_indep(cap, "out_indep_orig", sampler.indep_orig)
    # `Sampler.sample_init` unpacks `indep_uncond` into a LOCAL, so there is no
    # `sampler.indep_uncond` to read — an earlier version of this script fell
    # through to `indep` and captured the conditional structure twice under two
    # names. The unconditional one is taken from the transform's own output
    # instead (see the `diffuse` spy below).
    cap["out.is_diffused"] = sampler.is_diffused.detach().cpu().clone()
    cap["out.t_step_input"] = torch.tensor(int(t_step_input))
    cap["out.hal_idx0"] = torch.as_tensor(np.asarray(contig_map.hal_idx0)).long()
    cap["out.ref_idx0"] = torch.as_tensor(np.asarray(contig_map.ref_idx0)).long()
    cap["out.inpaint_seq"] = torch.as_tensor(np.asarray(contig_map.inpaint_seq))
    cap["out.inpaint_str"] = torch.as_tensor(np.asarray(contig_map.inpaint_str))

    meta.update({
        "pdb": os.path.basename(pdb),
        "ligand": "NAD,OXM",
        "contigs": "10,A106-106,10",
        "L": str(indep.length()),
        "n_diffused": str(int(sampler.is_diffused.sum())),
        "t_step_input": str(int(t_step_input)),
        "atomizer": type(atomizer).__name__ if atomizer else "None",
        "n_atomize_indices": str(len(contig_map.atomize_indices)),
        "tag": "pinned" if args.pinned else "stock",
    })
    print(f"L = {indep.length()}, {int(sampler.is_diffused.sum())} diffused, "
          f"t_step_input = {t_step_input}")
    print("transforms:", [meta[k] for k in sorted(meta) if k.startswith("transform_")])
    print(f"captured {len(cap)} tensors")
    common.write_fixture(args.out, "stages", cap, meta)
    _ = orig_getitem, idl


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
