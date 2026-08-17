#!/usr/bin/env python3
"""Rung 7 fixtures: `RFScoreModule.forward_from_rfi`, the wrapper around the net.

The network is complete, but the
network is not what the sampler calls. `frame_diffusion/rf_score/model.py:259`
wraps it with a layer that no document mentioned and that contains an RNG draw:

    rigids_t    = rigid_frames_from_atom_14(rfi.xyz)
    rfo         = model(**rfi)                       <- the 82.9 M-param network
    curr_rigids = rigids_from_rfo(rfo, rigids_t.rots)  quaternion compose, I=40
    psi_pred    = torch.rand((B, I, L, 2))           <- one draw per forward
    atom37      = compute_backbone(curr_rigids, psi_pred)[0]
    px0         = atom37[0, -1]

`px0` — the thing the sampler writes to the PDB — is produced *here*, not by the
trunk. This captures each stage plus the generator state on either side, so the
Rust side can be run from the reference's own input and bisected.

Of particular interest is `rots_t.get_quats()`: openfold's `rot_to_quat` builds
a symmetric 4x4 and takes the last eigenvector of `torch.linalg.eigh`. Under
pinning that is LAPACK in f64 with one narrowing, so the capture is what decides
whether a canonical Jacobi on the Rust side lands on the same fp32 values.

    PYTHONPATH=<ref> PYTORCH_JIT=0 .venv/bin/python python/gen_score.py --pinned
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
    ap.add_argument("--out", default="score")
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
    from rf_diffusion.frame_diffusion.rf_score import model as rfscore
    from rf_diffusion.frame_diffusion.data import all_atom as a37mod
    from openfold.utils import rigid_utils as ru

    cap = {}
    meta = {}

    # ---- the wrapper, stage by stage --------------------------------------
    orig_ffr = rfscore.RFScore.forward_from_rfi

    def ffr_spy(self, rfi, t, use_checkpoint=True, return_raw=False):
        n = meta.get("n_ffr", 0)
        meta["n_ffr"] = n + 1
        tag = f"ffr{n}"
        cap[f"{tag}.in_xyz"] = rfi.xyz.detach().cpu().clone()
        cap[f"{tag}.t"] = torch.as_tensor(t).detach().cpu().clone().float()
        cap[f"{tag}.rng_before"] = torch.get_rng_state().clone()
        out = orig_ffr(self, rfi, t, use_checkpoint=use_checkpoint,
                       return_raw=return_raw)
        cap[f"{tag}.rng_after"] = torch.get_rng_state().clone()
        cr = out["rigids_raw"]
        cap[f"{tag}.curr_trans"] = cr.get_trans().detach().cpu().clone()
        cap[f"{tag}.curr_rots"] = \
            cr.get_rots().get_rot_mats().detach().cpu().clone()
        cap[f"{tag}.curr_quats"] = \
            cr.get_rots().get_quats().detach().cpu().clone()
        cap[f"{tag}.psi"] = out["psi"].detach().cpu().clone()
        cap[f"{tag}.atom37"] = out["atom37"].detach().cpu().clone()
        cap[f"{tag}.atom14"] = out["atom14"].detach().cpu().clone()
        cap[f"{tag}.rfo_quat"] = out["rfo"].quat.detach().cpu().clone()
        cap[f"{tag}.rfo_xyz"] = out["rfo"].xyz.detach().cpu().clone()
        return out
    rfscore.RFScore.forward_from_rfi = ffr_spy

    # `rigids_from_rfo` is where the quaternion composition happens; capture the
    # frame it starts from, which is the one `rot_to_quat` had to convert.
    orig_rfr = rfscore.rigids_from_rfo

    def rfr_spy(rfo, rots_t, stopgrad_rotations):
        n = meta.get("n_rfr", 0)
        meta["n_rfr"] = n + 1
        tag = f"rfr{n}"
        cap[f"{tag}.rots_t_mats"] = rots_t.get_rot_mats().detach().cpu().clone()
        cap[f"{tag}.rots_t_quats"] = rots_t.get_quats().detach().cpu().clone()
        out = orig_rfr(rfo, rots_t, stopgrad_rotations)
        cap[f"{tag}.out_quats"] = out.get_rots().get_quats().detach().cpu().clone()
        cap[f"{tag}.out_trans"] = out.get_trans().detach().cpu().clone()
        meta["stopgrad_rotations"] = str(stopgrad_rotations)
        return out
    rfscore.rigids_from_rfo = rfr_spy

    # `compute_backbone` here receives a Rigid whose rotation came from
    # quaternions, unlike the `diffuse` call where it came from matrices.
    orig_cb = a37mod.compute_backbone

    def cb_spy(bb_rigids, psi):
        n = meta.get("n_cb", 0)
        meta["n_cb"] = n + 1
        if n < 2:
            cap[f"cb{n}.in_rots"] = \
                bb_rigids.get_rots().get_rot_mats().detach().cpu().clone()
            cap[f"cb{n}.in_trans"] = bb_rigids.get_trans().detach().cpu().clone()
            cap[f"cb{n}.psi"] = psi.detach().cpu().clone()
        out = orig_cb(bb_rigids, psi)
        if n < 2:
            cap[f"cb{n}.atom37"] = out[0].detach().cpu().clone()
        return out
    a37mod.compute_backbone = cb_spy
    rfscore.all_atom.compute_backbone = cb_spy

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
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)

    # one step is enough to fixture the wrapper; the loop itself is rung 7's own
    from rf_diffusion import features
    extra_tXd_names = getattr(sampler._conf, "extra_tXd", [])
    meta["extra_tXd"] = ",".join(extra_tXd_names)
    features_cache = features.init_tXd_inference(
        indep, extra_tXd_names, sampler._conf.extra_tXd_params,
        sampler._conf.inference.conditions)

    rfo = None
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": None}
    cap["step.rng_before"] = torch.get_rng_state().clone()
    cap["step.in_xyz"] = indep.xyz.detach().cpu().clone()
    px0, x_t, seq_t, rfo, extra = sampler.sample_step(
        int(t_step_input), indep, rfo, extra, features_cache)
    cap["step.extra_t1d"] = indep.extra_t1d.detach().cpu().clone()
    cap["step.extra_t2d"] = indep.extra_t2d.detach().cpu().clone()
    cap["step.rng_after"] = torch.get_rng_state().clone()
    cap["step.px0"] = px0.detach().cpu().clone()
    cap["step.x_t"] = x_t.detach().cpu().clone()
    cap["step.indep_xyz_after"] = indep.xyz.detach().cpu().clone()
    cap["step.is_diffused"] = sampler.is_diffused.detach().cpu().clone()

    meta.update({
        "t_step_input": str(int(t_step_input)),
        "T": str(int(conf.diffuser.T)),
        "model_runner": str(conf.inference.model_runner),
        "L": str(indep.length()),
        "tag": "pinned" if args.pinned else "stock",
        "trans_schedule": str(conf.diffuser.trans.sample_schedule),
        "rots_schedule": str(conf.diffuser.rots.sample_schedule),
        "rots_exp_rate": str(conf.diffuser.rots.exp_rate),
        "trans_exp_rate": str(conf.diffuser.trans.exp_rate),
        "final_step": str(int(conf.inference.final_step)),
    })
    print(f"captured {len(cap)} tensors; "
          f"n_ffr={meta.get('n_ffr')} n_cb={meta.get('n_cb')}")
    common.write_fixture(args.out, "step0", cap, meta)
    _ = (contig_map, atomizer, seq_t)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
