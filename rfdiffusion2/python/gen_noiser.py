#!/usr/bin/env python3
"""Rung 4e stage 3 fixtures: the inside of `diffuse`, plus the IGSO3 tables.

`diffuse` is the only step in `sample_init` that draws randomness, and it draws
it in a specific order:

    add_fake_frame_legs        torch.normal  x2   (the ligand rows' fake N/C)
    rigid_frames_from_atom_14  -
    forward_marginal
      _corrupt_trans           sample_gaussian    (translations)
      _corrupt_rotmats
        igso3.sample
          sample_vector        torch.randn        (axis)
          sample_angle         torch.rand         (inverse-transform angle)
    atom37_from_rigid          -

so this logs every draw with its shape and the RNG state before it, which is
what a port has to match before any value can agree.

It also exports the IGSO3 lookup tables. Those are numerically integrated from a
series expansion and then cached to `.cache/`, so the reference itself reads them
from disk — exporting what it actually uses is therefore the correct move, not a
shortcut.

    PYTHONPATH=<ref> .venv/bin/python python/gen_noiser.py --pinned
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
    ap.add_argument("--out", default="noiser")
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

    cap = {}
    meta = {}
    draws = []

    # ---- log every RNG draw, in order -------------------------------------
    active = {"on": False}
    for fn_name in ("randn", "rand", "normal"):
        orig = getattr(torch, fn_name)

        def make(fn_name=fn_name, orig=orig):
            def wrapper(*a, **k):
                if not active["on"]:
                    return orig(*a, **k)
                st = torch.get_rng_state().clone()
                out = orig(*a, **k)
                i = len(draws)
                cap[f"draw{i}.rng_before"] = st
                cap[f"draw{i}.out"] = out.detach().cpu().clone()
                import traceback
                fr = [f for f in traceback.extract_stack()[:-1]
                      if "site-packages" not in f.filename][-3:]
                where = " <- ".join(f"{os.path.basename(f.filename)}:{f.name}" for f in reversed(fr))
                draws.append((fn_name, tuple(out.shape), where))
                return wrapper_note(fn_name, out)
            return wrapper

        def wrapper_note(_n, out):
            return out
        setattr(torch, fn_name, make())

    from hydra import compose, initialize_config_dir
    from rf_diffusion.inference import model_runners
    from rf_diffusion import run_inference as ri
    import rf_diffusion.aa_model as aa_model

    # ---- capture the inside of `diffuse` ----------------------------------
    orig_diffuse = aa_model.diffuse

    def diffuse_spy(conf, diffuser, indep, is_diffused, t):
        n = meta.get("n_diffuse", 0)
        tag = f"d{n}"
        meta["n_diffuse"] = n + 1
        cap[f"{tag}.in_xyz"] = indep.xyz.detach().cpu().clone()
        cap[f"{tag}.in_is_sm"] = indep.is_sm.detach().cpu().clone()
        cap[f"{tag}.in_is_diffused"] = is_diffused.detach().cpu().clone()
        cap[f"{tag}.in_t"] = torch.tensor(float(t))
        cap[f"{tag}.rng_before"] = torch.get_rng_state().clone()
        out_indep, dout = orig_diffuse(conf, diffuser, indep, is_diffused, t)
        cap[f"{tag}.out_xyz"] = out_indep.xyz.detach().cpu().clone()
        r = dout["rigids_t"]
        cap[f"{tag}.rigids_t_trans"] = r.get_trans().detach().cpu().clone()
        cap[f"{tag}.rigids_t_rots"] = r.get_rots().get_rot_mats().detach().cpu().clone()
        r0 = dout["rigids_0_raw"]
        cap[f"{tag}.rigids_0_trans"] = r0.get_trans().detach().cpu().clone()
        cap[f"{tag}.rigids_0_rots"] = r0.get_rots().get_rot_mats().detach().cpu().clone()
        cap[f"{tag}.rng_after"] = torch.get_rng_state().clone()
        return out_indep, dout
    aa_model.diffuse = diffuse_spy

    # ---- inside the interpolant -------------------------------------------
    from se3_flow_matching.data import interpolant as itp
    from se3_flow_matching.data import so3_utils as s3
    from se3_flow_matching.data import utils as du

    orig_ct = itp.Interpolant._corrupt_trans_multi_t

    def ct_spy(self, trans_1, t, res_mask):
        cap["ct.trans_1"] = trans_1.detach().cpu().clone()
        cap["ct.t"] = t.detach().cpu().clone()
        out = orig_ct(self, trans_1, t, res_mask)
        cap["ct.out"] = out.detach().cpu().clone()
        return out
    itp.Interpolant._corrupt_trans_multi_t = ct_spy

    orig_ot = itp.Interpolant._batch_ot

    def ot_spy(self, trans_0, trans_1, res_mask):
        cap["ot.in_trans_0"] = trans_0.detach().cpu().clone()
        out = orig_ot(self, trans_0, trans_1, res_mask)
        cap["ot.out"] = out.detach().cpu().clone()
        return out
    itp.Interpolant._batch_ot = ot_spy

    orig_cr = itp.Interpolant._corrupt_rotmats_multi_t

    def cr_spy(self, rotmats_1, t, res_mask):
        cap["cr.rotmats_1"] = rotmats_1.detach().cpu().clone()
        out = orig_cr(self, rotmats_1, t, res_mask)
        cap["cr.out"] = out.detach().cpu().clone()
        return out
    itp.Interpolant._corrupt_rotmats_multi_t = cr_spy

    orig_sample = s3.BaseSampleSO3.sample

    def sample_spy(self, sigma, num_samples):
        out = orig_sample(self, sigma, num_samples)
        cap["igso3.sample_out"] = out.detach().cpu().clone()
        cap["igso3.sample_sigma"] = sigma.detach().cpu().clone()
        return out
    s3.BaseSampleSO3.sample = sample_spy

    orig_sv = s3.BaseSampleSO3.sample_vector
    def sv_spy(self, num_sigma, num_samples):
        out = orig_sv(self, num_sigma, num_samples)
        cap["igso3.vectors"] = out.detach().cpu().clone()
        return out
    s3.BaseSampleSO3.sample_vector = sv_spy

    orig_sa = s3.BaseSampleSO3.sample_angle
    def sa_spy(self, sigma, num_samples):
        out = orig_sa(self, sigma, num_samples)
        cap["igso3.angles"] = out.detach().cpu().clone()
        return out
    s3.BaseSampleSO3.sample_angle = sa_spy

    orig_geo = s3.geodesic_t
    def geo_spy(t, mat, base_mat, rot_vf=None):
        out = orig_geo(t, mat, base_mat, rot_vf)
        if "geo.out" not in cap:
            cap["geo.t"] = torch.as_tensor(t).detach().cpu().clone().float()
            cap["geo.mat"] = mat.detach().cpu().clone()
            cap["geo.base_mat"] = base_mat.detach().cpu().clone()
            cap["geo.out"] = out.detach().cpu().clone()
            quarter = torch.full_like(torch.as_tensor(t), 0.25)
            cap["geo.quarter_t"] = quarter.detach().cpu().clone().float()
            cap["geo.quarter_out"] = orig_geo(
                quarter, mat, base_mat, rot_vf
            ).detach().cpu().clone()
        return out
    s3.geodesic_t = geo_spy
    itp.so3_utils.geodesic_t = geo_spy

    meta["nm_to_ang"] = str(du.NM_TO_ANG_SCALE)

    # ---- atom37_from_rigid: the psi draw and the backbone it builds --------
    # This one looks purely geometric and is not: its first line is
    # `psi_pred = torch.rand(rigid.shape + (2,))`, which is draw 5 and draw 8 of
    # the nine. Capturing the rigid it was handed, the psi it drew and the
    # atom37 it produced lets the Rust side be run from the reference's own
    # input and compared alone.
    from rf_diffusion.frame_diffusion.data import all_atom as a37mod

    orig_a37 = a37mod.atom37_from_rigid

    def a37_spy(rigid, generator=None):
        n = meta.get("n_a37", 0)
        meta["n_a37"] = n + 1
        cap[f"a37_{n}.in_trans"] = rigid.get_trans().detach().cpu().clone()
        cap[f"a37_{n}.in_rots"] = \
            rigid.get_rots().get_rot_mats().detach().cpu().clone()
        cap[f"a37_{n}.rng_before"] = torch.get_rng_state().clone()
        out = orig_a37(rigid, generator)
        cap[f"a37_{n}.out"] = out.detach().cpu().clone()
        cap[f"a37_{n}.rng_after"] = torch.get_rng_state().clone()
        return out
    a37mod.atom37_from_rigid = a37_spy
    aa_model.all_atom.atom37_from_rigid = a37_spy

    orig_cb = a37mod.compute_backbone

    def cb_spy(bb_rigids, psi_torsions):
        n = meta.get("n_cb", 0)
        meta["n_cb"] = n + 1
        cap[f"cb_{n}.psi"] = psi_torsions.detach().cpu().clone()
        out = orig_cb(bb_rigids, psi_torsions)
        cap[f"cb_{n}.atom37"] = out[0].detach().cpu().clone()
        cap[f"cb_{n}.atom14"] = out[3].detach().cpu().clone()
        return out
    a37mod.compute_backbone = cb_spy

    orig_legs = aa_model.add_fake_frame_legs

    def legs_spy(xyz, is_atom, generator=None):
        n = meta.get("n_legs", 0)
        meta["n_legs"] = n + 1
        out = orig_legs(xyz, is_atom, generator)
        if n < 2:
            cap[f"legs{n}.in"] = xyz.detach().cpu().clone()
            cap[f"legs{n}.out"] = out.detach().cpu().clone()
        return out
    aa_model.add_fake_frame_legs = legs_spy

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

    active["on"] = True
    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    active["on"] = False

    # ---- the IGSO3 lookup tables -----------------------------------------
    ig = sampler.diffuser.igso3
    cap["igso3.sigma_grid"] = ig.sigma_grid.detach().cpu().clone().float()
    cap["igso3.omega_grid"] = ig.omega_grid.detach().cpu().clone().float()
    cap["igso3.cdf"] = ig.cdf_igso3.detach().cpu().clone().float()
    meta["igso3_tol"] = str(ig.tol)
    meta["igso3_interpolate"] = str(ig.interpolate)
    meta["igso3_sigma_used"] = "1.5"
    meta["igso3_sigma_idx"] = str(int(torch.bucketize(torch.tensor([1.5]), ig.sigma_grid)[0]))

    print(f"RNG draws inside sample_init, in order ({len(draws)}):")
    for i, (fn, shape, where) in enumerate(draws):
        print(f"  {i:>2} torch.{fn:<7} {str(shape):<12} {where}")
    meta["draw_order"] = ";".join(f"{fn}{list(s)}" for fn, s, _ in draws)
    meta["n_draws"] = str(len(draws))
    meta["tag"] = "pinned" if args.pinned else "stock"
    print(f"igso3: sigma_grid {tuple(ig.sigma_grid.shape)}, "
          f"omega_grid {tuple(ig.omega_grid.shape)}, cdf {tuple(ig.cdf_igso3.shape)}, "
          f"sigma_idx for 1.5 = {meta['igso3_sigma_idx']}")
    print(f"captured {len(cap)} tensors")
    common.write_fixture(args.out, "stages", cap, meta)
    _ = (contig_map, atomizer, t_step_input, indep)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
