#!/usr/bin/env python3
"""Bisect the reverse pass through `XYZConverter.compute_all_atom`.

`calc_lj_grads` gets `(dL/dxyz, dL/dalpha)` from autograd; the port writes that
reverse pass by hand, and the remaining disagreement is a *precision-boundary*
question — which f64/fp32 boundary each of ATen's derivative formulas rounds at
— not a chain-rule question (`cos = 1.0000000000` on both outputs).

So capture the gradient at every intermediate of the graph, not just at its
leaves: each einsum's output and inputs, each rotation constructor's output,
`NORM` and `angs`, and the rigid-frame rotation. Then every stage of the Rust
reverse pass can be checked from the reference's own upstream gradient.

    PYTHONPATH=<ref> .venv/bin/python python/dump_aa_bwd.py --pinned
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


def load_fixture(path):
    from safetensors.torch import load_file
    return load_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", default=True)
    ap.add_argument("--stock", dest="pinned", action="store_false")
    ap.add_argument("--out", default="refiner_io")
    args = ap.parse_args()

    if args.pinned:
        import pinned
        print(f"PINNED: {len(pinned.enable())} entry points")
    common.add_ref_to_path()

    # `rf_diffusion.chemical` is the entry point that supplies `params`; going
    # through `rf2aa.chemical` directly leaves the singleton uninitialised.
    from rf_diffusion.chemical import ChemicalData as ChemData
    import rf2aa.util_module as um
    import rf2aa.util as u

    io = load_fixture(os.path.join(common.FIXTURES, "refiner_io", "io.safetensors"))
    seq = io["lj0.seq"]
    xyz = io["lj0.xyz"].clone().requires_grad_(True)
    alpha = io["lj0.alpha"].clone().requires_grad_(True)
    dljEdx = io["ljf.dljEdx"]

    aamask = ChemData().allatom_mask
    natoms = torch.sum(aamask.bool()[seq])
    gout = (natoms * dljEdx).float()

    cap = {}

    # ---- rotation constructors -------------------------------------------
    # Replicas of upstream, line for line, so the autograd graph is the same
    # one; the forward output is asserted bit-identical to upstream's before
    # anything is read off the replica.
    def hook(name, t):
        if t.requires_grad:
            t.register_hook(lambda g, name=name: cap.__setitem__(name, g.detach().clone()))

    n_rot = {"X": 0, "Z": 0, "A": 0}

    def rot_replica(kind, angs, u_axis=None, eps=1e-6):
        i = n_rot[kind]
        n_rot[kind] = i + 1
        tag = f"rot{kind}{i}"
        B, L = angs.shape[:2]
        NORM = torch.linalg.norm(angs, dim=-1) + eps
        RTs = torch.eye(4, device=angs.device).repeat(B, L, 1, 1)
        if kind == "X":
            RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
            RTs[:, :, 1, 2] = -angs[:, :, 1] / NORM
            RTs[:, :, 2, 1] = angs[:, :, 1] / NORM
            RTs[:, :, 2, 2] = angs[:, :, 0] / NORM
        elif kind == "Z":
            RTs[:, :, 0, 0] = angs[:, :, 0] / NORM
            RTs[:, :, 0, 1] = -angs[:, :, 1] / NORM
            RTs[:, :, 1, 0] = angs[:, :, 1] / NORM
            RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
        else:
            ct = angs[:, :, 0] / NORM
            st = angs[:, :, 1] / NORM
            u0, u1, u2 = u_axis[:, :, 0], u_axis[:, :, 1], u_axis[:, :, 2]
            RTs[:, :, 0, 0] = ct + u0 * u0 * (1 - ct)
            RTs[:, :, 0, 1] = u0 * u1 * (1 - ct) - u2 * st
            RTs[:, :, 0, 2] = u0 * u2 * (1 - ct) + u1 * st
            RTs[:, :, 1, 0] = u0 * u1 * (1 - ct) + u2 * st
            RTs[:, :, 1, 1] = ct + u1 * u1 * (1 - ct)
            RTs[:, :, 1, 2] = u1 * u2 * (1 - ct) - u0 * st
            RTs[:, :, 2, 0] = u0 * u2 * (1 - ct) - u1 * st
            RTs[:, :, 2, 1] = u1 * u2 * (1 - ct) + u0 * st
            RTs[:, :, 2, 2] = ct + u2 * u2 * (1 - ct)
        cap[f"{tag}.angs"] = angs.detach().clone()
        cap[f"{tag}.norm"] = NORM.detach().clone()
        cap[f"{tag}.out"] = RTs.detach().clone()
        hook(f"{tag}.d_out", RTs)
        hook(f"{tag}.d_norm", NORM)
        hook(f"{tag}.d_angs", angs)
        return RTs

    orig_rotX, orig_rotZ, orig_rot_axis = um.make_rotX, um.make_rotZ, um.make_rot_axis

    def rotX(angs, eps=1e-6):
        want = orig_rotX(angs.detach(), eps)
        got = rot_replica("X", angs, eps=eps)
        assert torch.equal(want, got.detach()), "make_rotX replica diverged"
        return got

    def rotZ(angs, eps=1e-6):
        want = orig_rotZ(angs.detach(), eps)
        got = rot_replica("Z", angs, eps=eps)
        assert torch.equal(want, got.detach()), "make_rotZ replica diverged"
        return got

    def rot_axis(angs, uu, eps=1e-6):
        want = orig_rot_axis(angs.detach(), uu.detach(), eps)
        got = rot_replica("A", angs, uu, eps=eps)
        assert torch.equal(want, got.detach()), "make_rot_axis replica diverged"
        return got

    um.make_rotX, um.make_rotZ, um.make_rot_axis = rotX, rotZ, rot_axis

    # ---- einsums ----------------------------------------------------------
    # `compute_all_atom` is the only thing running, so every einsum seen here is
    # one of its frame chains, in source order.
    orig_einsum = torch.einsum
    n_es = {"i": 0}

    def einsum_spy(*a, **k):
        out = orig_einsum(*a, **k)
        if n_es.get("quiet"):
            return out          # the replica-check call on detached tensors
        i = n_es["i"]
        n_es["i"] = i + 1
        eq = a[0] if isinstance(a[0], str) else None
        ops = a[1:] if eq is not None else ()
        cap[f"es{i}.n"] = torch.tensor(len(ops))
        for j, o in enumerate(ops):
            if torch.is_tensor(o):
                hook(f"es{i}.d_in{j}", o)
        hook(f"es{i}.d_out", out)
        return out

    torch.einsum = einsum_spy

    # ---- rigid frame ------------------------------------------------------
    # Same replica-and-assert trick as the rotations: the body is upstream's,
    # line for line, so the autograd graph is identical; the forward is checked
    # bit for bit against upstream before any gradient is read off it.
    orig_rigid = u.rigid_from_3_points

    def rigid_replica(N, Ca, C, is_na=None, eps=1e-4):
        dims = N.shape[:-1]
        v1 = C - Ca
        v2 = N - Ca
        e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
        proj = torch.einsum('...li, ...li -> ...l', e1, v2)
        u2 = v2 - (proj[..., None] * e1)
        e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
        e3 = torch.cross(e1, e2, dim=-1)
        Rc = torch.cat([e1[..., None], e2[..., None], e3[..., None]], axis=-1)
        v2n = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
        cosref = torch.sum(e1 * v2n, dim=-1)
        costgt = torch.full(dims, -0.3616, device=N.device)
        if is_na is not None:
            costgt[is_na] = ChemData().costgtNA
        cos2del = torch.clamp(
            cosref * costgt + torch.sqrt((1 - cosref * cosref) * (1 - costgt * costgt) + eps),
            min=-1.0, max=1.0)
        cosdel = torch.sqrt(0.5 * (1 + cos2del) + eps)
        sindel = torch.sign(costgt - cosref) * torch.sqrt(1 - 0.5 * (1 + cos2del) + eps)
        Rp = torch.eye(3, device=N.device).repeat(*dims, 1, 1)
        Rp[..., 0, 0] = cosdel
        Rp[..., 0, 1] = -sindel
        Rp[..., 1, 0] = sindel
        Rp[..., 1, 1] = cosdel
        R = torch.einsum('...ij,...jk->...ik', Rc, Rp)
        for nm, t in [("v1", v1), ("v2", v2), ("e1", e1), ("proj", proj), ("u2", u2),
                      ("e2", e2), ("e3", e3), ("Rc", Rc), ("v2n", v2n),
                      ("cosref", cosref), ("cos2del", cos2del), ("cosdel", cosdel),
                      ("sindel", sindel), ("Rp", Rp), ("R", R)]:
            cap[f"rigid.{nm}"] = t.detach().clone()
            hook(f"rigid.d_{nm}", t)
        hook("rigid.d_N", N)
        hook("rigid.d_Ca", Ca)
        hook("rigid.d_C", C)
        return R, Ca

    def rigid_spy(N, Ca, C, is_na=None, eps=1e-4):
        n_es["quiet"] = True
        want, _ = orig_rigid(N.detach(), Ca.detach(), C.detach(), is_na, eps)
        n_es["quiet"] = False
        R, T = rigid_replica(N, Ca, C, is_na, eps)
        assert torch.equal(want, R.detach()), "rigid_from_3_points replica diverged"
        return R, T
    u.rigid_from_3_points = rigid_spy
    um.rigid_from_3_points = rigid_spy

    # ---- run --------------------------------------------------------------
    conv = um.XYZConverter()
    with torch.enable_grad():
        frames, xyzaa = conv.compute_all_atom(seq, xyz, alpha)
        hook("aa.d_frames", frames)
        hook("aa.d_xyzaa", xyzaa)
        dxyz, dalpha = torch.autograd.grad(xyzaa, (xyz, alpha), grad_outputs=gout)

    cap["aa.gout"] = gout
    cap["aa.natoms"] = natoms
    cap["aa.frames"] = frames.detach().clone()
    cap["aa.xyzaa"] = xyzaa.detach().clone()
    cap["out.dxyz"] = dxyz
    cap["out.dalpha"] = dalpha

    want_x, want_a = io["lj0.dxyz"], io["lj0.dalpha"]
    print(f"einsums seen: {n_es['i']}   rotations: {n_rot}")
    print(f"dxyz   reproduces the captured calc_lj_grads output: "
          f"{bool(torch.equal(dxyz, want_x))}")
    print(f"dalpha reproduces the captured calc_lj_grads output: "
          f"{bool(torch.equal(dalpha, want_a))}")
    print(f"captured {len(cap)} tensors")
    common.write_fixture(args.out, "aa_bwd", cap,
                         {"tag": "pinned" if args.pinned else "stock"})


if __name__ == "__main__":
    main()
