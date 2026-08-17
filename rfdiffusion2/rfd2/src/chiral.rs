//! `calc_chiral_grads` — the gradient of the chiral-dihedral loss with respect
//! to the coordinates, which every `Str2Str` feeds to the SE(3) transformer as
//! three extra degree-1 features.
//!
//! Upstream gets this from autograd. A port cannot, so the reverse pass is
//! written out here — and the interesting part is that the two passes do **not**
//! use the same arithmetic:
//!
//! * the **forward** runs under `python/pinned.py`, so `torch.norm`,
//!   `torch.sum`, `torch.cross` and `torch.atan2` each compute in f64 and round
//!   to fp32 once;
//! * the **backward** is mixed, and not in the obvious way. `python/pinned.py`
//!   wraps an op as `orig(x.double()).float()`, so the autograd graph it builds
//!   contains *three* nodes: a promotion, the op **on f64 tensors**, and a
//!   narrowing. Running that graph backwards therefore evaluates the patched
//!   op's derivative in **f64**, with an f32 rounding at each end — while the
//!   unpatched elementwise steps between them (`b0 - s0*b1n`, `b1/den`, the
//!   products feeding each `sum`) run in plain fp32.
//!
//! So the reverse pass below is f64 exactly where the forward op was patched
//! (`atan2`, `cross`, `norm`) and fp32 everywhere else. Doing it all in fp32
//! leaves ~8 % of the values 1-2 ULP out; doing it all in f64 is a different
//! wrong answer.

/// Number of the atom slot the chiral constraints index into. `calc_chiral_loss`
/// takes `pred[:, idx, 1]` — always the **CA** slot, never the atom's own slot.
const CA: usize = 1;

#[inline]
fn norm3_pinned(v: [f32; 3]) -> f32 {
    let (a, b, c) = (v[0] as f64, v[1] as f64, v[2] as f64);
    (a * a + b * b + c * c).sqrt() as f32
}

/// `torch.sum(a * b, dim=-1)`: the product is fp32, the reduction is pinned.
#[inline]
fn sum_mul3_pinned(a: [f32; 3], b: [f32; 3]) -> f32 {
    let mut s = 0.0f64;
    for k in 0..3 {
        s += (a[k] * b[k]) as f64;
    }
    s as f32
}

#[inline]
fn cross_pinned(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let f = |x: f32| x as f64;
    [
        (f(a[1]) * f(b[2]) - f(a[2]) * f(b[1])) as f32,
        (f(a[2]) * f(b[0]) - f(a[0]) * f(b[2])) as f32,
        (f(a[0]) * f(b[1]) - f(a[1]) * f(b[0])) as f32,
    ]
}

const EPS: f32 = 1e-4;

/// `dL/dxyz` for `L = mean((dihedral - target)^2)` over the chiral constraints.
///
/// `xyz` is `[L, n_atoms, 3]`; `chirals` is `[n_chiral, 5]` as
/// `(i0, i1, i2, i3, target_angle)`. Returns a tensor shaped like `xyz`.
///
/// Returns all zeros when there are no constraints — matching upstream, which
/// short-circuits on `l.item() == 0.0` and hands back `torch.zeros(xyz.shape)`
/// rather than a gradient.
pub fn chiral_grads(xyz: &[f32], n_res: usize, n_atoms: usize, chirals: &[f32]) -> Vec<f32> {
    let mut grad = vec![0.0f32; n_res * n_atoms * 3];
    let nc = chirals.len() / 5;
    if nc == 0 {
        return grad;
    }
    // `.mean()` over a [B=1, 1, nchiral] tensor
    let inv_n = 1.0f32 / nc as f32;

    let get = |i: usize| -> [f32; 3] {
        let o = (i * n_atoms + CA) * 3;
        [xyz[o], xyz[o + 1], xyz[o + 2]]
    };

    for c in 0..nc {
        let i0 = chirals[c * 5] as usize;
        let i1 = chirals[c * 5 + 1] as usize;
        let i2 = chirals[c * 5 + 2] as usize;
        let i3 = chirals[c * 5 + 3] as usize;
        let target = chirals[c * 5 + 4];
        let (pa, pb, pc, pd) = (get(i0), get(i1), get(i2), get(i3));

        // ---- forward (pinned where the reference is pinned) ---------------
        let b0 = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
        let b1 = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];
        let b2 = [pd[0] - pc[0], pd[1] - pc[1], pd[2] - pc[2]];
        // keep the f64 norm: `norm_backward` runs inside the promoted subgraph
        // and divides by the **f64** result, not by its f32 narrowing
        let nrm64 = {
            let (a, b, c) = (b1[0] as f64, b1[1] as f64, b1[2] as f64);
            (a * a + b * b + c * c).sqrt()
        };
        let nrm = nrm64 as f32;
        let den = nrm + EPS;
        let b1n = [b1[0] / den, b1[1] / den, b1[2] / den];
        let s0 = sum_mul3_pinned(b0, b1n);
        let v = [b0[0] - s0 * b1n[0], b0[1] - s0 * b1n[1], b0[2] - s0 * b1n[2]];
        let s2 = sum_mul3_pinned(b2, b1n);
        let w = [b2[0] - s2 * b1n[0], b2[1] - s2 * b1n[1], b2[2] - s2 * b1n[2]];
        let x = sum_mul3_pinned(v, w);
        let cr = cross_pinned(b1n, v);
        let y = sum_mul3_pinned(cr, w);
        let ay = y + EPS;
        let ax = x + EPS;
        let dih = (ay as f64).atan2(ax as f64) as f32;

        // ---- backward (fp32, as ATen's kernels are) -----------------------
        // ATen: `mean` backward is `grad/N`, `square` backward is
        // `grad * 2 * self`, and `atan2` backward multiplies by a
        // **reciprocal** rather than dividing:
        //   self:  grad * other * (self^2 + other^2).reciprocal()
        //   other: grad * -self  * (self^2 + other^2).reciprocal()
        let g_dih = (inv_n * 2.0f32) * (dih - target);
        // atan2's derivative is evaluated in the promoted (f64) subgraph:
        //   self:  grad * other * (self^2 + other^2).reciprocal()
        //   other: grad * -self  * (self^2 + other^2).reciprocal()
        let (ay64, ax64, g64) = (ay as f64, ax as f64, g_dih as f64);
        let recip = 1.0f64 / (ay64 * ay64 + ax64 * ax64);
        let g_y = ((g64 * ax64) * recip) as f32;
        let g_x = ((g64 * -ay64) * recip) as f32;

        // y = sum(cr * w) ; x = sum(v * w)
        let mut g_cr = [0.0f32; 3];
        let mut g_w = [0.0f32; 3];
        let mut g_v = [0.0f32; 3];
        for k in 0..3 {
            g_cr[k] = g_y * w[k];
            g_w[k] = g_y * cr[k] + g_x * v[k];
            g_v[k] = g_x * w[k];
        }

        // cr = cross(b1n, v): `self: other.cross(grad)`, `other: grad.cross(self)`
        // — also inside the promoted subgraph, so f64 with one rounding.
        let mut g_b1n = cross_pinned(v, g_cr);
        let g_v2 = cross_pinned(g_cr, b1n);
        for k in 0..3 {
            g_v[k] += g_v2[k];
        }

        // w = b2 - s2 * b1n
        let mut g_b2 = [0.0f32; 3];
        let mut g_s2 = 0.0f32;
        for k in 0..3 {
            g_b2[k] = g_w[k];
            g_s2 += -g_w[k] * b1n[k]; // broadcast multiply -> sum over the axis
            g_b1n[k] += -g_w[k] * s2;
        }
        // s2 = sum(b2 * b1n)
        for k in 0..3 {
            g_b2[k] += g_s2 * b1n[k];
            g_b1n[k] += g_s2 * b2[k];
        }

        // v = b0 - s0 * b1n
        let mut g_b0 = [0.0f32; 3];
        let mut g_s0 = 0.0f32;
        for k in 0..3 {
            g_b0[k] = g_v[k];
            g_s0 += -g_v[k] * b1n[k];
            g_b1n[k] += -g_v[k] * s0;
        }
        for k in 0..3 {
            g_b0[k] += g_s0 * b1n[k];
            g_b1n[k] += g_s0 * b0[k];
        }

        // b1n = b1 / den, den = norm(b1) + eps
        let mut g_b1 = [0.0f32; 3];
        let mut g_den = 0.0f32;
        for k in 0..3 {
            // div backward, ATen's association:
            //   self:  grad / other
            //   other: -grad * ((self / other) / other)
            g_b1[k] = g_b1n[k] / den;
            g_den += -g_b1n[k] * (b1n[k] / den);
        }
        // norm backward for p=2 is `self * (grad / norm)` — in f64, against the
        // f64 norm, then narrowed once.
        let gn = g_den as f64 / nrm64;
        for k in 0..3 {
            g_b1[k] += (b1[k] as f64 * gn) as f32;
        }

        // b0 = a - b ; b1 = c - b ; b2 = d - c
        let mut add = |i: usize, g: [f32; 3]| {
            let o = (i * n_atoms + CA) * 3;
            for k in 0..3 {
                grad[o + k] += g[k];
            }
        };
        add(i0, g_b0);
        add(i1, [-g_b0[0] - g_b1[0], -g_b0[1] - g_b1[1], -g_b0[2] - g_b1[2]]);
        add(i2, [g_b1[0] - g_b2[0], g_b1[1] - g_b2[1], g_b1[2] - g_b2[2]]);
        add(i3, g_b2);
    }
    grad
}
