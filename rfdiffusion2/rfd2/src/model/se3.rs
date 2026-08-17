//! The SE(3)-transformer refiner — NVIDIA's `se3_transformer` as vendored in
//! `rf2aa/SE3Transformer/`, plus the two third-party pieces it leans on:
//! `e3nn`'s real spherical harmonics and Wigner 3-j symbols, and DGL's graph
//! kernels.
//!
//! The checkpoint's `SE3_param` pins the shape of all of this, and the numbers
//! are much smaller than the module's generality suggests: `num_degrees = 2`, so
//! features are degree 0 (scalars) and degree 1 (vectors) only, and the bases
//! need spherical harmonics up to degree `2 * max_degree = 2`. That is why the
//! CG tables are 1 225 values rather than a library.
//!
//! Three things here are decided by the *config*, not by the defaults, and each
//! selects a different code path in upstream:
//!
//! * `tensor_cores = False` -> `ConvSE3FuseLevel.PARTIAL`, never `FULL`.
//! * In `to_key_value`, input channels are **not** uniform (64 scalars vs 3
//!   vectors), so the partial fusion is **per input degree** (`conv_in`); in the
//!   second attention block they are uniform, so it fuses **per output degree**
//!   (`conv_out`). The checkpoint's parameter names confirm which is which.
//! * `final_layer = "lin"` -> the last graph module is a `LinearSE3`, so there
//!   is no `ConvSE3` self-interaction or pooling on the output.

use crate::ops::acc::Acc;
use crate::nn::{LayerNorm, Linear, Params};
use crate::ops::relu_;
use crate::tensor::Tensor;
use crate::weights::Weights;
use std::sync::OnceLock;

#[inline]
pub fn degree_to_dim(d: usize) -> usize {
    2 * d + 1
}

// ---------------------------------------------------------------------------
// Fiber
// ---------------------------------------------------------------------------

/// `(degree, channels)` pairs. Upstream's `Fiber` sorts its `structure` by
/// *channel count*, which only affects iteration order for module registration;
/// everything numeric goes through `degrees` (sorted by degree), so this keeps
/// the degree-sorted view only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fiber {
    pub channels: Vec<usize>, // indexed by degree, 0 for absent
}

impl Fiber {
    pub fn new(pairs: &[(usize, usize)]) -> Self {
        let maxd = pairs.iter().map(|p| p.0).max().unwrap_or(0);
        let mut channels = vec![0usize; maxd + 1];
        for &(d, c) in pairs {
            channels[d] = c;
        }
        Fiber { channels }
    }

    pub fn degrees(&self) -> Vec<usize> {
        (0..self.channels.len()).filter(|&d| self.channels[d] > 0).collect()
    }

    pub fn get(&self, d: usize) -> usize {
        *self.channels.get(d).unwrap_or(&0)
    }

    /// `sum(channels * (2d+1))` — the flattened size, used for the attention
    /// scaling `1/sqrt(num_features)`.
    pub fn num_features(&self) -> usize {
        self.degrees().iter().map(|&d| self.get(d) * degree_to_dim(d)).sum()
    }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub struct Graph {
    pub n_nodes: usize,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
    /// `xyz[dst] - xyz[src]`, one 3-vector per edge.
    pub rel_pos: Vec<f32>,
    /// Incoming-edge lists per destination node, in edge order — the grouping
    /// that `edge_softmax` and `copy_e_sum` reduce over.
    pub in_edges: Vec<Vec<u32>>,
}

impl Graph {
    pub fn n_edges(&self) -> usize {
        self.src.len()
    }

    fn build(n_nodes: usize, src: Vec<u32>, dst: Vec<u32>, xyz: &[f32]) -> Self {
        let mut rel_pos = vec![0.0f32; src.len() * 3];
        for e in 0..src.len() {
            for k in 0..3 {
                rel_pos[e * 3 + k] =
                    xyz[dst[e] as usize * 3 + k] - xyz[src[e] as usize * 3 + k];
            }
        }
        let mut in_edges = vec![Vec::new(); n_nodes];
        for e in 0..src.len() {
            in_edges[dst[e] as usize].push(e as u32);
        }
        Graph { n_nodes, src, dst, rel_pos, in_edges }
    }
}

/// `util_module.make_full_graph`: an edge for every ordered pair whose residue
/// indices differ.
///
/// Note the predicate is on `idx`, not on position — two tokens that share a
/// residue index (an atomized residue's atoms, say) get no edge between them.
pub fn make_full_graph(xyz: &[f32], idx: &[i64]) -> Graph {
    let l = idx.len();
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    // torch.where scans in row-major (i, j) order, and the edge order decides
    // nothing numerically here (all reductions are pinned) but does decide the
    // edge feature layout, so it is reproduced exactly.
    for i in 0..l {
        for j in 0..l {
            if idx[j] - idx[i] != 0 {
                src.push(i as u32);
                dst.push(j as u32);
            }
        }
    }
    Graph::build(l, src, dst, xyz)
}

/// `util_module.make_topk_graph` with `topk_incl_local=True`.
///
/// The two-stage construction matters: local pairs (`|Δidx| < nlocal`) have
/// their distance **zeroed** before the top-k selection, so they always win a
/// slot, and then the final edge set is the union of the top-k and the local
/// band. Skipping the zeroing gives a graph that is right for compact
/// structures and quietly wrong for extended ones.
pub fn make_topk_graph(xyz: &[f32], idx: &[i64], top_k: usize) -> Graph {
    const NLOCAL: i64 = 33;
    const EPS: f32 = 1e-4;
    let l = idx.len();
    let d = crate::geom::cdist_self(xyz, l);
    let mut dmat = vec![0.0f32; l * l];
    let mut sep = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let s = (idx[j] - idx[i]).abs() as f32 + if i == j { 9999.9 } else { 0.0 };
            sep[i * l + j] = s;
            let base = d[i * l + j] + if i == j { 9999.9 } else { 0.0 };
            dmat[i * l + j] = base + s * EPS;
        }
    }
    for i in 0..l {
        for j in 0..l {
            if sep[i * l + j] < NLOCAL as f32 {
                dmat[i * l + j] = 0.0;
            }
        }
    }
    let k = top_k.min(l - 1);
    let mut keep = vec![false; l * l];
    let mut order: Vec<usize> = (0..l).collect();
    for i in 0..l {
        order.sort_by(|&a, &b| {
            dmat[i * l + a]
                .partial_cmp(&dmat[i * l + b])
                .unwrap()
                .then(a.cmp(&b))
        });
        for &j in order.iter().take(k) {
            keep[i * l + j] = true;
        }
    }
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for i in 0..l {
        for j in 0..l {
            if keep[i * l + j] || sep[i * l + j] < NLOCAL as f32 {
                src.push(i as u32);
                dst.push(j as u32);
            }
        }
    }
    Graph::build(l, src, dst, xyz)
}

// ---------------------------------------------------------------------------
// bases
// ---------------------------------------------------------------------------

static CG_BLOB: &[u8] = include_bytes!("../../data/se3_cg.safetensors");
static CG: OnceLock<Weights> = OnceLock::new();

fn cg() -> &'static Weights {
    CG.get_or_init(|| Weights::from_static(CG_BLOB).expect("embedded se3_cg is corrupt"))
}

/// `e3nn.o3.spherical_harmonics(range(0..=lmax), x, normalize=True)` with the
/// default `normalization='integral'`.
///
/// The polynomials are e3nn's own closed forms; the two normalisations around
/// them are the part that is easy to drop: the input is `F.normalize`d first
/// (pinned, so f64 then one rounding) and the whole output is divided by
/// `sqrt(4*pi)` afterwards.
pub fn spherical_harmonics(rel_pos: &[f32], n: usize, lmax: usize) -> Vec<Vec<f32>> {
    assert!(lmax <= 2, "SH beyond degree 2 not needed by this checkpoint");
    // e3nn does `sh.div_(math.sqrt(4*pi))` — a DIVISION by the fp32-narrowed
    // constant. Multiplying by `f32(1/sqrt(4*pi))` is a different fp32 value.
    let norm_div = (4.0f64 * std::f64::consts::PI).sqrt() as f32;
    let mut out: Vec<Vec<f32>> = (0..=lmax).map(|l| vec![0.0f32; n * degree_to_dim(l)]).collect();
    let sqrt3 = 3.0f64.sqrt() as f32;
    let sqrt15 = 15.0f64.sqrt() as f32;
    let sqrt5 = 5.0f64.sqrt() as f32;
    for e in 0..n {
        // F.normalize(x, dim=-1) — pinned: reduce in f64, round once.
        let (a, b, c) = (
            rel_pos[e * 3] as f64,
            rel_pos[e * 3 + 1] as f64,
            rel_pos[e * 3 + 2] as f64,
        );
        let nrm = (a * a + b * b + c * c).sqrt().max(1e-12);
        let x = (a / nrm) as f32;
        let y = (b / nrm) as f32;
        let z = (c / nrm) as f32;
        out[0][e] = 1.0 / norm_div;
        if lmax >= 1 {
            out[1][e * 3] = (sqrt3 * x) / norm_div;
            out[1][e * 3 + 1] = (sqrt3 * y) / norm_div;
            out[1][e * 3 + 2] = (sqrt3 * z) / norm_div;
        }
        if lmax >= 2 {
            let y2 = y * y;
            let x2z2 = x * x + z * z;
            let o = e * 5;
            out[2][o] = (sqrt15 * x * z) / norm_div;
            out[2][o + 1] = (sqrt15 * x * y) / norm_div;
            out[2][o + 2] = (sqrt5 * (y2 - 0.5 * x2z2)) / norm_div;
            out[2][o + 3] = (sqrt15 * y * z) / norm_div;
            out[2][o + 4] = (0.5 * sqrt15 * (z * z - x * x)) / norm_div;
        }
    }
    out
}

/// Per-`(d_in, d_out)` bases, plus the per-degree fused views the convolutions
/// actually index.
pub struct Basis {
    pub max_degree: usize,
    pub n_edges: usize,
    /// `in{d}_fused`: `[n, dim(d), sum_freq_in(d), sum_dim]`
    pub in_fused: Vec<Vec<f32>>,
    pub in_freq: Vec<usize>,
    /// `out{d}_fused`: `[n, sum_dim, sum_freq_out(d), dim(d)]`
    pub out_fused: Vec<Vec<f32>>,
    pub out_freq: Vec<usize>,
    pub sum_dim: usize,
}

pub fn build_basis(rel_pos: &[f32], n: usize, max_degree: usize) -> Basis {
    let sh = spherical_harmonics(rel_pos, n, 2 * max_degree);
    let sum_dim: usize = (0..=max_degree).map(degree_to_dim).sum();

    // basis[d_in][d_out] : [n, dim(d_in), n_freq, dim(d_out)]
    let mut pair: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); max_degree + 1]; max_degree + 1];
    for d_in in 0..=max_degree {
        for d_out in 0..=max_degree {
            let li = degree_to_dim(d_in);
            let ko = degree_to_dim(d_out);
            let js: Vec<usize> = (d_in.abs_diff(d_out)..=(d_in + d_out)).collect();
            let nf = js.len();
            let mut b = vec![0.0f32; n * li * nf * ko];
            for (fi, &j) in js.iter().enumerate() {
                let q = cg().get(&format!("cg_{d_in}_{d_out}_{j}")); // [ko, li, 2j+1], f32
                let fdim = degree_to_dim(j);
                for e in 0..n {
                    for l in 0..li {
                        for k in 0..ko {
                            // einsum('n f, k l f -> n l k'), pinned
                            let mut acc = Acc::new();
                            for f in 0..fdim {
                                acc.add(sh[j][e * fdim + f] as f64
                                    * q.data[(k * li + l) * fdim + f] as f64);
                            }
                            b[((e * li + l) * nf + fi) * ko + k] = acc.get() as f32;
                        }
                    }
                }
            }
            pair[d_in][d_out] = b;
        }
    }

    // fused per input degree
    let mut in_fused = Vec::new();
    let mut in_freq = Vec::new();
    for d_in in 0..=max_degree {
        let li = degree_to_dim(d_in);
        let sf: usize = (0..=max_degree).map(|d| degree_to_dim(d_in.min(d))).sum();
        let mut f = vec![0.0f32; n * li * sf * sum_dim];
        let (mut acc_d, mut acc_f) = (0usize, 0usize);
        for d_out in 0..=max_degree {
            let ko = degree_to_dim(d_out);
            let nf = degree_to_dim(d_in.min(d_out));
            let src = &pair[d_in][d_out];
            let src_nf = d_in + d_out - d_in.abs_diff(d_out) + 1;
            for e in 0..n {
                for l in 0..li {
                    for fi in 0..nf {
                        for k in 0..ko {
                            f[((e * li + l) * sf + acc_f + fi) * sum_dim + acc_d + k] =
                                src[((e * li + l) * src_nf + fi) * ko + k];
                        }
                    }
                }
            }
            acc_d += ko;
            acc_f += nf;
        }
        in_fused.push(f);
        in_freq.push(sf);
    }

    // fused per output degree
    let mut out_fused = Vec::new();
    let mut out_freq = Vec::new();
    for d_out in 0..=max_degree {
        let ko = degree_to_dim(d_out);
        let sf: usize = (0..=max_degree).map(|d| degree_to_dim(d_out.min(d))).sum();
        let mut f = vec![0.0f32; n * sum_dim * sf * ko];
        let (mut acc_d, mut acc_f) = (0usize, 0usize);
        for d_in in 0..=max_degree {
            let li = degree_to_dim(d_in);
            let nf = degree_to_dim(d_out.min(d_in));
            let src = &pair[d_in][d_out];
            let src_nf = d_in + d_out - d_in.abs_diff(d_out) + 1;
            for e in 0..n {
                for l in 0..li {
                    for fi in 0..nf {
                        for k in 0..ko {
                            f[((e * sum_dim + acc_d + l) * sf + acc_f + fi) * ko + k] =
                                src[((e * li + l) * src_nf + fi) * ko + k];
                        }
                    }
                }
            }
            acc_d += li;
            acc_f += nf;
        }
        out_fused.push(f);
        out_freq.push(sf);
    }

    Basis { max_degree, n_edges: n, in_fused, in_freq, out_fused, out_freq, sum_dim }
}

// ---------------------------------------------------------------------------
// RadialProfile / VersatileConvSE3 / ConvSE3
// ---------------------------------------------------------------------------

pub struct RadialProfile {
    pub l0: Linear,
    pub n1: LayerNorm,
    pub l3: Linear,
    pub n4: LayerNorm,
    pub l6: Linear,
}

impl RadialProfile {
    pub fn load(p: &Params) -> Self {
        let n = p.sub("net");
        RadialProfile {
            l0: Linear::load(&n.idx(0)),
            n1: LayerNorm::load(&n.idx(1)),
            l3: Linear::load(&n.idx(3)),
            n4: LayerNorm::load(&n.idx(4)),
            l6: Linear::load_nobias(&n.idx(6)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.l0.forward(x);
        let h = self.n1.forward(&h);
        let mut h = h;
        relu_(&mut h);
        let h = self.l3.forward(&h);
        let h = self.n4.forward(&h);
        let mut h = h;
        relu_(&mut h);
        self.l6.forward(&h)
    }
}

pub struct VersatileConv {
    pub radial: RadialProfile,
    pub freq_sum: usize,
    pub channels_in: usize,
    pub channels_out: usize,
}

impl VersatileConv {
    pub fn load(p: &Params, freq_sum: usize, channels_in: usize, channels_out: usize) -> Self {
        VersatileConv {
            radial: RadialProfile::load(&p.sub("radial_func")),
            freq_sum,
            channels_in,
            channels_out,
        }
    }

    /// `features [n, c_in, in_dim]`, `basis [n, in_dim, freq_sum, out_dim]`
    /// -> `[n, c_out, out_dim]`, exactly upstream's two chained `@`s.
    pub fn forward(
        &self,
        features: &[f32],
        n: usize,
        in_dim: usize,
        edge_feats: &Tensor,
        basis: &[f32],
        out_dim: usize,
    ) -> Vec<f32> {
        let (ci, co, fs) = (self.channels_in, self.channels_out, self.freq_sum);
        let rw = self.radial.forward(edge_feats); // [n, co*ci*fs]
        debug_assert_eq!(rw.last(), co * ci * fs);
        let mut out = vec![0.0f32; n * co * out_dim];
        let mut tmp = vec![0.0f32; ci * fs * out_dim];
        for e in 0..n {
            // tmp[c, f, k] = sum_l features[c, l] * basis[l, f, k]
            for c in 0..ci {
                for f in 0..fs {
                    for k in 0..out_dim {
                        let mut acc = Acc::new();
                        for l in 0..in_dim {
                            acc.add(features[(e * ci + c) * in_dim + l] as f64
                                * basis[((e * in_dim + l) * fs + f) * out_dim + k] as f64);
                        }
                        tmp[(c * fs + f) * out_dim + k] = acc.get() as f32;
                    }
                }
            }
            for o in 0..co {
                for k in 0..out_dim {
                    let mut acc = Acc::new();
                    for m in 0..ci * fs {
                        acc.add(rw.data[e * co * ci * fs + o * ci * fs + m] as f64
                            * tmp[m * out_dim + k] as f64);
                    }
                    out[(e * co + o) * out_dim + k] = acc.get() as f32;
                }
            }
        }
        out
    }
}

/// Which partial fusion a `ConvSE3` selected. Recorded rather than inferred so a
/// checkpoint whose channel layout changes fails loudly instead of loading a
/// differently-shaped `radial_func` into the wrong slot.
pub enum ConvFusion {
    PerInput(Vec<VersatileConv>),
    PerOutput(Vec<VersatileConv>),
}

pub struct ConvSE3 {
    pub fusion: ConvFusion,
    pub fiber_in: Fiber,
    pub fiber_out: Fiber,
    pub edge_channels: usize,
}

impl ConvSE3 {
    /// `fiber_edge` only ever carries degree 0 here, so the "concatenate edge
    /// features onto degree-`d` node features" branch is inactive for `d > 0`.
    pub fn load(p: &Params, fiber_in: &Fiber, fiber_out: &Fiber, max_degree: usize) -> Self {
        let degrees_up_to_max: Vec<usize> = (0..=max_degree).collect();
        let cin: Vec<usize> = fiber_in.degrees().iter().map(|&d| fiber_in.get(d)).collect();
        let cout: Vec<usize> = fiber_out.degrees().iter().map(|&d| fiber_out.get(d)).collect();
        let uniq_in = cin.iter().all(|c| *c == cin[0]);
        let uniq_out = cout.iter().all(|c| *c == cout[0]);

        if uniq_in && fiber_in.degrees() == degrees_up_to_max {
            // fused per OUTPUT degree
            let mut v = Vec::new();
            for &d_out in &fiber_out.degrees() {
                let sf: usize = fiber_in
                    .degrees()
                    .iter()
                    .map(|&d| degree_to_dim(d_out.min(d)))
                    .sum();
                v.push(VersatileConv::load(
                    &p.sub("conv_out").idx(d_out),
                    sf,
                    cin[0],
                    fiber_out.get(d_out),
                ));
            }
            ConvSE3 {
                fusion: ConvFusion::PerOutput(v),
                fiber_in: fiber_in.clone(),
                fiber_out: fiber_out.clone(),
                edge_channels: 0,
            }
        } else if uniq_out && fiber_out.degrees() == degrees_up_to_max {
            // fused per INPUT degree
            let mut v = Vec::new();
            for &d_in in &fiber_in.degrees() {
                let sf: usize = fiber_out
                    .degrees()
                    .iter()
                    .map(|&d| degree_to_dim(d_in.min(d)))
                    .sum();
                v.push(VersatileConv::load(
                    &p.sub("conv_in").idx(d_in),
                    sf,
                    fiber_in.get(d_in),
                    cout[0],
                ));
            }
            ConvSE3 {
                fusion: ConvFusion::PerInput(v),
                fiber_in: fiber_in.clone(),
                fiber_out: fiber_out.clone(),
                edge_channels: 0,
            }
        } else {
            panic!(
                "ConvSE3: neither partial fusion applies (in {:?}, out {:?}) — \
                 upstream would fall back to pairwise TFN convolutions, which \
                 this checkpoint never uses",
                fiber_in, fiber_out
            );
        }
    }

    /// Returns the **fused** edge output `[n_edges, c_out, sum_dim]`, which is
    /// what `AttentionBlockSE3` wants (`allow_fused_output=True`, `pool=False`,
    /// no self-interaction).
    pub fn forward_fused(
        &self,
        node_feats: &[Vec<f32>], // per degree, [N, C_d, dim(d)]
        graph: &Graph,
        basis: &Basis,
        edge_feats: &Tensor,
    ) -> Vec<f32> {
        let n = graph.n_edges();
        let sum_dim = basis.sum_dim;
        match &self.fusion {
            ConvFusion::PerInput(convs) => {
                let mut acc = vec![0.0f32; n * convs[0].channels_out * sum_dim];
                for (i, &d_in) in self.fiber_in.degrees().iter().enumerate() {
                    let li = degree_to_dim(d_in);
                    let c = self.fiber_in.get(d_in);
                    // gather source-node features onto edges
                    let mut feat = vec![0.0f32; n * c * li];
                    for e in 0..n {
                        let s = graph.src[e] as usize;
                        feat[e * c * li..(e + 1) * c * li]
                            .copy_from_slice(&node_feats[d_in][s * c * li..(s + 1) * c * li]);
                    }
                    let part = convs[i].forward(
                        &feat,
                        n,
                        li,
                        edge_feats,
                        &basis.in_fused[d_in],
                        sum_dim,
                    );
                    for (k, v) in acc.iter_mut().enumerate() {
                        *v += part[k];
                    }
                }
                acc
            }
            ConvFusion::PerOutput(convs) => {
                // inputs are concatenated along the last (degree) axis
                let cin = self.fiber_in.get(self.fiber_in.degrees()[0]);
                let mut fused = vec![0.0f32; n * cin * sum_dim];
                for e in 0..n {
                    let s = graph.src[e] as usize;
                    let mut off = 0;
                    for &d_in in &self.fiber_in.degrees() {
                        let li = degree_to_dim(d_in);
                        for c in 0..cin {
                            for l in 0..li {
                                fused[(e * cin + c) * sum_dim + off + l] =
                                    node_feats[d_in][(s * cin + c) * li + l];
                            }
                        }
                        off += li;
                    }
                }
                let mut out = vec![0.0f32; n * convs[0].channels_out * sum_dim];
                let mut acc_k = 0usize;
                for (i, &d_out) in self.fiber_out.degrees().iter().enumerate() {
                    let ko = degree_to_dim(d_out);
                    let co = convs[i].channels_out;
                    let part = convs[i].forward(
                        &fused,
                        n,
                        sum_dim,
                        edge_feats,
                        &basis.out_fused[d_out],
                        ko,
                    );
                    for e in 0..n {
                        for c in 0..co {
                            for k in 0..ko {
                                out[(e * co + c) * sum_dim + acc_k + k] =
                                    part[(e * co + c) * ko + k];
                            }
                        }
                    }
                    acc_k += ko;
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LinearSE3 / NormSE3 / AttentionSE3
// ---------------------------------------------------------------------------

pub struct LinearSE3 {
    pub weights: Vec<Option<Tensor>>, // by degree, [c_out, c_in]
}

impl LinearSE3 {
    pub fn load(p: &Params, fiber_out: &Fiber) -> Self {
        let maxd = fiber_out.channels.len();
        let mut weights = vec![None; maxd];
        for d in 0..maxd {
            if fiber_out.get(d) > 0 {
                weights[d] = Some(p.sub("weights").get(&d.to_string()));
            }
        }
        LinearSE3 { weights }
    }

    /// `w [c_out, c_in] @ x [N, c_in, dim]`, per degree, pinned.
    pub fn forward(&self, feats: &[Vec<f32>], n_nodes: usize) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::new(); self.weights.len()];
        for (d, w) in self.weights.iter().enumerate() {
            let Some(w) = w else { continue };
            let (co, ci) = (w.shape[0], w.shape[1]);
            let dim = degree_to_dim(d);
            let x = &feats[d];
            let mut o = vec![0.0f32; n_nodes * co * dim];
            for nnode in 0..n_nodes {
                for a in 0..co {
                    for k in 0..dim {
                        let mut acc = Acc::new();
                        for b in 0..ci {
                            acc.add(w.data[a * ci + b] as f64
                                * x[(nnode * ci + b) * dim + k] as f64);
                        }
                        o[(nnode * co + a) * dim + k] = acc.get() as f32;
                    }
                }
            }
            out[d] = o;
        }
        out
    }
}

pub struct NormSE3 {
    pub group_norm_w: Tensor,
    pub group_norm_b: Tensor,
    pub fiber: Fiber,
}

impl NormSE3 {
    /// Minimum positive subnormal for FP16 — upstream's `NORM_CLAMP`, applied
    /// even in fp32.
    const NORM_CLAMP: f32 = 1.0 / 16_777_216.0; // 2^-24, exactly representable
    const EPS: f64 = 1e-5;

    pub fn load(p: &Params, fiber: &Fiber) -> Self {
        NormSE3 {
            group_norm_w: p.sub("group_norm").get("weight"),
            group_norm_b: p.sub("group_norm").get("bias"),
            fiber: fiber.clone(),
        }
    }

    /// `x / ||x|| * relu(GroupNorm(||x||))`, with one group per degree.
    ///
    /// The `group_norm` branch is the live one because every hidden degree has
    /// the same channel count; the `layer_norms` fallback in upstream has its
    /// `rescale` arguments transposed and would be wrong if it ever ran.
    pub fn forward(&self, feats: &[Vec<f32>], n_nodes: usize) -> Vec<Vec<f32>> {
        let degrees = self.fiber.degrees();
        let ngroups = degrees.len();
        let total_c: usize = degrees.iter().map(|&d| self.fiber.get(d)).sum();
        // per-degree norms, concatenated along the channel axis
        let mut norms = vec![0.0f32; n_nodes * total_c];
        let mut off = 0usize;
        for &d in &degrees {
            let c = self.fiber.get(d);
            let dim = degree_to_dim(d);
            for nnode in 0..n_nodes {
                for ci in 0..c {
                    let mut acc = Acc::new();
                    for k in 0..dim {
                        let v = feats[d][(nnode * c + ci) * dim + k] as f64;
                        acc.add(v * v);
                    }
                    let v = acc.get().sqrt() as f32;
                    norms[nnode * total_c + off + ci] = v.max(Self::NORM_CLAMP);
                }
            }
            off += c;
        }
        // GroupNorm over `ngroups` contiguous channel groups
        let gsize = total_c / ngroups;
        let mut new_norms = vec![0.0f32; n_nodes * total_c];
        for nnode in 0..n_nodes {
            for g in 0..ngroups {
                let base = nnode * total_c + g * gsize;
                let mean = {
                    let mut a = Acc::new();
                    for i in 0..gsize {
                        a.add(norms[base + i] as f64);
                    }
                    a.get() / gsize as f64
                };
                let var = {
                    let mut a = Acc::new();
                    for i in 0..gsize {
                        let dv = norms[base + i] as f64 - mean;
                        a.add(dv * dv);
                    }
                    a.get() / gsize as f64
                };
                let rstd = 1.0 / (var + Self::EPS).sqrt();
                for i in 0..gsize {
                    let c = g * gsize + i;
                    // `F.group_norm` is pinned as a whole: the affine transform
                    // is part of the same op, so the single rounding to f32
                    // happens AFTER weight and bias, not before them.
                    let v = ((norms[base + i] as f64 - mean) * rstd
                        * self.group_norm_w.data[c] as f64
                        + self.group_norm_b.data[c] as f64) as f32;
                    new_norms[base + i] = crate::ops::relu_scalar(v);
                }
            }
        }
        let mut out = vec![Vec::new(); self.fiber.channels.len()];
        let mut off = 0usize;
        for &d in &degrees {
            let c = self.fiber.get(d);
            let dim = degree_to_dim(d);
            let mut o = vec![0.0f32; n_nodes * c * dim];
            for nnode in 0..n_nodes {
                for ci in 0..c {
                    let nrm = norms[nnode * total_c + off + ci];
                    let nn = new_norms[nnode * total_c + off + ci];
                    for k in 0..dim {
                        o[(nnode * c + ci) * dim + k] =
                            feats[d][(nnode * c + ci) * dim + k] / nrm * nn;
                    }
                }
            }
            out[d] = o;
            off += c;
        }
        out
    }
}

/// `dgl.ops.edge_softmax` — softmax over each destination node's incoming
/// edges, computed in f64 and rounded once (the op is patched in
/// `python/pinned.py`, which is what makes the edge order irrelevant).
fn edge_softmax(graph: &Graph, w: &[f32], heads: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    for edges in &graph.in_edges {
        if edges.is_empty() {
            continue;
        }
        for h in 0..heads {
            let mut m = f64::NEG_INFINITY;
            for &e in edges {
                let v = w[e as usize * heads + h] as f64;
                if v > m {
                    m = v;
                }
            }
            let s = {
                let mut a = Acc::new();
                for &e in edges {
                    a.add((w[e as usize * heads + h] as f64 - m).exp());
                }
                a.get()
            };
            for &e in edges {
                out[e as usize * heads + h] =
                    ((w[e as usize * heads + h] as f64 - m).exp() / s) as f32;
            }
        }
    }
    out
}

pub struct AttentionBlockSE3 {
    pub to_key_value: ConvSE3,
    pub to_query: LinearSE3,
    pub project: LinearSE3,
    pub fiber_in: Fiber,
    pub fiber_out: Fiber,
    pub value_fiber: Fiber,
    pub key_query_fiber: Fiber,
    pub num_heads: usize,
}

impl AttentionBlockSE3 {
    pub fn load(
        p: &Params,
        fiber_in: &Fiber,
        fiber_out: &Fiber,
        num_heads: usize,
        channels_div: usize,
        max_degree: usize,
    ) -> Self {
        let value_fiber = Fiber::new(
            &fiber_out
                .degrees()
                .iter()
                .map(|&d| (d, fiber_out.get(d) / channels_div))
                .collect::<Vec<_>>(),
        );
        let key_query_fiber = Fiber::new(
            &value_fiber
                .degrees()
                .iter()
                .filter(|d| fiber_in.get(**d) > 0)
                .map(|&d| (d, value_fiber.get(d)))
                .collect::<Vec<_>>(),
        );
        let kv_fiber = Fiber::new(
            &value_fiber
                .degrees()
                .iter()
                .map(|&d| (d, value_fiber.get(d) + key_query_fiber.get(d)))
                .collect::<Vec<_>>(),
        );
        AttentionBlockSE3 {
            to_key_value: ConvSE3::load(&p.sub("to_key_value"), fiber_in, &kv_fiber, max_degree),
            to_query: LinearSE3::load(&p.sub("to_query"), &key_query_fiber),
            project: LinearSE3::load(&p.sub("project"), fiber_out),
            fiber_in: fiber_in.clone(),
            fiber_out: fiber_out.clone(),
            value_fiber,
            key_query_fiber,
            num_heads,
        }
    }

    /// The block up to and including `AttentionSE3`, before the residual
    /// concatenation and `project`. Exists so the bisection can compare against
    /// the reference's captured `AttentionSE3` output.
    pub fn attention_only(
        &self,
        node_feats: &[Vec<f32>],
        graph: &Graph,
        basis: &Basis,
        edge_feats: &Tensor,
    ) -> Vec<Vec<f32>> {
        self.run(node_feats, graph, basis, edge_feats).0
    }

    pub fn forward(
        &self,
        node_feats: &[Vec<f32>],
        graph: &Graph,
        basis: &Basis,
        edge_feats: &Tensor,
    ) -> Vec<Vec<f32>> {
        let (z, n_nodes) = self.run(node_feats, graph, basis, edge_feats);
        // `aggregate_residual(node_features, z, 'cat')`: z first, input second
        let mut cat = vec![Vec::new(); z.len()];
        for d in 0..z.len() {
            if z[d].is_empty() {
                continue;
            }
            let dim = degree_to_dim(d);
            let cz = self.value_fiber.get(d);
            let cn = self.fiber_in.get(d);
            let mut o = vec![0.0f32; n_nodes * (cz + cn) * dim];
            for nnode in 0..n_nodes {
                let base = nnode * (cz + cn) * dim;
                o[base..base + cz * dim]
                    .copy_from_slice(&z[d][nnode * cz * dim..(nnode + 1) * cz * dim]);
                if cn > 0 {
                    o[base + cz * dim..base + (cz + cn) * dim].copy_from_slice(
                        &node_feats[d][nnode * cn * dim..(nnode + 1) * cn * dim],
                    );
                }
            }
            cat[d] = o;
        }
        self.project.forward(&cat, n_nodes)
    }

    fn run(
        &self,
        node_feats: &[Vec<f32>],
        graph: &Graph,
        basis: &Basis,
        edge_feats: &Tensor,
    ) -> (Vec<Vec<f32>>, usize) {
        let n_edges = graph.n_edges();
        let n_nodes = graph.n_nodes;
        let sum_dim = basis.sum_dim;
        let kv = self.to_key_value.forward_fused(node_feats, graph, basis, edge_feats);
        let c_kv = kv.len() / (n_edges * sum_dim);
        let half = c_kv / 2;
        // `torch.chunk(x, 2, dim=-2)` -> VALUE first, then KEY
        let heads = self.num_heads;
        let per_head = half * sum_dim / heads;

        let query = self.to_query.forward(node_feats, n_nodes);
        // cat over key_query degrees along the last axis, then split into heads
        let kq_degrees = self.key_query_fiber.degrees();
        let cq = self.key_query_fiber.get(kq_degrees[0]);
        let mut qflat = vec![0.0f32; n_nodes * cq * sum_dim];
        for nnode in 0..n_nodes {
            let mut off = 0;
            for &d in &kq_degrees {
                let dim = degree_to_dim(d);
                for c in 0..cq {
                    for k in 0..dim {
                        qflat[(nnode * cq + c) * sum_dim + off + k] =
                            query[d][(nnode * cq + c) * dim + k];
                    }
                }
                off += dim;
            }
        }

        // e_dot_v, then `/ np.sqrt(key_fiber.num_features)`, then softmax
        let sqrt_nf = (self.key_query_fiber.num_features() as f64).sqrt() as f32;
        let mut w = vec![0.0f32; n_edges * heads];
        for e in 0..n_edges {
            let d = graph.dst[e] as usize;
            for h in 0..heads {
                let mut acc = Acc::new();
                for j in 0..per_head {
                    let ki = e * c_kv * sum_dim + (half * sum_dim) + h * per_head + j;
                    let qi = d * cq * sum_dim + h * per_head + j;
                    acc.add(kv[ki] as f64 * qflat[qi] as f64);
                }
                w[e * heads + h] = (acc.get() as f32) / sqrt_nf;
            }
        }
        let w = edge_softmax(graph, &w, heads);

        // weighted sum of values over incoming edges
        let vch = half / heads;
        let mut fused = vec![0.0f32; n_nodes * half * sum_dim];
        for (dnode, edges) in graph.in_edges.iter().enumerate() {
            for h in 0..heads {
                for c in 0..vch {
                    for k in 0..sum_dim {
                        // `weights = edge_weights * v` is an ordinary fp32
                        // elementwise multiply; only the `copy_e_sum` that
                        // follows is pinned. Doing the product in f64 too is a
                        // different (and slightly more accurate) computation.
                        let mut acc = Acc::new();
                        for &e in edges {
                            let vi =
                                e as usize * c_kv * sum_dim + (h * vch + c) * sum_dim + k;
                            acc.add((w[e as usize * heads + h] * kv[vi]) as f64);
                        }
                        fused[(dnode * half + h * vch + c) * sum_dim + k] = acc.get() as f32;
                    }
                }
            }
        }

        // unfuse into degrees, then concatenate the residual and project
        let mut z = vec![Vec::new(); self.fiber_out.channels.len().max(sum_dim)];
        let mut off = 0usize;
        for &d in &self.value_fiber.degrees() {
            let dim = degree_to_dim(d);
            let c = self.value_fiber.get(d);
            let mut o = vec![0.0f32; n_nodes * c * dim];
            for nnode in 0..n_nodes {
                for ci in 0..c {
                    for k in 0..dim {
                        o[(nnode * c + ci) * dim + k] =
                            fused[(nnode * half + ci) * sum_dim + off + k];
                    }
                }
            }
            z[d] = o;
            off += dim;
        }
        (z, n_nodes)
    }
}

// ---------------------------------------------------------------------------
// SE3Transformer
// ---------------------------------------------------------------------------

pub struct Se3Transformer {
    pub blocks: Vec<(AttentionBlockSE3, NormSE3)>,
    pub final_lin: LinearSE3,
    pub fiber_out: Fiber,
    pub max_degree: usize,
}

impl Se3Transformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        num_layers: usize,
        fiber_in: &Fiber,
        fiber_hidden: &Fiber,
        fiber_out: &Fiber,
        num_heads: usize,
        channels_div: usize,
    ) -> Self {
        let max_degree = *[fiber_in.degrees(), fiber_hidden.degrees(), fiber_out.degrees()]
            .concat()
            .iter()
            .max()
            .unwrap();
        let gm = p.sub("graph_modules");
        let mut blocks = Vec::new();
        let mut fin = fiber_in.clone();
        for i in 0..num_layers {
            blocks.push((
                AttentionBlockSE3::load(
                    &gm.idx(2 * i),
                    &fin,
                    fiber_hidden,
                    num_heads,
                    channels_div,
                    max_degree,
                ),
                NormSE3::load(&gm.idx(2 * i + 1), fiber_hidden),
            ));
            fin = fiber_hidden.clone();
        }
        Se3Transformer {
            final_lin: LinearSE3::load(&gm.idx(2 * num_layers), fiber_out),
            blocks,
            fiber_out: fiber_out.clone(),
            max_degree,
        }
    }

    /// `edge_feats` is the pair-derived `[n_edges, C, 1]` block; the radial
    /// distance channel (`populate_edge='arcsin'`) is appended here.
    pub fn forward(
        &self,
        graph: &Graph,
        node_feats: &[Vec<f32>],
        edge_feats: &Tensor,
    ) -> Vec<Vec<f32>> {
        let n = graph.n_edges();
        let basis = build_basis(&graph.rel_pos, n, self.max_degree);
        let c = edge_feats.numel() / n;
        let mut ef = vec![0.0f32; n * (c + 1)];
        for e in 0..n {
            ef[e * (c + 1)..e * (c + 1) + c]
                .copy_from_slice(&edge_feats.data[e * c..(e + 1) * c]);
            // r = arcsinh(max(|rel_pos|, 4) - 4) / 3
            let (a, b, cc) = (
                graph.rel_pos[e * 3] as f64,
                graph.rel_pos[e * 3 + 1] as f64,
                graph.rel_pos[e * 3 + 2] as f64,
            );
            let r = (a * a + b * b + cc * cc).sqrt() as f32;
            let r = r.max(4.0) - 4.0;
            ef[e * (c + 1) + c] = crate::ops::elem::asinh_scalar(r) / 3.0;
        }
        let ef = Tensor::new(ef, vec![n, c + 1]);

        let mut feats: Vec<Vec<f32>> = node_feats.to_vec();
        for (attn, norm) in &self.blocks {
            feats = attn.forward(&feats, graph, &basis, &ef);
            feats = norm.forward(&feats, graph.n_nodes);
        }
        self.final_lin.forward(&feats, graph.n_nodes)
    }
}
