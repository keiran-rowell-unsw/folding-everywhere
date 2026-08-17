//! The last non-exact stage in the trunk: `main_block.0`'s `row_attn`, which
//! differs at 299 of 967 872 values by at most 1 ULP. Every differing cell
//! shares one query index, so this walks the four projections and the two
//! contractions to find which one carries it.

use rfd2::model::attention::scaling;
use rfd2::nn::{LayerNorm, Linear, Params};
use rfd2::ops::elem::{sigmoid_scalar, softmax_dim};
use rfd2::parity;
use rfd2::tensor::Tensor;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

fn chk(label: &str, got: &[f32], want: &[f32]) {
    let s = parity::compare(got, want);
    println!("{:<22} {:>8} values, {:>4} differ, max|d| {:.3e}", label, s.n, s.n - s.exact, s.max_abs);
}

const R: &str = "model.simulator.main_block.0.pair2pair.row_attn";

#[test]
fn row_attn_stages() {
    let Some(io) = open("fixtures/m0ra_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let p = Params::root(&w, "model")
        .sub("simulator")
        .sub("main_block")
        .sub("0")
        .sub("pair2pair")
        .sub("row_attn");

    let pair = io.get(&format!("in::{R}.0"));
    let bias = io.get(&format!("in::{R}.1"));
    let l = pair.shape[1];
    let (h, dim) = (6usize, 32usize);

    let pr = pair.permute(&[0, 2, 1, 3]);
    let br = bias.permute(&[0, 2, 1, 3]);
    let pn = LayerNorm::load(&p.sub("norm_pair")).forward(&pr);
    let bn = LayerNorm::load(&p.sub("norm_bias")).forward(&br);
    chk("norm_pair", &pn.data, &io.get(&format!("out::{R}.norm_pair")).data);
    chk("norm_bias", &bn.data, &io.get(&format!("out::{R}.norm_bias")).data);

    let mut q = Linear::load_nobias(&p.sub("to_q")).forward(&pn);
    let mut k = Linear::load_nobias(&p.sub("to_k")).forward(&pn);
    let v = Linear::load_nobias(&p.sub("to_v")).forward(&pn);
    let bb = Linear::load_nobias(&p.sub("to_b")).forward(&bn);
    let g = Linear::load(&p.sub("to_g")).forward(&pn);
    chk("to_q", &q.data, &io.get(&format!("out::{R}.to_q")).data);
    {
        let want = io.get(&format!("out::{R}.to_k")).data;
        let k0 = Linear::load_nobias(&p.sub("to_k")).forward(&pn);
        chk("to_k", &k0.data, &want);
        if let Some(bad) = k0.data.iter().zip(&want).position(|(g, w)| g.to_bits() != w.to_bits()) {
            // Recompute that one output several different ways, all in f64.
            // If the fp32 answer moves, the exact value straddles an fp32
            // midpoint and no f64 implementation can be relied on to agree.
            let wt = Linear::load_nobias(&p.sub("to_k")).weight;
            let kk = wt.shape[1];
            let (row, col) = (bad / wt.shape[0], bad % wt.shape[0]);
            let x = &pn.data[row * kk..row * kk + kk];
            let wrow = &wt.data[col * kk..col * kk + kk];
            let fwd: f64 = (0..kk).map(|i| x[i] as f64 * wrow[i] as f64).sum();
            let rev: f64 = (0..kk).rev().map(|i| x[i] as f64 * wrow[i] as f64).sum();
            let mut blocked = 0.0f64;
            for c in (0..kk).step_by(8) {
                let mut part = 0.0f64;
                for i in c..(c + 8).min(kk) {
                    part += x[i] as f64 * wrow[i] as f64;
                }
                blocked += part;
            }
            println!(
                "  to_k[{row},{col}]: want {:.9e}\n    fwd f64 {:.20e} -> {:e}\n    rev f64 {:.20e} -> {:e}\n    8-blocked {:.20e} -> {:e}",
                want[bad], fwd, fwd as f32, rev, rev as f32, blocked, blocked as f32
            );
            // Exact accumulation: the products of two fp32 values are exact in
            // f64, so a compensated (Neumaier) sum gives the correctly-rounded
            // fp32 of the *exact* dot product. If the reference agrees with that,
            // the remaining error is MKL's alone and matching it is possible.
            {
                let mut sum = 0.0f64;
                let mut comp = 0.0f64;
                for i in 0..kk {
                    let v = x[i] as f64 * wrow[i] as f64;
                    let t = sum + v;
                    comp += if sum.abs() >= v.abs() { (sum - t) + v } else { (v - t) + sum };
                    sum = t;
                }
                let exact = sum + comp;
                println!(
                    "    exact(KBN) {:.20e} -> {:e}  {}",
                    exact,
                    exact as f32,
                    if (exact as f32).to_bits() == want[bad].to_bits() { "MATCH" } else { "differs" }
                );
            }

            // lane-parallel variants, the shape a vectorised f64 kernel uses
            for lanes in [2usize, 4, 8] {
                let mut acc = vec![0.0f64; lanes];
                let n = kk / lanes;
                for c in 0..n {
                    for l2 in 0..lanes {
                        acc[l2] += x[c * lanes + l2] as f64 * wrow[c * lanes + l2] as f64;
                    }
                }
                // pairwise tree reduction of the lanes
                let mut v = acc.clone();
                let mut m = lanes;
                while m > 1 {
                    for i in 0..m / 2 {
                        v[i] = v[i] + v[i + m / 2];
                    }
                    m /= 2;
                }
                let mut tot = v[0];
                for i in n * lanes..kk {
                    tot += x[i] as f64 * wrow[i] as f64;
                }
                println!(
                    "    {lanes}-lane  {:.20e} -> {:e}  {}",
                    tot,
                    tot as f32,
                    if (tot as f32).to_bits() == want[bad].to_bits() { "MATCH" } else { "" }
                );
            }
        }
    }
    chk("to_v", &v.data, &io.get(&format!("out::{R}.to_v")).data);
    chk("to_b", &bb.data, &io.get(&format!("out::{R}.to_b")).data);
    chk("to_g", &g.data, &io.get(&format!("out::{R}.to_g")).data);

    let s = scaling(dim);
    for x in q.data.iter_mut() {
        *x *= s;
    }
    for x in k.data.iter_mut() {
        *x /= l as f32;
    }
    let mut attn = vec![0.0f32; l * l * h];
    for i in 0..l {
        for j in 0..l {
            for hh in 0..h {
                let mut acc = 0.0f64;
                for ni in 0..l {
                    let qo = (((ni) * l + i) * h + hh) * dim;
                    let ko = (((ni) * l + j) * h + hh) * dim;
                    for d in 0..dim {
                        acc += q.data[qo + d] as f64 * k.data[ko + d] as f64;
                    }
                }
                attn[((i * l) + j) * h + hh] = acc as f32;
            }
        }
    }
    for (i, a) in attn.iter_mut().enumerate() {
        *a += bb.data[i];
    }
    let attn = softmax_dim(&Tensor::new(attn, vec![1, l, l, h]), 2);

    let mut out = vec![0.0f32; l * l * h * dim];
    for ni in 0..l {
        for i in 0..l {
            for hh in 0..h {
                for d in 0..dim {
                    let mut acc = 0.0f64;
                    for j in 0..l {
                        acc += attn.data[((i * l) + j) * h + hh] as f64
                            * v.data[(((ni * l) + j) * h + hh) * dim + d] as f64;
                    }
                    out[(((ni * l) + i) * h + hh) * dim + d] = acc as f32;
                }
            }
        }
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o *= sigmoid_scalar(g.data[i]);
    }
    // `to_out`'s captured input is exactly this gated tensor
    chk("gated out (to_out input)", &out, &io.get(&format!("in::{R}.to_out.0")).data);
    let o = Linear::load(&p.sub("to_out")).forward(&Tensor::new(out, vec![1, l, l, h * dim]));
    chk("to_out", &o.data, &io.get(&format!("out::{R}.to_out")).data);
}
