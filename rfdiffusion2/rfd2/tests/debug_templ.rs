//! Bisection harness for `templ_emb`, driven by `python/dump_io.py`'s
//! input+output capture of every submodule. Each stage is fed the reference's
//! own input, so the first stage that disagrees is the one with the bug rather
//! than the first one downstream of it.

use rfd2::geom;
use rfd2::model::attention::{Attention, BiasedAxialAttention, TriangleMultiplication};
use rfd2::model::embeddings::{PairStr2Pair, TemplEmb};
use rfd2::nn::{Ctx, FeedForward, LayerNorm, Linear, Params};
use rfd2::rng::torch::Mt19937;
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

/// The generator state captured on entry to the top-level module. `templ_emb`
/// is the first thing in the forward pass that consumes it — the embeddings and
/// recycling contain no dropout — so a Ctx built here is correctly positioned.
fn ctx_from(io: &Weights) -> Ctx {
    let bytes: Vec<u8> = io
        .get_i64("rng_state_at_model_entry")
        .0
        .into_iter()
        .map(|v| v as u8)
        .collect();
    Ctx::new(Mt19937::from_torch_state(&bytes))
}

fn chk(label: &str, got: &[f32], want: &[f32]) -> bool {
    let s = parity::compare(got, want);
    let ok = s.exact == s.n && got.len() == want.len();
    println!("{:<58} {}", label, s.summary());
    ok
}

#[test]
fn templ_bisect() {
    let Some(io) = open("fixtures/templ_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let p = Params::root(&w, "model").sub("templ_emb");

    let t1d = io.get("in::model.templ_emb.0");
    let t2d = io.get("in::model.templ_emb.1");
    let alpha_t = io.get("in::model.templ_emb.2");
    let xyz_t = io.get("in::model.templ_emb.3");
    let mask_t: Vec<bool> =
        io.get_i64("in::model.templ_emb.4").0.into_iter().map(|x| x != 0).collect();
    let (b, t, l) = (t1d.shape[0], t1d.shape[1], t1d.shape[2]);
    let bt = b * t;
    let d1 = t1d.last();
    let d2 = t2d.last();

    // ---- stage 1: emb(cat(t2d, left, right)) ------------------------------
    let w_cat = d2 + 2 * d1;
    let mut cat = vec![0.0f32; bt * l * l * w_cat];
    for x in 0..bt {
        for i in 0..l {
            for j in 0..l {
                let o = ((x * l + i) * l + j) * w_cat;
                cat[o..o + d2].copy_from_slice(&t2d.data[((x * l + i) * l + j) * d2..][..d2]);
                cat[o + d2..o + d2 + d1].copy_from_slice(&t1d.data[(x * l + i) * d1..][..d1]);
                cat[o + d2 + d1..o + w_cat].copy_from_slice(&t1d.data[(x * l + j) * d1..][..d1]);
            }
        }
    }
    let cat = Tensor::new(cat, vec![bt, l, l, w_cat]);
    chk("cat -> emb input", &cat.data, &io.get("in::model.templ_emb.emb.0").data);
    let emb = Linear::load(&p.sub("emb"));
    let templ = emb.forward(&cat);
    chk("emb", &templ.data, &io.get("out::model.templ_emb.emb").data);

    // ---- stage 2: rbf feature ---------------------------------------------
    let mut rbf_all = vec![0.0f32; bt * l * l * geom::D_COUNT];
    for x in 0..bt {
        let pts = &xyz_t.data[x * l * 3..(x + 1) * l * 3];
        let d = geom::cdist_self(pts, l);
        let dst = &mut rbf_all[x * l * l * geom::D_COUNT..(x + 1) * l * l * geom::D_COUNT];
        geom::rbf_into(&d, dst);
        for k in 0..l * l {
            if !mask_t[x * l * l + k] {
                for c in 0..geom::D_COUNT {
                    dst[k * geom::D_COUNT + c] = 0.0;
                }
            }
        }
    }
    let rbf_feat = Tensor::new(rbf_all, vec![bt, l, l, geom::D_COUNT]);
    chk("rbf_feat (vs block.0 arg 1)", &rbf_feat.data,
        &io.get("in::model.templ_emb.templ_stack.block.0.1").data);

    // ---- stage 3: templ_stack ---------------------------------------------
    let t1d_flat = Tensor::new(t1d.data.clone(), vec![bt, l, d1]);
    let proj_t1d = Linear::load(&p.sub("templ_stack").sub("proj_t1d"));
    let state = proj_t1d.forward(&t1d_flat);
    chk("templ_stack.proj_t1d", &state.data,
        &io.get("out::model.templ_emb.templ_stack.proj_t1d").data);

    // block 0, piece by piece
    let bp = p.sub("templ_stack").sub("block").idx(0);
    let pair_in = io.get("in::model.templ_emb.templ_stack.block.0.0");
    let emb_rbf = Linear::load(&bp.sub("emb_rbf"));
    let rbf_p = emb_rbf.forward(&rbf_feat);
    chk("block0.emb_rbf", &rbf_p.data,
        &io.get("out::model.templ_emb.templ_stack.block.0.emb_rbf").data);

    let ns = LayerNorm::load(&bp.sub("norm_state"));
    let stn = ns.forward(&state);
    chk("block0.norm_state", &stn.data,
        &io.get("out::model.templ_emb.templ_stack.block.0.norm_state").data);
    let pl = Linear::load(&bp.sub("proj_left"));
    let left = pl.forward(&stn);
    chk("block0.proj_left", &left.data,
        &io.get("out::model.templ_emb.templ_stack.block.0.proj_left").data);

    // --- inside tri_mul_out, against the reference's own submodule captures --
    {
        let q = bp.sub("tri_mul_out");
        let pre = "out::model.templ_emb.templ_stack.block.0.tri_mul_out";
        let norm = LayerNorm::load(&q.sub("norm"));
        let np = norm.forward(&pair_in);
        chk("  tmo.norm", &np.data, &io.get(&format!("{pre}.norm")).data);
        let lp = Linear::load(&q.sub("left_proj"));
        let l0 = lp.forward(&np);
        chk("  tmo.left_proj", &l0.data, &io.get(&format!("{pre}.left_proj")).data);
        let lg = Linear::load(&q.sub("left_gate"));
        let g0 = lg.forward(&np);
        chk("  tmo.left_gate", &g0.data, &io.get(&format!("{pre}.left_gate")).data);
        let rp = Linear::load(&q.sub("right_proj"));
        let r0 = rp.forward(&np);
        let rg = Linear::load(&q.sub("right_gate"));
        let rg0 = rg.forward(&np);
        let dh = l0.last();
        let mut left = l0.clone();
        for (i, v) in left.data.iter_mut().enumerate() {
            *v *= rfd2::ops::elem::sigmoid_scalar(g0.data[i]);
        }
        let mut right = r0.clone();
        for (i, v) in right.data.iter_mut().enumerate() {
            *v *= rfd2::ops::elem::sigmoid_scalar(rg0.data[i]);
        }
        for v in right.data.iter_mut() {
            *v /= l as f32;
        }
        let mut out = vec![0.0f32; bt * l * l * dh];
        for x in 0..bt {
            for i in 0..l {
                for j in 0..l {
                    for d in 0..dh {
                        let mut acc = 0.0f64;
                        for k in 0..l {
                            acc += left.data[((x * l + i) * l + k) * dh + d] as f64
                                * right.data[((x * l + j) * l + k) * dh + d] as f64;
                        }
                        out[((x * l + i) * l + j) * dh + d] = acc as f32;
                    }
                }
            }
        }
        chk("  tmo.einsum (vs norm_out input)", &out,
            &io.get(&format!("in::model.templ_emb.templ_stack.block.0.tri_mul_out.norm_out.0")).data);
    }

    let tmo = TriangleMultiplication::load(&bp.sub("tri_mul_out"), true);
    let d = tmo.forward(&pair_in);
    chk("block0.tri_mul_out", &d.data,
        &io.get("out::model.templ_emb.templ_stack.block.0.tri_mul_out").data);

    // From here on the block is run whole, because dropout makes the stages
    // order-dependent through the shared RNG as well as through the residual.
    let mut ctx = ctx_from(&io);
    let full = PairStr2Pair::load(&bp, 4, 64, 0.25f64);
    let got = full.forward(&pair_in, &rbf_feat, &state, &mut ctx);
    chk("block0 output (with dropout)", &got.data,
        &io.get("out::model.templ_emb.templ_stack.block.0").data);

    // ---- stage 4: the two pointwise attentions ----------------------------
    let attn_tor = Attention::load(&p.sub("attn_tor"), 4, 64);
    let q = io.get("in::model.templ_emb.attn_tor.0");
    let k = io.get("in::model.templ_emb.attn_tor.1");
    let got = attn_tor.forward(&q, &k, &k);
    chk("attn_tor (reference inputs)", &got.data, &io.get("out::model.templ_emb.attn_tor").data);

    let attn = Attention::load(&p.sub("attn"), 4, 64);
    let q = io.get("in::model.templ_emb.attn.0");
    let k = io.get("in::model.templ_emb.attn.1");
    let got = attn.forward(&q, &k, &k);
    chk("attn (reference inputs)", &got.data, &io.get("out::model.templ_emb.attn").data);

    // ---- the whole module, from the reference's own inputs ----------------
    let te = TemplEmb::load(&p, 4, 64, 2);
    let mut ctx = ctx_from(&io);
    let (pair_o, state_o) = te.forward(
        &t1d,
        &t2d,
        &alpha_t,
        &xyz_t,
        &mask_t,
        &io.get("in::model.templ_emb.5"),
        &io.get("in::model.templ_emb.6"),
        &mut ctx,
    );
    assert!(chk("templ_emb.pair", &pair_o.data, &io.get("out::model.templ_emb.0").data));
    assert!(chk("templ_emb.state", &state_o.data, &io.get("out::model.templ_emb.1").data));
}
