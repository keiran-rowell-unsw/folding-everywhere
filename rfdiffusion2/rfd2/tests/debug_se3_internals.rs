//! Bisection *inside* the SE(3) transformer, against `python/dump_se3.py`'s
//! capture of the pieces that are plain functions rather than modules: the
//! spherical harmonics, the CG bases, the fused basis views, each
//! `VersatileConvSE3`, the attention, and `NormSE3`.

use rfd2::model::se3::{self, Fiber, Se3Transformer};
use rfd2::nn::Params;
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

fn chk(label: &str, got: &[f32], want: &[f32]) -> bool {
    let s = parity::compare(got, want);
    println!("{:<34} {}", label, s.summary());
    s.exact == s.n && got.len() == want.len()
}

#[test]
fn se3_internals() {
    let Some(io) = open("fixtures/se3_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let p = Params::root(&w, "model")
        .sub("simulator")
        .sub("extra_block")
        .sub("0")
        .sub("str2str")
        .sub("se3")
        .sub("se3");

    let rel_pos = io.get("rel_pos");
    let n = rel_pos.shape[0];

    // ---- spherical harmonics ---------------------------------------------
    let sh = se3::spherical_harmonics(&rel_pos.data, n, 2);
    for (j, s) in sh.iter().enumerate() {
        chk(&format!("sh[{j}]"), s, &io.get(&format!("sh.{j}")).data);
    }

    // ---- fused bases ------------------------------------------------------
    let basis = se3::build_basis(&rel_pos.data, n, 1);
    chk("basis.in0_fused", &basis.in_fused[0], &io.get("basis.in0_fused").data);
    chk("basis.in1_fused", &basis.in_fused[1], &io.get("basis.in1_fused").data);
    chk("basis.out0_fused", &basis.out_fused[0], &io.get("basis.out0_fused").data);
    chk("basis.out1_fused", &basis.out_fused[1], &io.get("basis.out1_fused").data);

    // ---- the two VersatileConvSE3 calls of the first block ----------------
    let kv = p.sub("graph_modules").idx(0).sub("to_key_value");
    for (i, d_in) in [(0usize, 0usize), (1, 1)] {
        let vc = se3::VersatileConv::load(
            &kv.sub("conv_in").idx(d_in),
            basis.in_freq[d_in],
            if d_in == 0 { 64 } else { 6 },
            16,
        );
        let feats = io.get(&format!("vconv{i}.features"));
        let edge = io.get(&format!("vconv{i}.edge"));
        let radial = vc.radial.forward(&edge);
        chk(&format!("vconv{i}.radial"), &radial.data, &io.get(&format!("vconv{i}.radial")).data);
        let out = vc.forward(
            &feats.data,
            n,
            se3::degree_to_dim(d_in),
            &edge,
            &basis.in_fused[d_in],
            basis.sum_dim,
        );
        chk(&format!("vconv{i}.out"), &out, &io.get(&format!("vconv{i}.out")).data);
    }

    // ---- the fused key/value, and the chunk order ------------------------
    {
        let bio = open("fixtures/extra0_io/io.safetensors").unwrap();
        let s = "model.simulator.extra_block.0.str2str";
        let xyz = bio.get(&format!("in::{s}.2"));
        let l = xyz.shape[1];
        let idx = bio.get_i64(&format!("in::{s}.4")).0;
        let ca: Vec<f32> =
            (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
        let graph = se3::make_full_graph(&ca, &idx);
        let edge = bio.get(&format!("in::{s}.se3.3"));
        // rebuild the invariant edge features the way the transformer does
        let c = edge.numel() / graph.n_edges();
        let mut ef = vec![0.0f32; graph.n_edges() * (c + 1)];
        for e in 0..graph.n_edges() {
            ef[e * (c + 1)..e * (c + 1) + c]
                .copy_from_slice(&edge.data[e * c..(e + 1) * c]);
            let (a, b, cc) = (
                graph.rel_pos[e * 3] as f64,
                graph.rel_pos[e * 3 + 1] as f64,
                graph.rel_pos[e * 3 + 2] as f64,
            );
            let r = (a * a + b * b + cc * cc).sqrt() as f32;
            let r = r.max(4.0) - 4.0;
            ef[e * (c + 1) + c] = rfd2::ops::elem::asinh_scalar(r) / 3.0;
        }
        let ef = Tensor::new(ef, vec![graph.n_edges(), c + 1]);
        chk("edge feats (vs vconv0.edge)", &ef.data, &io.get("vconv0.edge").data);

        let conv = se3::ConvSE3::load(&kv, &Fiber::new(&[(0, 64), (1, 6)]),
                                      &Fiber::new(&[(0, 16), (1, 16)]), 1);
        let node = [
            bio.get(&format!("in::{s}.se3.1")).data,
            bio.get(&format!("in::{s}.se3.2")).data,
        ];
        let fused = conv.forward_fused(&node, &graph, &basis, &ef);
        let ne = graph.n_edges();
        let sd = basis.sum_dim;
        let value: Vec<f32> = (0..ne)
            .flat_map(|e| fused[e * 16 * sd..e * 16 * sd + 8 * sd].to_vec())
            .collect();
        let key: Vec<f32> = (0..ne)
            .flat_map(|e| fused[e * 16 * sd + 8 * sd..(e + 1) * 16 * sd].to_vec())
            .collect();
        chk("kv chunk -> value", &value, &io.get("attn.value").data);
        chk("kv chunk -> key", &key, &io.get("attn.key").data);

        // to_query, then the attention itself
        let kq = Fiber::new(&[(0, 8), (1, 8)]);
        let toq = se3::LinearSE3::load(&p.sub("graph_modules").idx(0).sub("to_query"), &kq);
        let q = toq.forward(&node, l);
        chk("to_query['0']", &q[0], &io.get("attn.query.0").data);
        chk("to_query['1']", &q[1], &io.get("attn.query.1").data);

        let blk = se3::AttentionBlockSE3::load(
            &p.sub("graph_modules").idx(0),
            &Fiber::new(&[(0, 64), (1, 6)]),
            &Fiber::new(&[(0, 32), (1, 32)]),
            4,
            4,
            1,
        );
        let z = blk.attention_only(&node, &graph, &basis, &ef);
        chk("attention out['0']", &z[0], &io.get("attn.out.0").data);
        chk("attention out['1']", &z[1], &io.get("attn.out.1").data);

        // the residual concat + project — its output is what NormSE3 receives
        let pj = blk.forward(&node, &graph, &basis, &ef);
        chk("project['0'] (vs norm.in.0)", &pj[0], &io.get("norm.in.0").data);
        chk("project['1'] (vs norm.in.1)", &pj[1], &io.get("norm.in.1").data);
    }

    // ---- NormSE3 ----------------------------------------------------------
    let fiber_hidden = Fiber::new(&[(0, 32), (1, 32)]);
    let nrm = se3::NormSE3::load(&p.sub("graph_modules").idx(1), &fiber_hidden);
    let nin = vec![io.get("norm.in.0").data, io.get("norm.in.1").data];
    let nout = nrm.forward(&nin, io.get("norm.in.0").shape[0]);
    chk("norm.out.0", &nout[0], &io.get("norm.out.0").data);
    chk("norm.out.1", &nout[1], &io.get("norm.out.1").data);

    // final LinearSE3, fed the reference's own NormSE3 output
    {
        let bio = open("fixtures/extra0_io/io.safetensors").unwrap();
        let s = "model.simulator.extra_block.0.str2str";
        let fin = se3::LinearSE3::load(&p.sub("graph_modules").idx(2),
                                       &Fiber::new(&[(0, 64), (1, 2)]));
        let refnorm = vec![io.get("norm.out.0").data, io.get("norm.out.1").data];
        let o = fin.forward(&refnorm, io.get("norm.out.0").shape[0]);
        chk("final_lin['0']", &o[0], &bio.get(&format!("out::{s}.se3.0")).data);
        chk("final_lin['1']", &o[1], &bio.get(&format!("out::{s}.se3.1")).data);

        // hypothesis check: is the reference doing this matmul in fp32?
        let wl = w.get("model.simulator.extra_block.0.str2str.se3.se3.graph_modules.2.weights.0");
        let (co, ci) = (wl.shape[0], wl.shape[1]);
        let nn_ = io.get("norm.out.0").shape[0];
        let x = &refnorm[0];
        let mut f32acc = vec![0.0f32; nn_ * co];
        for nd in 0..nn_ {
            for a in 0..co {
                let mut acc = 0.0f32;
                for b in 0..ci {
                    acc += wl.data[a * ci + b] * x[nd * ci + b];
                }
                f32acc[nd * co + a] = acc;
            }
        }
        chk("final_lin['0'] fp32 seq", &f32acc, &bio.get(&format!("out::{s}.se3.0")).data);
    }

    // ---- whole transformer, for context ----------------------------------
    let Some(bio) = open("fixtures/extra0_io/io.safetensors") else { return };
    let s = "model.simulator.extra_block.0.str2str";
    let xyz = bio.get(&format!("in::{s}.2"));
    let l = xyz.shape[1];
    let idx = bio.get_i64(&format!("in::{s}.4")).0;
    let ca: Vec<f32> = (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let graph = se3::make_full_graph(&ca, &idx);
    chk("rel_pos (rebuilt graph)", &graph.rel_pos, &rel_pos.data);
    let edge = bio.get(&format!("in::{s}.se3.3"));
    let t = Se3Transformer::load(
        &p,
        1,
        &Fiber::new(&[(0, 64), (1, 6)]),
        &fiber_hidden,
        &Fiber::new(&[(0, 64), (1, 2)]),
        4,
        4,
    );
    let ef = Tensor::new(edge.data.clone(), vec![graph.n_edges(), edge.shape[1]]);
    let out = t.forward(
        &graph,
        &[bio.get(&format!("in::{s}.se3.1")).data, bio.get(&format!("in::{s}.se3.2")).data],
        &ef,
    );
    chk("se3['0']", &out[0], &bio.get(&format!("out::{s}.se3.0")).data);
    chk("se3['1']", &out[1], &bio.get(&format!("out::{s}.se3.1")).data);
}
