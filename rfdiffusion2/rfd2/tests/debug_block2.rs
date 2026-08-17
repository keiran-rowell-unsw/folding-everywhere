//! Why does `extra_block.2` disagree at exactly one node when blocks 0, 1 and 3
//! are bit-exact? This walks `Str2Str` stage by stage for that one block.

use rfd2::model::str2str::Str2Str;
use rfd2::nn::{Ctx, LayerNorm, Linear, Params};
use rfd2::rng::torch::Mt19937;
use rfd2::{geom, parity};
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
    println!("{:<32} {}", label, s.summary());
    if s.exact != s.n {
        let bad: Vec<usize> = got
            .iter()
            .zip(want)
            .enumerate()
            .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
            .map(|(i, _)| i)
            .collect();
        println!("      {} differ, first at {} ({} vs {})", bad.len(), bad[0],
                 got[bad[0]], want[bad[0]]);
    }
}

const S: &str = "model.simulator.extra_block.2.str2str";

#[test]
fn block2_str2str() {
    let Some(io) = open("fixtures/b2_io/io.safetensors") else { return };
    let Some(blocks) = open("fixtures/blocks_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let p = Params::root(&w, "model").sub("simulator").sub("extra_block").sub("2").sub("str2str");

    let msa = io.get(&format!("in::{S}.0"));
    let pair = io.get(&format!("in::{S}.1"));
    let xyz = io.get(&format!("in::{S}.2"));
    let state = io.get(&format!("in::{S}.3"));
    let l = state.shape[1];
    let idx = io.get_i64(&format!("in::{S}.4")).0;
    let bond_feats = io.get_i64(&format!("in::{S}.6")).0;
    let dist_matrix = step.get("rfi.dist_matrix").data;
    let chirals = step.get("rfi.chirals").data;
    let seq_unmasked = blocks.get_i64("in::model.simulator.extra_block.2.4").0;
    let rotation_mask: Vec<bool> = seq_unmasked.iter().map(|&t| geom::is_atom(t)).collect();

    // node branch
    let d_msa = msa.last();
    let seq = LayerNorm::load(&p.sub("norm_msa"))
        .forward(&Tensor::new(msa.data[..l * d_msa].to_vec(), vec![1, l, d_msa]));
    let st = LayerNorm::load(&p.sub("norm_state")).forward(&state);
    let ds = st.last();
    let mut cat = vec![0.0f32; l * (d_msa + ds)];
    for i in 0..l {
        cat[i * (d_msa + ds)..i * (d_msa + ds) + d_msa]
            .copy_from_slice(&seq.data[i * d_msa..(i + 1) * d_msa]);
        cat[i * (d_msa + ds) + d_msa..(i + 1) * (d_msa + ds)]
            .copy_from_slice(&st.data[i * ds..(i + 1) * ds]);
    }
    chk("embed_node input", &cat, &io.get(&format!("in::{S}.embed_node.0")).data);
    let node0 = Linear::load(&p.sub("embed_node"))
        .forward(&Tensor::new(cat, vec![1, l, d_msa + ds]));
    chk("embed_node", &node0.data, &io.get(&format!("out::{S}.embed_node")).data);

    let bytes: Vec<u8> =
        io.get_i64(&format!("rng::{S}")).0.into_iter().map(|v| v as u8).collect();
    let mut ctx = Ctx::new(Mt19937::from_torch_state(&bytes));
    let ffn = rfd2::nn::FeedForward::load(&p.sub("ff_node"), 0.15);
    let d = ffn.forward(&node0, &mut ctx);
    chk("ff_node", &d.data, &io.get(&format!("out::{S}.ff_node")).data);
    let mut node = node0.clone();
    for (i, v) in node.data.iter_mut().enumerate() {
        *v += d.data[i];
    }
    let node = LayerNorm::load(&p.sub("norm_node")).forward(&node);
    chk("norm_node", &node.data, &io.get(&format!("out::{S}.norm_node")).data);

    // the SE(3) inputs the reference actually built
    chk("se3 node l0", &node.data, &io.get(&format!("in::{S}.se3.1")).data);
    let frames = Str2Str::xyz_frame(
        &xyz.data,
        l,
        3,
        &rotation_mask,
        &step.get_i64("rfi.atom_frames").0,
    );
    let extra_l1 = rfd2::chiral::chiral_grads(&xyz.data, l, 3, &chirals);
    let mut l1 = vec![0.0f32; l * 6 * 3];
    for i in 0..l {
        let mid = [frames[i * 9 + 3], frames[i * 9 + 4], frames[i * 9 + 5]];
        for a in 0..3 {
            for k in 0..3 {
                l1[(i * 6 + a) * 3 + k] = frames[(i * 3 + a) * 3 + k] - mid[k];
            }
        }
        for a in 0..3 {
            for k in 0..3 {
                l1[(i * 6 + 3 + a) * 3 + k] = extra_l1[(i * 3 + a) * 3 + k];
            }
        }
    }
    chk("se3 node l1", &l1, &io.get(&format!("in::{S}.se3.2")).data);

    // edge branch
    let neighbor = geom::seqsep_protein_sm(&idx, &bond_feats, &dist_matrix, &rotation_mask);
    let ca: Vec<f32> = (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let rbf = geom::rbf_ca(&ca, l);
    let pair_n = LayerNorm::load(&p.sub("norm_pair")).forward(&pair);
    let dp = pair_n.last();
    let wid = dp + geom::D_COUNT + 1;
    let mut edge = vec![0.0f32; l * l * wid];
    for k in 0..l * l {
        let o = k * wid;
        edge[o..o + dp].copy_from_slice(&pair_n.data[k * dp..(k + 1) * dp]);
        edge[o + dp..o + dp + geom::D_COUNT]
            .copy_from_slice(&rbf.data[k * geom::D_COUNT..(k + 1) * geom::D_COUNT]);
        edge[o + wid - 1] = neighbor[k];
    }
    chk("embed_edge input", &edge, &io.get(&format!("in::{S}.embed_edge.0")).data);
    let mut edge = Linear::load(&p.sub("embed_edge"))
        .forward(&Tensor::new(edge, vec![1, l, l, wid]));
    chk("embed_edge", &edge.data, &io.get(&format!("out::{S}.embed_edge")).data);
    let ffe = rfd2::nn::FeedForward::load(&p.sub("ff_edge"), 0.15);
    let d = ffe.forward(&edge, &mut ctx);
    chk("ff_edge", &d.data, &io.get(&format!("out::{S}.ff_edge")).data);
    for (i, v) in edge.data.iter_mut().enumerate() {
        *v += d.data[i];
    }
    let edge = LayerNorm::load(&p.sub("norm_edge")).forward(&edge);
    chk("norm_edge", &edge.data, &io.get(&format!("out::{S}.norm_edge")).data);

    // gather onto the graph's edges and run the transformer
    let de = edge.last();
    let graph = rfd2::model::se3::make_full_graph(&ca, &idx);
    let ne = graph.n_edges();
    let mut ef = vec![0.0f32; ne * de];
    for e in 0..ne {
        let (i, j) = (graph.src[e] as usize, graph.dst[e] as usize);
        ef[e * de..(e + 1) * de]
            .copy_from_slice(&edge.data[(i * l + j) * de..(i * l + j + 1) * de]);
    }
    chk("se3 edge feats", &ef, &io.get(&format!("in::{S}.se3.3")).data);

    use rfd2::model::se3::{Fiber, Se3Transformer};
    let t = Se3Transformer::load(
        &p.sub("se3").sub("se3"),
        1,
        &Fiber::new(&[(0, 64), (1, 6)]),
        &Fiber::new(&[(0, 32), (1, 32)]),
        &Fiber::new(&[(0, 64), (1, 2)]),
        4,
        4,
    );
    let out = t.forward(&graph, &[node.data.clone(), l1.clone()],
                        &Tensor::new(ef, vec![ne, de]));
    chk("se3['0']", &out[0], &io.get(&format!("out::{S}.se3.0")).data);
    chk("se3['1']", &out[1], &io.get(&format!("out::{S}.se3.1")).data);
}
