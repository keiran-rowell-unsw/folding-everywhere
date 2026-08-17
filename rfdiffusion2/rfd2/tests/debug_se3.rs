//! Bisection inside `Str2Str` — the chiral gradients, the node/edge embedding
//! and the SE(3) transformer itself, each against the reference's own captured
//! inputs.

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
    println!("{:<44} {}", label, s.summary());
    s.exact == s.n && got.len() == want.len()
}

const S: &str = "model.simulator.extra_block.0.str2str";

#[test]
fn se3_bisect() {
    let Some(io) = open("fixtures/extra0_io/io.safetensors") else { return };
    let Some(w) = open("fixtures/weights/model_state_dict.safetensors") else { return };
    let Some(step) = open("fixtures/model_pinned/step0.safetensors") else { return };
    let p = Params::root(&w, "model").sub("simulator").sub("extra_block").sub("0").sub("str2str");

    let xyz = io.get(&format!("in::{S}.2")); // [1,L,3,3]
    let l = xyz.shape[1];
    let chirals = step.get("rfi.chirals").data;

    // ---- chiral gradients (arg 11) ---------------------------------------
    let got = rfd2::chiral::chiral_grads(&xyz.data, l, 3, &chirals);
    chk("chiral_grads", &got, &io.get(&format!("in::{S}.11")).data);

    // ---- SE(3) transformer, from the reference's own node/edge features ---
    let node0 = io.get(&format!("in::{S}.se3.1")); // [L, 64, 1]
    let node1 = io.get(&format!("in::{S}.se3.2")); // [L, 6, 3]
    let edge = io.get(&format!("in::{S}.se3.3")); // [n_edges, 64, 1]
    let n_e = edge.shape[0];

    // The graph is rebuilt here rather than captured: `rel_pos` lives on the
    // DGLGraph, which no forward hook sees. Reproducing it is part of the port,
    // and the edge count is the first thing that would betray a wrong one.
    let idx = io.get_i64(&format!("in::{S}.4")).0;
    let ca: Vec<f32> = (0..l).flat_map(|i| xyz.data[i * 9 + 3..i * 9 + 6].to_vec()).collect();
    let graph = se3::make_full_graph(&ca, &idx);
    println!("graph: {} nodes, {} edges (reference edge features: {n_e})",
             graph.n_nodes, graph.n_edges());
    assert_eq!(graph.n_edges(), n_e, "edge count mismatch — the graph is wrong");

    let fiber_in = Fiber::new(&[(0, 64), (1, 6)]);
    let fiber_hidden = Fiber::new(&[(0, 32), (1, 32)]);
    let fiber_out = Fiber::new(&[(0, 64), (1, 2)]);
    let t = Se3Transformer::load(&p.sub("se3").sub("se3"), 1, &fiber_in, &fiber_hidden,
                                 &fiber_out, 4, 4);
    let ef = Tensor::new(edge.data.clone(), vec![n_e, edge.shape[1]]);
    let out = t.forward(&graph, &[node0.data.clone(), node1.data.clone()], &ef);
    let ok0 = chk("se3['0']", &out[0], &io.get(&format!("out::{S}.se3.0")).data);
    let ok1 = chk("se3['1']", &out[1], &io.get(&format!("out::{S}.se3.1")).data);
    assert!(ok0 && ok1, "SE(3) transformer is not bit-exact");
}
