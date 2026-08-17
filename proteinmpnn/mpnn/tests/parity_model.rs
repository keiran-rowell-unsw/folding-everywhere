//! Stage-by-stage parity against a PyTorch dump (`python/ref_dump.py`).
//!
//! The pipeline is rebuilt here from the public building blocks rather than
//! calling `ProteinMpnn::forward`, so a divergence is attributed to the exact
//! layer that introduced it instead of only showing up at the output.

use proteinmpnn::features::{protein_features, FeatureWeights};
use proteinmpnn::featurize::featurize;
use proteinmpnn::layers::{cat_neighbors_nodes, DecLayer, EncLayer};
use proteinmpnn::model::ProteinMpnn;
use proteinmpnn::parity::compare;
use proteinmpnn::pdb::parse_pdb;
use proteinmpnn::rng::{randn, Mt19937};
use proteinmpnn::tensor::Tensor;
use proteinmpnn::weights::Weights;
use proteinmpnn::{ops, ALPHABET};

const CASE: &str = "5L33";
const PDB: &str = "../../ref_ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb";
const SEED: u64 = 37;
const TEMPERATURE: f64 = 0.1;

fn root() -> &'static str {
    env!("CARGO_MANIFEST_DIR")
}

fn fixture() -> Weights {
    let p = format!("{}/../fixtures/model/{CASE}.safetensors", root());
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}\nrun python/ref_dump.py first"))
}

fn ckpt() -> Weights {
    let p = std::env::var("PROTEINMPNN_WEIGHTS").unwrap_or_else(|_| {
        format!("{}/../../ref_ProteinMPNN/vanilla_model_weights/v_48_020.pt", root())
    });
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}"))
}

/// Record a stage's statistics to `results/stage_parity/` so the benchmark
/// figures are generated from the same numbers the tests assert on, rather than
/// from a second, separately-maintained comparison.
fn record(label: &str, s: &proteinmpnn::parity::Stats) {
    let dir = format!("{}/../results/stage_parity", root());
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let slug: String = label
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    let json = format!(
        "{{\"stage\":\"{}\",\"n\":{},\"max_abs\":{:.6e},\"mean_abs\":{:.6e},\
         \"max_ulp\":{},\"cosine\":{:.15},\"bitexact_frac\":{:.6}}}",
        label, s.n, s.max_abs, s.mean_abs, s.max_ulp, s.cosine, s.exact_frac()
    );
    let _ = std::fs::write(format!("{dir}/{slug}.json"), json);
}

fn check(label: &str, got: &[f32], want: &[f32], tol: f32) {
    let s = compare(got, want);
    println!("{label:26} {}", s.summary());
    record(label, &s);
    assert!(!s.any_nan, "{label}: NaN");
    assert!(s.max_abs <= tol, "{label}: max_abs {:.3e} > {:.3e}", s.max_abs, tol);
}

/// Everything the staged tests need, built once.
struct Ctx {
    fx: Weights,
    w: Weights,
    b: proteinmpnn::featurize::Batch,
    l: usize,
    k: usize,
}

fn ctx() -> Ctx {
    let fx = fixture();
    let w = ckpt();
    let st = parse_pdb(&format!("{}/{PDB}", root())).expect("parse pdb");
    let designed: Vec<char> = st.chain_ids();
    let b = featurize(&st, &designed, &[]);
    let (e_idx, shape) = fx.get_i64("E_idx");
    let _ = e_idx;
    let (l, k) = (shape[1], shape[2]);
    Ctx { fx, w, b, l, k }
}

#[test]
fn stage1_featurization() {
    let c = ctx();
    assert_eq!(c.b.l, c.l, "sequence length");

    check("X", &c.b.x, &c.fx.get("X").data, 0.0);
    check("mask", &c.b.mask, &c.fx.get("mask").data, 0.0);
    check("chain_M", &c.b.chain_m, &c.fx.get("chain_M").data, 0.0);
    check("chain_M_pos", &c.b.chain_m_pos, &c.fx.get("chain_M_pos").data, 0.0);

    let (s, _) = c.fx.get_i64("S");
    assert_eq!(c.b.s, s, "native sequence indices");
    let (ridx, _) = c.fx.get_i64("residue_idx");
    assert_eq!(c.b.residue_idx, ridx, "residue_idx");
    let (chain, _) = c.fx.get_i64("chain_encoding_all");
    assert_eq!(c.b.chain_encoding, chain, "chain_encoding_all");
    println!("featurization: L={} exact", c.b.l);
}

/// Geometry, before any large reduction.
///
/// `Cb` is required to be bit-exact. The distance/RBF values are not, and cannot
/// be: PyTorch's fp32 `sqrt` is an approximation that lands 1 ULP below the
/// correctly-rounded result in ~0.6% of inputs, and its fp32 `exp` is SLEEF
/// rather than libm. Both are pure-function differences with no accumulation, so
/// the error stays at 1 ULP instead of growing.
#[test]
fn stage2a_geometry() {
    let c = ctx();
    let l = c.b.l;

    // virtual Cb
    let want_cb = c.fx.get("Cb");
    let got_cb: Vec<f32> = proteinmpnn::features::virtual_cb(&c.b.x, l)
        .into_iter()
        .flatten()
        .collect();
    check("virtual Cb", &got_cb, &want_cb.data, 0.0);

    // Ca-Ca neighbour distances: sqrt of a 3-term sum. The 3-term sum is
    // bit-exact; the residual is PyTorch's inexact fp32 sqrt (<= 1 ULP).
    let (_k, _idx, dn) = proteinmpnn::features::ca_knn(&c.b.x, &c.b.mask, 48);
    let s = compare(&dn, &c.fx.get("D_neighbors").data);
    println!("{:26} {}", "D_neighbors", s.summary());
    record("D_neighbors", &s);
    assert!(s.max_ulp <= 1, "D_neighbors should be within 1 ULP: {}", s.summary());

    // The RBF transform of exactly those distances, isolating `exp()`.
    let got_rbf: Vec<f32> = dn.iter().flat_map(|&d| proteinmpnn::features::rbf(d)).collect();
    let s = compare(&got_rbf, &c.fx.get("RBF_D_neighbors").data);
    println!("{:26} {}", "RBF(D_neighbors)", s.summary());
    record("RBF(D_neighbors)", &s);
    // 78% bit-exact vs 80% for the distances themselves: `exp` contributes
    // essentially nothing, the residual is the inherited sqrt ULP amplified by
    // the Gaussian slope (up to ~25x near the narrow bins).
    assert!(s.max_abs < 5e-6, "RBF: {}", s.summary());

    // The full 416-wide edge input: kNN distances, 25 RBF blocks, positional
    // one-hot. No accumulation anywhere, so the error must stay at the 1-ULP
    // level of its inputs rather than growing.
    let fw = FeatureWeights::load(&c.w);
    let (_, _, _, raw) = proteinmpnn::features::edge_input(
        &fw, &c.b.x, &c.b.mask, &c.b.residue_idx, &c.b.chain_encoding, 48,
    );
    let s = compare(&raw.data, &c.fx.get("E_input").data);
    println!("{:26} {}", "E_input (416-d)", s.summary());
    record("E_input (416-d)", &s);
    assert!(s.max_abs < 1e-5, "E_input: {}", s.summary());
    assert!(s.exact_frac() > 0.7, "E_input mostly bit-exact: {}", s.summary());
}

#[test]
fn stage2b_graph_and_edges() {
    let c = ctx();
    let fw = FeatureWeights::load(&c.w);
    let g = protein_features(&fw, &c.b.x, &c.b.mask, &c.b.residue_idx, &c.b.chain_encoding, 48);

    assert_eq!(g.k, c.k, "K");
    let (want_idx, _) = c.fx.get_i64("E_idx");
    let bad = g.e_idx.iter().zip(&want_idx).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "E_idx: {bad}/{} neighbour indices differ", want_idx.len());
    println!("E_idx: {} indices integer-identical", want_idx.len());

    // The edge features are LayerNorm(Linear_416x128(...)). The 416-wide
    // reduction is the first place PyTorch's blocked MKL GEMM and our pinned
    // sequential fold can disagree, so ~1e-5 here is expected fp32 noise.
    check("E (edge features)", &g.e.data, &c.fx.get("E").data, 5e-5);
}

#[test]
fn stage3_encoder() {
    let c = ctx();
    let fw = FeatureWeights::load(&c.w);
    let g = protein_features(&fw, &c.b.x, &c.b.mask, &c.b.residue_idx, &c.b.chain_encoding, 48);
    let (l, k) = (g.l, g.k);

    let mut h_v = Tensor::zeros(&[l, 128]);
    let mut h_e = ops::linear(&g.e, &c.w.get("W_e.weight"), Some(&c.w.get("W_e.bias")));
    check("h_E init (W_e)", &h_e.data, &c.fx.get("h_E_init").data, 1e-4);

    let mut mask_attend = vec![0.0f32; l * k];
    for i in 0..l {
        for t in 0..k {
            mask_attend[i * k + t] = c.b.mask[i] * c.b.mask[g.e_idx[i * k + t] as usize];
        }
    }
    check("enc mask_attend", &mask_attend, &c.fx.get("enc_mask_attend").data, 0.0);

    for i in 0..3 {
        let layer = EncLayer::load(&c.w, i);
        let (v, e) = layer.forward(&h_v, &h_e, &g.e_idx, k, &c.b.mask, &mask_attend);
        h_v = v;
        h_e = e;
        check(&format!("enc{i} h_V"), &h_v.data, &c.fx.get(&format!("enc{i}_h_V")).data, 2e-4);
        check(&format!("enc{i} h_E"), &h_e.data, &c.fx.get(&format!("enc{i}_h_E")).data, 2e-4);
    }
}

#[test]
fn stage4_decoder_and_log_probs() {
    let c = ctx();
    let fw = FeatureWeights::load(&c.w);
    let g = protein_features(&fw, &c.b.x, &c.b.mask, &c.b.residue_idx, &c.b.chain_encoding, 48);
    let (l, k) = (g.l, g.k);

    let mut h_v = Tensor::zeros(&[l, 128]);
    let mut h_e = ops::linear(&g.e, &c.w.get("W_e.weight"), Some(&c.w.get("W_e.bias")));
    let mut mask_attend0 = vec![0.0f32; l * k];
    for i in 0..l {
        for t in 0..k {
            mask_attend0[i * k + t] = c.b.mask[i] * c.b.mask[g.e_idx[i * k + t] as usize];
        }
    }
    for i in 0..3 {
        let (v, e) = EncLayer::load(&c.w, i).forward(&h_v, &h_e, &g.e_idx, k, &c.b.mask, &mask_attend0);
        h_v = v;
        h_e = e;
    }

    // decoding order from randn_1 (the scoring draw)
    let mut gen = Mt19937::new(SEED);
    let r1 = randn(&mut gen, l);
    check("randn_1", &r1, &c.fx.get("randn_1").data, 0.0);
    let order = ProteinMpnn::decoding_order(&r1, &c.b.design_mask());
    let (want_order, _) = c.fx.get_i64("decoding_order_fwd");
    assert_eq!(order, want_order, "forward decoding order");

    let mask_attend = ProteinMpnn::attend_mask(&order, &g.e_idx, l, k);
    check("mask_bw", &mask_attend, &c.fx.get("mask_bw").data, 0.0);

    let w_s = c.w.get("W_s.weight");
    let h_s = ops::embedding(&c.b.s, &w_s, &[l, 128]);
    let h_es = cat_neighbors_nodes(&h_s, &h_e, &g.e_idx, k);
    let zeros = Tensor::zeros(&[l, 128]);
    let h_ex = cat_neighbors_nodes(&zeros, &h_e, &g.e_idx, k);
    let mut h_exv_fw = cat_neighbors_nodes(&h_v, &h_ex, &g.e_idx, k);
    let wdim = h_exv_fw.last();
    for i in 0..l {
        for t in 0..k {
            let f = c.b.mask[i] * (1.0 - mask_attend[i * k + t]);
            for v in h_exv_fw.data[(i * k + t) * wdim..(i * k + t) * wdim + wdim].iter_mut() {
                *v *= f;
            }
        }
    }

    for i in 0..3 {
        let mut h_esv = cat_neighbors_nodes(&h_v, &h_es, &g.e_idx, k);
        for p in 0..l {
            for t in 0..k {
                let bw = c.b.mask[p] * mask_attend[p * k + t];
                let base = (p * k + t) * wdim;
                for ci in 0..wdim {
                    h_esv.data[base + ci] = h_esv.data[base + ci] * bw + h_exv_fw.data[base + ci];
                }
            }
        }
        h_v = DecLayer::load(&c.w, i).forward(&h_v, &h_esv, &c.b.mask);
        check(&format!("dec{i} h_V"), &h_v.data, &c.fx.get(&format!("dec{i}_h_V")).data, 2e-4);
    }

    let logits = ops::linear(&h_v, &c.w.get("W_out.weight"), Some(&c.w.get("W_out.bias")));
    check("logits", &logits.data, &c.fx.get("logits").data, 3e-4);
    let lp = ops::log_softmax_last(&logits);
    check("log_probs", &lp.data, &c.fx.get("log_probs").data, 3e-4);
}

/// The headline claim: same seed -> same designed sequence, residue for residue.
#[test]
fn stage5_sampling() {
    let c = ctx();
    let model = ProteinMpnn::load(&c.w, 48, 3, 3);
    let l = c.b.l;

    let mut gen = Mt19937::new(SEED);
    let r2 = randn(&mut gen, l);
    check("randn_2", &r2, &c.fx.get("randn_2").data, 0.0);

    let order = ProteinMpnn::decoding_order(&r2, &c.b.design_mask());
    let (want_order, _) = c.fx.get_i64("sample_decoding_order");
    assert_eq!(order, want_order, "sample decoding order");

    let mut omit = [0.0f64; 21];
    omit[ALPHABET.find('X').unwrap()] = 1.0;
    let bias = [0.0f64; 21];
    let out = model.sample(&c.b, &mut gen, &order, TEMPERATURE, &omit, &bias);

    let (want_s, _) = c.fx.get_i64("sample_S");
    let got: String = out.s.iter().map(|&i| proteinmpnn::idx_to_aa(i as usize)).collect();
    let want: String = want_s.iter().map(|&i| proteinmpnn::idx_to_aa(i as usize)).collect();
    println!("rust   : {got}");
    println!("torch  : {want}");
    let ident = got.bytes().zip(want.bytes()).filter(|(a, b)| a == b).count();
    println!("identity: {ident}/{l} ({:.2}%)", 100.0 * ident as f64 / l as f64);
    assert_eq!(out.s, want_s, "sampled sequence");

    check("sample probs", &out.probs, &c.fx.get("sample_probs").data, 3e-4);
}
