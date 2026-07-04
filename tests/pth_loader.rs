//! Verify the PyTorch .bin reader matches the known-good safetensors, tensor for tensor.

use esmfold::parity::compare;
use esmfold::weights::Weights;

fn cache(name: &str) -> Option<String> {
    let base = format!("{}/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots", std::env::var("HOME").unwrap());
    for e in std::fs::read_dir(&base).ok()?.flatten() {
        let p = e.path().join(name);
        if p.exists() {
            return Some(p.to_string_lossy().into_owned());
        }
    }
    None
}

#[test]
fn pth_matches_safetensors() {
    let (st, bin) = match (cache("model.safetensors"), cache("pytorch_model.bin")) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("SKIP: need both model.safetensors and pytorch_model.bin in HF cache");
            return;
        }
    };
    let wst = Weights::open(&st).unwrap();
    let wbin = Weights::open(&bin).unwrap();

    // representative tensors across the whole model (F16 esm + F32 folding)
    let names = [
        "esm_s_combine",
        "esm.embeddings.word_embeddings.weight",
        "esm.encoder.layer.0.attention.self.query.weight",
        "esm.encoder.layer.35.output.dense.weight",
        "esm.encoder.emb_layer_norm_after.weight",
        "trunk.blocks.0.tri_mul_out.linear_a_p.weight",
        "trunk.blocks.47.mlp_pair.mlp.1.weight",
        "trunk.structure_module.ipa.linear_q.weight",
        "trunk.structure_module.ipa.head_weights",
        "distogram_head.weight",
        "lddt_head.3.weight",
        "trunk.recycle_disto.weight",
    ];
    let mut worst = 0i64;
    for n in names {
        assert!(wbin.has(n), "bin missing {n}");
        let a = wst.get(n);
        let b = wbin.get(n);
        assert_eq!(a.shape, b.shape, "shape mismatch {n}: {:?} vs {:?}", a.shape, b.shape);
        let s = compare(&a.data, &b.data);
        println!("{n:55} {}", s.summary());
        assert_eq!(s.max_abs, 0.0, "{n} not bit-identical: {}", s.summary());
        worst = worst.max(s.max_ulp);
    }
    println!("all {} tensors bit-identical (worst ulp {worst})", names.len());
}
