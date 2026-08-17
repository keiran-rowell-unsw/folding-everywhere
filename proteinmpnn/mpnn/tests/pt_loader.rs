//! The `.pt` checkpoint reader must reproduce `torch.load(...)['model_state_dict']`
//! exactly — same names, same shapes, same bytes.

use proteinmpnn::weights::Weights;

fn ckpt() -> String {
    std::env::var("PROTEINMPNN_WEIGHTS").unwrap_or_else(|_| {
        format!(
            "{}/../../ref_ProteinMPNN/vanilla_model_weights/v_48_020.pt",
            env!("CARGO_MANIFEST_DIR")
        )
    })
}

#[test]
fn pt_index_matches_state_dict() {
    let w = Weights::open(&ckpt()).expect("open checkpoint");
    let names = w.names();
    assert_eq!(names.len(), 118, "expected 118 parameter tensors, got {}", names.len());

    // Spot-check the shapes that pin the architecture.
    let expect: &[(&str, &[usize])] = &[
        ("features.embeddings.linear.weight", &[16, 66]),
        ("features.edge_embedding.weight", &[128, 416]),
        ("features.norm_edges.weight", &[128]),
        ("W_e.weight", &[128, 128]),
        ("W_s.weight", &[21, 128]),
        ("encoder_layers.0.W1.weight", &[128, 384]),
        ("encoder_layers.2.W11.weight", &[128, 384]),
        ("encoder_layers.0.dense.W_in.weight", &[512, 128]),
        ("decoder_layers.0.W1.weight", &[128, 512]),
        ("decoder_layers.2.dense.W_out.weight", &[128, 512]),
        ("W_out.weight", &[21, 128]),
        ("W_out.bias", &[21]),
    ];
    for (n, s) in expect {
        assert!(w.has(n), "missing {n}");
        assert_eq!(w.shape(n).unwrap(), *s, "shape of {n}");
    }

    let total: usize = names.iter().map(|n| w.shape(n).unwrap().iter().product::<usize>()).sum();
    assert_eq!(total, 1_660_485, "total parameter count");
    println!("checkpoint: {} tensors, {total} parameters", names.len());
}

/// Cross-check raw values against a safetensors dump of the same state dict
/// produced by `python/gen_weight_fixture.py` (byte-for-byte identical expected).
#[test]
fn pt_values_match_safetensors() {
    let a = Weights::open(&ckpt()).expect("open checkpoint");
    let p = format!("{}/../fixtures/weights/v_48_020.safetensors", env!("CARGO_MANIFEST_DIR"));
    let b = match Weights::open(&p) {
        Ok(w) => w,
        Err(_) => {
            eprintln!("skip: run python/gen_weight_fixture.py first");
            return;
        }
    };
    let mut checked = 0usize;
    for n in a.names() {
        let (x, y) = (a.get(&n), b.get(&n));
        assert_eq!(x.shape, y.shape, "{n} shape");
        let bad = x
            .data
            .iter()
            .zip(&y.data)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        assert_eq!(bad, 0, "{n}: {bad} values differ");
        checked += x.numel();
    }
    println!("{checked} parameter values bit-identical to torch.load");
}
