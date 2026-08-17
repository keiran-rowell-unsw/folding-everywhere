//! Does `Mt19937::new(seed)` reproduce `torch.manual_seed(seed)`?
use rfd2::rng::torch::Mt19937;
use rfd2::weights::Weights;
use std::path::Path;

#[test]
fn manual_seed_matches_capture() {
    let p = format!("{}/../fixtures/sample_init/stages.safetensors", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&p).exists() { return; }
    let f = Weights::open(&p).unwrap();
    let bytes: Vec<u8> = f.get_i64("rng.at_sample_init").0.into_iter().map(|v| v as u8).collect();
    let mut want = Mt19937::from_torch_state(&bytes);
    let mut got = Mt19937::new(0);
    let a: Vec<u32> = (0..8).map(|_| want.random()).collect();
    let b: Vec<u32> = (0..8).map(|_| got.random()).collect();
    println!("captured manual_seed(0): {a:?}");
    println!("Mt19937::new(0):         {b:?}");
    println!("match: {}", a == b);
}
