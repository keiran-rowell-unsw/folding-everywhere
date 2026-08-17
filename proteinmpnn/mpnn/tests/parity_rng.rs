//! Parity of the reimplemented torch CPU RNG vs PyTorch fixtures.
//!
//! These must be *bit*-exact (not merely close): the whole point is that
//! `--seed N` selects the same decoding order and the same sampled residues as
//! a PyTorch run at that seed.

use proteinmpnn::rng::{multinomial1, randn, Mt19937};
use proteinmpnn::weights::Weights;

fn fx(name: &str) -> Weights {
    let p = format!("{}/../fixtures/rng/{}.safetensors", env!("CARGO_MANIFEST_DIR"), name);
    Weights::open(&p).unwrap_or_else(|e| panic!("open {p}: {e}"))
}

fn bit_equal(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

const SIZES: [usize; 14] = [16, 17, 20, 37, 64, 100, 128, 129, 200, 256, 257, 301, 512, 1000];
const SEEDS: [u64; 5] = [0, 1, 37, 12345, 999999];

#[test]
fn rng_randn_bit_exact() {
    let mut total = 0usize;
    for seed in SEEDS {
        for size in SIZES {
            let f = fx(&format!("randn_s{seed}_n{size}"));
            let want = f.get("y");
            let mut g = Mt19937::new(seed);
            let got = randn(&mut g, size);
            let bad = bit_equal(&got, &want.data);
            assert_eq!(bad, 0, "randn seed={seed} size={size}: {bad}/{size} bits differ");
            total += size;
        }
    }
    println!("randn: {total} values bit-exact across {} cases", SEEDS.len() * SIZES.len());
}

#[test]
fn rng_exponential_bit_exact() {
    for seed in [0u64, 7, 4242] {
        for size in [21usize, 105, 512] {
            let f = fx(&format!("exp_s{seed}_n{size}"));
            let want = f.get("y");
            let mut g = Mt19937::new(seed);
            let got: Vec<f32> = (0..size).map(|_| g.exponential_f32()).collect();
            let bad = bit_equal(&got, &want.data);
            assert_eq!(bad, 0, "exponential seed={seed} size={size}: {bad} differ");
        }
    }
}

#[test]
fn rng_multinomial_matches() {
    for seed in [0u64, 5, 2024] {
        let f = fx(&format!("multinomial_s{seed}"));
        let probs = f.get("probs");
        let (picks, _) = f.get_i64("picks");
        let n = probs.shape[0];
        let k = probs.shape[1];
        let mut g = Mt19937::new(seed);
        for i in 0..n {
            let got = multinomial1(&mut g, &probs.data[i * k..i * k + k]);
            assert_eq!(got as i64, picks[i], "multinomial seed={seed} step={i}");
        }
        println!("multinomial seed={seed}: {n} sequential draws identical");
    }
}

/// The decoding order is `argsort((chain_mask + 1e-4) * |randn|)`. This is the
/// only way `randn` reaches the model, so it gets its own end-to-end check.
#[test]
fn rng_decoding_order_matches() {
    for seed in [0u64, 37] {
        for l in [50usize, 137, 256] {
            let f = fx(&format!("decorder_s{seed}_L{l}"));
            let (want_order, _) = f.get_i64("order");
            let want_randn = f.get("randn");

            let mut g = Mt19937::new(seed);
            let r = randn(&mut g, l);
            assert_eq!(bit_equal(&r, &want_randn.data), 0, "randn seed={seed} L={l}");

            let keys: Vec<f32> = r.iter().map(|v| (1.0f32 + 0.0001) * v.abs()).collect();
            let mut order: Vec<usize> = (0..l).collect();
            order.sort_by(|&a, &b| {
                keys[a].partial_cmp(&keys[b]).unwrap().then(a.cmp(&b))
            });
            for i in 0..l {
                assert_eq!(order[i] as i64, want_order[i], "order seed={seed} L={l} step={i}");
            }
        }
    }
}
