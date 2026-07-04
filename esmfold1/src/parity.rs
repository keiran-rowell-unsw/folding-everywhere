//! Comparison primitives for parity tests against PyTorch fixtures.

#[derive(Debug, Clone)]
pub struct Stats {
    pub n: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub mean_abs: f64,
    pub max_ulp: i64,
    pub cosine: f64,
    pub any_nan: bool,
}

#[inline]
fn ordered_key(f: f32) -> i64 {
    let b = f.to_bits();
    if b & 0x8000_0000 != 0 {
        0x8000_0000i64 - (b as i64)
    } else {
        b as i64
    }
}

/// Compare two equal-length fp32 slices.
pub fn compare(a: &[f32], b: &[f32]) -> Stats {
    assert_eq!(a.len(), b.len(), "compare length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_ulp = 0i64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut any_nan = false;
    for i in 0..a.len() {
        let (x, y) = (a[i], b[i]);
        if x.is_nan() || y.is_nan() {
            any_nan = true;
            continue;
        }
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = y.abs().max(x.abs()).max(1e-12);
        let rel = d / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        sum_abs += d as f64;
        let ulp = (ordered_key(x) - ordered_key(y)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    Stats {
        n: a.len(),
        max_abs,
        max_rel,
        mean_abs: sum_abs / a.len() as f64,
        max_ulp,
        cosine,
        any_nan,
    }
}

impl Stats {
    pub fn summary(&self) -> String {
        format!(
            "n={} max_abs={:.3e} max_rel={:.3e} mean_abs={:.3e} max_ulp={} cos={:.10}{}",
            self.n,
            self.max_abs,
            self.max_rel,
            self.mean_abs,
            self.max_ulp,
            self.cosine,
            if self.any_nan { " [NaN!]" } else { "" }
        )
    }
}
