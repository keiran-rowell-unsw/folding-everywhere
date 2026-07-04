//! Test-time helpers: a minimal `.npy` reader and tensor comparison metrics.
//! Fixtures are individual `.npy` files (saved with `np.save`) under `fixtures/`.

use crate::tensor::Tensor;
use std::path::PathBuf;

/// Directory holding `.npy` fixtures (CARGO_MANIFEST_DIR/fixtures).
pub fn fixtures_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("fixtures");
    p
}

pub fn fixture_path(name: &str) -> PathBuf {
    let mut p = fixtures_dir();
    p.push(name);
    p
}

struct Npy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn read_npy_raw(path: &std::path::Path) -> Npy {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("cannot read npy {:?}", path));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "bad npy magic {:?}", path);
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12usize)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
    // descr
    let descr = {
        let k = header.find("'descr'").unwrap();
        let q1 = header[k + 7..].find('\'').unwrap() + k + 7;
        let q2 = header[q1 + 1..].find('\'').unwrap() + q1 + 1;
        header[q1 + 1..q2].to_string()
    };
    assert!(
        header.contains("'fortran_order': False"),
        "fortran-order npy unsupported {:?}",
        path
    );
    // shape tuple
    let shape = {
        let k = header.find("'shape'").unwrap();
        let lp = header[k..].find('(').unwrap() + k;
        let rp = header[lp..].find(')').unwrap() + lp;
        header[lp + 1..rp]
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                if s.is_empty() { None } else { Some(s.parse::<usize>().unwrap()) }
            })
            .collect::<Vec<_>>()
    };
    let data = bytes[header_start + header_len..].to_vec();
    Npy { descr, shape, data }
}

/// Read an `.npy` as an fp32 Tensor. Supports `<f4` and `<f8` (downcast).
pub fn read_npy_f32(path: &std::path::Path) -> Tensor {
    let n = read_npy_raw(path);
    let data: Vec<f32> = match n.descr.as_str() {
        "<f4" => n.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "<f8" => n.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        d => panic!("read_npy_f32: dtype {d} in {:?}", path),
    };
    let shape = if n.shape.is_empty() { vec![1] } else { n.shape };
    Tensor::new(data, shape)
}

/// Read an `.npy` as i64 (supports `<i8`, `<i4`, `|b1`).
pub fn read_npy_i64(path: &std::path::Path) -> (Vec<i64>, Vec<usize>) {
    let n = read_npy_raw(path);
    let data: Vec<i64> = match n.descr.as_str() {
        "<i8" => n.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        "<i4" => n.data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        "|b1" => n.data.iter().map(|&b| b as i64).collect(),
        "|i1" => n.data.iter().map(|&b| b as i8 as i64).collect(),
        d => panic!("read_npy_i64: dtype {d} in {:?}", path),
    };
    (data, n.shape)
}

pub fn load_f32(name: &str) -> Tensor {
    read_npy_f32(&fixture_path(name))
}

// ---- comparison metrics ---------------------------------------------------

pub struct Diff {
    pub max_abs: f32,
    pub max_rel: f32,
    pub cosine: f64,
    pub n: usize,
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_abs={:.3e} max_rel={:.3e} cosine={:.12} n={}",
            self.max_abs, self.max_rel, self.cosine, self.n
        )
    }
}

/// Compare two equal-length slices: max abs diff, max relative diff
/// (|a-b|/(|b|+1e-6)), and cosine similarity (f64 accumulation).
pub fn compare(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len(), "compare length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d > max_abs { max_abs = d; }
        let r = d / (y.abs() + 1e-6);
        if r > max_rel { max_rel = r; }
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let cosine = if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) } else { 1.0 };
    Diff { max_abs, max_rel, cosine, n: a.len() }
}

pub fn compare_t(a: &Tensor, b: &Tensor) -> Diff {
    assert_eq!(a.shape, b.shape, "compare_t shape {:?} vs {:?}", a.shape, b.shape);
    compare(&a.data, &b.data)
}
