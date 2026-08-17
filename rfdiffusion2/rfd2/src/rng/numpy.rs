//! `numpy.random.RandomState` — the **legacy** generator, bit-exact.
//!
//! This is the stream `np.random.seed(N)` seeds and that `np.random.*` module
//! functions draw from. RFdiffusion2 needs it because
//! `se3_flow_matching/data/interpolant.py:_uniform_so3` builds the initial
//! rotation noise with `scipy.spatial.transform.Rotation.random`, which uses
//! the global numpy `RandomState` — *not* torch. See `docs/RECON.md` §1.3(7).
//!
//! Sources (numpy 1.26.4):
//! - `numpy/random/src/mt19937/mt19937.c` — `mt19937_seed`, `mt19937_gen`
//! - `numpy/random/src/distributions/random_mtrand.c` — `legacy_gauss`
//! - `numpy/random/mtrand.pyx` — `random_sample`, `shuffle`, `choice`
//!
//! The three details that matter and are easy to get wrong:
//! 1. `random_sample` is a **53-bit** double built from *two* u32 draws as
//!    `(a>>5)*2^26 + (b>>6)) / 2^53` — not a single draw scaled.
//! 2. `standard_normal` is the **polar (Marsaglia) method with a one-value
//!    cache**: it produces two normals per accepted pair, returns `f*x2` first
//!    and caches `f*x1`. An odd-length draw therefore leaves a value behind
//!    that the *next* call consumes. Rejected pairs still consume the stream.
//! 3. `seed()` clears the cache (`has_gauss = 0`).

use super::{init_by_array, init_genrand, temper, twist_state, MT_N};

#[derive(Clone)]
pub struct RandomState {
    state: [u32; MT_N],
    pos: usize,
    /// The polar-method cache: `legacy_gauss` returns one value and keeps one.
    gauss: f64,
    has_gauss: bool,
}

impl RandomState {
    /// `np.random.seed(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        let state = if seed <= u32::MAX as u64 {
            init_genrand(seed as u32)
        } else {
            // numpy converts a wide seed to a u32 array, little-endian
            let mut key = Vec::new();
            let mut v = seed;
            while v > 0 {
                key.push((v & 0xffff_ffff) as u32);
                v >>= 32;
            }
            init_by_array(&key)
        };
        RandomState { state, pos: MT_N, gauss: 0.0, has_gauss: false }
    }

    /// `mt19937_next` — one tempered 32-bit draw.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos == MT_N {
            twist_state(&mut self.state);
            self.pos = 0;
        }
        let y = self.state[self.pos];
        self.pos += 1;
        temper(y)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // numpy's `next_uint64` for the MT19937 bit generator: low word first.
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// `random_sample()` — a 53-bit double in [0, 1).
    #[inline]
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `legacy_gauss()` — polar/Marsaglia with the one-value cache.
    pub fn standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            let g = self.gauss;
            self.gauss = 0.0;
            return g;
        }
        loop {
            let x1 = 2.0 * self.random_sample() - 1.0;
            let x2 = 2.0 * self.random_sample() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 >= 1.0 || r2 == 0.0 {
                continue; // rejected pairs still advanced the stream
            }
            let f = (-2.0 * r2.ln() / r2).sqrt();
            self.gauss = f * x1;
            self.has_gauss = true;
            return f * x2;
        }
    }

    /// `np.random.normal(loc, scale, size)` filled in C order.
    pub fn normal(&mut self, n: usize, loc: f64, scale: f64) -> Vec<f64> {
        (0..n).map(|_| loc + scale * self.standard_normal()).collect()
    }

    /// `np.random.random_sample(size)`.
    pub fn random(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.random_sample()).collect()
    }

    /// `random_interval(max)` — masked rejection sampling, inclusive of `max`.
    /// This is what `shuffle` uses to pick its swap partner.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        // smallest 2^k - 1 that is >= max
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;

        if max <= u32::MAX as u64 {
            loop {
                let v = (self.next_u32() as u64) & mask;
                if v <= max {
                    return v;
                }
            }
        } else {
            loop {
                let v = self.next_u64() & mask;
                if v <= max {
                    return v;
                }
            }
        }
    }

    /// `np.random.shuffle(x)` — Fisher-Yates from the top, using
    /// `random_interval(i)` (inclusive), i.e. `j <= i`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        if x.len() < 2 {
            return;
        }
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `np.random.permutation(n)` = shuffle of `arange(n)`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        self.shuffle(&mut v);
        v
    }
}

// ---------------------------------------------------------------------------
// SciPy's `Rotation.random`, which is what actually consumes the numpy stream
// on the RFdiffusion2 inference path.
// ---------------------------------------------------------------------------

/// `scipy.spatial.transform.Rotation.random(n)` as
/// `Rotation.from_quat(np.random.normal(size=(n, 4)))`, verified bit-exact
/// against scipy 1.13.1 / numpy 1.26.4 for n in {1, 2, 5, 17}.
///
/// Returns row-major 3x3 matrices, f64 — `_uniform_so3` narrows to fp32 only
/// at the very end, so the whole construction must stay in double.
///
/// SciPy's quaternion order is **(x, y, z, w)**, and the normalisation happens
/// *inside* `from_quat`; pre-normalising with a different expression differs in
/// the last bit (measured 3.3e-16).
pub fn scipy_random_rotations(rs: &mut RandomState, n: usize) -> Vec<[f64; 9]> {
    let q = rs.normal(4 * n, 0.0, 1.0);
    (0..n)
        .map(|i| quat_to_matrix(&q[4 * i..4 * i + 4]))
        .collect()
}

/// `Rotation.from_quat(q).as_matrix()` for a single, *unnormalised* quaternion
/// in (x, y, z, w) order.
pub fn quat_to_matrix(q: &[f64]) -> [f64; 9] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (x, y, z, w) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let w2 = w * w;

    let xy = x * y;
    let zw = z * w;
    let xz = x * z;
    let yw = y * w;
    let yz = y * z;
    let xw = x * w;

    [
        x2 - y2 - z2 + w2,
        2.0 * (xy - zw),
        2.0 * (xz + yw),
        2.0 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2.0 * (yz - xw),
        2.0 * (xz - yw),
        2.0 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
}
