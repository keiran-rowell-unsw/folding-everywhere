//! CPython's `random` module, bit-exact.
//!
//! `run_inference.py:seed_all` calls `random.seed(seed)`, and the contig
//! machinery draws from it (`contigs.py:184 random.randint(...)`,
//! `aa_model.py:2655 random.choice(paths)`); see `docs/RECON.md` §1.3.
//!
//! Source: CPython `Modules/_randommodule.c` and `Lib/random.py`.
//!
//! Differences from numpy that matter:
//! - **Seeding.** `random.seed(n)` for an int does *not* use `init_genrand`.
//!   It takes `abs(n)`, splits it into 32-bit little-endian words, and calls
//!   `init_by_array` on that key. For n = 0 the key is `[0]` (one word), not
//!   empty.
//! - `random()` is the same 53-bit construction as numpy's `random_sample`.
//! - `randrange`/`randint` do **not** use rejection on a mask over a full word;
//!   they use `_randbelow_with_getrandbits`, which draws exactly
//!   `k = bit_length(n)` bits at a time and rejects until `< n`.
//! - `getrandbits(k)` for k <= 32 is **one** draw shifted right by `32 - k`.
//!   For larger k it consumes words little-endian.

use super::{init_by_array, temper, twist_state, MT_N};

#[derive(Clone)]
pub struct PyRandom {
    state: [u32; MT_N],
    pos: usize,
}

impl PyRandom {
    /// `random.seed(n)` for a non-negative integer `n`.
    pub fn new(seed: u64) -> Self {
        // CPython: key = abs(n) as little-endian 32-bit words; zero -> [0].
        let mut key: Vec<u32> = Vec::new();
        let mut v = seed;
        if v == 0 {
            key.push(0);
        }
        while v > 0 {
            key.push((v & 0xffff_ffff) as u32);
            v >>= 32;
        }
        PyRandom { state: init_by_array(&key), pos: MT_N }
    }

    #[inline]
    pub fn genrand_uint32(&mut self) -> u32 {
        if self.pos >= MT_N {
            twist_state(&mut self.state);
            self.pos = 0;
        }
        let y = self.state[self.pos];
        self.pos += 1;
        temper(y)
    }

    /// `random.random()` — 53-bit double, identical construction to numpy's.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64;
        let b = (self.genrand_uint32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `random.getrandbits(k)`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        assert!(k <= 64, "getrandbits > 64 not needed here");
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_uint32() >> (32 - k)) as u64;
        }
        // words little-endian; the final (most significant) word is shifted
        let lo = self.genrand_uint32() as u64;
        let rem = k - 32;
        let hi = (self.genrand_uint32() >> (32 - rem)) as u64;
        (hi << 32) | lo
    }

    /// `Random._randbelow_with_getrandbits(n)` — returns a value in `[0, n)`.
    ///
    /// `k` is `n.bit_length()`, **not** `(n-1).bit_length()`. CPython's source
    /// carries the comment "don't use (n-1) here because n can be 1", and the
    /// two differ at every power of two: for n = 2 the correct k is 2, so the
    /// draw rejects half the time. Getting this wrong passes every non-power-of-
    /// two case and fails `randint(0, 1)` — which is exactly how the fixture
    /// caught it.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros(); // n.bit_length()
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.randrange(start, stop)`.
    pub fn randrange(&mut self, start: i64, stop: i64) -> i64 {
        let width = stop - start;
        assert!(width > 0, "empty range");
        start + self.randbelow(width as u64) as i64
    }

    /// `random.randint(a, b)` — **inclusive** of `b`.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        self.randrange(a, b + 1)
    }

    /// `random.choice(seq)` = `seq[self._randbelow(len(seq))]`.
    pub fn choice_index(&mut self, len: usize) -> usize {
        assert!(len > 0, "choice from empty sequence");
        self.randbelow(len as u64) as usize
    }

    /// `random.shuffle(x)` — Fisher-Yates from the top with `_randbelow(i+1)`,
    /// i.e. exclusive bound, unlike numpy's inclusive `random_interval(i)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        if x.len() < 2 {
            return;
        }
        for i in (1..x.len()).rev() {
            let j = self.randbelow((i + 1) as u64) as usize;
            x.swap(i, j);
        }
    }
}
