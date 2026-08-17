//! Bit-exact reimplementations of the **three** generators RFdiffusion2 consumes.
//!
//! `run_inference.py:seed_all(seed)` seeds all three, and the inference path
//! draws from all three in an interleaved order (see `docs/RECON.md` §1.3):
//!
//! | module | reference | drives |
//! |---|---|---|
//! | [`torch`] | `at::mt19937` + ATen distributions | translation noise, `psi_pred`, `torch.randint` |
//! | [`numpy`] | `np.random.RandomState` (legacy MT19937) | rotation noise via SciPy, contig shuffles |
//! | [`pyrandom`] | CPython's `random` module | contig lengths, path choices |
//!
//! All three are Mersenne-Twister-based and, for an integer seed, share the
//! *same underlying u32 stream* — the seeding routines are equivalent
//! (`mt19937_seed` == `init_genrand`). They diverge entirely in how that stream
//! is consumed into distributions, which is where every bug lives.
//!
//! Rung 2 of the SOP ladder requires **exactly 0** difference on every one of
//! these. Nothing here is believed until `tests/parity_rng.rs` says so.

pub mod numpy;
pub mod pyrandom;
pub mod torch;

/// The shared Mersenne Twister core: 624-word state, standard twist and
/// tempering. `at::mt19937`, numpy's `mt19937_state` and CPython's
/// `_randommodule` all use this; they differ only in bookkeeping and seeding.
pub const MT_N: usize = 624;
pub const MT_M: usize = 397;
pub const MATRIX_A: u32 = 0x9908_b0df;
pub const UMASK: u32 = 0x8000_0000;
pub const LMASK: u32 = 0x7fff_ffff;

#[inline]
pub(crate) fn temper(mut y: u32) -> u32 {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c_5680;
    y ^= (y << 15) & 0xefc6_0000;
    y ^= y >> 18;
    y
}

/// One full regeneration pass over the 624-word state (the "twist").
pub(crate) fn twist_state(s: &mut [u32; MT_N]) {
    #[inline]
    fn mix(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }
    #[inline]
    fn tw(u: u32, v: u32) -> u32 {
        (mix(u, v) >> 1) ^ (if v & 1 != 0 { MATRIX_A } else { 0 })
    }
    for i in 0..(MT_N - MT_M) {
        s[i] = s[i + MT_M] ^ tw(s[i], s[i + 1]);
    }
    for i in (MT_N - MT_M)..(MT_N - 1) {
        s[i] = s[i + MT_M - MT_N] ^ tw(s[i], s[i + 1]);
    }
    let last = MT_N - 1;
    s[last] = s[last + MT_M - MT_N] ^ tw(s[last], s[0]);
}

/// `init_genrand(seed)` — Knuth's linear seeding, shared by all three
/// implementations for a plain integer seed.
///
/// numpy spells this `mt19937_seed` with the assignment before the advance and
/// `pos + 1` instead of `i`; it is algebraically the same recurrence.
pub(crate) fn init_genrand(seed: u32) -> [u32; MT_N] {
    let mut s = [0u32; MT_N];
    s[0] = seed;
    for j in 1..MT_N {
        let prev = s[j - 1];
        s[j] = 1812433253u32
            .wrapping_mul(prev ^ (prev >> 30))
            .wrapping_add(j as u32);
    }
    s
}

/// `init_by_array(key)` — CPython's `random.seed(int)` path for arbitrary-width
/// integers, and numpy's path for seeds >= 2^32.
pub(crate) fn init_by_array(key: &[u32]) -> [u32; MT_N] {
    let mut s = init_genrand(19650218);
    let mut i = 1usize;
    let mut j = 0usize;
    let mut k = MT_N.max(key.len());
    while k > 0 {
        let prev = s[i - 1];
        s[i] = (s[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
        i += 1;
        j += 1;
        if i >= MT_N {
            s[0] = s[MT_N - 1];
            i = 1;
        }
        if j >= key.len() {
            j = 0;
        }
        k -= 1;
    }
    let mut k = MT_N - 1;
    while k > 0 {
        let prev = s[i - 1];
        s[i] = (s[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
        i += 1;
        if i >= MT_N {
            s[0] = s[MT_N - 1];
            i = 1;
        }
        k -= 1;
    }
    s[0] = 0x8000_0000;
    s
}
