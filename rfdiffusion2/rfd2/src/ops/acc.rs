//! The reduction accumulator, and the `exact` feature that makes it provably
//! correctly-rounded.
//!
//! ## Why this exists
//!
//! `docs/BITEXACT.md` argues that accumulating in f64 and rounding once to fp32
//! makes the result independent of the reduction order, because an f64 rounding
//! error (~1e-16 relative) is ~9 orders below an fp32 ULP (~6e-8). That is a
//! *probability*, not a guarantee: two f64 orders disagree whenever the exact
//! value happens to lie within ~1e-16 of the midpoint between two fp32 numbers,
//! which `tests/probe_f64_tie.rs` measures at ~2e-9 of values. A forward pass
//! evaluates ~1e9 reduction outputs, so a handful of 1-ULP flips per pass is the
//! *expected* behaviour — and one of them is exactly what the port and the
//! reference disagree on (`row_attn.to_k[4427, 157]` in `main_block.0`).
//!
//! Expected is not proven. Building with `--features exact` swaps every
//! reduction in the network onto a **double-double** accumulator, which makes
//! the port's answer the correctly-rounded one and turns "we differ by 1 ULP and
//! we think they are wrong" into a measurement.
//!
//! ## Why double-double is enough to call it exact
//!
//! Every product this accumulator is fed is a product of two values that were
//! fp32: `a * b` with `a, b` representable in 24 bits is **exact** in f64,
//! because 24 + 24 = 48 <= 53. So a dot product over fp32 inputs is a sum of
//! *exact* f64 terms, and the only error is in the summation.
//!
//! Double-double (Dekker/Knuth `two_sum`, carrying an unevaluated `hi + lo`)
//! holds ~106 significand bits, so the accumulated relative error is ~2^-104.
//! Rounding that to fp32 gives the correctly-rounded result unless the exact
//! value sits within 2^-104 of an fp32 midpoint — about 2^-80 of values, i.e.
//! never at 1e9 values per pass. Contrast the plain f64 path at ~2^-53, which
//! is 2e-9 of values and therefore fires several times per pass.
//!
//! ## Cost, and why it is a feature and not a runtime flag
//!
//! The default build must not pay for this: the GEMM is tuned to 17 GFLOP/s and
//! a branch in its inner loop would undo that. `Acc` is a `#[repr(transparent)]`
//! newtype over `f64` in the default build with `#[inline(always)]` methods, so
//! it compiles to exactly the arithmetic that was there before. Under `exact` it
//! becomes two fields and ~6 extra flops per term — 3-5x slower, which is fine
//! for a correctness mode that runs twice.

/// Knuth's `two_sum`: the exact sum of `a + b` as an unevaluated pair, with no
/// assumption about which operand is larger.
#[inline(always)]
#[cfg(feature = "exact")]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// A reduction accumulator over exactly-representable f64 terms.
///
/// Default build: a plain `f64`. With `--features exact`: double-double.
#[cfg(not(feature = "exact"))]
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
pub struct Acc(f64);

#[cfg(not(feature = "exact"))]
impl Acc {
    #[inline(always)]
    pub fn new() -> Self {
        Acc(0.0)
    }

    #[inline(always)]
    pub fn add(&mut self, v: f64) {
        self.0 += v;
    }

    /// Fold another accumulator in — used where a reduction is split into
    /// independent lanes and then combined.
    #[inline(always)]
    pub fn merge(&mut self, other: Acc) {
        self.0 += other.0;
    }

    #[inline(always)]
    pub fn get(self) -> f64 {
        self.0
    }
}

#[cfg(feature = "exact")]
#[derive(Clone, Copy, Default, Debug)]
pub struct Acc {
    hi: f64,
    lo: f64,
}

#[cfg(feature = "exact")]
impl Acc {
    #[inline(always)]
    pub fn new() -> Self {
        Acc { hi: 0.0, lo: 0.0 }
    }

    #[inline(always)]
    pub fn add(&mut self, v: f64) {
        let (s, e) = two_sum(self.hi, v);
        let lo = self.lo + e;
        let (hi, lo) = two_sum(s, lo);
        self.hi = hi;
        self.lo = lo;
    }

    #[inline(always)]
    pub fn merge(&mut self, other: Acc) {
        // both parts, largest first, so the compensation term is not dropped
        self.add(other.hi);
        self.add(other.lo);
    }

    #[inline(always)]
    pub fn get(self) -> f64 {
        self.hi + self.lo
    }
}

/// True when the binary was built with the correctness accumulator.
pub const fn exact_mode() -> bool {
    cfg!(feature = "exact")
}
