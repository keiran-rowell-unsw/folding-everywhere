//! Bit-exact reimplementation of PyTorch's **CPU** random number generation.
//!
//! This is what lets `mpnn --seed 37` produce byte-identical output to
//! `python protein_mpnn_run.py --seed 37`: ProteinMPNN draws randomness in
//! exactly two places, and both are reproduced here.
//!
//!   1. `torch.randn([1, L])`  -> the residue decoding order
//!   2. `torch.multinomial(p, 1)` -> the amino acid sampled at each step
//!
//! ## What PyTorch actually does
//!
//! * `torch.manual_seed(s)` seeds `at::mt19937` (a 32-bit-state MT19937 variant,
//!   `ATen/core/MT19937RNGEngine.h`). `random()` = one 32-bit draw,
//!   `random64()` = two draws combined `(hi << 32) | lo`.
//! * `uniform_real_distribution<float>(0,1)` = `(random() & (2^24-1)) * 2^-24`;
//!   the `double` version uses `random64()` with a 53-bit mask.
//! * `torch.randn` on a contiguous fp32 tensor of size >= 16 takes the
//!   `normal_fill` path: fill the whole buffer with uniforms *first*, then
//!   convert in place, 16 at a time, by Box-Muller over the pairs (j, j+8). If
//!   the size is not a multiple of 16 the **last 16 values are redrawn** and
//!   recomputed, which is why the tail of a `randn` is not a prefix property.
//! * The Box-Muller itself runs through `normal_fill_16_AVX2`, which calls
//!   `log256_ps` / `sincos256_ps` from `ATen/native/cpu/avx_mathfun.h`. Those are
//!   Cephes-derived fp32 polynomial approximations — *not* libm — and PyTorch
//!   compiles them with FMA contraction enabled (GCC's default
//!   `-ffp-contract=fast`). Both facts are load-bearing: using libm `logf`/`cosf`
//!   here, or plain mul+add instead of `mul_add`, changes the last bits.
//! * `torch.multinomial(p, 1)` does **not** walk a CDF. Per
//!   `aten/src/ATen/native/Multinomial.cpp` it uses the Gumbel/exponential trick:
//!   draw `q ~ Exp(1)` elementwise and return `argmax(p / q)`.
//! * `exponential_(1)` uses `exponential_distribution<double>`, i.e.
//!   `-log1p(-uniform_double)`, then narrows to fp32.
//!
//! Every one of those claims is pinned by a parity test against torch 2.7.1
//! (`tests/parity_rng.rs`), including the >= 256-element cases where the FMA
//! contraction is the only thing that differs.

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// `at::mt19937` — note the `left_/next_` bookkeeping, which makes the *first*
/// draw after seeding trigger a state regeneration.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; MT_N],
    left: i32,
    next: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; MT_N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..MT_N {
            let prev = state[j - 1];
            state[j] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        Mt19937 { state, left: 1, next: 0 }
    }

    #[inline]
    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }

    #[inline]
    fn twist(u: u32, v: u32) -> u32 {
        (Self::mix_bits(u, v) >> 1) ^ (if v & 1 != 0 { MATRIX_A } else { 0 })
    }

    fn next_state(&mut self) {
        self.left = MT_N as i32;
        self.next = 0;
        let s = &mut self.state;
        // for (j = N-M+1; --j; p++)  =>  N-M iterations
        for i in 0..(MT_N - MT_M) {
            s[i] = s[i + MT_M] ^ Self::twist(s[i], s[i + 1]);
        }
        // for (j = M; --j; p++)  =>  M-1 iterations
        for i in (MT_N - MT_M)..(MT_N - 1) {
            s[i] = s[i + MT_M - MT_N] ^ Self::twist(s[i], s[i + 1]);
        }
        let last = MT_N - 1;
        s[last] = s[last + MT_M - MT_N] ^ Self::twist(s[last], s[0]);
    }

    /// One 32-bit draw (`CPUGeneratorImpl::random()`).
    #[inline]
    pub fn random(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Two 32-bit draws combined (`CPUGeneratorImpl::random64()`).
    #[inline]
    pub fn random64(&mut self) -> u64 {
        let hi = self.random() as u64;
        let lo = self.random() as u64;
        (hi << 32) | lo
    }

    /// `uniform_real_distribution<float>(0, 1)`.
    #[inline]
    pub fn uniform_f32(&mut self) -> f32 {
        ((self.random() & ((1 << 24) - 1)) as f32) * (1.0 / (1u32 << 24) as f32)
    }

    /// `uniform_real_distribution<double>(0, 1)`.
    #[inline]
    pub fn uniform_f64(&mut self) -> f64 {
        ((self.random64() & ((1u64 << 53) - 1)) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Advance the stream by `n` 32-bit draws without materialising them.
    /// Used to skip the draws PyTorch burns initialising weights it then
    /// overwrites (see `model::torch_init_draws`).
    pub fn skip(&mut self, n: u64) {
        for _ in 0..n {
            let _ = self.random();
        }
    }

    /// `Tensor.exponential_(1)` narrowed to fp32.
    #[inline]
    pub fn exponential_f32(&mut self) -> f32 {
        let v = self.uniform_f64();
        (-(-v).ln_1p()) as f32
    }
}

// ---------------------------------------------------------------------------
// avx_mathfun: Cephes fp32 polynomial log / sincos, with the FMA contraction
// GCC applies when PyTorch builds `normal_fill_16_AVX2`.
// ---------------------------------------------------------------------------

/// `log256_ps` — one lane.
fn log_ps(x0: f32) -> f32 {
    let invalid = x0 <= 0.0;
    // cut off denormals
    let x0 = if x0 < f32::MIN_POSITIVE { f32::MIN_POSITIVE } else { x0 };
    let bits = x0.to_bits();
    let imm0 = (bits >> 23) as i32 - 0x7f;
    // keep only the mantissa, then force the exponent to 2^-1 => x in [0.5, 1)
    let mut x = f32::from_bits((bits & !0x7f80_0000) | 0x3f00_0000);
    let mut e = imm0 as f32 + 1.0;

    let mask = x < 0.707106781186547524_f64 as f32;
    let tmp = if mask { x } else { 0.0 };
    x -= 1.0;
    if mask {
        e -= 1.0;
    }
    x += tmp;

    let z = x * x;
    let mut y = 7.0376836292E-2_f64 as f32;
    y = y.mul_add(x, -1.1514610310E-1_f64 as f32);
    y = y.mul_add(x, 1.1676998740E-1_f64 as f32);
    y = y.mul_add(x, -1.2420140846E-1_f64 as f32);
    y = y.mul_add(x, 1.4249322787E-1_f64 as f32);
    y = y.mul_add(x, -1.6668057665E-1_f64 as f32);
    y = y.mul_add(x, 2.0000714765E-1_f64 as f32);
    y = y.mul_add(x, -2.4999993993E-1_f64 as f32);
    y = y.mul_add(x, 3.3333331174E-1_f64 as f32);
    y *= x;
    // GCC fuses `y = add(mul(y, z), mul(e, q1))` by contracting the FIRST operand
    // of the add, i.e. `fma(y, z, fl(e*q1))` — not `fma(e, q1, fl(y*z))`. The two
    // differ in the last bit and this is the only place in the kernel where the
    // choice is observable.
    y = y.mul_add(z, e * (-2.12194440e-4_f64 as f32));
    y = (-z).mul_add(0.5, y);
    x += y;
    x = e.mul_add(0.693359375_f64 as f32, x);
    if invalid {
        f32::NAN
    } else {
        x
    }
}

/// `sincos256_ps` — one lane. Returns `(sin, cos)`.
fn sincos_ps(x0: f32) -> (f32, f32) {
    let sign_bit_sin_in = x0.to_bits() & 0x8000_0000;
    let mut x = f32::from_bits(x0.to_bits() & 0x7fff_ffff); // |x|

    let y = x * (1.27323954473516_f64 as f32); // 4/pi
    let mut imm2 = y as i32; // _mm256_cvttps_epi32 truncates toward zero
    imm2 = (imm2 + 1) & !1;
    let y = imm2 as f32;
    let imm4 = imm2;

    let imm0 = (imm2 & 4) << 29;
    let swap_sign_bit_sin = imm0 as u32;
    let poly_mask = (imm2 & 2) == 0;

    // extended-precision modular arithmetic: x = ((x - y*DP1) - y*DP2) - y*DP3
    x = y.mul_add(-0.78515625_f64 as f32, x);
    x = y.mul_add(-2.4187564849853515625e-4_f64 as f32, x);
    x = y.mul_add(-3.77489497744594108e-8_f64 as f32, x);

    let sign_bit_cos = ((!(imm4 - 2)) & 4) << 29;

    let sign_bit_sin = sign_bit_sin_in ^ swap_sign_bit_sin;

    let z = x * x;
    // cos polynomial
    let mut y1 = 2.443315711809948E-005_f64 as f32;
    y1 = y1.mul_add(z, -1.388731625493765E-003_f64 as f32);
    y1 = y1.mul_add(z, 4.166664568298827E-002_f64 as f32);
    y1 *= z;
    // Same first-operand contraction rule as in log_ps: `sub(mul(y1,z), mul(z,0.5))`
    // becomes `fma(y1, z, -fl(z*0.5))`.
    y1 = y1.mul_add(z, -(z * 0.5));
    y1 += 1.0;
    // sin polynomial
    let mut y2 = -1.9515295891E-4_f64 as f32;
    y2 = y2.mul_add(z, 8.3321608736E-3_f64 as f32);
    y2 = y2.mul_add(z, -1.6666654611E-1_f64 as f32);
    y2 *= z;
    y2 = y2.mul_add(x, x);

    let (ysin, ycos) = if poly_mask { (y2, y1) } else { (y1, y2) };
    let s = f32::from_bits(ysin.to_bits() ^ sign_bit_sin);
    let c = f32::from_bits(ycos.to_bits() ^ sign_bit_cos as u32);
    (s, c)
}

/// `normal_fill_16_AVX2` with mean 0 / std 1: converts `data[off..off+16]` from
/// uniforms to normals in place, pairing lane `j` with lane `j+8`.
fn normal_fill_16(data: &mut [f32], off: usize) {
    let two_pi = 2.0 * std::f32::consts::PI;
    for j in 0..8 {
        let u1 = 1.0f32 - data[off + j];
        let u2 = data[off + j + 8];
        let radius = (-2.0f32 * log_ps(u1)).sqrt();
        let theta = two_pi * u2;
        let (sintheta, costheta) = sincos_ps(theta);
        data[off + j] = radius * costheta;
        data[off + j + 8] = radius * sintheta;
    }
}

/// `torch.randn(size)` for a contiguous fp32 tensor, bit-exact on x86-64 builds
/// of PyTorch (the `normal_fill_AVX2` path). Requires `size >= 16`; smaller
/// tensors take a different ATen path that ProteinMPNN never hits.
pub fn randn(gen: &mut Mt19937, size: usize) -> Vec<f32> {
    assert!(size >= 16, "randn: ATen uses a different path below 16 elements");
    let mut d = vec![0.0f32; size];
    for v in d.iter_mut() {
        *v = gen.uniform_f32();
    }
    let mut i = 0usize;
    while i + 16 <= size {
        normal_fill_16(&mut d, i);
        i += 16;
    }
    if size % 16 != 0 {
        // ATen redraws and recomputes the final 16 values.
        let off = size - 16;
        for j in 0..16 {
            d[off + j] = gen.uniform_f32();
        }
        normal_fill_16(&mut d, off);
    }
    d
}

/// `torch.multinomial(probs, 1)` = `argmax(probs / Exp(1))`, drawing one
/// exponential per category in order.
pub fn multinomial1(gen: &mut Mt19937, probs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        let q = gen.exponential_f32();
        let v = p / q;
        // argmax keeps the FIRST maximum, matching ATen's argmax reduction.
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
