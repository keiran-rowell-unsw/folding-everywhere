//! Bit-exact reproduction of PyTorch's **CPU** RNG (`torch.manual_seed` →
//! `CPUGeneratorImpl` over MT19937), enabling a fully standalone, dependency-free
//! ESMFold2 fold whose stochastic diffusion matches a PyTorch run pinned at a given
//! seed. Validated primitive-by-primitive against torch seed-0 (see tests + the
//! `python/torch_rng_prototype.py` reference).
//!
//! Reproduced exactly:
//! - MT19937 (`init_genrand` + standard tempering)
//! - float32 uniform = (u32 & 0xFFFFFF) * 2^-24                 [torch.rand f32]
//! - float64 uniform = ((a<<32|b) & (2^53-1)) * 2^-53          [torch.rand f64, a first]
//! - `normal_fill` (randn numel>=16): fill f32 uniforms, transform blocks of 16
//!   pairing j with j+8; non-multiple-of-16 tail re-fills the last 16 fresh.
//! - scalar normal (randn numel<16): Box-Muller on double uniforms with a persistent
//!   double-normal cache that carries across calls (and across a normal_fill).
//! - `trunc_normal_`: f32-uniform fill + torch `calc_erfinv` + scale + clamp.
//! - `dropout(p)`: bernoulli mask via **double** uniform (`u64 < 1-p`), scale 1/(1-p).
//!
//! Transcendentals use the `libm` crate (identical on Linux/Windows) so the build is
//! reproducible across platforms.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const TWO_PI_F32: f32 = (2.0 * std::f64::consts::PI) as f32;

pub struct TorchRng {
    mt: [u32; N],
    idx: usize,
    /// torch's `next_double_normal_sample_` (stored as f64; cast to f32 on output).
    cache: Option<f64>,
}

impl TorchRng {
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u32; N];
        mt[0] = (seed & 0xffffffff) as u32;
        for i in 1..N {
            mt[i] = (1812433253u32)
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        TorchRng { mt, idx: N, cache: None }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                self.mt[i] = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    self.mt[i] ^= MATRIX_A;
                }
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    #[inline]
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u32() & 0xFFFFFF) as f32 * (2.0f32.powi(-24))
    }

    #[inline]
    pub fn uniform_f64(&mut self) -> f64 {
        let a = self.next_u32() as u64;
        let b = self.next_u32() as u64;
        (((a << 32) | b) & ((1u64 << 53) - 1)) as f64 * (2.0f64.powi(-53))
    }

    /// torch `calc_erfinv` (float): matlab rational approximation + 2 Newton steps.
    pub fn erfinv_f32(y: f32) -> f32 {
        const CENTRAL_RANGE: f32 = 0.7;
        let a = [0.886226899f32, -1.645349621, 0.914624893, -0.140543331];
        let b = [-2.118377725f32, 1.442710462, -0.329097515, 0.012229801];
        let c = [-1.970840454f32, -1.624906493, 3.429567803, 1.641345311];
        let d = [3.543889200f32, 1.637067800];
        let y_abs = y.abs();
        if y_abs > 1.0 {
            return f32::NAN;
        }
        if y_abs == 1.0 {
            return f32::INFINITY.copysign(y);
        }
        let mut x;
        if y_abs <= CENTRAL_RANGE {
            let z = y * y;
            let num = ((a[3] * z + a[2]) * z + a[1]) * z + a[0];
            let dem = (((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0;
            x = y * num / dem;
        } else {
            let z = libm::sqrtf(-libm::logf((1.0 - y_abs) / 2.0));
            let num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
            let dem = (d[1] * z + d[0]) * z + 1.0;
            x = num.copysign(y) / dem;
        }
        // torch uses (2 / sqrt(pi<double>)) as the Newton denominator constant.
        let two_over_sqrt_pi = (2.0f64 / std::f64::consts::PI.sqrt()) as f32;
        x -= (libm::erff(x) - y) / (two_over_sqrt_pi * libm::expf(-x * x));
        x -= (libm::erff(x) - y) / (two_over_sqrt_pi * libm::expf(-x * x));
        x
    }

    /// Fill `buf` with `randn` (standard normal), matching torch's path selection by length.
    pub fn fill_randn(&mut self, buf: &mut [f32]) {
        let n = buf.len();
        if n >= 16 {
            self.normal_fill(buf);
        } else {
            for v in buf.iter_mut() {
                if let Some(c) = self.cache.take() {
                    *v = c as f32;
                    continue;
                }
                let u1 = self.uniform_f64();
                let u2 = self.uniform_f64();
                let rad = libm::sqrt(-2.0 * libm::log1p(-u2));
                let th = 2.0 * std::f64::consts::PI * u1;
                *v = (rad * libm::cos(th)) as f32;
                self.cache = Some(rad * libm::sin(th));
            }
        }
    }

    fn normal_fill(&mut self, buf: &mut [f32]) {
        let n = buf.len();
        for v in buf.iter_mut() {
            *v = self.uniform_f32();
        }
        let mut off = 0;
        while off + 16 <= n {
            Self::normal_fill_16(&mut buf[off..off + 16]);
            off += 16;
        }
        if n % 16 != 0 {
            let start = n - 16;
            for i in 0..16 {
                buf[start + i] = self.uniform_f32();
            }
            Self::normal_fill_16(&mut buf[start..start + 16]);
        }
    }

    #[inline]
    fn normal_fill_16(d: &mut [f32]) {
        for j in 0..8 {
            let u1 = 1.0f32 - d[j];
            let u2 = d[j + 8];
            let rad = libm::sqrtf(-2.0 * libm::logf(u1));
            let th = TWO_PI_F32 * u2;
            d[j] = rad * libm::cosf(th);
            d[j + 8] = rad * libm::sinf(th);
        }
    }

    /// `nn.init.trunc_normal_(buf, mean, std, a, b)` filling `buf` in place.
    pub fn fill_trunc_normal(&mut self, buf: &mut [f32], mean: f32, std: f32, a: f32, b: f32) {
        let norm_cdf = |x: f64| 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
        let l = norm_cdf(((a - mean) / std) as f64);
        let u = norm_cdf(((b - mean) / std) as f64);
        let lo = (2.0 * l - 1.0) as f32;
        let span = (2.0 * u - 2.0 * l) as f32;
        let scale = std * std::f32::consts::SQRT_2;
        for v in buf.iter_mut() {
            let fu = self.uniform_f32();
            let x = lo + span * fu;
            let mut val = Self::erfinv_f32(x) * scale + mean;
            if val < a {
                val = a;
            }
            if val > b {
                val = b;
            }
            *v = val;
        }
    }

    /// Dropout keep-mask*scale: `out[i] = (u64 < keep_prob) ? 1/keep_prob : 0`, where
    /// `keep_prob = 1 - p`. Multiply your activations by this. Consumes a double uniform per element.
    pub fn fill_dropout_scale(&mut self, buf: &mut [f32], p: f32) {
        let keep = 1.0f32 - p;
        let inv = 1.0f32 / keep;
        for v in buf.iter_mut() {
            let kept = (self.uniform_f64() as f32) < keep;
            *v = if kept { inv } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_matches_torch_seed0() {
        let mut g = TorchRng::new(0);
        let exp = [0.4962565899, 0.7682217956, 0.0884774327];
        for e in exp {
            assert!((g.uniform_f32() - e).abs() < 1e-9, "uniform_f32");
        }
        let mut g = TorchRng::new(0);
        assert!((g.uniform_f64() - 0.97005300180655307).abs() < 1e-15, "uniform_f64");
    }

    #[test]
    fn randn_scalar_matches_torch_seed0() {
        let mut g = TorchRng::new(0);
        let mut v = [0.0f32; 4];
        g.fill_randn(&mut v);
        let exp = [1.5409961, -0.2934289, -2.1787894, 0.5684313];
        for (a, b) in v.iter().zip(exp) {
            assert!((a - b).abs() < 1e-5, "randn scalar got {a} want {b}");
        }
    }

    #[test]
    fn normal_fill_matches_torch_seed0() {
        let mut g = TorchRng::new(0);
        let mut v = [0.0f32; 16];
        g.fill_randn(&mut v);
        // torch.manual_seed(0); torch.randn(16)[0] == -1.1258398294
        assert!((v[0] - (-1.1258398294)).abs() < 1e-6, "normal_fill got {}", v[0]);
    }

    #[test]
    fn cache_carryover_across_calls() {
        // torch: manual_seed(0); randn(3); randn(4)  -- the odd randn(3) leaves a cached
        // normal that feeds randn(4)[0].
        let mut g = TorchRng::new(0);
        let mut a = [0.0f32; 3];
        g.fill_randn(&mut a);
        let mut b = [0.0f32; 4];
        g.fill_randn(&mut b);
        // torch randn(4) after randn(3): [0.568431, -1.084522, -1.398595, 0.403347]
        let exp = [0.568431, -1.084522, -1.398595, 0.403347];
        for (x, e) in b.iter().zip(exp) {
            assert!((x - e).abs() < 1e-5, "carryover got {x} want {e}");
        }
    }
}
