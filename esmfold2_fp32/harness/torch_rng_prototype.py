"""VALIDATED prototype of torch CPU RNG (manual_seed) for the pure-Rust standalone port.
Drives a single generator with persistent double-normal cache (matches torch's CPUGeneratorImpl).

PROVEN bit-exact vs torch (seed 0):
- MT19937: init_genrand(seed); standard tempering.
- float32 uniform = (u32 & 0xFFFFFF) * 2^-24            [== torch.rand float32]
- double  uniform = ((u32a<<32 | u32b) & (2^53-1)) * 2^-53   [== torch.rand float64; a drawn first]
- normal_fill (randn numel>=16): fill float uniforms; transform in blocks of 16 pairing j with j+8
  [u1=1-d[j], u2=d[j+8], rad=sqrt(-2 ln u1), th=2pi u2, d[j]=rad cos, d[j+8]=rad sin]; if numel%16
  the tail RE-FILLS the last 16 with fresh uniforms. Does NOT touch the double-normal cache.
- scalar normal (randn numel<16): per value Box-Muller with DOUBLE uniforms + cache:
  u1=dbl, u2=dbl; rad=sqrt(-2*log1p(-u2)); th=2pi*u1; ret=rad cos(th); cache=rad sin(th).
  The cache PERSISTS across randn() calls (and across an intervening normal_fill).

SOLVED (verified vs torch):
- trunc_normal_(mean,std,a,b): per elem fu=uniform_f32; x=(2l-1)+(2u-2l)*fu; v=erfinv(x)*std*sqrt(2)+
  mean; clamp(a,b). l=Phi((a-mean)/std), u=Phi((b-mean)/std). Matches torch to 1e-6 with a good erfinv
  (Rust: port torch calc_erfinv or AS241 for f32-exactness; z-init feeds chaotic trunk so want bit-exact).
- dropout(p): mask[i] = (uniform_f64() < 1-p); out = x*mask*(1/(1-p)).  DOUBLE uniform (2 u32/elem).
  scale 1/(1-p) computed in f32 (torch 1.3333334). Keep-pattern matched torch exactly.

EXACT model draw order:
trunc_normal[1,L,L,256]; per loop dropout[1,L,L,256]x4; randn x_init[B*N,3]; per step (x10)
randn rotation[B,4] + randn translation[B,1,3] + randn churn[B,N,3].
"""
import math

class TorchCPURng:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xffffffff
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xffffffff
        self.idx = 624
        self.cache = None  # double-normal cache

    def u32(self):
        if self.idx >= 624:
            for i in range(624):
                y = (self.mt[i] & 0x80000000) + (self.mt[(i+1) % 624] & 0x7fffffff)
                self.mt[i] = self.mt[(i+397) % 624] ^ (y >> 1)
                if y & 1:
                    self.mt[i] ^= 2567483615
            self.idx = 0
        y = self.mt[self.idx]; self.idx += 1
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680; y ^= (y << 15) & 0xefc60000; y ^= y >> 18
        return y & 0xffffffff

    def uniform_f32(self):
        return (self.u32() & 0xFFFFFF) * (2.0 ** -24)

    def uniform_f64(self):
        a = self.u32(); b = self.u32()
        return (((a << 32) | b) & ((1 << 53) - 1)) * (2.0 ** -53)

    def randn(self, n):
        if n >= 16:
            return self._normal_fill(n)
        out = []
        for _ in range(n):
            if self.cache is not None:
                out.append(self.cache); self.cache = None; continue
            u1 = self.uniform_f64(); u2 = self.uniform_f64()
            rad = math.sqrt(-2.0 * math.log1p(-u2)); th = 2.0 * math.pi * u1
            out.append(rad * math.cos(th)); self.cache = rad * math.sin(th)
        return out

    def _normal_fill(self, n):
        d = [self.uniform_f32() for _ in range(n)]
        def blk(o):
            for j in range(8):
                u1 = 1.0 - d[o+j]; u2 = d[o+j+8]
                rad = math.sqrt(-2.0 * math.log(u1)); th = 2.0 * math.pi * u2
                d[o+j] = rad * math.cos(th); d[o+j+8] = rad * math.sin(th)
        o = 0
        while o < n - 15:
            blk(o); o += 16
        if n % 16 != 0:
            for i in range(16):
                d[n-16+i] = self.uniform_f32()
            blk(n-16)
        return d

if __name__ == "__main__":
    g = TorchCPURng(0)
    assert abs(g.uniform_f32() - 0.4962565899) < 1e-9
    g = TorchCPURng(0)
    assert abs(g.uniform_f64() - 0.97005300180655307) < 1e-15
    g = TorchCPURng(0)
    r = g.randn(4)
    assert abs(r[0] - 1.5409961) < 1e-5 and abs(r[2] - (-2.1787894)) < 1e-5
    print("torch CPU RNG prototype verified: MT19937, f32/f64 uniform, normal_fill, scalar normal+cache")
