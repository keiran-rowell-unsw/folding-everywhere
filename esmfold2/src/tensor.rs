//! Minimal row-major, contiguous, fp32 tensor (shared with the v1 design).
//!
//! Bit-exactness conventions:
//! - Always contiguous, row-major (C order). `permute`/`transpose` materialize a
//!   new contiguous buffer so downstream ops never depend on stride trickery.
//! - All reductions/matmuls live in `ops` with a pinned accumulation order so the
//!   result is independent of thread count.

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape {:?} ({}) != data len {}", shape, n, data.len());
        Tensor { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn filled(shape: &[usize], v: f32) -> Self {
        let n: usize = shape.iter().product();
        Tensor { data: vec![v; n], shape: shape.to_vec() }
    }

    #[inline]
    pub fn numel(&self) -> usize { self.data.len() }
    #[inline]
    pub fn ndim(&self) -> usize { self.shape.len() }
    #[inline]
    pub fn dim(&self, i: usize) -> usize { self.shape[i] }

    /// Last-axis length (the "feature" dim for most ops).
    #[inline]
    pub fn last(&self) -> usize { self.shape[self.shape.len() - 1] }
    /// Product of all but the last axis (number of feature vectors).
    #[inline]
    pub fn rows(&self) -> usize { self.numel() / self.last() }

    pub fn reshape(mut self, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.data.len(), "reshape {:?} -> {:?}", self.shape, shape);
        self.shape = shape.to_vec();
        self
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1];
        }
        s
    }

    /// General permutation of axes; returns a new contiguous tensor.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let nd = self.shape.len();
        assert_eq!(axes.len(), nd);
        let in_strides = self.strides();
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides_of_old: Vec<usize> = axes.iter().map(|&a| in_strides[a]).collect();
        let n = new_shape.len();
        let total = self.data.len();
        let mut out = vec![0.0f32; total];
        let mut idx = vec![0usize; n];
        for o in 0..total {
            let mut src = 0usize;
            for d in 0..n {
                src += idx[d] * new_strides_of_old[d];
            }
            out[o] = self.data[src];
            for d in (0..n).rev() {
                idx[d] += 1;
                if idx[d] < new_shape[d] { break; }
                idx[d] = 0;
            }
        }
        Tensor::new(out, new_shape)
    }

    pub fn t(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        self.permute(&[1, 0])
    }

    pub fn amax(&self) -> f32 {
        self.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }

    /// Elementwise add (same shape).
    pub fn add(&self, o: &Tensor) -> Tensor {
        assert_eq!(self.shape, o.shape, "add shape {:?} vs {:?}", self.shape, o.shape);
        let data = self.data.iter().zip(&o.data).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn add_assign(&mut self, o: &Tensor) {
        assert_eq!(self.shape, o.shape);
        for (a, b) in self.data.iter_mut().zip(&o.data) { *a += *b; }
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|x| x * s).collect();
        Tensor::new(data, self.shape.clone())
    }
}

/// Round an fp32 value to bfloat16 and back (round-to-nearest-even), matching
/// what an explicit `.to(torch.bfloat16)` cast does inside the reference graph.
#[inline]
pub fn bf16_round(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// Elementwise bf16 round of a whole tensor.
pub fn bf16_round_tensor(t: &Tensor) -> Tensor {
    let data = t.data.iter().map(|&x| bf16_round(x)).collect();
    Tensor::new(data, t.shape.clone())
}
