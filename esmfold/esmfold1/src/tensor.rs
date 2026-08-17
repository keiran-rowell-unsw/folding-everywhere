//! Minimal row-major, contiguous, fp32 tensor.
//!
//! Design choices for bit-close parity with PyTorch:
//! - Always contiguous, row-major (C order). `permute`/`transpose` materialize a
//!   new contiguous buffer so downstream ops never depend on stride trickery.
//! - All reductions/matmuls live in `ops` with a pinned accumulation order.

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "shape {:?} ({}) != data len {}",
            shape,
            n,
            data.len()
        );
        Tensor { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn filled(shape: &[usize], v: f32) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![v; n],
            shape: shape.to_vec(),
        }
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }

    /// Reshape in place (must preserve element count).
    pub fn reshape(mut self, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.data.len(), "reshape {:?} -> {:?}", self.shape, shape);
        self.shape = shape.to_vec();
        self
    }

    /// Row-major strides for the current shape.
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
        let n: usize = new_shape.len();
        let total: usize = self.data.len();
        let mut out = vec![0.0f32; total];
        // iterate output in row-major order, map to source offset
        let mut idx = vec![0usize; n];
        for o in 0..total {
            // compute source offset from multi-index
            let mut src = 0usize;
            for d in 0..n {
                src += idx[d] * new_strides_of_old[d];
            }
            out[o] = self.data[src];
            // increment multi-index (row-major over new_shape)
            for d in (0..n).rev() {
                idx[d] += 1;
                if idx[d] < new_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Tensor::new(out, new_shape)
    }

    /// 2D transpose convenience.
    pub fn t(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        self.permute(&[1, 0])
    }

    /// Max absolute element (debug helper).
    pub fn amax(&self) -> f32 {
        self.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }
}
