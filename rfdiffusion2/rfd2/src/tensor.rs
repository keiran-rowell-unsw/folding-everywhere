//! Minimal row-major, contiguous, fp32 tensor.
//!
//! Same design contract as the ESMFold port: always contiguous C order, and
//! `permute` materializes a fresh buffer so no downstream op depends on stride
//! trickery. Every reduction lives in `ops` with a pinned accumulation order,
//! which is what makes the port deterministic and thread-count-independent.

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
        Tensor { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn filled(shape: &[usize], v: f32) -> Self {
        let n: usize = shape.iter().product();
        Tensor { data: vec![v; n], shape: shape.to_vec() }
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

    /// Size of the last axis.
    #[inline]
    pub fn last(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }

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
        let src_strides: Vec<usize> = axes.iter().map(|&a| in_strides[a]).collect();
        let total = self.data.len();
        let mut out = vec![0.0f32; total];
        let mut idx = vec![0usize; nd];
        for o in 0..total {
            let mut src = 0usize;
            for d in 0..nd {
                src += idx[d] * src_strides[d];
            }
            out[o] = self.data[src];
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < new_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Tensor::new(out, new_shape)
    }

    pub fn t(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        self.permute(&[1, 0])
    }

    /// Elementwise add (same shape).
    pub fn add(&self, o: &Tensor) -> Tensor {
        assert_eq!(self.shape, o.shape, "add shape");
        let data = self.data.iter().zip(&o.data).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Elementwise multiply (same shape).
    pub fn mul(&self, o: &Tensor) -> Tensor {
        assert_eq!(self.shape, o.shape, "mul shape");
        let data = self.data.iter().zip(&o.data).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn scale(&self, s: f32) -> Tensor {
        Tensor::new(self.data.iter().map(|v| v * s).collect(), self.shape.clone())
    }

    /// Concatenate a list of tensors along the last axis.
    pub fn cat_last(parts: &[&Tensor]) -> Tensor {
        assert!(!parts.is_empty());
        let lead: usize = parts[0].numel() / parts[0].last();
        let widths: Vec<usize> = parts.iter().map(|p| p.last()).collect();
        let total_w: usize = widths.iter().sum();
        for p in parts {
            assert_eq!(p.numel() / p.last(), lead, "cat_last leading mismatch");
        }
        // `with_capacity` + `extend` rather than `vec![0.0; n]`: these buffers can
        // be tens of MB in the decoder, and zero-filling them first doubles the
        // page-fault traffic for data we immediately overwrite.
        let mut out: Vec<f32> = Vec::with_capacity(lead * total_w);
        for r in 0..lead {
            for (p, &w) in parts.iter().zip(&widths) {
                out.extend_from_slice(&p.data[r * w..r * w + w]);
            }
        }
        let mut shape = parts[0].shape.clone();
        let n = shape.len();
        shape[n - 1] = total_w;
        Tensor::new(out, shape)
    }

    pub fn amax(&self) -> f32 {
        self.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }
}
