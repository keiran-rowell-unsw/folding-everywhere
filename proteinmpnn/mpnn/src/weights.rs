//! mmap'd weight loader. Serves fp32 `Tensor`s by name from either a PyTorch
//! `.pt` checkpoint (ZIP + pickle, see `pth.rs`) or a safetensors file — the
//! format is auto-detected from the magic bytes. Fixtures written by the Python
//! harness are safetensors; the shipped ProteinMPNN weights are `.pt`.

use crate::tensor::Tensor;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

#[derive(Clone, Debug)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize, // ABSOLUTE byte offset in the file
    end: usize,
}

/// Where the bytes live: an mmap'd file, or a slice baked into the binary.
///
/// ProteinMPNN checkpoints are only ~6.7 MB, so unlike ESMFold the whole model
/// can be embedded with `include_bytes!` and the executable needs no companion
/// data file at all.
enum Backing {
    File(Mmap),
    Static(&'static [u8]),
}

impl Backing {
    #[inline]
    fn bytes(&self) -> &[u8] {
        match self {
            Backing::File(m) => m,
            Backing::Static(s) => s,
        }
    }
}

pub struct Weights {
    backing: Backing,
    index: HashMap<String, Entry>,
}

impl Weights {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let index = Self::build_index(&mmap)?;
        Ok(Weights { backing: Backing::File(mmap), index })
    }

    /// Build a `Weights` over bytes compiled into the binary.
    pub fn from_static(bytes: &'static [u8]) -> std::io::Result<Self> {
        let index = Self::build_index(bytes)?;
        Ok(Weights { backing: Backing::Static(bytes), index })
    }

    fn build_index(mmap: &[u8]) -> std::io::Result<HashMap<String, Entry>> {
        let index = if mmap.len() >= 4 && &mmap[0..4] == b"PK\x03\x04" {
            crate::pth::index_pth(&mmap)
                .into_iter()
                .map(|e| {
                    (e.name, Entry { dtype: e.dtype, shape: e.shape, start: e.start, end: e.end })
                })
                .collect()
        } else {
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let json: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let data_start = 8 + header_len;
            let mut index = HashMap::new();
            if let serde_json::Value::Object(map) = json {
                for (k, v) in map {
                    if k == "__metadata__" {
                        continue;
                    }
                    let dtype = v["dtype"].as_str().unwrap().to_string();
                    let shape: Vec<usize> = v["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect();
                    let offs = v["data_offsets"].as_array().unwrap();
                    let start = data_start + offs[0].as_u64().unwrap() as usize;
                    let end = data_start + offs[1].as_u64().unwrap() as usize;
                    index.insert(k, Entry { dtype, shape, start, end });
                }
            }
            index
        };
        Ok(index)
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|e| e.shape.as_slice())
    }

    /// Fetch a tensor as fp32. Panics if the name is missing.
    pub fn get(&self, name: &str) -> Tensor {
        let e = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = &self.backing.bytes()[e.start..e.end];
        let data: Vec<f32> = match e.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F64" => bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            "I64" => bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            "I32" => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            other => panic!("unsupported dtype {other} for {name}"),
        };
        Tensor::new(data, e.shape.clone())
    }

    /// Fetch an integer tensor as `i64` (fixtures store E_idx, S, decoding_order).
    pub fn get_i64(&self, name: &str) -> (Vec<i64>, Vec<usize>) {
        let e = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = &self.backing.bytes()[e.start..e.end];
        let data: Vec<i64> = match e.dtype.as_str() {
            "I64" => bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            "I32" => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            other => panic!("unsupported int dtype {other} for {name}"),
        };
        (data, e.shape.clone())
    }
}
