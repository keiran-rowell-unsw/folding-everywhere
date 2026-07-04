//! mmap'd safetensors loader. Returns fp32 `Tensor`s by name, upcasting F16
//! losslessly. Header is parsed manually to avoid self-referential borrows and
//! to sidestep alignment requirements (we convert via `from_le_bytes`).

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

pub struct Weights {
    mmap: Mmap,
    index: HashMap<String, Entry>,
}

impl Weights {
    /// Open either a safetensors file or a PyTorch `pytorch_model.bin` (ZIP+pickle);
    /// auto-detected by magic bytes.
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let index = if mmap.len() >= 4 && &mmap[0..4] == b"PK\x03\x04" {
            // PyTorch .bin (ZIP archive)
            crate::pth::index_pth(&mmap)
                .into_iter()
                .map(|e| (e.name, Entry { dtype: e.dtype, shape: e.shape, start: e.start, end: e.end }))
                .collect()
        } else {
            // safetensors
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
                    let shape: Vec<usize> = v["shape"].as_array().unwrap().iter()
                        .map(|x| x.as_u64().unwrap() as usize).collect();
                    let offs = v["data_offsets"].as_array().unwrap();
                    let start = data_start + offs[0].as_u64().unwrap() as usize;
                    let end = data_start + offs[1].as_u64().unwrap() as usize;
                    index.insert(k, Entry { dtype, shape, start, end });
                }
            }
            index
        };
        Ok(Weights { mmap, index })
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

    /// Fetch a tensor as fp32 (F16 upcast losslessly). Panics if name missing.
    pub fn get(&self, name: &str) -> Tensor {
        let e = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = &self.mmap[e.start..e.end];
        let data: Vec<f32> = match e.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => panic!("get() unsupported dtype {other} for {name}"),
        };
        Tensor::new(data, e.shape.clone())
    }

    pub fn get_i64(&self, name: &str) -> Vec<i64> {
        let e = self.index.get(name).unwrap_or_else(|| panic!("weight not found: {name}"));
        assert_eq!(e.dtype, "I64");
        let bytes = &self.mmap[e.start..e.end];
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}
