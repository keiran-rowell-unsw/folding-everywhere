//! Safetensors loader returning fp32 `Tensor`s by name (F16/BF16 upcast
//! losslessly). `_extra_state` (U8 TE junk) is indexed but never fetched.
//!
//! Two backends:
//! * **native** (default): mmap'd single-file and sharded checkpoints from disk.
//! * **wasm**: byte-slice loader — caller passes the file bytes from JS.

use crate::tensor::Tensor;
use std::collections::HashMap;

#[cfg(feature = "native")]
use memmap2::Mmap;

#[derive(Clone, Debug)]
struct Entry {
    shard: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize, // absolute byte offset within its shard
    end: usize,
}

// ---------------------------------------------------------------------------
// Native (mmap) backend
// ---------------------------------------------------------------------------

#[cfg(feature = "native")]
pub struct Weights {
    mmaps: Vec<Mmap>,
    index: HashMap<String, Entry>,
}

#[cfg(feature = "native")]
fn parse_header(mmap: &Mmap, shard: usize, index: &mut HashMap<String, Entry>) {
    parse_header_bytes(mmap.as_ref(), shard, index);
}

#[cfg(feature = "native")]
impl Weights {
    /// Open a single safetensors file.
    pub fn open(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut index = HashMap::new();
        parse_header(&mmap, 0, &mut index);
        Ok(Weights { mmaps: vec![mmap], index })
    }

    /// Open a sharded checkpoint from its `*.index.json` file.
    pub fn open_sharded(index_json_path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::path::Path;
        let dir = Path::new(index_json_path).parent().unwrap();
        let txt = std::fs::read_to_string(index_json_path)?;
        let j: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let wm = j["weight_map"].as_object().expect("weight_map");
        // Unique shard filenames in deterministic order.
        let mut shard_files: Vec<String> =
            wm.values().map(|v| v.as_str().unwrap().to_string()).collect();
        shard_files.sort();
        shard_files.dedup();
        let mut mmaps = Vec::new();
        let mut index = HashMap::new();
        for (i, fname) in shard_files.iter().enumerate() {
            let p = dir.join(fname);
            let file = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            parse_header(&mmap, i, &mut index);
            mmaps.push(mmap);
        }
        Ok(Weights { mmaps, index })
    }

    fn shard_bytes(&self, shard: usize, start: usize, end: usize) -> &[u8] {
        &self.mmaps[shard][start..end]
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

    /// Fetch a tensor as fp32 (F16/BF16 upcast). Panics if name missing.
    pub fn get(&self, name: &str) -> Tensor {
        let e = self.index.get(name).unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = self.shard_bytes(e.shard, e.start, e.end);
        decode_tensor(bytes, &e.dtype, &e.shape, name)
    }

    /// Fetch as a flat Vec<f32> (no shape assertion beyond element count).
    pub fn get_vec(&self, name: &str) -> Vec<f32> {
        self.get(name).data
    }
}

// ---------------------------------------------------------------------------
// Wasm (byte-slice) backend
// ---------------------------------------------------------------------------

/// In wasm builds the caller passes raw safetensors bytes from JS.
/// Multiple shards are supported by calling `add_shard` in order.
#[cfg(not(feature = "native"))]
pub struct Weights {
    shards: Vec<Vec<u8>>,
    index: HashMap<String, Entry>,
}

#[cfg(not(feature = "native"))]
impl Weights {
    /// Create an empty weight store. Call `add_shard` for each shard buffer.
    pub fn new() -> Self {
        Weights { shards: Vec::new(), index: HashMap::new() }
    }

    /// Load a single safetensors blob provided as bytes (e.g., from JS).
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let mut w = Self::new();
        w.add_shard(data);
        w
    }

    /// Append one shard's bytes and parse its header.
    pub fn add_shard(&mut self, data: Vec<u8>) {
        let shard = self.shards.len();
        parse_header_bytes(&data, shard, &mut self.index);
        self.shards.push(data);
    }

    fn shard_bytes(&self, shard: usize, start: usize, end: usize) -> &[u8] {
        &self.shards[shard][start..end]
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

    /// Fetch a tensor as fp32 (F16/BF16 upcast). Panics if name missing.
    pub fn get(&self, name: &str) -> Tensor {
        let e = self.index.get(name).unwrap_or_else(|| panic!("weight not found: {name}"));
        let bytes = self.shard_bytes(e.shard, e.start, e.end);
        decode_tensor(bytes, &e.dtype, &e.shape, name)
    }

    /// Fetch as a flat Vec<f32> (no shape assertion beyond element count).
    pub fn get_vec(&self, name: &str) -> Vec<f32> {
        self.get(name).data
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn parse_header_bytes(data: &[u8], shard: usize, index: &mut HashMap<String, Entry>) {
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let json: serde_json::Value =
        serde_json::from_slice(&data[8..8 + header_len]).expect("safetensors header json");
    let data_start = 8 + header_len;
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
            index.insert(k, Entry { shard, dtype, shape, start, end });
        }
    }
}

fn decode_tensor(bytes: &[u8], dtype: &str, shape: &[usize], name: &str) -> Tensor {
    let data: Vec<f32> = match dtype {
        "F32" => {
            // Fast path on little-endian hosts: the file is LE F32, so the bytes
            // are already in the correct in-memory layout — one bulk copy suffices.
            // On big-endian hosts we fall back to per-element byte-swapping.
            #[cfg(target_endian = "little")]
            {
                let n = bytes.len() / 4;
                let mut v = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * 4);
                }
                v
            }
            #[cfg(not(target_endian = "little"))]
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
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
    Tensor::new(data, shape.to_vec())
}
