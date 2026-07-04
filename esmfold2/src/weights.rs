//! mmap'd safetensors loader supporting both single-file and sharded
//! (index.json) checkpoints. Returns fp32 `Tensor`s by name (F16/BF16 upcast
//! losslessly). `_extra_state` (U8 TE junk) is indexed but never fetched.

use crate::tensor::Tensor;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Debug)]
struct Entry {
    shard: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize, // absolute byte offset within its shard
    end: usize,
}

pub struct Weights {
    mmaps: Vec<Mmap>,
    index: HashMap<String, Entry>,
}

fn parse_header(mmap: &Mmap, shard: usize, index: &mut HashMap<String, Entry>) {
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let json: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + header_len]).expect("safetensors header json");
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

impl Weights {
    /// Open a single safetensors file.
    pub fn open(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut index = HashMap::new();
        parse_header(&mmap, 0, &mut index);
        Ok(Weights { mmaps: vec![mmap], index })
    }

    /// Open a sharded checkpoint from its `*.index.json` file.
    pub fn open_sharded(index_json_path: &str) -> std::io::Result<Self> {
        let dir = Path::new(index_json_path).parent().unwrap();
        let txt = std::fs::read_to_string(index_json_path)?;
        let j: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let wm = j["weight_map"].as_object().expect("weight_map");
        // Unique shard filenames in deterministic order.
        let mut shard_files: Vec<String> = wm.values().map(|v| v.as_str().unwrap().to_string()).collect();
        shard_files.sort();
        shard_files.dedup();
        let mut mmaps = Vec::new();
        let mut shard_idx: HashMap<String, usize> = HashMap::new();
        let mut index = HashMap::new();
        for (i, fname) in shard_files.iter().enumerate() {
            let p = dir.join(fname);
            let file = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&file)? };
            parse_header(&mmap, i, &mut index);
            mmaps.push(mmap);
            shard_idx.insert(fname.clone(), i);
        }
        Ok(Weights { mmaps, index })
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
        let bytes = &self.mmaps[e.shard][e.start..e.end];
        let data: Vec<f32> = match e.dtype.as_str() {
            // Bulk memcpy: the file stores little-endian F32, so this is bit-identical
            // to per-element from_le_bytes on LE hosts, but a single fast copy.
            "F32" => {
                let n = bytes.len() / 4;
                let mut v = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, n * 4);
                }
                v
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
        Tensor::new(data, e.shape.clone())
    }

    /// Fetch as a flat Vec<f32> (no shape assertion beyond element count).
    pub fn get_vec(&self, name: &str) -> Vec<f32> {
        self.get(name).data
    }
}
