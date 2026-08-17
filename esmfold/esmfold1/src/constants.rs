//! Residue constants for the structure module / all-atom reconstruction.
//! Loaded from the exported safetensors (python/export_constants.py).

use crate::weights::Weights;

/// AF2 restype order (A=0 .. V=19, X=20). Matches residue_constants.restype_order_with_x.
pub const RESTYPES: &str = "ARNDCQEGHILKMFPSTWYV";

pub fn aa_to_restype(c: char) -> usize {
    RESTYPES.find(c.to_ascii_uppercase()).unwrap_or(20)
}

pub fn seq_to_aatype(seq: &str) -> Vec<usize> {
    seq.chars().map(aa_to_restype).collect()
}

pub struct Constants {
    pub default_frames: Vec<f32>,            // [21,8,4,4]
    pub atom14_to_rigid_group: Vec<usize>,   // [21,14]
    pub atom14_mask: Vec<f32>,               // [21,14]
    pub atom14_rigid_group_positions: Vec<f32>, // [21,14,3]
    pub atom14_to_atom37: Vec<usize>,        // [21,14]
    pub atom37_to_atom14: Vec<usize>,        // [21,37]
    pub atom37_mask: Vec<f32>,               // [21,37]
}

/// Residue constants embedded at compile time so the binary is self-contained.
static EMBEDDED: &[u8] = include_bytes!("../fixtures/constants/residue_constants.safetensors");

impl Constants {
    pub fn load(path: &str) -> Self {
        let w = Weights::open(path).expect("constants safetensors");
        Self::from_getter(|n| w.get(n).data)
    }

    /// Load from the embedded constants (no external file needed).
    pub fn embedded() -> Self {
        let m = parse_st(EMBEDDED);
        Self::from_getter(|n| m.get(n).cloned().unwrap_or_else(|| panic!("const {n}")))
    }

    fn from_getter(f: impl Fn(&str) -> Vec<f32>) -> Self {
        let u = |n: &str| f(n).iter().map(|&x| x.round() as usize).collect::<Vec<_>>();
        Constants {
            default_frames: f("restype_rigid_group_default_frame"),
            atom14_to_rigid_group: u("restype_atom14_to_rigid_group"),
            atom14_mask: f("restype_atom14_mask"),
            atom14_rigid_group_positions: f("restype_atom14_rigid_group_positions"),
            atom14_to_atom37: u("restype_atom14_to_atom37"),
            atom37_to_atom14: u("restype_atom37_to_atom14"),
            atom37_mask: f("restype_atom37_mask"),
        }
    }
}

/// Minimal in-memory safetensors -> {name: f32 vec} (F32/F16 only, for constants).
fn parse_st(bytes: &[u8]) -> std::collections::HashMap<String, Vec<f32>> {
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
    let data = &bytes[8 + hlen..];
    let mut out = std::collections::HashMap::new();
    if let serde_json::Value::Object(map) = json {
        for (k, v) in map {
            if k == "__metadata__" {
                continue;
            }
            let dt = v["dtype"].as_str().unwrap();
            let o = v["data_offsets"].as_array().unwrap();
            let (s, e) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            let slice = &data[s..e];
            let vals: Vec<f32> = match dt {
                "F32" => slice.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                "F16" => slice.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
                other => panic!("const dtype {other}"),
            };
            out.insert(k, vals);
        }
    }
    out
}
