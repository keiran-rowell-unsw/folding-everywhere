//! Standalone featurization: turn a bare amino-acid sequence into the exact
//! feature tensors that `ESMFold2.forward` expects, reproducing PyTorch's
//! `prepare_protein_features` (single chain, no MSA / mods / bonds). The constant
//! tables (atom templates `PROTEIN_REF_POS`, heavy-atom order, ESM-C vocab, charges)
//! are embedded from `featurize_tables.json`, so the fold needs no Python.

use serde_json::Value;
use std::collections::HashMap;

const TABLES_JSON: &str = include_str!("featurize_tables.json");

pub struct Tables {
    one_to_three: HashMap<String, String>,
    res_to_restype: HashMap<String, i64>,
    vocab: HashMap<String, i64>,
    heavy_atoms: HashMap<String, Vec<String>>,
    ref_pos: HashMap<String, HashMap<String, [f32; 3]>>,
    charged: HashMap<(String, String), i64>,
    element_num: HashMap<String, i64>,
    unk_res_type: i64,
}

/// All per-protein features (single chain). Token-level arrays have length L,
/// atom-level arrays length n_atoms (= ceil(n_real/32)*32).
pub struct ProteinFeatures {
    pub l: usize,
    pub n_atoms: usize,
    pub res_type: Vec<i64>,             // [L]
    pub input_ids: Vec<i64>,           // [L]  (ESM-C tokens, no BOS/EOS)
    pub ref_pos: Vec<f32>,             // [n_atoms*3]
    pub ref_element: Vec<i64>,         // [n_atoms]
    pub ref_charge: Vec<i64>,          // [n_atoms]
    pub ref_atom_name_chars: Vec<i64>, // [n_atoms*4]
    pub ref_space_uid: Vec<i64>,       // [n_atoms]
    pub atom_attention_mask: Vec<bool>,// [n_atoms]
    pub atom_to_token: Vec<i64>,       // [n_atoms]
    pub distogram_atom_idx: Vec<i64>,  // [L]
}

fn obj_str_map(v: &Value) -> HashMap<String, String> {
    v.as_object().unwrap().iter().map(|(k, x)| (k.clone(), x.as_str().unwrap().to_string())).collect()
}
fn obj_int_map(v: &Value) -> HashMap<String, i64> {
    v.as_object().unwrap().iter().map(|(k, x)| (k.clone(), x.as_i64().unwrap())).collect()
}

impl Tables {
    pub fn load() -> Self {
        let j: Value = serde_json::from_str(TABLES_JSON).expect("featurize_tables.json");
        let heavy_atoms = j["PROTEIN_HEAVY_ATOMS"].as_object().unwrap().iter()
            .map(|(k, v)| (k.clone(), v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()))
            .collect();
        let ref_pos = j["PROTEIN_REF_POS"].as_object().unwrap().iter()
            .map(|(res, atoms)| {
                let m = atoms.as_object().unwrap().iter().map(|(a, p)| {
                    let arr = p.as_array().unwrap();
                    (a.clone(), [arr[0].as_f64().unwrap() as f32, arr[1].as_f64().unwrap() as f32, arr[2].as_f64().unwrap() as f32])
                }).collect();
                (res.clone(), m)
            }).collect();
        let charged = j["PROTEIN_CHARGED_ATOMS"].as_array().unwrap().iter()
            .map(|t| { let a = t.as_array().unwrap();
                ((a[0].as_str().unwrap().to_string(), a[1].as_str().unwrap().to_string()), a[2].as_i64().unwrap()) })
            .collect();
        Tables {
            one_to_three: obj_str_map(&j["PROTEIN_1TO3"]),
            res_to_restype: obj_int_map(&j["PROTEIN_RESIDUE_TO_RES_TYPE"]),
            vocab: obj_int_map(&j["ESM_PROTEIN_VOCAB"]),
            heavy_atoms,
            ref_pos,
            charged,
            element_num: obj_int_map(&j["ELEMENT_TO_ATOMIC_NUM"]),
            unk_res_type: j["PROTEIN_UNK_RES_TYPE"].as_i64().unwrap(),
        }
    }

    pub fn featurize(&self, sequence: &str) -> ProteinFeatures {
        assert!(!sequence.is_empty(), "sequence must be non-empty");
        let unk = "UNK".to_string();
        let xid = *self.vocab.get("X").unwrap();
        let l = sequence.chars().count();

        let mut res_type = Vec::with_capacity(l);
        let mut input_ids = Vec::with_capacity(l);
        let mut distogram_atom_idx = Vec::with_capacity(l);
        // (t_idx, name, element, charge, ref_pos)
        let mut atoms: Vec<(i64, String, String, i64, [f32; 3])> = Vec::new();

        for (t_idx, c) in sequence.chars().enumerate() {
            let letter = c.to_string();
            let res_3 = self.one_to_three.get(&letter).unwrap_or(&unk);
            let names = self.heavy_atoms.get(res_3).unwrap_or_else(|| panic!("no heavy atoms for {res_3}"));
            res_type.push(*self.res_to_restype.get(res_3).unwrap_or(&self.unk_res_type));
            input_ids.push(*self.vocab.get(&letter).unwrap_or(&xid));

            let atom_start = atoms.len();
            for name in names {
                let charge = *self.charged.get(&(res_3.clone(), name.clone())).unwrap_or(&0);
                let element = name.chars().next().unwrap().to_string();
                let pos = self.ref_pos[res_3][name];
                atoms.push((t_idx as i64, name.clone(), element, charge, pos));
            }
            let rep = if names.iter().any(|n| n == "CB") { "CB" } else { "CA" };
            let rep_off = names.iter().position(|n| n == rep).unwrap();
            distogram_atom_idx.push((atom_start + rep_off) as i64);
        }

        let n_real = atoms.len();
        let n_atoms = if n_real > 0 { ((n_real + 31) / 32) * 32 } else { 32 };

        let mut ref_pos = vec![0.0f32; n_atoms * 3];
        let mut ref_element = vec![0i64; n_atoms];
        let mut ref_charge = vec![0i64; n_atoms];
        let mut ref_atom_name_chars = vec![0i64; n_atoms * 4];
        let mut ref_space_uid = vec![0i64; n_atoms];
        let mut atom_attention_mask = vec![false; n_atoms];
        let mut atom_to_token = vec![0i64; n_atoms];

        for (i, (t_idx, name, element, charge, pos)) in atoms.iter().enumerate() {
            ref_pos[i * 3] = pos[0];
            ref_pos[i * 3 + 1] = pos[1];
            ref_pos[i * 3 + 2] = pos[2];
            ref_element[i] = self.element_num[element];
            ref_charge[i] = *charge;
            let enc = encode_atom_name(name);
            ref_atom_name_chars[i * 4..i * 4 + 4].copy_from_slice(&enc);
            ref_space_uid[i] = *t_idx;
            atom_attention_mask[i] = true;
            atom_to_token[i] = *t_idx;
        }

        ProteinFeatures {
            l, n_atoms, res_type, input_ids, ref_pos, ref_element, ref_charge,
            ref_atom_name_chars, ref_space_uid, atom_attention_mask, atom_to_token, distogram_atom_idx,
        }
    }
}

impl ProteinFeatures {
    /// One-hot atomic-number features [n_atoms*128], zeroed for padding atoms.
    pub fn ref_element_onehot(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.n_atoms * 128];
        for i in 0..self.n_atoms {
            if self.atom_attention_mask[i] {
                v[i * 128 + self.ref_element[i] as usize] = 1.0;
            }
        }
        v
    }
    /// One-hot atom-name features [n_atoms*256] = 4 chars x 64 (char-major), zeroed for padding.
    pub fn ref_atom_name_onehot(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.n_atoms * 256];
        for i in 0..self.n_atoms {
            if self.atom_attention_mask[i] {
                for j in 0..4 {
                    let c = self.ref_atom_name_chars[i * 4 + j] as usize;
                    v[i * 256 + j * 64 + c] = 1.0;
                }
            }
        }
        v
    }
    /// One-hot over 33 res-type classes [L*33] (used for aatype / profile / msa one-hot).
    pub fn res_type_onehot(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; self.l * 33];
        for i in 0..self.l {
            v[i * 33 + self.res_type[i] as usize] = 1.0;
        }
        v
    }
}

/// `name.ljust(4)[:4]` then ord(c)-32 (space -> 0). Matches torch `_encode_atom_name`.
fn encode_atom_name(name: &str) -> [i64; 4] {
    let mut out = [0i64; 4];
    for (i, c) in name.chars().take(4).enumerate() {
        out[i] = (c as i64) - 32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn crambin_features_match_reference() {
        // sequence from python/proteins10.py crambin46
        let seq = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN";
        let t = Tables::load();
        let f = t.featurize(seq);
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let rj: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("feat_crambin46.json")).unwrap()).unwrap();
        let as_i64 = |k: &str| rj[k].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>();
        assert_eq!(f.l, 46);
        assert_eq!(f.n_atoms, rj["n_atoms"].as_i64().unwrap() as usize);
        assert_eq!(f.res_type, as_i64("res_type"), "res_type");
        assert_eq!(f.input_ids, as_i64("input_ids"), "input_ids");
        assert_eq!(f.atom_to_token, as_i64("atom_to_token"), "atom_to_token");
        assert_eq!(f.distogram_atom_idx, as_i64("distogram_atom_idx"), "distogram_atom_idx");
        assert_eq!(f.ref_element, as_i64("ref_element"), "ref_element");
        assert_eq!(f.ref_charge, as_i64("ref_charge"), "ref_charge");
        assert_eq!(f.ref_space_uid, as_i64("ref_space_uid"), "ref_space_uid");
        // ref_pos + atom_name_chars vs npy
        let rp = crate::parity::read_npy_f32(&dir.join("feat_crambin46_ref_pos.npy"));
        let maxd = f.ref_pos.iter().zip(&rp.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(maxd == 0.0, "ref_pos max diff {maxd}");
        let (anc, _) = crate::parity::read_npy_i64(&dir.join("feat_crambin46_atom_name_chars.npy"));
        assert_eq!(f.ref_atom_name_chars, anc, "ref_atom_name_chars");
    }

    #[test]
    fn crambin_atom_onehots_match_fold_fixtures() {
        // Validate the masked one-hot element/name features against the actual fold
        // fixtures (ie_ref_element [N,128], ie_ref_atom_name_chars [N,256]) used by bench_fold.
        let seq = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN";
        let f = Tables::load().featurize(seq);
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let el = crate::parity::read_npy_f32(&dir.join("fold_crambin46_ie_ref_element.npy"));
        let nm = crate::parity::read_npy_f32(&dir.join("fold_crambin46_ie_ref_atom_name_chars.npy"));
        let my_el = f.ref_element_onehot();
        let my_nm = f.ref_atom_name_onehot();
        assert_eq!(my_el.len(), el.data.len(), "element onehot len");
        assert_eq!(my_nm.len(), nm.data.len(), "name onehot len");
        let de = my_el.iter().zip(&el.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dn = my_nm.iter().zip(&nm.data).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(de == 0.0, "ref_element onehot max diff {de}");
        assert!(dn == 0.0, "ref_atom_name onehot max diff {dn}");
    }
}
