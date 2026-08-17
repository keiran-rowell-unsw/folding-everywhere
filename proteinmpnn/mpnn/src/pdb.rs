//! PDB backbone parser — a faithful port of `parse_PDB` / `parse_PDB_biounits`
//! from `protein_mpnn_utils.py`.
//!
//! The quirks below are all deliberate, because ProteinMPNN's residue numbering
//! is defined by that code rather than by the PDB spec:
//!
//! * `MSE` HETATM records are rewritten to `MET` ATOM records.
//! * Residues are keyed by `resSeq - 1` and the output runs over the *dense*
//!   range `min_resn ..= max_resn`, so gaps in the numbering become residues
//!   with unknown identity and NaN coordinates (later masked out).
//! * Insertion codes split a residue number into several entries, ordered by the
//!   insertion-code string ("" sorts first), matching Python's `sorted(...)`.
//! * The first occurrence of an atom wins (altloc A, in practice).
//! * Coordinates are parsed as f64 (numpy's default) and narrowed to f32 only at
//!   the end, matching `torch.from_numpy(X).to(torch.float32)`.

use std::collections::BTreeMap;

/// Three-letter residue names in ProteinMPNN's `alpha_3` order; the index into
/// this table is also the index into `alpha_1` = "ARNDCQEGHILKMFPSTWYV-".
const ALPHA_3: [&str; 21] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "GAP",
];
const ALPHA_1: &[u8] = b"ARNDCQEGHILKMFPSTWYV-";

/// Backbone atoms, in the order ProteinMPNN stacks them.
pub const BACKBONE: [&str; 4] = ["N", "CA", "C", "O"];

#[derive(Debug, Clone)]
pub struct Chain {
    pub id: char,
    /// One-letter sequence; '-' marks an unknown or absent residue.
    pub seq: String,
    /// `[L][4][3]` N/CA/C/O coordinates; NaN where an atom is missing.
    pub coords: Vec<[[f32; 3]; 4]>,
}

#[derive(Debug, Clone)]
pub struct Structure {
    pub name: String,
    pub chains: Vec<Chain>,
}

impl Structure {
    pub fn chain(&self, id: char) -> Option<&Chain> {
        self.chains.iter().find(|c| c.id == id)
    }
    pub fn chain_ids(&self) -> Vec<char> {
        self.chains.iter().map(|c| c.id).collect()
    }
}

#[derive(Default)]
struct ResEntry {
    resi: String,
    atoms: BTreeMap<String, [f64; 3]>,
}

/// Parse every chain of a PDB file. Chains come back in the order of
/// ProteinMPNN's `chain_alphabet` (A-Z, then a-z, then the numeric names).
pub fn parse_pdb(path: &str) -> std::io::Result<Structure> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();

    // chain -> resn -> insertion code -> entry. BTreeMap reproduces Python's
    // `sorted(...)` iteration order.
    let mut chains: BTreeMap<char, BTreeMap<i64, BTreeMap<String, ResEntry>>> = BTreeMap::new();

    for raw in text.lines() {
        let line = raw.trim_end_matches(['\r', '\n']);
        // HETATM/MSE -> ATOM/MET rewrite (selenomethionine).
        let owned;
        let line = if line.len() >= 20 && line.starts_with("HETATM") && &line[17..20] == "MSE" {
            owned = line.replacen("HETATM", "ATOM  ", 1).replace("MSE", "MET");
            owned.as_str()
        } else {
            line
        };
        if line.len() < 54 || !line.starts_with("ATOM") || !line.is_char_boundary(54) {
            continue;
        }
        let ch = line.as_bytes()[21] as char;
        let atom = line[12..16].trim().to_string();
        let resi = line[17..20].to_string();
        let resn_s = line[22..27].trim().to_string();
        if resn_s.is_empty() {
            continue;
        }
        let last = resn_s.chars().last().unwrap();
        let (resa, resn) = if last.is_alphabetic() {
            (last.to_string(), resn_s[..resn_s.len() - 1].parse::<i64>())
        } else {
            (String::new(), resn_s.parse::<i64>())
        };
        let resn = match resn {
            Ok(v) => v - 1,
            Err(_) => continue,
        };
        let xyz = match (
            line[30..38].trim().parse::<f64>(),
            line[38..46].trim().parse::<f64>(),
            line[46..54].trim().parse::<f64>(),
        ) {
            (Ok(x), Ok(y), Ok(z)) => [x, y, z],
            _ => continue,
        };

        let e = chains
            .entry(ch)
            .or_default()
            .entry(resn)
            .or_default()
            .entry(resa)
            .or_default();
        if e.resi.is_empty() {
            e.resi = resi;
        }
        e.atoms.entry(atom).or_insert(xyz); // first occurrence wins
    }

    let mut out = Vec::new();
    for letter in chain_alphabet() {
        let per_res = match chains.get(&letter) {
            Some(m) if !m.is_empty() => m,
            _ => continue,
        };
        let min_resn = *per_res.keys().next().unwrap();
        let max_resn = *per_res.keys().next_back().unwrap();

        let mut seq = String::new();
        let mut coords: Vec<[[f32; 3]; 4]> = Vec::new();
        for resn in min_resn..=max_resn {
            match per_res.get(&resn) {
                Some(insertions) => {
                    for entry in insertions.values() {
                        let idx = ALPHA_3.iter().position(|&a| a == entry.resi).unwrap_or(20);
                        seq.push(ALPHA_1[idx] as char);
                        let mut c = [[f32::NAN; 3]; 4];
                        for (ai, aname) in BACKBONE.iter().enumerate() {
                            if let Some(v) = entry.atoms.get(*aname) {
                                c[ai] = [v[0] as f32, v[1] as f32, v[2] as f32];
                            }
                        }
                        coords.push(c);
                    }
                }
                None => {
                    seq.push('-');
                    coords.push([[f32::NAN; 3]; 4]);
                }
            }
        }
        out.push(Chain { id: letter, seq, coords });
    }

    let stem = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Structure { name: stem, chains: out })
}

/// ProteinMPNN's `chain_alphabet` restricted to names that can occupy the
/// single-character chain column: A-Z, a-z, then "0".."9".
fn chain_alphabet() -> Vec<char> {
    let mut v: Vec<char> = ('A'..='Z').collect();
    v.extend('a'..='z');
    v.extend('0'..='9');
    v
}
