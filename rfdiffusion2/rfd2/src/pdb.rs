//! Rung 4c — `rf_diffusion/parsers.py:parse_pdb_lines_target`.
//!
//! A faithful port, quirks included. The quirks *are* the specification: each
//! one changes which atom lands in which slot, and a slot shift is invisible
//! until geometry comes out wrong many rungs later.
//!
//! What the reference does, in order:
//!
//! 1. **Residue identity comes from first occurrence.** `first_atom_iter`
//!    yields the first ATOM line for each `(chain, resSeq)` pair and ignores
//!    every later one, so a residue's *name* is taken from its first atom line
//!    even if later lines disagree.
//! 2. **Unknown residue names become token 20 (UNK)**, not an error.
//! 3. **Atom names are matched stripped**, against only the first `NHEAVY` (23)
//!    entries of `aa2long` for that residue, and the **first match wins**
//!    (`break`) — a duplicated atom name keeps the earlier coordinate.
//! 4. Missing atoms stay NaN, become the mask, and are then **zeroed**.
//! 5. Duplicate `(chain, resSeq)` rows are removed a second time, keeping the
//!    first.
//! 6. HETATM lines are taken only when `parse_hetatom`, and are skipped when
//!    column 78 (`l[77]`) is `H` — i.e. hydrogens are dropped by *element
//!    column*, not by atom name.
//!
//! Column indices are byte offsets into the raw line, exactly as the reference
//! slices them; PDB is a fixed-column format and using whitespace splitting
//! here would break on ligands with 4-character atom names.

use crate::chemical_gen::{AA2LONG, NHEAVY, NUM2AA};

/// One parsed polymer residue.
#[derive(Debug, Clone)]
pub struct Residue {
    pub chain: String,
    pub res_seq: i64,
    pub name: String,
    /// Token index, or 20 (UNK) for an unrecognised residue name.
    pub token: i64,
}

/// One parsed heteroatom (ligand / ion / ORI marker).
#[derive(Debug, Clone)]
pub struct HetAtom {
    pub idx: i64,
    pub atom_id: String,
    pub atom_type: String,
    pub name: String,
    pub res_idx: i64,
    pub xyz: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct TargetFeats {
    pub residues: Vec<Residue>,
    /// `[L, NHEAVY, 3]`, missing atoms zeroed.
    pub xyz: Vec<f32>,
    /// `[L, NHEAVY]`, true where the PDB supplied the atom.
    pub mask: Vec<bool>,
    pub het: Vec<HetAtom>,
}

impl TargetFeats {
    pub fn len(&self) -> usize {
        self.residues.len()
    }
    pub fn is_empty(&self) -> bool {
        self.residues.is_empty()
    }
    pub fn seq(&self) -> Vec<i64> {
        self.residues.iter().map(|r| r.token).collect()
    }
    pub fn idx(&self) -> Vec<i64> {
        self.residues.iter().map(|r| r.res_seq).collect()
    }
}

/// Fixed-column slice, tolerating short lines the way Python slicing does
/// (`l[30:38]` on a short line yields whatever exists rather than panicking).
fn col(line: &[u8], a: usize, b: usize) -> &str {
    let n = line.len();
    let (a, b) = (a.min(n), b.min(n));
    std::str::from_utf8(&line[a..b]).unwrap_or("")
}

fn parse_f32(s: &str) -> f32 {
    // The reference uses Python `float()`, which parses in f64 and is then
    // stored into a float32 array — so parse wide and narrow once.
    s.trim().parse::<f64>().unwrap_or(f64::NAN) as f32
}

/// Residue name -> token index, falling back to 20 (UNK).
fn res_token(name: &str) -> i64 {
    NUM2AA
        .iter()
        .position(|&n| n == name)
        .map(|i| i as i64)
        .unwrap_or(20)
}

/// `parse_pdb_lines_target(lines, parse_hetatom, ignore_het_h)`.
pub fn parse_pdb_lines_target(
    lines: &[&[u8]],
    parse_hetatom: bool,
    ignore_het_h: bool,
) -> TargetFeats {
    // ---- 1. residue order and identity, first occurrence wins -------------
    let mut residues: Vec<Residue> = Vec::new();
    let mut seen: Vec<(String, i64)> = Vec::new();
    for l in lines {
        if col(l, 0, 4) != "ATOM" {
            continue;
        }
        let chain = col(l, 21, 22).trim().to_string();
        let res_seq: i64 = match col(l, 22, 26).trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let key = (chain.clone(), res_seq);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let name = col(l, 17, 20).to_string();
        residues.push(Residue { chain, res_seq, token: res_token(&name), name });
    }

    let n = residues.len();
    let mut xyz = vec![f32::NAN; n * NHEAVY * 3];

    // ---- 2/3. place atoms -------------------------------------------------
    for l in lines {
        if col(l, 0, 4) != "ATOM" {
            continue;
        }
        let chain = col(l, 21, 22).trim();
        let res_seq: i64 = match col(l, 22, 26).trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let atom = col(l, 12, 16).trim();
        let aa = col(l, 17, 20);

        let Some(idx) = residues
            .iter()
            .position(|r| r.chain == chain && r.res_seq == res_seq)
        else {
            continue;
        };

        // NOTE: the reference indexes aa2long by the residue name on THIS line,
        // which can differ from the name recorded for the residue (which came
        // from its first line). Reproduce that rather than "fixing" it.
        let tok = res_token(aa) as usize;
        if tok >= AA2LONG.len() {
            continue; // unknown residue: no atom template, so nothing is placed
        }
        let row = &AA2LONG[tok];
        for (i_atm, tgt) in row.iter().enumerate().take(NHEAVY) {
            if tgt.is_empty() {
                continue;
            }
            if tgt.trim() == atom {
                let base = (idx * NHEAVY + i_atm) * 3;
                xyz[base] = parse_f32(col(l, 30, 38));
                xyz[base + 1] = parse_f32(col(l, 38, 46));
                xyz[base + 2] = parse_f32(col(l, 46, 54));
                break; // first match wins
            }
        }
    }

    // ---- 4. mask from NaN, then zero --------------------------------------
    let mut mask = vec![false; n * NHEAVY];
    for i in 0..n * NHEAVY {
        let x = xyz[i * 3];
        mask[i] = !x.is_nan();
        if x.is_nan() {
            xyz[i * 3] = 0.0;
            xyz[i * 3 + 1] = 0.0;
            xyz[i * 3 + 2] = 0.0;
        }
    }

    // ---- 6. heteroatoms ---------------------------------------------------
    let mut het = Vec::new();
    if parse_hetatom {
        for l in lines {
            if col(l, 0, 6) != "HETATM" {
                continue;
            }
            let elem = col(l, 77, 78);
            if ignore_het_h && elem == "H" {
                continue;
            }
            let idx: i64 = col(l, 7, 11).trim().parse().unwrap_or(0);
            let res_idx: i64 = col(l, 22, 26).trim().parse().unwrap_or(0);
            het.push(HetAtom {
                idx,
                atom_id: col(l, 12, 16).to_string(),
                atom_type: elem.to_string(),
                name: col(l, 17, 20).to_string(),
                res_idx,
                xyz: [
                    parse_f32(col(l, 30, 38)),
                    parse_f32(col(l, 38, 46)),
                    parse_f32(col(l, 46, 54)),
                ],
            });
        }
    }

    TargetFeats { residues, xyz, mask, het }
}

/// Convenience: parse a whole PDB file's text.
pub fn parse_pdb_str(text: &str, parse_hetatom: bool, ignore_het_h: bool) -> TargetFeats {
    let lines: Vec<&[u8]> = text.lines().map(|l| l.as_bytes()).collect();
    parse_pdb_lines_target(&lines, parse_hetatom, ignore_het_h)
}
