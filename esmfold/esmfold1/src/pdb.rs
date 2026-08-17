//! Minimal PDB writer: atom37 coordinates -> ATOM records, pLDDT in the B-factor.

use crate::constants::Constants;

pub const ATOM37_NAMES: [&str; 37] = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1",
    "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2",
    "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
];

const RES3: [&str; 21] = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
];

/// Mean pLDDT over the atoms that actually exist, on the 0..100 scale — the exact
/// statistic upstream ESMFold reports as `output["mean_plddt"]`:
///
/// ```text
/// structure["plddt"] = 100 * plddt
/// output["mean_plddt"] = (plddt * atom37_atom_exists).sum() / atom37_atom_exists.sum()
/// ```
///
/// `c.atom37_mask[a*37+at]` is AF2's `restype_atom37_mask`, i.e. exactly the
/// `atom37_atom_exists` upstream builds via `make_atom14_masks` — so this averages
/// over precisely the atoms `to_pdb` writes, and equals the mean B-factor of the
/// output PDB.
///
/// NB: a plain mean over the whole `[L,37]` tensor is **not** the same number. The
/// pLDDT head emits a value for all 37 slots including ones the residue does not
/// have, and those extra values are low but non-zero, so the unmasked mean reads
/// several points low (ubiquitin: 77.35 unmasked vs 85.82 masked).
///
/// plddt [L,37] (0..1), aatype [L].
pub fn mean_plddt(plddt: &[f32], aatype: &[usize], c: &Constants, l: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for li in 0..l {
        let a = aatype[li];
        for at in 0..37 {
            let m = c.atom37_mask[a * 37 + at];
            sum += plddt[li * 37 + at] * 100.0 * m;
            n += m;
        }
    }
    sum / n
}

/// atom37 [L,37,3], plddt [L,37] (0..1), aatype [L].
pub fn to_pdb(atom37: &[f32], plddt: &[f32], aatype: &[usize], c: &Constants, l: usize) -> String {
    let mut s = String::new();
    let mut serial = 1;
    for li in 0..l {
        let a = aatype[li];
        let resname = RES3[a.min(20)];
        for at in 0..37 {
            if c.atom37_mask[a * 37 + at] < 0.5 {
                continue;
            }
            let name = ATOM37_NAMES[at];
            let (x, y, z) = (atom37[(li * 37 + at) * 3], atom37[(li * 37 + at) * 3 + 1], atom37[(li * 37 + at) * 3 + 2]);
            let b = plddt[li * 37 + at] * 100.0;
            let element = &name[0..1];
            // PDB ATOM record (fixed columns)
            let atname = if name.len() >= 4 {
                name.to_string()
            } else {
                format!(" {:<3}", name)
            };
            s.push_str(&format!(
                "ATOM  {:>5} {:<4} {:>3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                serial, atname, resname, li + 1, x, y, z, 1.0, b, element
            ));
            serial += 1;
        }
    }
    s.push_str("TER\nEND\n");
    s
}
