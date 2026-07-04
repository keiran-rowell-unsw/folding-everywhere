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
