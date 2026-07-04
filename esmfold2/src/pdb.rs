//! Minimal PDB writer for predicted all-atom structures. Atom names are decoded
//! from the `ref_atom_name_chars` one-hot (char = chr(argmax+32)); residue names
//! from the 1-letter sequence; B-factor carries per-residue pLDDT (0..100).

const CHAR_VOCAB: usize = 64;

fn one_to_three(c: u8) -> &'static str {
    match c {
        b'A' => "ALA", b'R' => "ARG", b'N' => "ASN", b'D' => "ASP", b'C' => "CYS",
        b'Q' => "GLN", b'E' => "GLU", b'G' => "GLY", b'H' => "HIS", b'I' => "ILE",
        b'L' => "LEU", b'K' => "LYS", b'M' => "MET", b'F' => "PHE", b'P' => "PRO",
        b'S' => "SER", b'T' => "THR", b'W' => "TRP", b'Y' => "TYR", b'V' => "VAL",
        _ => "UNK",
    }
}

fn decode_atom_name(name_chars: &[f32]) -> String {
    // name_chars: [4, 64] one-hot; char = chr(argmax + 32)
    let mut s = String::new();
    for c in 0..4 {
        let row = &name_chars[c * CHAR_VOCAB..c * CHAR_VOCAB + CHAR_VOCAB];
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() { if v > bv { bv = v; best = i; } }
        let ch = (best as u8) + 32;
        if ch != b' ' { s.push(ch as char); }
    }
    s
}

fn element_of(atom_name: &str) -> &str {
    match atom_name.chars().next() {
        Some('C') => "C", Some('N') => "N", Some('O') => "O", Some('S') => "S",
        Some('H') => "H", Some('P') => "P", _ => "C",
    }
}

/// Write a PDB string.
/// - `coords`: [N,3]
/// - `ref_atom_name_chars`: [N,4,64] one-hot
/// - `atom_to_token`: [N] token index per atom
/// - `residue_index`: [L] residue number per token
/// - `seq`: 1-letter sequence (len L)
/// - `asym_id`: [L] chain id per token
/// - `atom_mask`: [N] (1.0 = real atom)
/// - `plddt`: [L] per-token confidence in 0..1 (scaled to 0..100 B-factor)
#[allow(clippy::too_many_arguments)]
pub fn write_pdb(
    coords: &[f32],
    ref_atom_name_chars: &[f32],
    atom_to_token: &[i64],
    residue_index: &[i64],
    seq: &[u8],
    asym_id: &[i64],
    atom_mask: &[f32],
    plddt: &[f32],
) -> String {
    let n = atom_to_token.len();
    let mut out = String::new();
    let mut serial = 1;
    for a in 0..n {
        if atom_mask[a] == 0.0 { continue; }
        let tok = atom_to_token[a] as usize;
        let name = decode_atom_name(&ref_atom_name_chars[a * 4 * CHAR_VOCAB..(a + 1) * 4 * CHAR_VOCAB]);
        let resname = one_to_three(seq[tok]);
        let chain = (b'A' + (asym_id[tok] as u8 % 26)) as char;
        let resseq = residue_index[tok];
        let (x, y, z) = (coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2]);
        let bf = (plddt[tok] * 100.0).clamp(0.0, 100.0);
        let elem = element_of(&name);
        // PDB ATOM record (name left-justified in a 4-col field starting at col 14
        // for short names, matching the common convention for single-char elements).
        let name_field = if name.len() >= 4 { name.clone() } else { format!(" {:<3}", name) };
        out.push_str(&format!(
            "ATOM  {serial:>5} {name_field:<4} {resname:>3} {chain}{resseq:>4}    {x:8.3}{y:8.3}{z:8.3}{occ:6.2}{bf:6.2}          {elem:>2}\n",
            serial = serial, name_field = name_field, resname = resname, chain = chain,
            resseq = resseq, x = x, y = y, z = z, occ = 1.0, bf = bf, elem = elem,
        ));
        serial += 1;
    }
    out.push_str("END\n");
    out
}
