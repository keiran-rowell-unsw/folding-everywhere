//! Rung 7's output half — `run_inference.save_outputs` and everything it calls.
//!
//! SOP §5.6: the output format is part of the port, so the gate here is a
//! byte-for-byte diff of the `.pdb`, not a coordinate tolerance. The chain from
//! the sampler's last `px0` to the file on disk is longer than it looks:
//!
//! ```text
//! px0_xyz_stack                                       src/sampler.rs
//!   add_implicit_side_chain_atoms   motif sidechains back from indep_orig
//!   idealize_bb_atoms               idealize_reference_frame + ideal O
//! final_seq                         argmax, then mask/unknown -> ALA
//! write_traj -> writepdb_file       the text, with MODEL/ENDMDL
//! idealize_backbone.rewrite         RE-PARSES that text and idealizes again
//! rename_ligand_atoms               ligand atom names copied from the input
//! ```
//!
//! ## `rewrite` really does round-trip through text
//!
//! It does not operate on the structure in memory: it parses the PDB stream it
//! was just handed, rebuilds an `Indep` from it, idealizes the protein
//! backbone, and writes it out again. So the coordinates that feed the second
//! idealization have been through `%8.3f` — three decimals — and reproducing
//! the output means reproducing that rounding, not skipping it.
//!
//! That also means **OpenBabel perceives the ligand a second time**, from the
//! rounded coordinates. Measured before relying on it: the reference's own
//! output carries 106 `CONECT` records and the sidecar
//! (`fixtures/ligand/M0584_1ldm.safetensors`) has 106 directed bonds, so the
//! second perception agrees with the first and the sidecar can be reused.

use crate::chemical_gen::{AA2LONG, ATOM_NUM, FRAME_PRIORITY2ATOM, NHEAVY, NTOTAL, NUM2AA};
use crate::indep::Indep;

/// The corrected element table `writepdb_file` carries inline, with the comment
/// "correct mistake in atomic number assignment in RF2-allatom".
const WRITER_ATOM_NAMES: [&str; 48] = [
    "F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As", "Sb", "C", "Si", "Ge",
    "Sn", "Pb", "B", "Al", "Zn", "Hg", "Cu", "Au", "Ni", "Pd", "Pt", "Co", "Rh", "Ir",
    "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", "Mo", "W", "V", "U", "Tb", "Y", "Be",
    "Mg", "Ca", "Li", "K", "ATM",
];
const WRITER_ATOM_NUM: [i64; 48] = [
    9, 17, 35, 53, 8, 16, 34, 52, 7, 15, 33, 51, 6, 14, 32, 50, 82, 5, 13, 30, 80, 29,
    79, 28, 46, 78, 27, 45, 77, 59, 26, 44, 76, 25, 75, 24, 42, 74, 23, 92, 65, 39, 4,
    12, 20, 3, 19, 0,
];

/// Tokens below this index are polymer residues with an `aa2long` row; at or
/// above it a row is a single ligand atom and is written as `HETATM`.
const N_AA2LONG: usize = AA2LONG.len();

/// `writepdb_file`'s `atomtype_map`: `ChemData`'s element name for a token, run
/// through the corrected atomic-number table.
///
/// `ChemData().atomnum2atomtype` is `zip(atom_num, frame_priority2atom)` and is
/// the one with the mistake; the writer maps its *names* back through the
/// corrected numbering. For most elements this is the identity, and for the few
/// that are shifted it is not.
fn atomtype_map(name: &str) -> &'static str {
    let idx = FRAME_PRIORITY2ATOM
        .iter()
        .position(|n| *n == name)
        .unwrap_or_else(|| panic!("element {name:?} is not in frame_priority2atom"));
    let num = ATOM_NUM[idx];
    let widx = WRITER_ATOM_NUM
        .iter()
        .position(|n| *n == num)
        .unwrap_or_else(|| panic!("atomic number {num} is not in the writer's table"));
    WRITER_ATOM_NAMES[widx]
}

/// `run_inference.add_implicit_side_chain_atoms`.
///
/// Copies sidechain atoms of the marked residues out of `xyz_with_sc`. The
/// mask is "atoms this residue type has" minus "atoms its *masked* form has",
/// so backbone slots are never touched — the point is to restore the motif's
/// chemistry, which the diffusion threw away.
pub fn add_implicit_side_chain_atoms(
    seq: &[i64],
    act_on_residue: &[bool],
    xyz: &mut [f32],
    xyz_with_sc: &[f32],
    n_atoms: usize,
) {
    let (mask, shape) = crate::chemical::allatom_mask();
    let width = shape[1];
    let l = seq.len();
    for i in 0..l {
        if !act_on_residue[i] {
            continue;
        }
        let t = seq[i] as usize;
        let masked = mask_token_for(seq[i]);
        for a in 0..n_atoms.min(width) {
            let has = mask[t * width + a];
            // `backbone_atom_mask[:, NHEAVY:] = False`
            let is_backbone = a < NHEAVY && mask[masked * width + a];
            if has && !is_backbone {
                for c in 0..3 {
                    xyz[(i * n_atoms + a) * 3 + c] = xyz_with_sc[(i * n_atoms + a) * 3 + c];
                }
            }
        }
    }
}

/// `nucl_utils.inds_to_mol_class_mask` — the mask token of a token's molecule
/// class. Protein tokens map to `MAS`, bare elements to `ATM`; the nucleic
/// classes never occur on this path and are refused rather than guessed, the
/// same way `sample_init::mask_indep` refuses them.
fn mask_token_for(t: i64) -> usize {
    const MAS: usize = 21;
    const ATM: usize = 79;
    match t {
        0..=21 => MAS,
        33..=79 => ATM,
        _ => panic!(
            "token {t} is nucleic or HIS_D; its molecule-class mask token has not \
             been measured on any run"
        ),
    }
}

/// `dev/idealize_backbone.get_o` — place the peptide O from the frame, using
/// the *next* residue's N when the two are sequential.
///
/// `is_adj[i]` is `idx[i+1] - idx[i] == 1`, with a sentinel `-1` appended so the
/// last residue is never adjacent. The two ideal positions differ because a
/// C-terminal O sits in the residue's own N/CA/C frame while an internal one is
/// placed in the CA/C/N(next) frame.
fn get_o(xyz: &[f32], idx: &[i64], l: usize, n_atoms: usize) -> Vec<f32> {
    const IDEAL_TERMINAL: [f32; 3] = [2.1428, 0.7350, -0.7413];
    const IDEAL_INTERNAL: [f32; 3] = [-0.7247, -1.0032, -0.0003];
    let at = |i: usize, a: usize| -> [f32; 3] {
        let o = (i * n_atoms + a) * 3;
        [xyz[o], xyz[o + 1], xyz[o + 2]]
    };
    let mut out = vec![0.0f32; l * 4 * 3];
    for i in 0..l {
        for a in 0..3 {
            let p = at(i, a);
            for c in 0..3 {
                out[(i * 4 + a) * 3 + c] = p[c];
            }
        }
    }
    // `rigid_from_3_points` with `is_na = None`, i.e. the protein `costgt`
    // everywhere; the frame is built per row so the two branches can share it.
    let no_na = [false];
    for i in 0..l {
        let adj = i + 1 < l && idx[i + 1] - idx[i] == 1;
        let (a, b, c, ideal) = if adj {
            (at(i, 1), at(i, 2), at(i + 1, 0), IDEAL_INTERNAL)
        } else {
            (at(i, 0), at(i, 1), at(i, 2), IDEAL_TERMINAL)
        };
        let (r, t) = crate::geom::rigid_from_3_points(
            &a,
            &b,
            &c,
            1,
            &no_na,
            crate::chemical_gen::COSTGTNA,
        );
        // `torch.einsum('...lij,...j->...li', R, ideal) + T` — pinned, f64
        for row in 0..3 {
            let mut acc = 0.0f64;
            for k in 0..3 {
                acc += r[row * 3 + k] as f64 * ideal[k] as f64;
            }
            out[(i * 4 + 3) * 3 + row] = (acc as f32) + t[row];
        }
    }
    out
}

/// `dev/idealize_backbone.idealize_bb_atoms`, applied to the protein rows.
///
/// `idealize_reference_frame` is called with an all-alanine sequence, so N and
/// C are rebuilt in the *protein* frame convention regardless of the real
/// residue type.
pub fn idealize_bb_atoms(xyz: &mut [f32], idx: &[i64], rows: &[usize], n_atoms: usize) {
    let n = rows.len();
    if n == 0 {
        return;
    }
    let mut sub = vec![0.0f32; n * NTOTAL * 3];
    for (j, &i) in rows.iter().enumerate() {
        for a in 0..NTOTAL.min(n_atoms) {
            for c in 0..3 {
                sub[(j * NTOTAL + a) * 3 + c] = xyz[(i * n_atoms + a) * 3 + c];
            }
        }
    }
    let ala = vec![0i64; n];
    let ideal = crate::torsions::idealize_reference_frame(&ala, &sub, n);
    let sub_idx: Vec<i64> = rows.iter().map(|&i| idx[i]).collect();
    let o = get_o(&ideal, &sub_idx, n, NTOTAL);
    for (j, &i) in rows.iter().enumerate() {
        for a in 0..4 {
            for c in 0..3 {
                xyz[(i * n_atoms + a) * 3 + c] = o[(j * 4 + a) * 3 + c];
            }
        }
    }
}

/// Would `write_file.fix_null_sidechains` rebuild this residue?
///
/// It is not ported — `build_ideal_sidechains` is a separate sub-port — but it
/// is *detected*, because a silently-unrebuilt sidechain would be a wrong
/// output file rather than an error. Measured on the demo configuration: no
/// residue triggers it (designed rows are alanine, whose only sidechain atom is
/// CB and which is never at the origin; the motif row carries real coordinates).
fn would_fix_sidechain(atoms: &[f32], seq: i64, n_atoms: usize) -> bool {
    const TOO_CLOSE_SQ: f32 = 0.01 * 0.01;
    if seq >= 20 {
        return false;
    }
    let at = |a: usize| -> [f32; 3] {
        [atoms[a * 3], atoms[a * 3 + 1], atoms[a * 3 + 2]]
    };
    let d2 = |a: [f32; 3], b: [f32; 3]| {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    };
    if at(0).iter().any(|v| v.is_nan()) || at(1).iter().any(|v| v.is_nan()) {
        return false;
    }
    if d2(at(0), at(1)) < TOO_CLOSE_SQ || d2(at(1), at(2)) < TOO_CLOSE_SQ {
        return false;
    }
    let names = &AA2LONG[seq as usize];
    let n = n_atoms.min(names.len());
    let mut sel: Vec<usize> = (0..n)
        .filter(|&a| !names[a].is_empty() && !at(a).iter().any(|v| v.is_nan()))
        .collect();
    sel.retain(|&a| a >= 4 || a == 1);
    if sel.is_empty() {
        return false;
    }
    for (x, &a) in sel.iter().enumerate() {
        for &b in sel.iter().skip(x + 1) {
            if d2(at(a), at(b)) <= TOO_CLOSE_SQ {
                return true;
            }
        }
    }
    // "exactly at the origin", ignoring CA
    sel.iter().skip(1).any(|&a| at(a).iter().all(|v| v.abs() < 0.0001))
}

/// Everything `writepdb_file` needs that is not the coordinates.
pub struct PdbSpec<'a> {
    pub seq: &'a [i64],
    pub idx: &'a [i64],
    pub chain_letters: &'a [u8],
    pub ligand_names: &'a [String],
    /// `[L, L]`; `CONECT` is written for `0 < bond_feats < 5`
    pub bond_feats: Option<&'a [i64]>,
    pub modelnum: Option<usize>,
}

/// `rf_diffusion/write_file.py:writepdb_file`.
///
/// Format strings are transcribed rather than approximated:
/// `%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f` for `ATOM`, the same
/// plus `          %+2s` for `HETATM` (Python ignores `+` for `%s`, so that is
/// a width-2 right-justified field).
pub fn writepdb_file(atoms: &[f32], n_atoms: usize, spec: &PdbSpec) -> String {
    let l = spec.seq.len();
    // "PDBs coordinate range is (-999.99, 9999.99)"
    let clamp = |v: f32| {
        if v >= 10000.0 {
            9999.99
        } else if v <= -1000.0 {
            -999.99
        } else {
            v
        }
    };

    if n_atoms > 4 {
        for i in 0..l {
            let row = &atoms[i * n_atoms * 3..(i + 1) * n_atoms * 3];
            assert!(
                !would_fix_sidechain(row, spec.seq[i], n_atoms),
                "row {i} (token {}) would trigger write_file.fix_null_sidechains, \
                 which rebuilds the sidechain from build_ideal_sidechains — a path \
                 no measured run exercises and which is therefore not ported",
                spec.seq[i]
            );
        }
    }

    let mut out = String::new();
    if let Some(m) = spec.modelnum {
        out.push_str(&format!("MODEL        {m}\n"));
    }
    let max_idx = *spec.idx.iter().max().unwrap_or(&0);
    let mut ctr = 1usize;
    let mut atom_idxs: Vec<Option<usize>> = vec![None; l];
    let mut ligand_count = 0usize;
    let mut writing_ligand = false;
    // `atom_count_by_res[res_idx][atom_type]`, keyed the same way upstream does
    let mut counts: std::collections::HashMap<(i64, &str), usize> =
        std::collections::HashMap::new();

    for i in 0..l {
        let s = spec.seq[i] as usize;
        if s >= N_AA2LONG {
            writing_ligand = true;
            atom_idxs[i] = Some(ctr);
            let res_idx = max_idx + 10 * (ligand_count as i64 + 1);
            let atom_type = atomtype_map(NUM2AA[s]);
            let e = counts.entry((res_idx, atom_type)).or_insert(0);
            *e += 1;
            let atom_name = format!("{atom_type}{}", *e);
            let o = (i * n_atoms + 1) * 3;
            out.push_str(&format!(
                "{:<6}{:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}\n",
                "HETATM",
                ctr,
                atom_name,
                spec.ligand_names[i],
                spec.chain_letters[i] as char,
                res_idx,
                clamp(atoms[o]),
                clamp(atoms[o + 1]),
                clamp(atoms[o + 2]),
                1.0,
                0.0,
                atom_type
            ));
            ctr += 1;
        } else {
            if writing_ligand {
                ligand_count += 1;
            }
            writing_ligand = false;
            assert!(max_idx <= 9999, "PDB residue index overflow");
            for a in 0..n_atoms.min(AA2LONG[s].len()) {
                let name = AA2LONG[s][a];
                if name.is_empty() {
                    continue;
                }
                let o = (i * n_atoms + a) * 3;
                if atoms[o..o + 3].iter().any(|v| v.is_nan()) {
                    continue;
                }
                out.push_str(&format!(
                    "{:<6}{:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}\n",
                    "ATOM",
                    ctr,
                    name,
                    NUM2AA[s],
                    spec.chain_letters[i] as char,
                    spec.idx[i],
                    clamp(atoms[o]),
                    clamp(atoms[o + 1]),
                    clamp(atoms[o + 2]),
                    1.0,
                    0.0
                ));
                ctr += 1;
            }
        }
    }

    if let Some(bf) = spec.bond_feats {
        for i in 0..l {
            for j in 0..l {
                let b = bf[i * l + j];
                if b > 0 && b < 5 {
                    let (a, c) = (atom_idxs[i], atom_idxs[j]);
                    if let (Some(a), Some(c)) = (a, c) {
                        out.push_str(&format!("CONECT{a:5}{c:5}\n"));
                    }
                }
            }
        }
    }
    if spec.modelnum.is_some() {
        out.push_str("ENDMDL\n");
    }
    out
}

/// `Indep.write_pdb_file` — the writer as `rewrite` calls it.
///
/// The two sequence substitutions are load-bearing: UNK (20) and MASK (21)
/// both become ALA (0), which is why every designed residue comes out as ALA
/// with five atoms.
pub fn write_indep_pdb(indep: &Indep, ligand_names: &[String]) -> String {
    let seq: Vec<i64> = indep
        .seq
        .iter()
        .map(|&s| if s == 20 || s == 21 { 0 } else { s })
        .collect();
    let chains = chain_letters(indep);
    // `torch.nan_to_num(self.xyz[:, :NHEAVY])`
    let mut atoms = vec![0.0f32; indep.len() * NHEAVY * 3];
    for i in 0..indep.len() {
        for a in 0..NHEAVY {
            for c in 0..3 {
                let v = indep.xyz[(i * NTOTAL + a) * 3 + c];
                atoms[(i * NHEAVY + a) * 3 + c] = if v.is_nan() { 0.0 } else { v };
            }
        }
    }
    writepdb_file(
        &atoms,
        NHEAVY,
        &PdbSpec {
            seq: &seq,
            idx: &indep.idx,
            chain_letters: &chains,
            ligand_names,
            bond_feats: Some(&indep.bond_feats),
            modelnum: None,
        },
    )
}

/// `chain_letters_from_same_chain` — chain index to letter.
pub fn chain_letters(indep: &Indep) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    indep.chains().into_iter().map(|c| ALPHABET[c]).collect()
}

/// `aa_model.write_traj` for a single model — pads to `NHEAVY` and stamps a
/// `MODEL` record.
pub fn write_traj(
    xyz: &[f32],
    n_atoms_in: usize,
    spec_seq: &[i64],
    idx: &[i64],
    chain_letters: &[u8],
    ligand_names: &[String],
    bond_feats: Option<&[i64]>,
    modelnum: usize,
) -> String {
    let l = spec_seq.len();
    let mut atoms = vec![0.0f32; l * NHEAVY * 3];
    for i in 0..l {
        for a in 0..NHEAVY {
            for c in 0..3 {
                atoms[(i * NHEAVY + a) * 3 + c] = if a < n_atoms_in {
                    xyz[(i * n_atoms_in + a) * 3 + c]
                } else {
                    0.0
                };
            }
        }
    }
    writepdb_file(
        &atoms,
        NHEAVY,
        &PdbSpec {
            seq: spec_seq,
            idx,
            chain_letters,
            ligand_names,
            bond_feats,
            modelnum: Some(modelnum),
        },
    )
}

/// `aa_model.rename_ligand_atoms` — put the input PDB's ligand atom names back.
///
/// Also **strips every line**, which is why the final file has no trailing
/// whitespace even though the `HETATM` format string ends in a padded element
/// field.
pub fn rename_ligand_atoms(input_pdb_text: &str, stream: &str) -> String {
    // `hetatm_names` then `without_H`
    let mut names: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for line in input_pdb_text.lines() {
        if !line.starts_with("HETATM") {
            continue;
        }
        let lig = line[17..20].trim().to_string();
        let atom = line[12..16].trim().to_string();
        let elem = if line.len() >= 78 {
            line[76..78].trim().to_string()
        } else {
            String::new()
        };
        if elem == "H" {
            continue;
        }
        match names.iter_mut().find(|(k, _)| *k == lig) {
            Some((_, v)) => v.push((atom, elem)),
            None => names.push((lig, vec![(atom, elem)])),
        }
    }

    let mut counters: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut out = String::new();
    for raw in stream.lines() {
        let line = raw.trim();
        let mut line = line.to_string();
        if line.starts_with("HETATM") {
            let lig = line[17..20].trim().to_string();
            let elem = line[76..78].trim().to_string();
            assert_ne!(elem, "H", "a hydrogen reached the ligand renamer");
            let entry = names
                .iter()
                .find(|(k, _)| *k == lig)
                .unwrap_or_else(|| panic!("ligand {lig:?} is not in the input PDB"));
            let k = counters.entry(lig.clone()).or_insert(0);
            let (ref_name, ref_elem) = &entry.1[*k];
            assert_eq!(
                elem.to_uppercase(),
                ref_elem.to_uppercase(),
                "ligand {lig} atom {k}: element {elem} does not match the input's {ref_elem}"
            );
            *k += 1;
            line = format!("{}{:<4}{}", &line[..12], ref_name, &line[16..]);
            line = format!("{}{:>2}{}", &line[..76], ref_elem, &line[78..]);
        }
        if line.starts_with("MODEL") {
            counters.clear();
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// `dev/idealize_backbone.rewrite`.
///
/// Upstream re-parses the stream it was handed and rebuilds an `Indep` from it,
/// which is why the second idealization sees coordinates that have been through
/// `%8.3f`. That round-trip is reproduced literally: the text is parsed with the
/// port's own PDB parser and `make_indep`, so the writer and the reader cannot
/// drift apart.
///
/// The ligand order is taken from **first appearance in the stream**, not from
/// the caller's `--ligand` list: upstream derives it from `get_ligands`, which
/// reads the HETATM records back out of the text.
pub fn rewrite(stream: &str, topo: &crate::ligand::LigandSet, ligand_names: &[String]) -> String {
    let mut order: Vec<String> = Vec::new();
    for line in stream.lines() {
        if line.starts_with("HETATM") {
            let n = line[17..21].trim().to_string();
            if !order.contains(&n) {
                order.push(n);
            }
        }
    }
    let feats = crate::pdb::parse_pdb_str(stream, true, true);
    let mut indep = crate::indep::make_indep(&feats, &order, topo)
        .expect("re-parsing the written PDB should reproduce the same Indep");

    let rows: Vec<usize> = (0..indep.len())
        .filter(|&i| crate::geom::is_protein(indep.seq[i]))
        .collect();
    let idx = indep.idx.clone();
    idealize_bb_atoms(&mut indep.xyz, &idx, &rows, NTOTAL);

    write_indep_pdb(&indep, ligand_names)
}
