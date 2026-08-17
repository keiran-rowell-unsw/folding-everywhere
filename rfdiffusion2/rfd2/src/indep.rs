//! `rf_diffusion/aa_model.py:make_indep` — the structure every later stage
//! transforms, built from a PDB plus the named ligands.
//!
//! Rung 4c already parses the PDB and rung 4d already supplies ligand topology,
//! so this is assembly rather than parsing. What it adds, and what has to be
//! right, is the *layout*: which residue lands at which row, how the ligand
//! atoms are numbered, and which of `xyz`'s 36 atom slots are NaN.

use crate::chemical_gen::NTOTAL;
use crate::ligand::LigandSet;
use crate::pdb::TargetFeats;

/// `terminus_type` codes (`aa_model.N_TERMINUS` / `C_TERMINUS`).
pub const N_TERMINUS: f32 = 1.0;
pub const C_TERMINUS: f32 = 2.0;

/// `ChemData().NHEAVYPROT` — protein atom slots above this are hydrogens and
/// are zeroed on the way in.
const NHEAVYPROT: usize = 14;

/// The chain-letter order `find_protein_dna_chains` walks. Chains are emitted in
/// **this** order, not in the order they appear in the file.
const CHAIN_ORDER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// The bond-feature code for a polymer backbone link (`get_protein_bond_feats`).
const BOND_POLYMER: i64 = 5;

#[derive(Clone, Debug)]
pub struct Indep {
    /// `[L]` token per row
    pub seq: Vec<i64>,
    /// `[L, NTOTAL, 3]`, NaN where an atom is absent
    pub xyz: Vec<f32>,
    /// `[L]` PDB numbering; ligand atoms continue past the protein with a
    /// deliberate +200 gap so no positional feature can bridge them
    pub idx: Vec<i64>,
    /// `[L, L]`
    pub bond_feats: Vec<i64>,
    /// `[n_chiral, 5]`
    pub chirals: Vec<f32>,
    /// `[L, L]`
    pub same_chain: Vec<bool>,
    /// `[L]` — guidepost flags, all false unless `contig_as_guidepost`
    pub is_gp: Vec<bool>,
    /// `[L]`
    pub terminus_type: Vec<f32>,
    /// `[L]` — true for ligand (small-molecule) rows
    pub is_sm: Vec<bool>,
}

impl Indep {
    pub fn len(&self) -> usize {
        self.seq.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// `Indep.chains()` — a chain id per row, derived from `same_chain`.
    ///
    /// Upstream recovers chain letters from the boolean matrix rather than
    /// storing them, so this does the same: a row's chain is the lowest row it
    /// shares a chain with, and those representatives are then numbered in
    /// increasing order. Two ligands are two chains, which is load-bearing —
    /// they get separate index runs and a false `same_chain` block.
    pub fn chains(&self) -> Vec<usize> {
        let l = self.len();
        let rep: Vec<usize> = (0..l)
            .map(|i| (0..l).find(|&j| self.same_chain[i * l + j]).unwrap_or(i))
            .collect();
        let mut order: Vec<usize> = rep.clone();
        order.sort_unstable();
        order.dedup();
        rep.iter().map(|r| order.iter().position(|o| o == r).unwrap()).collect()
    }
}

/// Chain segmentation, in the reference's own order.
///
/// `find_protein_dna_chains` collects the chain letters that carry protein and
/// the ones that carry nucleic acid, then walks `CHAIN_ORDER` and emits each
/// chain's length. So the output order is alphabetical by chain letter, which
/// is only incidentally the file order.
fn chain_segments(feats: &TargetFeats) -> (Vec<usize>, Vec<bool>) {
    let mut prot: Vec<u8> = Vec::new();
    let mut na: Vec<u8> = Vec::new();
    for r in &feats.residues {
        let c = r.chain.as_bytes()[0];
        if r.token >= 22 && r.token <= 31 {
            if !na.contains(&c) {
                na.push(c);
            }
        } else if r.token < 20 && !prot.contains(&c) {
            prot.push(c);
        }
    }
    let mut ls = Vec::new();
    let mut is_prot_chain = Vec::new();
    for &c in CHAIN_ORDER {
        for (isna, set) in [(true, &na), (false, &prot)] {
            if set.contains(&c) {
                let n = feats.residues.iter().filter(|r| r.chain.as_bytes()[0] == c).count();
                ls.push(n);
                is_prot_chain.push(!isna);
            }
        }
    }
    (ls, is_prot_chain)
}

/// Errors that must stop the run rather than be papered over.
#[derive(Debug)]
pub enum IndepError {
    /// The reference detects protein-ligand covalent links from CONECT records
    /// and rewires `same_chain` around them. None of the demo inputs has one,
    /// so rather than ship an untested branch this refuses.
    CovalentLink,
    NucleicAcid,
}

impl std::fmt::Display for IndepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndepError::CovalentLink => write!(
                f,
                "input has a protein-ligand covalent bond; `same_chain_with_covale` \
                 is not ported (no demo input exercises it) and guessing would be worse \
                 than stopping"
            ),
            IndepError::NucleicAcid => write!(
                f,
                "input contains nucleic-acid chains; the NA branch of make_indep is \
                 not ported and is untested"
            ),
        }
    }
}

impl std::error::Error for IndepError {}

/// `make_indep(pdb, ligand)` for a protein + small-molecule input.
///
/// `feats` must come from `pdb::parse_pdb_str(.., parse_hetatom = true)` on a
/// stream with the non-target ligands already removed, and `ligands` must name
/// them in the order the caller passed to `--ligand`, because that order sets
/// the row numbering of every ligand atom.
pub fn make_indep(
    feats: &TargetFeats,
    ligands: &[String],
    topo: &LigandSet,
) -> Result<Indep, IndepError> {
    let (mut ls, is_prot_chain) = chain_segments(feats);
    if is_prot_chain.iter().any(|p| !p) {
        return Err(IndepError::NucleicAcid);
    }
    let n_poly_chains = ls.len();
    let n_poly: usize = ls.iter().sum();

    // ---- ligand rows -----------------------------------------------------
    // `chirals[:, :-1] += sum(Ls)` is applied per ligand with the running total
    // *before* that ligand's rows are appended, so a second ligand's chirals are
    // offset by the first ligand's length too.
    let mut chirals: Vec<f32> = Vec::new();
    let mut lig_elems: Vec<i64> = Vec::new();
    let mut lig_bonds: Vec<(usize, usize, i64)> = Vec::new();
    for name in ligands {
        let t = topo.get(name).expect("ligand not in sidecar; LigandSet::load should have refused");
        let off: usize = ls.iter().sum();
        for row in t.chirals.chunks(5) {
            for k in 0..4 {
                chirals.push(row[k] + off as f32);
            }
            chirals.push(row[4]);
        }
        for i in 0..t.n_atoms {
            for j in 0..t.n_atoms {
                let b = t.bond(i, j);
                if b != 0 {
                    lig_bonds.push((off + i, off + j, b));
                }
            }
        }
        lig_elems.extend_from_slice(&t.elem);
        ls.push(t.n_atoms);
    }
    if ligands.is_empty() {
        ls.push(0);
    }
    let l: usize = ls.iter().sum();

    // ---- seq -------------------------------------------------------------
    let mut seq: Vec<i64> = feats.seq();
    seq.extend_from_slice(&lig_elems);
    assert_eq!(seq.len(), l);

    // ---- idx -------------------------------------------------------------
    // Ligand atoms restart at `max(protein idx) + 200`, and each further ligand
    // restarts 200 past the previous one's last index.
    let idx_poly = feats.idx();
    let mut idx: Vec<i64> = idx_poly.clone();
    let mut last = *idx_poly.iter().max().unwrap_or(&0);
    for n in &ls[n_poly_chains..] {
        let mut newest = last;
        for a in 0..*n {
            let v = a as i64 + 200 + last;
            idx.push(v);
            newest = newest.max(v);
        }
        last = newest;
    }
    assert_eq!(idx.len(), l);

    // ---- xyz -------------------------------------------------------------
    let mut xyz = vec![f32::NAN; l * NTOTAL * 3];
    let n_atoms_in = feats.xyz.len() / (n_poly.max(1) * 3);
    for i in 0..n_poly {
        for a in 0..n_atoms_in.min(NTOTAL) {
            // hydrogens above NHEAVYPROT are dropped for protein residues
            let v = if a >= NHEAVYPROT { [0.0, 0.0, 0.0] } else {
                let o = (i * n_atoms_in + a) * 3;
                [feats.xyz[o], feats.xyz[o + 1], feats.xyz[o + 2]]
            };
            let o = (i * NTOTAL + a) * 3;
            xyz[o..o + 3].copy_from_slice(&v);
        }
    }
    // ligand atoms occupy slot 1 (the CA slot) of their own row
    let mut het_iter = feats.het.iter();
    for i in n_poly..l {
        let h = het_iter.next().expect("fewer HETATM records than ligand atoms");
        let o = (i * NTOTAL + 1) * 3;
        xyz[o..o + 3].copy_from_slice(&h.xyz);
    }

    // ---- bond_feats ------------------------------------------------------
    let mut bond_feats = vec![0i64; l * l];
    let mut base = 0usize;
    for (ci, &n) in ls[..n_poly_chains].iter().enumerate() {
        if is_prot_chain[ci] {
            for r in 0..n.saturating_sub(1) {
                bond_feats[(base + r) * l + base + r + 1] = BOND_POLYMER;
                bond_feats[(base + r + 1) * l + base + r] = BOND_POLYMER;
            }
        }
        base += n;
    }
    for (i, j, b) in lig_bonds {
        bond_feats[i * l + j] = b;
    }

    // ---- same_chain ------------------------------------------------------
    // One chain letter per segment, in `ls` order, then the outer equality.
    let mut chain_of = vec![0usize; l];
    let mut base = 0usize;
    for (ci, &n) in ls.iter().enumerate() {
        for r in 0..n {
            chain_of[base + r] = ci;
        }
        base += n;
    }
    let mut same_chain = vec![false; l * l];
    for i in 0..l {
        for j in 0..l {
            same_chain[i * l + j] = chain_of[i] == chain_of[j];
        }
    }

    // ---- is_sm / terminus_type -------------------------------------------
    let mut is_sm = vec![false; l];
    for v in is_sm.iter_mut().skip(n_poly) {
        *v = true;
    }
    let mut terminus_type = vec![0.0f32; l];
    let mut base = 0usize;
    for (ci, &n) in ls[..n_poly_chains].iter().enumerate() {
        if is_prot_chain[ci] && n > 0 {
            terminus_type[base] = N_TERMINUS;
            terminus_type[base + n - 1] = C_TERMINUS;
        }
        base += n;
    }

    // The network's `compute_all_atom` needs N and C coordinates even for
    // ligand rows, which have none; upstream writes literal zeros there.
    for (i, &sm) in is_sm.iter().enumerate() {
        if sm {
            for a in [0usize, 2] {
                let o = (i * NTOTAL + a) * 3;
                xyz[o..o + 3].copy_from_slice(&[0.0, 0.0, 0.0]);
            }
        }
    }

    Ok(Indep {
        seq,
        xyz,
        idx,
        bond_feats,
        chirals,
        same_chain,
        is_gp: vec![false; l],
        terminus_type,
        is_sm,
    })
}
