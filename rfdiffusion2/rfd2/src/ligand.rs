//! Ligand topology — loaded from a sidecar, **not** perceived.
//!
//! # Why this is a loader and not a port
//!
//! RFdiffusion2 gets ligand bond features by handing the HETATM block to
//! OpenBabel. `python/probe_ligand_bonds.py` measured what that actually does
//! on the real demo inputs (`results/ligand_bond_probe.txt`):
//!
//! | input | atoms | bonds | CONECT edges |
//! |---|---:|---:|---:|
//! | NAD (M0584_1ldm) | 44 | 48 | **0** |
//! | OXM (M0584_1ldm) | 6 | 5 | **0** |
//! | M0151_1q0n ligand | 47 | 49 | **0** |
//! | PH2 (trimmed_ec2) | 14 | 15 | 15 |
//!
//! For three of the four there are **no CONECT records at all**, so OpenBabel
//! is perceiving connectivity from interatomic distances and covalent radii
//! (`ConnectTheDots`) and then bond orders including aromaticity
//! (`PerceiveBondOrders`: hybridisation, ring perception, aromaticity,
//! kekulisation) purely from 3D coordinates. NAD comes out with 10 aromatic
//! bonds, the M0151 ligand with 16.
//!
//! Those orders are not cosmetic: `bond_feats` feeds `bond_emb` directly, so
//! single-vs-aromatic changes the network's input. Reproducing that heuristic
//! stack bit-for-bit is a cheminformatics sub-project, and a half-ported version
//! would emit *plausible but wrong* topology — the worst failure mode available,
//! because nothing downstream would complain.
//!
//! So the scope boundary is drawn here and stated: **ligand topology is an
//! input to rfd2.** `python/gen_ligand_bonds.py` runs the reference's own path
//! once per PDB and writes a sidecar; this module loads it and **hard-errors**
//! on anything not covered. Loudly wrong beats silently wrong.
//!
//! Proteins are unaffected — their bonds come from the chemical tables
//! (`num_bonds`, `aabonds`), which are already embedded and exact.

use crate::weights::Weights;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LigandTopology {
    pub name: String,
    pub n_atoms: usize,
    /// `[n, n]` bond orders: 0 none, 1 single, 2 double, 3 triple, 4 aromatic.
    pub bond_feats: Vec<i64>,
    /// Element token per atom, indexing the chemical alphabet.
    pub elem: Vec<i64>,
    /// `[n, 3]`
    pub xyz: Vec<f32>,
    /// `[n_chiral, 5]` — `(i0, i1, i2, i3, target_angle)`. The first four are
    /// atom indices **local to this ligand**; `make_indep` shifts them by the
    /// number of rows already placed.
    pub chirals: Vec<f32>,
    /// `[n, 3, 2]` — the local frame chosen for each atom, as (offset, 1) pairs.
    ///
    /// Also a sidecar field, for a reason of its own. `get_atom_frames` ranks
    /// candidate frames by atom priority and breaks ties by the order of
    /// `list(set(allpaths))` — i.e. by **CPython's set iteration order** over
    /// tuples of ints. Measured on the demo ligand, **20 of 50 atoms** have two
    /// or more frames sharing the minimum priority, so that order really does
    /// select the frame. Reproducing it would mean reimplementing CPython's
    /// tuple hash and set probe sequence.
    pub atom_frames: Vec<i64>,
    /// `[n]` PDB atom names, 4 characters each, in the order the sidecar was
    /// built from. Present only in sidecars written after 2026-08-12.
    ///
    /// Load-bearing for any SHARED library: `make_indep` consumes this topology
    /// **positionally** (`bond(i, j)`, `elem[i]`) while the coordinates come from
    /// the input PDB in file order, so a sidecar built from one file is valid for
    /// another only if the atoms appear in the same order. With the names
    /// recorded, `align_to_pdb` can permute instead of silently pairing the wrong
    /// bonds with the wrong atoms.
    pub atom_names: Option<Vec<[u8; 4]>>,
}

impl LigandTopology {
    /// Reorder every per-atom field so that row `i` is `perm[i]` of the old
    /// order. `inv` is the inverse permutation.
    ///
    /// The three encodings differ and all three must move together:
    ///   * `bond_feats` is `[n, n]` — both axes permute;
    ///   * `chirals` holds ABSOLUTE atom indices in its first four columns;
    ///   * `atom_frames` holds RELATIVE offsets (`neighbour = i + delta`), so a
    ///     delta has to be re-derived in the new numbering, not just moved.
    fn permute(&mut self, perm: &[usize]) {
        let n = self.n_atoms;
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let mut bf = vec![0i64; n * n];
        for i in 0..n {
            for j in 0..n {
                bf[i * n + j] = self.bond_feats[perm[i] * n + perm[j]];
            }
        }
        self.bond_feats = bf;
        self.elem = perm.iter().map(|&o| self.elem[o]).collect();
        if self.xyz.len() == n * 3 {
            let mut x = vec![0.0f32; n * 3];
            for i in 0..n {
                x[i * 3..i * 3 + 3].copy_from_slice(&self.xyz[perm[i] * 3..perm[i] * 3 + 3]);
            }
            self.xyz = x;
        }
        let mut af = vec![0i64; n * 3 * 2];
        for i in 0..n {
            let old_i = perm[i];
            for k in 0..3 {
                let d = self.atom_frames[(old_i * 3 + k) * 2];
                let flag = self.atom_frames[(old_i * 3 + k) * 2 + 1];
                let nbr_old = old_i as i64 + d;
                let new_d = if nbr_old >= 0 && (nbr_old as usize) < n {
                    inv[nbr_old as usize] as i64 - i as i64
                } else {
                    d // out of range in the source too; carry it through unchanged
                };
                af[(i * 3 + k) * 2] = new_d;
                af[(i * 3 + k) * 2 + 1] = flag;
            }
        }
        self.atom_frames = af;
        for row in self.chirals.chunks_mut(5) {
            for v in row.iter_mut().take(4) {
                let old = *v as usize;
                if old < n {
                    *v = inv[old] as f32;
                }
            }
        }
        if let Some(names) = &self.atom_names {
            self.atom_names = Some(perm.iter().map(|&o| names[o]).collect());
        }
    }

    pub fn bond(&self, i: usize, j: usize) -> i64 {
        self.bond_feats[i * self.n_atoms + j]
    }

    pub fn n_bonds(&self) -> usize {
        let mut n = 0;
        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                if self.bond(i, j) > 0 {
                    n += 1;
                }
            }
        }
        n
    }
}

#[derive(Debug)]
pub enum LigandError {
    SidecarMissing { path: String },
    LigandNotCovered { name: String, available: Vec<String> },
    /// The library's atoms for this ligand are not the file's atoms. Reordering
    /// is safe; a different atom SET is not, so it is refused.
    AtomSetMismatch { name: String, expected: String, found: String },
    /// Asked for a ligand the input structure does not contain.
    NotInStructure { name: String },
}

impl std::fmt::Display for LigandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LigandError::SidecarMissing { path } => write!(
                f,
                "ligand topology sidecar not found: {path}\n\
                 rfd2 does not perceive ligand bonds from coordinates (see \
                 src/ligand.rs). Generate it once with:\n  \
                 PYTHONPATH=<ref> python python/gen_ligand_bonds.py <input.pdb> <LIG,LIG>"
            ),
            LigandError::LigandNotCovered { name, available } => write!(
                f,
                "ligand {name:?} is not in the sidecar (has: {available:?}).\n\
                 Refusing to guess its bond orders — regenerate the sidecar \
                 including {name:?}."
            ),
            LigandError::NotInStructure { name } => write!(
                f,
                "ligand {name:?} is not present in the input structure.\n\
                 The ligand ATOMS and their coordinates come from your PDB; this \
                 program only supplies their bond orders and aromaticity. Choose \
                 a ligand the file actually contains."
            ),
            LigandError::AtomSetMismatch { name, expected, found } => write!(
                f,
                "ligand {name:?}: the input PDB's atoms are not the ones this \
                 topology was built for.\n  topology: {expected}\n  this file: {found}\n\
                 Reordering is handled automatically; a different atom SET (a \
                 different protonation state, a truncated ligand) is not \
                 something to guess at. Build a sidecar from THIS file:\n  \
                 PYTHONPATH=<ref> python python/gen_ligand_bonds.py <input.pdb> <LIG,LIG>"
            ),
        }
    }
}

impl std::error::Error for LigandError {}

/// Ligand topologies for one input structure.
pub struct LigandSet {
    ligands: HashMap<String, LigandTopology>,
    order: Vec<String>,
    /// Authoritative frames from a reference run, when the sidecar carries them.
    combined_atom_frames: Option<Vec<i64>>,
}

impl LigandSet {
    /// Load a sidecar written by `python/gen_ligand_bonds.py`.
    pub fn load(sidecar: &str, names: &[String]) -> Result<Self, LigandError> {
        let w = Weights::open(sidecar).map_err(|_| LigandError::SidecarMissing {
            path: sidecar.to_string(),
        })?;

        let available: Vec<String> = w
            .names()
            .iter()
            .filter_map(|n| n.strip_suffix(".bond_feats").map(|s| s.to_string()))
            .collect();

        let mut ligands = HashMap::new();
        let mut order = Vec::new();
        let mut needs_frames = false;
        for name in names {
            let key = format!("{name}.bond_feats");
            if !w.has(&key) {
                return Err(LigandError::LigandNotCovered {
                    name: name.clone(),
                    available,
                });
            }
            let (bond_feats, shape) = w.get_i64(&key);
            let n_atoms = shape[0];
            let (elem, _) = w.get_i64(&format!("{name}.elem"));
            let xyz = w.get(&format!("{name}.xyz")).data;
            // Optional. A shared LIBRARY omits them on purpose: frames are a
            // function of the whole ligand block's bond matrix in THIS file's
            // atom order, so they are recomputed (see `recompute_frames`) rather
            // than carried. A per-file sidecar still ships its own.
            let atom_frames = if w.has(&format!("{name}.atom_frames")) {
                w.get_i64(&format!("{name}.atom_frames")).0
            } else {
                needs_frames = true;
                Vec::new()
            };
            let chirals = w.get(&format!("{name}.chirals")).data;
            let atom_names = if w.has(&format!("{name}.atom_names")) {
                let (v, _) = w.get_i64(&format!("{name}.atom_names"));
                Some(v.chunks(4).map(|c| [c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8]).collect())
            } else {
                None
            };
            ligands.insert(
                name.clone(),
                LigandTopology {
                    name: name.clone(), n_atoms, bond_feats, elem, xyz, chirals, atom_frames,
                    atom_names,
                },
            );
            order.push(name.clone());
        }
        let combined_atom_frames = if w.has("combined.atom_frames") {
            Some(w.get_i64("combined.atom_frames").0)
        } else {
            None
        };
        let mut set = LigandSet { ligands, order, combined_atom_frames };
        if needs_frames {
            // frames for the file's own atom order; `align_to_pdb` redoes this if
            // the order turns out to differ
            set.combined_atom_frames = Some(set.recompute_frames());
        }
        Ok(set)
    }

    /// Reorder every ligand's topology to the atom order of `pdb_text`.
    ///
    /// Call this whenever the sidecar was not built from *this* file — i.e. for
    /// any shared ligand library. A sidecar is consumed positionally, so without
    /// it a file that lists the same atoms in a different order gets the wrong
    /// bonds, silently and with no error.
    ///
    /// Refuses (rather than guessing) when the atom SETS differ: a different
    /// protonation state or a truncated ligand is not something to paper over.
    /// A sidecar with no recorded names is left untouched and reported.
    pub fn align_to_pdb(&mut self, pdb_text: &str) -> Result<Vec<String>, LigandError> {
        let mut unnamed = Vec::new();
        let mut permuted = false;
        for name in &self.order.clone() {
            let t = self.ligands.get_mut(name).expect("order/ligands agree");
            let Some(lib_names) = t.atom_names.clone() else {
                unnamed.push(name.clone());
                continue;
            };
            // the file's own order for this residue name, HETATM lines as written
            let mut want: Vec<[u8; 4]> = Vec::new();
            for l in pdb_text.lines() {
                if l.len() >= 20 && l.starts_with("HETATM") && l[17..20].trim() == name.as_str() {
                    let b = l.as_bytes();
                    want.push([b[12], b[13], b[14], b[15]]);
                }
            }
            if want.is_empty() {
                // The requested ligand has no atoms in this structure. Its
                // coordinates would come from the PDB and there are none, so
                // `make_indep` would reserve rows it cannot fill. Refuse here,
                // where the message can still say something useful.
                return Err(LigandError::NotInStructure { name: name.clone() });
            }
            if want == lib_names {
                continue; // already in the right order
            }
            let show = |v: &[[u8; 4]]| -> String {
                v.iter().map(|n| String::from_utf8_lossy(n).trim().to_string())
                    .collect::<Vec<_>>().join(",")
            };
            if want.len() != lib_names.len() {
                return Err(LigandError::AtomSetMismatch {
                    name: name.clone(),
                    expected: show(&lib_names),
                    found: show(&want),
                });
            }
            let mut perm = Vec::with_capacity(want.len());
            let mut used = vec![false; lib_names.len()];
            for w in &want {
                match lib_names.iter().enumerate()
                    .find(|(j, n)| !used[*j] && *n == w) {
                    Some((j, _)) => { used[j] = true; perm.push(j); }
                    None => return Err(LigandError::AtomSetMismatch {
                        name: name.clone(),
                        expected: show(&lib_names),
                        found: show(&want),
                    }),
                }
            }
            t.permute(&perm);
            permuted = true;
        }
        if permuted || self.combined_atom_frames.is_none() {
            // Frames CANNOT be permuted into correctness: the reference picks
            // among tied candidates by CPython set-iteration order, which is a
            // function of the whole ligand block's insertion sequence, not of any
            // one atom. So recompute them the way `Indep.atom_frames` does — from
            // the (now correctly ordered) bond matrix. Verified atom-for-atom
            // against the pipeline on 1178 atoms; see tests/parity_atom_frames.rs.
            self.combined_atom_frames = Some(self.recompute_frames());
        }
        Ok(unnamed)
    }

    pub fn get(&self, name: &str) -> Option<&LigandTopology> {
        self.ligands.get(name)
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn total_atoms(&self) -> usize {
        self.order
            .iter()
            .map(|n| self.ligands[n].n_atoms)
            .sum()
    }

    /// Element tokens for all ligands, concatenated in load order — this is the
    /// small-molecule tail of `indep.seq`.
    pub fn elements(&self) -> Vec<i64> {
        let mut v = Vec::new();
        for n in &self.order {
            v.extend_from_slice(&self.ligands[n].elem);
        }
        v
    }

    /// Atom frames for all ligands — `rfi.atom_frames`.
    ///
    /// Prefers `combined.atom_frames`, which the sidecar generator takes from an
    /// actual reference pipeline run. That matters because `get_atom_frames`
    /// breaks priority ties by `list(set(allpaths))` order: on the demo ligand
    /// 20 of 50 atoms tie, and for OXM atom 3 the tie is between `(4,3,5)` and
    /// `(5,3,4)` — the same frame with its neighbours swapped, both scoring
    /// `[4, 4]`. A recomputation picks whichever the set happens to yield first,
    /// which is not necessarily what the pipeline picked.
    ///
    /// Falls back to per-ligand concatenation only when the sidecar predates
    /// this field.
    pub fn atom_frames(&self) -> Vec<i64> {
        if let Some(c) = &self.combined_atom_frames {
            return c.clone();
        }
        let mut v = Vec::new();
        for n in &self.order {
            v.extend_from_slice(&self.ligands[n].atom_frames);
        }
        v
    }

    /// `Indep.atom_frames` for the whole ligand block, from the bond matrix.
    ///
    /// Computed over the CONCATENATED ligands, not per ligand: every path from
    /// every ligand goes into one Python `set`, so the table layout — and with it
    /// the tie-break — depends on the combined insertion sequence.
    pub fn recompute_frames(&self) -> Vec<i64> {
        let (bf, n) = self.block_diag_bond_feats();
        let elems = self.elements();
        crate::atom_frames::get_atom_frames(&elems, &bf, n)
    }

    /// `torch.block_diag` of the per-ligand bond matrices — the ligand block of
    /// `indep.bond_feats`. Different ligands are never bonded to each other, so
    /// the off-diagonal blocks are zero.
    pub fn block_diag_bond_feats(&self) -> (Vec<i64>, usize) {
        let n = self.total_atoms();
        let mut out = vec![0i64; n * n];
        let mut off = 0usize;
        for name in &self.order {
            let lig = &self.ligands[name];
            for i in 0..lig.n_atoms {
                for j in 0..lig.n_atoms {
                    out[(off + i) * n + (off + j)] = lig.bond(i, j);
                }
            }
            off += lig.n_atoms;
        }
        (out, n)
    }
}
