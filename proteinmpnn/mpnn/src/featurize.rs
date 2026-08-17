//! Batch packing — the single-protein path through `tied_featurize`.
//!
//! Chain order is load-bearing: designed chains first (sorted), then fixed
//! chains (sorted). `residue_idx` restarts with a +100 offset per chain, which
//! is how the model learns to treat chain breaks; `chain_encoding` is a 1-based
//! counter over that same concatenation order.

use crate::pdb::Structure;

#[derive(Debug, Clone)]
pub struct Batch {
    /// Total residues across all included chains.
    pub l: usize,
    /// `[L*4*3]` N/CA/C/O coordinates, NaN replaced by 0.
    pub x: Vec<f32>,
    /// `[L]` native sequence as indices into `ALPHABET`.
    pub s: Vec<i64>,
    /// `[L]` 1.0 where all four backbone atoms are present.
    pub mask: Vec<f32>,
    /// `[L]` 1.0 for residues on a designed chain.
    pub chain_m: Vec<f32>,
    /// `[L]` 1.0 unless the position is explicitly fixed.
    pub chain_m_pos: Vec<f32>,
    /// `[L]` `100*(c-1) + global_index`.
    pub residue_idx: Vec<i64>,
    /// `[L]` 1-based chain counter.
    pub chain_encoding: Vec<i64>,
    /// Concatenated one-letter native sequence (with 'X' for unknown).
    pub seq: String,
    /// Chain ids in concatenation order, with their lengths.
    pub chain_order: Vec<(char, usize)>,
    pub designed: Vec<char>,
    pub fixed: Vec<char>,
}

/// `tied_featurize(batch=[one protein], ...)` for the common case: no tied
/// positions, no PSSM, no per-residue bias.
///
/// `designed` / `fixed` are chain ids; each is sorted, then concatenated
/// designed-first, exactly as upstream does.
pub fn featurize(st: &Structure, designed: &[char], fixed: &[char]) -> Batch {
    let mut designed: Vec<char> = designed.to_vec();
    let mut fixed: Vec<char> = fixed.to_vec();
    designed.sort_unstable();
    fixed.sort_unstable();

    let mut b = Batch {
        l: 0,
        x: Vec::new(),
        s: Vec::new(),
        mask: Vec::new(),
        chain_m: Vec::new(),
        chain_m_pos: Vec::new(),
        residue_idx: Vec::new(),
        chain_encoding: Vec::new(),
        seq: String::new(),
        chain_order: Vec::new(),
        designed: designed.clone(),
        fixed: fixed.clone(),
    };

    // all_chains = masked (designed) + visible (fixed), processed in that order.
    let all: Vec<(char, bool)> = designed
        .iter()
        .map(|&c| (c, true))
        .chain(fixed.iter().map(|&c| (c, false)))
        .collect();

    let mut c = 1i64; // 1-based chain counter
    let mut l0 = 0usize; // running global residue offset
    for (letter, is_designed) in all {
        let chain = match st.chain(letter) {
            Some(ch) => ch,
            None => continue,
        };
        let n = chain.seq.len();
        for (i, aa) in chain.seq.bytes().enumerate() {
            let aa = if aa == b'-' { b'X' } else { aa };
            b.seq.push(aa as char);
            b.s.push(crate::aa_to_idx(aa) as i64);
            let c4 = &chain.coords[i];
            let mut finite = true;
            for a in 0..4 {
                for k in 0..3 {
                    let v = c4[a][k];
                    if !v.is_finite() {
                        finite = false;
                    }
                }
            }
            for a in 0..4 {
                for k in 0..3 {
                    let v = c4[a][k];
                    b.x.push(if v.is_finite() { v } else { 0.0 });
                }
            }
            b.mask.push(if finite { 1.0 } else { 0.0 });
            b.chain_m.push(if is_designed { 1.0 } else { 0.0 });
            b.chain_m_pos.push(1.0);
            b.residue_idx.push(100 * (c - 1) + (l0 + i) as i64);
            b.chain_encoding.push(c);
        }
        b.chain_order.push((letter, n));
        l0 += n;
        c += 1;
    }
    b.l = b.s.len();
    b
}

impl Batch {
    /// `chain_M * chain_M_pos * mask` — the positions the sampler actually
    /// redesigns (everything else is copied from the native sequence).
    ///
    /// This is also `mask_for_loss` in the reference, i.e. the weighting for the
    /// reported `score` and for sequence recovery (as opposed to `global_score`,
    /// which weights by `mask` alone).
    pub fn design_mask(&self) -> Vec<f32> {
        (0..self.l)
            .map(|i| self.chain_m[i] * self.chain_m_pos[i] * self.mask[i])
            .collect()
    }

    /// The designed chains, in output order, with their lengths.
    ///
    /// `featurize` lays designed chains out first (sorted), so these are the
    /// leading entries of `chain_order`. The reference's FASTA reports only
    /// these chains, `/`-separated.
    pub fn designed_chains(&self) -> Vec<(char, usize)> {
        self.chain_order
            .iter()
            .copied()
            .filter(|(c, _)| self.designed.contains(c))
            .collect()
    }

    /// Render a sequence the way `protein_mpnn_run.py` writes it: only residues
    /// on designed chains (`_S_to_seq(S, chain_M)`), split into per-chain blocks
    /// joined with '/'.
    pub fn format_seq(&self, s: &[i64]) -> String {
        let mut out = String::new();
        let mut pos = 0usize;
        for (ci, (_letter, len)) in self.designed_chains().iter().enumerate() {
            if ci > 0 {
                out.push('/');
            }
            for i in pos..pos + len {
                if self.chain_m[i] > 0.0 {
                    out.push(crate::idx_to_aa(s[i] as usize));
                }
            }
            pos += len;
        }
        out
    }
}
