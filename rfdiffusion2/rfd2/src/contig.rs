//! `rf_diffusion/contigs.py:ContigMap` — the map from the reference structure's
//! residues to the designed chain's rows.
//!
//! A contig string like `"10,A106-106,10"` means: ten designed residues, then
//! reference residue A106, then ten more designed residues. Underscore-pairs
//! mark the designed positions, which have no reference to point at.
//!
//! ## What consumes randomness, and from which stream
//!
//! `get_sampled_mask` calls **`random.randint`** — CPython's Mersenne Twister,
//! not numpy's and not torch's — for any length written as a range (`10-20`).
//! A fixed length draws nothing, which is why `python/probe_featurize.py`
//! measured zero RNG consumption for the demo configuration. [`ContigMap::parse`]
//! therefore refuses ranges rather than pretending to be deterministic: a range
//! needs `rng::pyrandom` threaded in, and needs its own fixture.
//!
//! Two numbering details that are easy to get wrong and silent when wrong:
//!
//! * designed positions are numbered from 1, continuing across a motif, and
//! * a chain break adds **32** to the running index, not the +200 that separates
//!   ligands.

use crate::pdb::TargetFeats;

const CHAIN_ORDER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Gap inserted in the output numbering between contig chains.
const CHAIN_GAP: i64 = 32;

/// One row of the map: either a reference residue or a designed position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ref {
    /// `(chain, residue number)` in the reference PDB
    Res(u8, i64),
    /// `('_', '_')` — a designed position
    Gap,
}

#[derive(Debug)]
pub struct ContigMap {
    /// per row, where it came from in the reference
    pub reference: Vec<Ref>,
    /// per row, `(chain, residue number)` in the designed output
    pub hal: Vec<(u8, i64)>,
    /// rows that carry a reference residue
    pub hal_idx0: Vec<usize>,
    /// the reference row each of those came from, as an index into the PDB
    pub ref_idx0: Vec<usize>,
    /// `true` where the row's structure is *shown* to the model. Note the name
    /// is upstream's and reads backwards: `inpaint_str[i] == false` means
    /// "diffuse this residue's structure".
    pub inpaint_str: Vec<bool>,
    pub inpaint_seq: Vec<bool>,
    /// number of contig chains (`_`-separated groups)
    pub n_inpaint_chains: usize,
    /// total designed length, excluding ligands
    pub contig_length: usize,
    /// `get_sampled_mask`'s output — the contig with every length range
    /// resolved, e.g. `["10-10,A106-106,10-10"]`. The `.trb` writes it.
    pub sampled_mask: Vec<String>,
    /// True when no segment was a range, i.e. the contig needed no randomness.
    pub deterministic: bool,
}

/// `contigs.parse_length` — `'10-20'` becomes the half-open `[10, 21)`, and a
/// bare `'180'` becomes `[180, 181)`. The half-open form is why
/// `contigmap.length=180-180` means exactly 180 rather than nothing.
pub fn parse_length(length: &str) -> Result<(usize, usize), ContigError> {
    let bad = || ContigError::Malformed(length.to_string());
    match length.split_once('-') {
        Some((a, b)) => Ok((
            a.parse().map_err(|_| bad())?,
            b.parse::<usize>().map_err(|_| bad())? + 1,
        )),
        None => {
            let v: usize = length.parse().map_err(|_| bad())?;
            Ok((v, v + 1))
        }
    }
}

#[derive(Debug)]
pub enum ContigError {
    VariableLength(String),
    NotInPdb(u8, i64),
    Malformed(String),
}

impl std::fmt::Display for ContigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContigError::VariableLength(s) => write!(
                f,
                "contig segment {s:?} is a length RANGE. Upstream samples it with \
                 CPython's `random.randint`, so reproducing it needs the pyrandom \
                 stream threaded through contig parsing and its own fixture. \
                 Refusing rather than silently picking a length."
            ),
            ContigError::NotInPdb(c, i) => {
                write!(f, "contig references residue {}{i}, which is not in the PDB",
                       *c as char)
            }
            ContigError::Malformed(s) => write!(f, "cannot parse contig segment {s:?}"),
        }
    }
}

impl std::error::Error for ContigError {}

impl ContigMap {
    /// Parse one contig string against a parsed PDB.
    ///
    /// `contigs` is upstream's list; chains are separated by `_` *within* the
    /// single string, and segments by `,`.
    pub fn parse(feats: &TargetFeats, contigs: &str) -> Result<Self, ContigError> {
        Self::parse_with(feats, contigs, None, &mut None)
    }

    /// `ContigMap.__init__`, including `get_sampled_mask`.
    ///
    /// A length *range* in a designed segment is sampled with **CPython's**
    /// `random.randint` — not numpy's and not torch's — and the whole contig is
    /// then re-sampled until its total length falls in `length`. Both facts are
    /// load-bearing: drawing from the wrong generator shifts every later draw,
    /// and the rejection loop consumes a variable number of them.
    ///
    /// `rng` is `None` for a fixed-length contig, and a range is refused in
    /// that case rather than silently picking a length.
    pub fn parse_with(
        feats: &TargetFeats,
        contigs: &str,
        length: Option<(usize, usize)>,
        rng: &mut Option<&mut crate::rng::pyrandom::PyRandom>,
    ) -> Result<Self, ContigError> {
        let (sampled, deterministic) = Self::get_sampled_mask(contigs, length, rng)?;
        let resolved = sampled.join("_");
        let mut me = Self::parse_resolved(feats, &resolved)?;
        me.sampled_mask = sampled;
        me.deterministic = deterministic;
        Ok(me)
    }

    /// `get_sampled_mask` — resolve every length range, retrying until the
    /// total is compatible with `length`.
    fn get_sampled_mask(
        contigs: &str,
        length: Option<(usize, usize)>,
        rng: &mut Option<&mut crate::rng::pyrandom::PyRandom>,
    ) -> Result<(Vec<String>, bool), ContigError> {
        const MAX_ATTEMPTS: usize = 100_000_000;
        let mut attempts = 0usize;
        let (mut min_seen, mut max_seen) = (usize::MAX, 0usize);
        loop {
            let mut deterministic = true;
            let mut sampled: Vec<String> = Vec::new();
            let mut total = 0usize;
            for group in contigs.trim().split('_') {
                let mut out: Vec<String> = Vec::new();
                for seg in group.split(',') {
                    let first =
                        seg.chars().next().ok_or_else(|| ContigError::Malformed(seg.into()))?;
                    if first.is_alphabetic() {
                        out.push(seg.to_string());
                        // the atom spec after '/' is not part of the length
                        let body = seg.split('/').next().unwrap_or(seg);
                        total += match body[1..].split_once('-') {
                            Some((a, b)) => {
                                let lo: i64 = a
                                    .parse()
                                    .map_err(|_| ContigError::Malformed(seg.into()))?;
                                let hi: i64 = b
                                    .parse()
                                    .map_err(|_| ContigError::Malformed(seg.into()))?;
                                (hi - lo + 1) as usize
                            }
                            None => 1,
                        };
                    } else if let Some((a, b)) = seg.split_once('-') {
                        let lo: i64 =
                            a.parse().map_err(|_| ContigError::Malformed(seg.into()))?;
                        let hi: i64 =
                            b.parse().map_err(|_| ContigError::Malformed(seg.into()))?;
                        let n = if lo == hi {
                            lo
                        } else {
                            deterministic = false;
                            match rng {
                                Some(r) => r.randint(lo, hi),
                                None => {
                                    return Err(ContigError::VariableLength(seg.into()))
                                }
                            }
                        };
                        out.push(format!("{n}-{n}"));
                        total += n as usize;
                    } else if seg == "0" {
                        // a zero-length run is emitted verbatim and adds nothing
                        out.push("0".into());
                    } else {
                        let n: usize =
                            seg.parse().map_err(|_| ContigError::Malformed(seg.into()))?;
                        out.push(format!("{n}-{n}"));
                        total += n;
                    }
                }
                sampled.push(out.join(","));
            }
            min_seen = min_seen.min(total);
            max_seen = max_seen.max(total);
            match length {
                None => return Ok((sampled, deterministic)),
                Some((lo, hi)) if total >= lo && total < hi => {
                    return Ok((sampled, deterministic))
                }
                Some(_) => {}
            }
            attempts += 1;
            if attempts == MAX_ATTEMPTS || (deterministic && attempts > 1) {
                return Err(ContigError::Malformed(format!(
                    "contig {contigs:?} is incompatible with length {length:?}: \
                     sampled lengths {min_seen}..={max_seen} in {attempts} attempts"
                )));
            }
        }
    }

    fn parse_resolved(feats: &TargetFeats, contigs: &str) -> Result<Self, ContigError> {
        let mut reference: Vec<Ref> = Vec::new();
        let mut hal: Vec<(u8, i64)> = Vec::new();
        let mut hal_idx: i64 = 1;
        let mut n_chains = 0usize;
        let mut contig_length = 0usize;

        for (chain_i, group) in contigs.trim().split('_').enumerate() {
            n_chains += 1;
            let out_chain = CHAIN_ORDER[chain_i];
            for seg in group.split(',') {
                let seg = seg.split('/').next().unwrap_or(seg);
                let first = seg.chars().next().ok_or_else(|| ContigError::Malformed(seg.into()))?;
                if first.is_alphabetic() {
                    // a motif segment: <chain><start>-<end>, or <chain><n>
                    let c = first as u8;
                    let body = &seg[1..];
                    let (lo, hi) = match body.split_once('-') {
                        Some((a, b)) => (
                            a.parse::<i64>().map_err(|_| ContigError::Malformed(seg.into()))?,
                            b.parse::<i64>().map_err(|_| ContigError::Malformed(seg.into()))?,
                        ),
                        None => {
                            let v = body
                                .parse::<i64>()
                                .map_err(|_| ContigError::Malformed(seg.into()))?;
                            (v, v)
                        }
                    };
                    for r in lo..=hi {
                        reference.push(Ref::Res(c, r));
                        hal.push((out_chain, hal_idx));
                        hal_idx += 1;
                        contig_length += 1;
                    }
                } else {
                    // a designed run: a plain integer, or a refused range
                    if seg.contains('-') {
                        let (a, b) = seg.split_once('-').unwrap();
                        if a != b {
                            return Err(ContigError::VariableLength(seg.into()));
                        }
                    }
                    let n: usize = seg
                        .split('-')
                        .next()
                        .unwrap()
                        .parse()
                        .map_err(|_| ContigError::Malformed(seg.into()))?;
                    for _ in 0..n {
                        reference.push(Ref::Gap);
                        hal.push((out_chain, hal_idx));
                        hal_idx += 1;
                        contig_length += 1;
                    }
                }
            }
            hal_idx += CHAIN_GAP;
        }

        // `pdb_idx` order is the order residues appear in the parsed structure.
        let pdb_idx: Vec<(u8, i64)> = feats
            .residues
            .iter()
            .map(|r| (r.chain.as_bytes()[0], r.res_seq))
            .collect();
        let mut hal_idx0 = Vec::new();
        let mut ref_idx0 = Vec::new();
        for (i, r) in reference.iter().enumerate() {
            if let Ref::Res(c, n) = r {
                let p = pdb_idx
                    .iter()
                    .position(|x| x == &(*c, *n))
                    .ok_or(ContigError::NotInPdb(*c, *n))?;
                hal_idx0.push(i);
                ref_idx0.push(p);
            }
        }

        // With no explicit inpaint_seq/inpaint_str given, both are simply
        // "this row has a reference residue".
        let mask: Vec<bool> = reference.iter().map(|r| *r != Ref::Gap).collect();

        Ok(ContigMap {
            reference,
            hal,
            hal_idx0,
            ref_idx0,
            inpaint_str: mask.clone(),
            inpaint_seq: mask,
            n_inpaint_chains: n_chains,
            contig_length,
            sampled_mask: Vec::new(),
            deterministic: true,
        })
    }

    /// `chain_start_end_from_hal` — the `[start, end)` row range of each output
    /// chain, in `hal` order.
    pub fn chain_start_end(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0usize;
        for i in 1..self.hal.len() {
            if self.hal[i].0 != self.hal[i - 1].0 {
                out.push((start, i));
                start = i;
            }
        }
        if !self.hal.is_empty() {
            out.push((start, self.hal.len()));
        }
        out
    }
}
