//! `rf2aa.util.get_atom_frames`, in Rust — including CPython's set iteration
//! order, which is what actually decides the answer.
//!
//! # Why this exists
//!
//! Each ligand atom needs a local frame built from two bonded neighbours. Often
//! several candidates tie on priority, and the reference resolves the tie by
//! taking the FIRST one out of `list(set(allpaths))` — i.e. by CPython's set
//! iteration order over tuples of small ints. Measured across the 41 benchmark
//! inputs, ties decide the frame for a large fraction of atoms, so this is not a
//! detail that can be approximated.
//!
//! It is reproducible because integer and tuple hashes are **not** randomised by
//! `PYTHONHASHSEED` (only `str`/`bytes` are), so the table layout is a pure
//! function of the values and their insertion order.
//!
//! # What the frames depend on
//!
//! Only the **bond matrix** and the sequence tokens. Inference builds the graph
//! with `nx.from_numpy_matrix(bond_matrix)` (`Indep.atom_frames`), so node order
//! is `0..n-1` and neighbours are in index order — nothing about OpenBabel's
//! internal ordering enters. That is what makes a ligand library usable on a
//! new PDB: permute the bonds by atom name, then recompute the frames here.
//!
//! (`python/gen_ligand_bonds.py` builds its graph from the OpenBabel molecule
//! instead, which is a *different* node order and therefore a different tie
//! break — the source of the 20 % disagreement seen when frames are copied
//! rather than recomputed.)

include!("atom_frames_priority.rs");

// ---- CPython hashing -------------------------------------------------------

const XXPRIME_1: u64 = 11400714785074694791;
const XXPRIME_2: u64 = 14029467366897019727;
const XXPRIME_5: u64 = 2870177450012600261;
const MODULUS: u64 = (1u64 << 61) - 1; // _PyHASH_MODULUS

/// `hash(int)` for CPython: sign-magnitude modulo `2**61 - 1`, and never -1.
fn hash_i64(v: i64) -> u64 {
    let neg = v < 0;
    let m = (v.unsigned_abs()) % MODULUS;
    let h = if neg { (!m).wrapping_add(1) } else { m }; // -m as u64
    if h == u64::MAX { u64::MAX - 1 } else { h } // hash never returns -1
}

/// `hash(tuple)` for CPython >= 3.8 (the xxPrime construction in tupleobject.c).
fn hash_tuple(items: &[i64]) -> u64 {
    let mut acc = XXPRIME_5;
    for &it in items {
        let lane = hash_i64(it);
        acc = acc.wrapping_add(lane.wrapping_mul(XXPRIME_2));
        acc = (acc << 31) | (acc >> 33); // XXROTATE
        acc = acc.wrapping_mul(XXPRIME_1);
    }
    acc = acc.wrapping_add((items.len() as u64) ^ (XXPRIME_5 ^ 3527539));
    if acc == u64::MAX { 1546275796 } else { acc }
}

// ---- CPython's set ---------------------------------------------------------

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const MINSIZE: usize = 8;

/// A faithful-enough `set` of int tuples: same probe sequence, same growth
/// schedule, so `iter()` yields the same order `list(set(...))` would.
struct PySet {
    keys: Vec<Option<Vec<i64>>>,
    hashes: Vec<u64>,
    mask: usize,
    fill: usize,
    used: usize,
}

impl PySet {
    fn new() -> Self {
        PySet { keys: vec![None; MINSIZE], hashes: vec![0; MINSIZE], mask: MINSIZE - 1,
                fill: 0, used: 0 }
    }

    fn contains_at(&self, i: usize, h: u64, key: &[i64]) -> bool {
        self.hashes[i] == h && self.keys[i].as_deref() == Some(key)
    }

    fn add(&mut self, key: Vec<i64>) {
        let h = hash_tuple(&key);
        let mut i = (h as usize) & self.mask;
        let mut perturb = h;
        loop {
            if self.keys[i].is_none() {
                break;
            }
            if self.contains_at(i, h, &key) {
                return;
            }
            // the linear-probe window, only when it fits without wrapping
            if i + LINEAR_PROBES <= self.mask {
                let mut j = i;
                let mut placed = None;
                for _ in 0..LINEAR_PROBES {
                    j += 1;
                    if self.keys[j].is_none() {
                        placed = Some(j);
                        break;
                    }
                    if self.contains_at(j, h, &key) {
                        return;
                    }
                }
                if let Some(j) = placed {
                    i = j;
                    break;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & self.mask;
        }
        self.keys[i] = Some(key);
        self.hashes[i] = h;
        self.fill += 1;
        self.used += 1;
        if self.fill * 5 >= self.mask * 3 {
            let minused = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
            self.resize(minused);
        }
    }

    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        let old: Vec<(u64, Vec<i64>)> = self
            .keys
            .iter()
            .zip(&self.hashes)
            .filter_map(|(k, h)| k.as_ref().map(|k| (*h, k.clone())))
            .collect();
        self.keys = vec![None; newsize];
        self.hashes = vec![0; newsize];
        self.mask = newsize - 1;
        self.fill = old.len();
        self.used = old.len();
        // set_insert_clean: no equality checks, the keys are known distinct
        for (h, k) in old {
            let mut i = (h as usize) & self.mask;
            let mut perturb = h;
            loop {
                if self.keys[i].is_none() {
                    break;
                }
                if i + LINEAR_PROBES <= self.mask {
                    let mut j = i;
                    let mut placed = None;
                    for _ in 0..LINEAR_PROBES {
                        j += 1;
                        if self.keys[j].is_none() {
                            placed = Some(j);
                            break;
                        }
                    }
                    if let Some(j) = placed {
                        i = j;
                        break;
                    }
                }
                perturb >>= PERTURB_SHIFT;
                i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & self.mask;
            }
            self.keys[i] = Some(k);
            self.hashes[i] = h;
        }
    }

    /// Slot order — exactly what `list(set(...))` produces.
    fn into_list(self) -> Vec<Vec<i64>> {
        self.keys.into_iter().flatten().collect()
    }
}

// ---- get_atom_frames -------------------------------------------------------

/// `[n, 3, 2]` frames as `(offset, 1)` pairs, matching `Indep.atom_frames`.
///
/// `bond_feats` is the `[n, n]` ligand bond matrix; an entry in `1..=4` is a
/// covalent bond. `seq` is the per-atom element token.
pub fn get_atom_frames(seq: &[i64], bond_feats: &[i64], n: usize) -> Vec<i64> {
    // neighbours in ascending index order: `nx.from_numpy_matrix` inserts edges
    // row-major over the upper triangle, so node u's adjacency ends up ordered
    // by the other endpoint's index.
    let nbrs: Vec<Vec<usize>> = (0..n)
        .map(|u| {
            (0..n)
                .filter(|&v| v != u && (1..=4).contains(&bond_feats[u * n + v]))
                .collect()
        })
        .collect();

    // allpaths = [tuple(p) for node in G for p in findPaths(G, node, 2)]
    // findPaths(G,u,2) walks u -> nb -> nb2 with nb2 != nb and nb2 != u.
    let mut set = PySet::new();
    for u in 0..n {
        for &nb in &nbrs[u] {
            for &nb2 in &nbrs[nb] {
                if nb2 != nb && nb2 != u {
                    set.add(vec![u as i64, nb as i64, nb2 as i64]);
                }
            }
        }
    }
    let frames = set.into_list();

    let pri = |tok: i64| -> i64 {
        TOKEN_FRAME_PRIORITY
            .iter()
            .find(|(t, _)| *t == tok)
            .map(|(_, p)| *p)
            .unwrap_or(i64::MAX)
    };

    let mut out = Vec::with_capacity(n * 6);
    for a in 0..n {
        let ai = a as i64;
        // frames centred on `a`, then any frame containing it
        let mut cands: Vec<&Vec<i64>> = frames.iter().filter(|f| f[1] == ai).collect();
        if cands.is_empty() {
            cands = frames.iter().filter(|f| f.contains(&ai)).collect();
        }
        if cands.is_empty() {
            out.extend_from_slice(&[0, 1, 0, 1, 0, 1]);
            continue;
        }
        // omit_permutation=False -> priorities in frame order, NOT sorted
        let keys: Vec<Vec<i64>> = cands
            .iter()
            .map(|f| f.iter().filter(|&&i| i != ai).map(|&i| pri(seq[i as usize])).collect())
            .collect();
        // stable sort by the key list, take the first minimum
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&x, &y| keys[x].cmp(&keys[y]));
        let best = cands[order[0]];
        for &f in best {
            out.push(f - ai);
            out.push(1);
        }
    }
    out
}
