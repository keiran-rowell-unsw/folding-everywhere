//! Rung 8 — a **variable-length** contig, and `contigmap.length`.
//!
//! `get_sampled_mask` resolves each length range with **CPython's**
//! `random.randint` (not numpy's, not torch's) and then re-samples the whole
//! contig until the total matches `contigmap.length`. The rejection loop means
//! the number of draws is not a function of the contig alone, so reproducing
//! the sampled mask is a real test of both the generator and the loop.
//!
//! Measured on the reference: `'5-15,A106-106,5-15'` with `length=25-25`
//! resolves to `10-10,A106-106,14-14` and leaves CPython's generator at
//! position 55 (from 624, i.e. it twisted).

use rfd2::contig::{parse_length, ContigMap};
use rfd2::pdb;
use rfd2::rng::pyrandom::PyRandom;
use std::path::Path;

const PDB: &str = "../ref_RFdiffusion2/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb";
const CONTIGS: &str = "5-15,A106-106,5-15";
/// what the reference sampled, from `seed_all(0)`
const EXPECTED_MASK: &str = "10-10,A106-106,14-14";

fn root(rel: &str) -> String {
    format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn length_parsing_is_half_open() {
    // '180-180' -> [180, 181): the form every mcsa_41 benchmark entry uses, and
    // the reason it means "exactly 180" rather than "nothing".
    assert_eq!(parse_length("180-180").unwrap(), (180, 181));
    assert_eq!(parse_length("25-25").unwrap(), (25, 26));
    assert_eq!(parse_length("180").unwrap(), (180, 181));
    assert_eq!(parse_length("10-20").unwrap(), (10, 21));
    println!("parse_length: 180-180 -> [180, 181), 10-20 -> [10, 21)");
}

#[test]
fn variable_contig_is_refused_without_a_generator() {
    let p = root(PDB);
    if !Path::new(&p).exists() {
        return;
    }
    let feats = pdb::parse_pdb_str(&std::fs::read_to_string(&p).unwrap(), true, true);
    let err = ContigMap::parse(&feats, CONTIGS).unwrap_err();
    println!("without a generator: {err}");
}

#[test]
fn sampled_mask_matches_the_reference() {
    let p = root(PDB);
    if !Path::new(&p).exists() {
        eprintln!("SKIP: {p} missing");
        return;
    }
    let feats = pdb::parse_pdb_str(&std::fs::read_to_string(&p).unwrap(), true, true);

    // `run_inference.seed_all(i_des + seed_offset)` -> `random.seed(0)`
    let mut py = PyRandom::new(0);
    let cmap = {
        let mut r: Option<&mut PyRandom> = Some(&mut py);
        ContigMap::parse_with(&feats, CONTIGS, Some(parse_length("25-25").unwrap()), &mut r)
            .expect("contig parse")
    };

    let got = cmap.sampled_mask.join("_");
    println!("sampled_mask   got {got:?}   want {EXPECTED_MASK:?}");
    println!("deterministic  {}", cmap.deterministic);
    println!("contig_length  {} (25 designed + the ligand rows come later)", cmap.contig_length);

    assert_eq!(got, EXPECTED_MASK, "the sampled contig does not match the reference");
    assert!(!cmap.deterministic, "a range contig must report deterministic = false");
    assert_eq!(cmap.contig_length, 25, "the length constraint was not honoured");
}
