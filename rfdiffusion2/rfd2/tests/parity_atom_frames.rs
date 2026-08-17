//! `get_atom_frames` in Rust vs the pipeline's own frames, on every ligand
//! captured from a reference run.
//!
//! This is the test that decides whether a ligand library can be used on a PDB
//! it was not built from. The frames are chosen by CPython's set iteration
//! order, so "close" is not a category here: either all 1178 atoms match or the
//! reproduction is wrong.

use rfd2::atom_frames::get_atom_frames;
use rfd2::weights::Weights;
use std::path::Path;

#[test]
fn frames_match_the_pipeline_on_every_captured_ligand() {
    let root = format!("{}/..", env!("CARGO_MANIFEST_DIR"));
    let dir = format!("{root}/bench/ligand_runs");
    if !Path::new(&dir).exists() {
        eprintln!("SKIP: {dir} missing — run bench/build_ligand_library.sh");
        return;
    }
    let mut entries: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    let (mut ok, mut tot, mut n_in) = (0usize, 0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();
    for e in entries {
        let p = e.path().join("rfi.safetensors");
        if !p.exists() {
            continue;
        }
        let w = Weights::open(&p.to_string_lossy()).expect("open rfi");
        let (seq, seq_shape) = w.get_i64("rfi.seq");
        let (bf, bf_shape) = w.get_i64("rfi.bond_feats");
        let (want, _) = w.get_i64("rfi.atom_frames");
        let l = seq_shape[seq_shape.len() - 1];
        let bf = &bf[bf.len() - l * l..]; // drop the leading batch axis

        // ligand rows are the atom tokens, and they are contiguous at the end
        let is_sm: Vec<bool> = seq[seq.len() - l..].iter().map(|&t| rfd2::geom::is_atom(t)).collect();
        let idx: Vec<usize> = (0..l).filter(|&i| is_sm[i]).collect();
        if idx.is_empty() {
            continue;
        }
        let n = idx.len();
        let sub_seq: Vec<i64> = idx.iter().map(|&i| seq[seq.len() - l + i]).collect();
        let mut sub_bf = vec![0i64; n * n];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                sub_bf[a * n + b] = bf[i * l + j];
            }
        }
        let got = get_atom_frames(&sub_seq, &sub_bf, n);
        let m = (0..n).filter(|&a| got[a * 6..a * 6 + 6] == want[a * 6..a * 6 + 6]).count();
        ok += m;
        tot += n;
        n_in += 1;
        if m != n {
            bad.push(format!("{}: {m}/{n}", e.file_name().to_string_lossy()));
        }
    }
    println!("atom_frames reproduced in Rust: {ok}/{tot} atoms across {n_in} inputs");
    if !bad.is_empty() {
        println!("mismatched inputs:");
        for b in &bad {
            println!("   {b}");
        }
    }
    assert!(tot > 0, "no captured ligands found");
    assert_eq!(ok, tot, "{} of {} inputs disagree", bad.len(), n_in);
}
