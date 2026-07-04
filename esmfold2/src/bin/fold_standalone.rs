//! Standalone ESMFold2 fold from a bare sequence (no PyTorch, no fixtures).
//! usage: fold_standalone <SEQUENCE> [seed] [out.npy] [num_loops] [num_sampling_steps]
//! Writes coords .npy and the all-atom .pdb (if out given) and prints a metrics JSON line.
//!
//! `num_loops` / `num_sampling_steps` default to 3 / 14 — the reduced setting used for the
//! bit-exact fp32 benchmark, so a bare `fold_standalone <seq> <seed> [out]` reproduces the
//! published numbers. Pass `20 68` for the official ESMFold2 release trunk/diffusion depth.
//! (A single diffusion sample is produced; see the README.)

use esmfold2::standalone;
use esmfold2::weights::Weights;
use std::time::Instant;

fn home() -> String { std::env::var("HOME").unwrap() }
fn esmc_index() -> String {
    let base = format!("{}/.cache/huggingface/hub/models--biohub--ESMC-6B/snapshots", home());
    let s = std::fs::read_dir(&base).unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .find(|p| p.is_dir()).unwrap();
    s.join("model.safetensors.index.json").to_string_lossy().to_string()
}
fn head_path() -> String {
    let base = format!("{}/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots", home());
    let s = std::fs::read_dir(&base).unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .find(|p| p.is_dir()).unwrap();
    s.join("model.safetensors").to_string_lossy().to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seq = &args[1];
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = args.get(3).cloned();
    let num_loops: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let num_steps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(14);

    let w_esmc = Weights::open_sharded(&esmc_index()).unwrap();
    let w = Weights::open(&head_path()).unwrap();

    let t = Instant::now();
    // Progress to stderr (JSON metrics still go to stdout). A carriage return keeps
    // it on one self-updating line; the ESM-C layer / trunk-block / diffusion-step
    // messages mirror the GUI and the ESMFold1 CLI.
    use std::io::Write;
    let o = standalone::fold_cb(seq, seed, &w_esmc, &w, num_loops, num_steps, &mut |msg, frac| {
        eprint!("\r\x1b[K[{:3.0}%] {msg}", frac * 100.0);
        let _ = std::io::stderr().flush();
    });
    eprintln!();
    let secs = t.elapsed().as_secs_f32();

    if let Some(path) = out {
        write_npy(std::path::Path::new(&path), &o.coords, &[o.n_atoms, 3]);
        // Also emit the all-atom PDB (same coords/writer the GUI uses) for direct
        // PDB-to-PDB comparison: <out with .npy->.pdb>, or <out>.pdb otherwise.
        let pdb_path = if let Some(stem) = path.strip_suffix(".npy") {
            format!("{stem}.pdb")
        } else {
            format!("{path}.pdb")
        };
        std::fs::write(&pdb_path, &o.pdb).unwrap();
    }
    println!(
        "{{\"L\":{},\"n_atoms\":{},\"seed\":{seed},\"fold_s\":{secs:.2},\"plddt_mean\":{:.5},\"ptm\":{:.5},\"complex_plddt\":{:.5}}}",
        o.l, o.n_atoms, o.plddt_mean, o.ptm, o.complex_plddt
    );
}

fn write_npy(path: &std::path::Path, data: &[f32], shape: &[usize]) {
    use std::io::Write;
    let shape_str = shape.iter().map(|s| format!("{s}")).collect::<Vec<_>>().join(", ");
    let shape_tuple = if shape.len() == 1 { format!("({shape_str},)") } else { format!("({shape_str})") };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
    let total = 10 + header.len() + 1;
    let pad = (64 - (total % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes).unwrap();
}
