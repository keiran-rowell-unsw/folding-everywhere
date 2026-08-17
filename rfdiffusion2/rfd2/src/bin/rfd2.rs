//! `rfd2` — the command-line entry point, mirroring `run_inference.py`'s
//! options for the configurations this port has been measured against.
//!
//! Deliberately not a Hydra clone: only the flags that change what the port
//! computes are exposed, and anything the port has not been compared against is
//! refused by the modules underneath rather than silently accepted.

use rfd2::design::{run_design, DesignConfig};
use rfd2::ligand::LigandSet;
use rfd2::model::rf::{Arch, RoseTTAFold};
use rfd2::nn::Params;
use rfd2::noiser::Igso3;
use rfd2::weights::Weights;

const USAGE: &str = "\
rfd2 - pure-Rust RFdiffusion2 (bit-exact against the pinned reference)

USAGE:
    rfd2 --input-pdb <FILE> --contigs <STR> --weights <FILE> --ligand-topology <FILE>
         --igso3 <FILE> [--ligand NAD,OXM] [--T 100] [--final-step 1]
         [--num-designs 1] [--seed-offset 0] [--output-prefix out/design]

    --input-pdb        the target structure
    --contigs          e.g. '10,A106-106,10'; a range like '5-15' needs --length
    --ligand           comma-separated residue names present in the PDB
    --ligand-topology  sidecar produced by python/gen_ligand_bonds.py for this PDB
    --weights          the official RFD_173.pt, or a safetensors export of it
    --igso3            IGSO(3) table safetensors (fixtures/noiser/stages.safetensors)
    --T                diffusion steps (default 100)
    --length           total designed length, e.g. 180-180 (needed for a range contig)
    --partial-t        start the trajectory at this step instead of T
    --str-self-cond    show the model its own previous prediction (template 1)
    --output-prefix    files are written as <prefix>_<i>-atomized-bb-False.pdb
";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(());
    }
    let need = |n: &str| -> String {
        arg(&args, n).unwrap_or_else(|| {
            eprintln!("missing required argument {n}\n\n{USAGE}");
            std::process::exit(2);
        })
    };

    let input_pdb = need("--input-pdb");
    let contigs = need("--contigs");
    let weights_path = need("--weights");
    let topo_path = need("--ligand-topology");
    let igso3_path = need("--igso3");
    let ligands: Vec<String> = arg(&args, "--ligand")
        .map(|s| {
            s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
        })
        .unwrap_or_default();
    let big_t: usize = arg(&args, "--T").map(|s| s.parse()).transpose()?.unwrap_or(100);
    let final_step: usize =
        arg(&args, "--final-step").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let num_designs: usize =
        arg(&args, "--num-designs").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let seed_offset: u64 =
        arg(&args, "--seed-offset").map(|s| s.parse()).transpose()?.unwrap_or(0);
    let prefix = arg(&args, "--output-prefix").unwrap_or_else(|| "design".into());

    let pdb_text = std::fs::read_to_string(&input_pdb)?;
    let mut topo = LigandSet::load(&topo_path, &ligands)?;
    // A sidecar is consumed POSITIONALLY, so if it was not built from this very
    // file its atom order may differ. Align by atom name (a no-op when the order
    // already matches); refuses on a different atom set rather than guessing.
    match topo.align_to_pdb(&pdb_text) {
        Ok(unnamed) if !unnamed.is_empty() => eprintln!(
            "note: ligand topology for {unnamed:?} predates atom-name recording; \
             its atom order is assumed to match this file"),
        Ok(_) => {}
        // Display, not Debug: these errors exist to explain what to do next, and
        // `Box<dyn Error>` from main prints the Debug form.
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
    eprintln!("loading {weights_path} ...");
    let w = Weights::open(&weights_path)?;
    // Accept either the official `RFD_173.pt` or a converted safetensors export.
    // The .pt holds BOTH state dicts (7208 EMA + 7208 final, 14419 names), so its
    // keys are prefixed; `inference.state_dict_to_load` is `model_state_dict`,
    // the EMA weights, which is what the reference loads.
    let root = if w.has("model_state_dict.model.latent_emb.emb.weight") {
        "model_state_dict.model"
    } else {
        "model"
    };
    let model = RoseTTAFold::load(&Params::root(&w, root), Arch::rfd173());

    // Only the last row of the 1000x1000 CDF is reachable: the interpolant
    // hard-codes sigma = 1.5 and `bucketize` puts that at index 999.
    let ig = Weights::open(&igso3_path)?;
    let omega = ig.get("igso3.omega_grid").data;
    let cdf = ig.get("igso3.cdf");
    let n = cdf.shape[1];
    let igso3 = Igso3::new(omega, cdf.data[(cdf.shape[0] - 1) * n..].to_vec());

    let cfg = DesignConfig {
        input_pdb: input_pdb.clone(),
        ligands,
        contigs,
        big_t,
        final_step,
        seed_offset,
        deterministic: true,
        rots_exp_rate: 10,
        str_self_cond: args.iter().any(|a| a == "--str-self-cond"),
        partial_t: arg(&args, "--partial-t").map(|s| s.parse()).transpose()?,
        length: arg(&args, "--length"),
    };

    if let Some(dir) = std::path::Path::new(&prefix).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    for i in 0..num_designs {
        eprintln!("design {i} of {num_designs}: T = {big_t}");
        let t0 = std::time::Instant::now();
        let out = run_design(&model, &cfg, &pdb_text, &topo, &igso3, i)?;
        let path = format!("{prefix}_{i}-atomized-bb-False.pdb");
        std::fs::write(&path, &out.pdb)?;
        eprintln!("  wrote {path}  ({:.1} s)", t0.elapsed().as_secs_f32());
    }
    Ok(())
}
