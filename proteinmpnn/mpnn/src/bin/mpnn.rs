//! `mpnn` — the ProteinMPNN CLI.
//!
//! Mirrors `protein_mpnn_run.py`'s flags, RNG consumption order and FASTA output
//! so results can be diffed directly against the reference implementation.
//!
//!   mpnn --pdb 5L33.pdb --num_seq_per_target 8 --sampling_temp 0.1 --seed 37

use std::io::Write;
use std::time::Instant;

use proteinmpnn::featurize::featurize;
use proteinmpnn::model::{score, ProteinMpnn};
use proteinmpnn::pdb::parse_pdb;
use proteinmpnn::rng::{randn, Mt19937};
use proteinmpnn::weights::Weights;
use proteinmpnn::{embedded, ALPHABET};

struct Args {
    pdb: String,
    weights: Option<String>,
    model_name: String,
    out: Option<String>,
    num_seq: usize,
    temps: Vec<f64>,
    seed: u64,
    chains: Option<Vec<char>>,
    omit_aas: String,
    score_only: bool,
    quiet: bool,
    dump: Option<String>,
}

fn usage() -> ! {
    eprintln!(
        r#"usage: mpnn --pdb FILE [options]

  --pdb FILE                 input backbone (required)
  --weights FILE             path to a ProteinMPNN .pt checkpoint
  --model_name NAME          v_48_002 | v_48_010 | v_48_020 | v_48_030 (default v_48_020)
  --out FILE                 write FASTA here (default: stdout)
  --num_seq_per_target N     sequences to sample (default 1)
  --sampling_temp "T [T..]"  sampling temperature(s) (default 0.1)
  --seed N                   RNG seed (default 37; matches torch.manual_seed)
  --pdb_path_chains "A B"    chains to design (default: all)
  --omit_AAs STR             amino acids to forbid (default X)
  --score_only               only score the native sequence
  --dump FILE                write raw fp32 log-probs for benchmarking
  --quiet"#
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        pdb: String::new(),
        weights: None,
        model_name: "v_48_020".into(),
        out: None,
        num_seq: 1,
        temps: vec![0.1],
        seed: 37,
        chains: None,
        omit_aas: "X".into(),
        score_only: false,
        quiet: false,
        dump: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].clone();
        let mut val = || {
            i += 1;
            argv.get(i).cloned().unwrap_or_else(|| usage())
        };
        match key.as_str() {
            "--pdb" | "--pdb_path" => a.pdb = val(),
            "--weights" => a.weights = Some(val()),
            "--model_name" => a.model_name = val(),
            "--out" => a.out = Some(val()),
            "--num_seq_per_target" => a.num_seq = val().parse().unwrap_or_else(|_| usage()),
            "--sampling_temp" => {
                a.temps = val()
                    .split_whitespace()
                    .map(|s| s.parse().unwrap_or_else(|_| usage()))
                    .collect()
            }
            "--seed" => a.seed = val().parse().unwrap_or_else(|_| usage()),
            "--pdb_path_chains" => {
                a.chains =
                    Some(val().split_whitespace().filter_map(|s| s.chars().next()).collect())
            }
            "--omit_AAs" => a.omit_aas = val(),
            "--score_only" => a.score_only = true,
            "--dump" => a.dump = Some(val()),
            "--quiet" => a.quiet = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage()
            }
        }
        i += 1;
    }
    if a.pdb.is_empty() {
        usage()
    }
    a
}

/// Resolve the checkpoint: an explicit `--weights` path wins, otherwise use the
/// copy embedded in this binary, otherwise look on disk (for checkpoints we do
/// not ship, e.g. the soluble or CA-only variants).
fn load_weights(a: &Args) -> Weights {
    if let Some(p) = &a.weights {
        return Weights::open(p).unwrap_or_else(|e| {
            eprintln!("failed to open {p}: {e}");
            std::process::exit(1)
        });
    }
    if let Some(bytes) = embedded::by_name(&a.model_name) {
        return Weights::from_static(bytes).expect("embedded checkpoint is corrupt");
    }
    let file = format!("{}.pt", a.model_name);
    for rel in [
        "weights",
        "../ref_ProteinMPNN/vanilla_model_weights",
        "../../ref_ProteinMPNN/vanilla_model_weights",
    ] {
        let p = std::path::Path::new(rel).join(&file);
        if p.exists() {
            return Weights::open(&p.to_string_lossy()).expect("open checkpoint");
        }
    }
    eprintln!(
        "unknown model {:?}. Embedded models: {}. Or pass --weights /path/to/{file}",
        a.model_name,
        embedded::names().join(", ")
    );
    std::process::exit(1)
}

fn main() {
    let a = parse_args();
    let t_load = Instant::now();

    let st = parse_pdb(&a.pdb).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", a.pdb);
        std::process::exit(1)
    });
    let all: Vec<char> = st.chain_ids();
    if all.is_empty() {
        eprintln!("no protein chains found in {}", a.pdb);
        std::process::exit(1);
    }
    let designed: Vec<char> = a.chains.clone().unwrap_or_else(|| all.clone());
    let fixed: Vec<char> = all.iter().copied().filter(|c| !designed.contains(c)).collect();
    let b = featurize(&st, &designed, &fixed);

    let w = load_weights(&a);
    let model = ProteinMpnn::load(&w, 48, 3, 3);
    let load_s = t_load.elapsed().as_secs_f64();

    if !a.quiet {
        eprintln!(
            "{}: L={} chains={:?} designed={:?} model={} (load {:.2}s)",
            st.name,
            b.l,
            all,
            designed,
            a.weights.as_deref().unwrap_or(&a.model_name),
            load_s
        );
    }

    let mut omit = [0.0f64; 21];
    for c in a.omit_aas.bytes() {
        if let Some(i) = ALPHABET.bytes().position(|x| x == c) {
            omit[i] = 1.0;
        }
    }
    let bias = [0.0f64; 21];

    // Reproduce protein_mpnn_run.py's RNG consumption order exactly:
    //   manual_seed(seed) -> ProteinMPNN.__init__ burns ~3.3M draws initialising
    //   weights that load_state_dict then overwrites -> randn_1 (native scoring)
    //   -> per sequence randn_2 -> 21 exponentials per decoded residue.
    // `protein_mpnn_run.py` does `if args.seed: seed = args.seed else: <random>`,
    // and 0 is falsy in Python — so `--seed 0` means "pick a random seed", not
    // "seed with zero". Matched here so the flag means the same thing; use any
    // non-zero seed for reproducible output.
    let seed = if a.seed == 0 { random_seed() } else { a.seed };
    let mut gen = Mt19937::new(seed);
    let burn = proteinmpnn::model::torch_init_draws(&w);
    gen.skip(burn);
    let t0 = Instant::now();
    // The encoder sees only the backbone, so it is computed once and shared by
    // the native scoring pass, every sampling pass and every rescoring pass.
    let enc = model.encode(&b);
    let r1 = randn(&mut gen, b.l);
    let order1 = ProteinMpnn::decoding_order(&r1, &b.design_mask());
    let lp_native = model.forward_with(&enc, &b, &b.s, &order1);
    // The reference reports two numbers: `score` over designed positions only
    // (mask * chain_M * chain_M_pos) and `global_score` over every resolved
    // residue. They coincide for a fully designed monomer and differ as soon as
    // any chain is held fixed.
    let mask_for_loss = b.design_mask();
    let native_score = score(&b.s, &lp_native, &mask_for_loss);
    let native_global = score(&b.s, &lp_native, &b.mask);

    let mut out: Box<dyn Write> = match &a.out {
        Some(p) => Box::new(std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("cannot write {p}: {e}");
            std::process::exit(1)
        })),
        None => Box::new(std::io::stdout()),
    };

    // Python list repr, so the header diffs cleanly against the reference.
    let pylist = |v: &[char]| -> String {
        format!("[{}]", v.iter().map(|c| format!("'{c}'")).collect::<Vec<_>>().join(", "))
    };
    let fixed_s = pylist(&fixed);
    let designed_s = pylist(&designed);
    writeln!(
        out,
        ">{}, score={native_score:.4}, global_score={native_global:.4}, \
         fixed_chains={fixed_s}, designed_chains={designed_s}, \
         model_name={}, seed={seed}",
        st.name, a.model_name
    )
    .unwrap();
    writeln!(out, "{}", b.format_seq(&b.s)).unwrap();

    if a.score_only {
        if let Some(p) = &a.dump {
            // Raw fp32 log-probs [L,21] for the native sequence — used by
            // python/compare_logprobs.py to measure continuous agreement at
            // full precision rather than through the 4-decimal FASTA header.
            let bytes: Vec<u8> = lp_native.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(p, bytes).unwrap();
        }
        if !a.quiet {
            eprintln!("native score {native_score:.4} in {:.3}s", t0.elapsed().as_secs_f64());
        }
        return;
    }

    let mut dump_buf: Vec<f32> = Vec::new();
    let mut sample_no = 0usize;
    for &temp in &a.temps {
        for _ in 0..a.num_seq {
            // batch_size 1: one randn draw per sampled sequence
            let r2 = randn(&mut gen, b.l);
            let order2 = ProteinMpnn::decoding_order(&r2, &b.design_mask());
            let s = model.sample_with(&enc, &b, &mut gen, &order2, temp, &omit, &bias);
            let lp = model.forward_with(&enc, &b, &s.s, &order2);
            let sc = score(&s.s, &lp, &mask_for_loss);
            let gsc = score(&s.s, &lp, &b.mask);
            let seq = b.format_seq(&s.s);
            // Recovery is measured over the designed positions only.
            let denom: f64 = mask_for_loss.iter().map(|&m| m as f64).sum::<f64>().max(1.0);
            let rec: f64 = (0..b.l)
                .filter(|&i| b.s[i] == s.s[i])
                .map(|i| mask_for_loss[i] as f64)
                .sum();
            sample_no += 1;
            writeln!(
                out,
                ">T={temp}, sample={sample_no}, score={sc:.4}, global_score={gsc:.4}, seq_recovery={:.4}",
                rec / denom
            )
            .unwrap();
            writeln!(out, "{seq}").unwrap();
            if a.dump.is_some() {
                dump_buf.extend_from_slice(&lp.data);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();

    if let Some(p) = &a.dump {
        let bytes: Vec<u8> = dump_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).unwrap();
    }
    if !a.quiet {
        eprintln!(
            "{} sequences in {:.3}s ({:.1} ms/seq), peak RSS {:.0} MB",
            sample_no,
            dt,
            1000.0 * dt / sample_no.max(1) as f64,
            peak_rss_mb()
        );
    }
}

/// A non-reproducible seed, for `--seed 0` (see above). The reference draws
/// `np.random.randint(0, 999)` from numpy's default entropy; the exact
/// distribution does not matter since neither side is reproducible.
fn random_seed() -> u64 {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
        .unwrap_or(1);
    (n % 998) + 1
}

/// Peak resident set size in MB (Linux; 0 elsewhere).
fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(v) = line.strip_prefix("VmHWM:") {
                    if let Some(kb) = v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok())
                    {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    0.0
}
