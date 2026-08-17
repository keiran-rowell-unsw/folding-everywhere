//! Bisect the reverse pass through `compute_all_atom` against the reference's
//! own per-intermediate gradients (`python/dump_aa_bwd.py`).
//!
//! `tests/parity_lj_grads.rs` only checks the two leaves, so a disagreement
//! there says nothing about *where* it came from. This walks the same graph the
//! reference does — every einsum's output gradient, every rotation's matrix,
//! `NORM` and `angs` gradients — and prints one line per stage. Ordered from the
//! top of the graph down, so the first red line is the defect.

use rfd2::parity;
use rfd2::weights::Weights;
use rfd2::xyzconv_bwd::{self, Trace};
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

/// einsum index -> the frame it produces (`es0`/`es1` are inside
/// `rigid_from_3_points`, `es18` is the atom placement).
const ES_FRAME: [(usize, usize); 16] = [
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 8),
    (6, 4),
    (7, 5),
    (8, 6),
    (9, 7),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 12),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 16),
];

/// rotation-capture tag -> the alpha slot it belongs to.
const ROT_SLOT: [(&str, usize); 18] = [
    ("rotX0", 0),
    ("rotX1", 1),
    ("rotX2", 2),
    ("rotX3", 3),
    ("rotX4", 4),
    ("rotX5", 5),
    ("rotX6", 6),
    ("rotX7", 12),
    ("rotX8", 13),
    ("rotX9", 14),
    ("rotX10", 15),
    ("rotX11", 16),
    ("rotX12", 17),
    ("rotX13", 18),
    ("rotX14", 19),
    ("rotZ0", 9),
    ("rotA0", 7),
    ("rotA1", 8),
];

fn line(label: &str, got: &[f32], want: &[f32], bad: &mut Vec<String>) {
    let s = parity::compare(got, want);
    let mark = if s.exact == s.n { "OK  " } else { "FAIL" };
    println!("  {mark} {label:<22} {}", s.summary());
    if s.exact != s.n {
        bad.push(label.to_string());
    }
}

#[test]
fn aa_bwd_stages() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let Some(bw) = open("fixtures/refiner_io/aa_bwd.safetensors") else { return };
    let seq = io.get_i64("lj0.seq").0;
    let xyz = io.get("lj0.xyz");
    let alpha = io.get("lj0.alpha");
    let l = seq.len();

    let gout = bw.get("aa.gout").data;
    let mut tr = Trace::default();
    let g = xyzconv_bwd::backward_traced(&seq, &xyz.data, 3, &alpha.data, &gout, Some(&mut tr));

    let mut bad = Vec::new();

    println!("frame gradients (dL/dRTF{{t}}), from the producing einsum:");
    for (es, t) in ES_FRAME {
        let want = bw.get(&format!("es{es}.d_out"));
        // fixture is [1, L, 4, 4]; the trace is [L, 17, 4, 4]
        let mut got = vec![0.0f32; l * 16];
        for i in 0..l {
            got[i * 16..(i + 1) * 16]
                .copy_from_slice(&tr.dframe[(i * 17 + t) * 16..(i * 17 + t + 1) * 16]);
        }
        line(&format!("RTF{t} (es{es})"), &got, &want.data, &mut bad);
    }
    // RTF0 has no producing einsum of its own; it is `es2`'s first input.
    {
        let want = bw.get("es2.d_in0");
        let mut got = vec![0.0f32; l * 16];
        for i in 0..l {
            got[i * 16..(i + 1) * 16].copy_from_slice(&tr.dframe[(i * 17) * 16..(i * 17 + 1) * 16]);
        }
        line("RTF0 (es2.in0)", &got, &want.data, &mut bad);
    }

    println!("rotation matrices (dL/dR):");
    for (tag, slot) in ROT_SLOT {
        let want = bw.get(&format!("{tag}.d_out"));
        let mut got = vec![0.0f32; l * 16];
        for i in 0..l {
            got[i * 16..(i + 1) * 16]
                .copy_from_slice(&tr.drot[(i * 20 + slot) * 16..(i * 20 + slot + 1) * 16]);
        }
        line(&format!("{tag} -> a{slot}"), &got, &want.data, &mut bad);
    }

    println!("NORM gradients:");
    for (tag, slot) in ROT_SLOT {
        let want = bw.get(&format!("{tag}.d_norm"));
        let got: Vec<f32> = (0..l).map(|i| tr.dnorm[i * 20 + slot]).collect();
        line(&format!("{tag}.norm"), &got, &want.data, &mut bad);
    }

    println!("angs gradients (per rotation, before they meet in `alphas`):");
    for (tag, slot) in ROT_SLOT {
        let want = bw.get(&format!("{tag}.d_angs"));
        let ndof = alpha.data.len() / (l * 2);
        let got: Vec<f32> = (0..l)
            .flat_map(|i| {
                [
                    g.dalpha[(i * ndof + slot) * 2],
                    g.dalpha[(i * ndof + slot) * 2 + 1],
                ]
            })
            .collect();
        line(&format!("{tag}.angs"), &got, &want.data, &mut bad);
    }

    println!("inside rigid_from_3_points, top of the graph down:");
    line("rigid.d_R", &tr.drigid, &bw.get("rigid.d_R").data, &mut bad);
    let rt = &tr.rigid;
    let m3 = |f: &dyn Fn(&rfd2::geom::RigidTrace) -> [[f32; 3]; 3]| -> Vec<f32> {
        rt.iter().flat_map(|t| f(t).into_iter().flatten()).collect()
    };
    let v3 = |f: &dyn Fn(&rfd2::geom::RigidTrace) -> [f32; 3]| -> Vec<f32> {
        rt.iter().flat_map(|t| f(t)).collect()
    };
    let s1 = |f: &dyn Fn(&rfd2::geom::RigidTrace) -> f32| -> Vec<f32> {
        rt.iter().map(f).collect()
    };
    for (name, got) in [
        ("Rp", m3(&|t| t.d_rp)),
        ("Rc", m3(&|t| t.d_rc)),
        ("cosdel", s1(&|t| t.d_cosdel)),
        ("sindel", s1(&|t| t.d_sindel)),
        ("cos2del", s1(&|t| t.d_cos2del)),
        ("cosref", s1(&|t| t.d_cosref)),
        ("e3", v3(&|t| t.d_e3)),
        ("e2", v3(&|t| t.d_e2)),
        ("v2n", v3(&|t| t.d_v2n)),
        ("u2", v3(&|t| t.d_u2)),
        ("proj", s1(&|t| t.d_proj)),
        ("e1", v3(&|t| t.d_e1)),
        ("v1", v3(&|t| t.d_v1)),
        ("v2", v3(&|t| t.d_v2)),
    ] {
        line(&format!("rigid.d_{name}"), &got, &bw.get(&format!("rigid.d_{name}")).data, &mut bad);
    }
    println!("leaves:");
    line("out.dxyz", &g.dxyz, &bw.get("out.dxyz").data, &mut bad);
    line("out.dalpha", &g.dalpha, &bw.get("out.dalpha").data, &mut bad);

    println!("\n{} of {} stages disagree", bad.len(), 16 + 1 + 18 * 3 + 3);
    for b in &bad {
        println!("  - {b}");
    }
    assert!(bad.is_empty(), "stages not bit-exact: {bad:?}");
}
