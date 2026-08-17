//! `XYZConverter.compute_all_atom` — backbone frames + torsions -> all-atom
//! coordinates. Checked against the reference's own captured call.
//!
//! This is on the critical path twice: it produces the `xyzallatom` the sampler
//! writes, and the LJ gradient is back-propagated through it.

use rfd2::model::xyzconv::XyzConverter;
use rfd2::parity;
use rfd2::weights::Weights;
use std::path::Path;

fn open(rel: &str) -> Option<Weights> {
    let path = format!("{}/../{rel}", env!("CARGO_MANIFEST_DIR"));
    if !Path::new(&path).exists() {
        eprintln!("SKIP: {path} missing");
        return None;
    }
    Some(Weights::open(&path).expect("open"))
}

#[test]
fn compute_all_atom_matches() {
    let Some(io) = open("fixtures/refiner_io/io.safetensors") else { return };
    let conv = XyzConverter::new();
    for call in 0..2 {
        let key = format!("aa{call}");
        if !io.has(&format!("{key}.xyzaa")) {
            continue;
        }
        let seq = io.get_i64(&format!("{key}.seq")).0;
        let xyz = io.get(&format!("{key}.xyz")); // [1, L, n, 3]
        let alphas = io.get(&format!("{key}.alphas"));
        let l = xyz.shape[1];
        let n_in = xyz.shape[2];
        let (frames, got) = conv.compute_all_atom_with_frames(&seq, &xyz.data, n_in, &alphas.data);
        if io.has(&format!("{key}.frames")) {
            let wf = io.get(&format!("{key}.frames"));
            let sf = parity::compare(&frames, &wf.data);
            println!("  RTframes[{call}]: {}", sf.summary());
            let badf: Vec<usize> = frames
                .iter()
                .zip(&wf.data)
                .enumerate()
                .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
                .map(|(i, _)| i)
                .collect();
            if !badf.is_empty() {
                let rt: std::collections::BTreeSet<(usize, usize)> =
                    badf.iter().map(|i| ((i / 16) / 17, (i / 16) % 17)).collect();
                let prot: Vec<&(usize, usize)> = rt.iter().filter(|x| x.1 < 9).collect();
                println!(
                    "    frames 0..8 differ at (res, frame): {:?}  [{} NA-frame entries ignored]",
                    prot.iter().take(12).collect::<Vec<_>>(),
                    rt.len() - prot.len()
                );
            }
        }
        let want = io.get(&format!("{key}.xyzaa"));
        let s = parity::compare(&got, &want.data);
        println!("compute_all_atom[{call}] (L={l}, n_in={n_in}): {}", s.summary());
        let bad: Vec<usize> = got
            .iter()
            .zip(&want.data)
            .enumerate()
            .filter(|(_, (g, w))| g.to_bits() != w.to_bits() && !(g.is_nan() && w.is_nan()))
            .map(|(i, _)| i)
            .collect();
        if !bad.is_empty() {
            let ra: std::collections::BTreeSet<(usize, usize)> =
                bad.iter().map(|i| ((i / 3) / 36, (i / 3) % 36)).collect();
            let res: std::collections::BTreeSet<usize> = ra.iter().map(|x| x.0).collect();
            let atoms: std::collections::BTreeSet<usize> = ra.iter().map(|x| x.1).collect();
            println!("   {} values; residues {:?}; atom slots {:?}", bad.len(), res, atoms);
            let tok: Vec<i64> = res.iter().map(|&r| seq[r]).collect();
            println!("   tokens {:?}", tok);
        }
        // Which association does torch.einsum actually use for RTF4?
        if call == 0 && io.has(&format!("{key}.frames")) {
            let wf = io.get(&format!("{key}.frames"));
            let res = 10usize;
            let tok = seq[res] as usize;
            let conv2 = XyzConverter::new();
            let f8 = {
                let mut m = [[0.0f32; 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        m[r][c] = frames[((res * 17 + 8) * 4 + r) * 4 + c];
                    }
                }
                m
            };
            let ndof = alphas.data.len() / (l * 2);
            let a3 = (
                alphas.data[(res * ndof + 3) * 2],
                alphas.data[(res * ndof + 3) * 2 + 1],
            );
            let a9 = (
                alphas.data[(res * ndof + 9) * 2],
                alphas.data[(res * ndof + 9) * 2 + 1],
            );
            for w in 0..5 {
                let m = rfd2::model::xyzconv::mm4_4_assoc(
                    &f8,
                    &conv2.rt_pub(tok, 3),
                    &rfd2::model::xyzconv::make_rot_x_pub(a3.0, a3.1),
                    &rfd2::model::xyzconv::make_rot_z_pub(a9.0, a9.1),
                    w,
                );
                let mut ok = true;
                for r in 0..4 {
                    for c in 0..4 {
                        let g = m[r][c] as f32;
                        let want = wf.data[((res * 17 + 4) * 4 + r) * 4 + c];
                        if g.to_bits() != want.to_bits() {
                            ok = false;
                        }
                    }
                }
                println!(
                    "    RTF4 association {w}: {}  [0][0] = {:.15}",
                    if ok { "MATCH" } else { "differs" },
                    m[0][0] as f32
                );
            }
            {
                let rxm = rfd2::model::xyzconv::make_rot_x_pub(a3.0, a3.1);
                let rzm = rfd2::model::xyzconv::make_rot_z_pub(a9.0, a9.1);
                println!("    a3 = ({:e}, {:e})  a9 = ({:e}, {:e})", a3.0, a3.1, a9.0, a9.1);
                println!("    rotX bits = {:?}", rxm.iter().flatten().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>());
                println!("    rotZ bits = {:?}", rzm.iter().flatten().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>());
                let b3 = conv2.rt_pub(tok, 3);
                println!("    B3 bits = {:?}", b3.iter().flatten().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>());
                println!("    F8 bits = {:?}", f8.iter().flatten().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>());
                println!("    F8 row0 = {:?}", f8[0].iter().map(|v| format!("{:.9e}", v)).collect::<Vec<_>>());
                let wf8: Vec<String> = (0..4)
                    .map(|c| format!("{:.9e}", wf.data[((res * 17 + 8) * 4) * 4 + c]))
                    .collect();
                println!("    F8ref r0= {:?}", wf8);
                // local, dependency-free triple loop, to rule out the library
                let mul = |x: &[[f64; 4]; 4], y: &[[f64; 4]; 4]| {
                    let mut o = [[0.0f64; 4]; 4];
                    for i in 0..4 {
                        for k in 0..4 {
                            let mut a = 0.0f64;
                            for j in 0..4 {
                                a += x[i][j] * y[j][k];
                            }
                            o[i][k] = a;
                        }
                    }
                    o
                };
                let cv = |m: &[[f32; 4]; 4]| {
                    let mut o = [[0.0f64; 4]; 4];
                    for i in 0..4 {
                        for j in 0..4 {
                            o[i][j] = m[i][j] as f64;
                        }
                    }
                    o
                };
                let ab = mul(&cv(&f8), &cv(&b3));
                println!("    local F8@B3 row0 = {:?}", (0..4).map(|c| format!("{:.12e}", ab[0][c])).collect::<Vec<_>>());
                let abc = mul(&ab, &cv(&rxm));
                let abcd = mul(&abc, &cv(&rzm));
                println!("    local F4    row0 = {:?}", (0..4).map(|c| format!("{:.12e}", abcd[0][c])).collect::<Vec<_>>());
            }

            // Is the rotation factor's normalisation the difference?
            {
                let mk = |c: f32, s: f32, mode: usize, z: bool| -> [[f32; 4]; 4] {
                    let n = match mode {
                        0 => ((c as f64) * (c as f64) + (s as f64) * (s as f64)).sqrt() as f32,
                        1 => (c * c + s * s).sqrt(),
                        2 => (c as f64).hypot(s as f64) as f32,
                        _ => c.hypot(s),
                    } + 1e-6f32;
                    let mut m = [[0.0f32; 4]; 4];
                    for i in 0..4 {
                        m[i][i] = 1.0;
                    }
                    if z {
                        m[0][0] = c / n;
                        m[0][1] = -s / n;
                        m[1][0] = s / n;
                        m[1][1] = c / n;
                    } else {
                        m[1][1] = c / n;
                        m[1][2] = -s / n;
                        m[2][1] = s / n;
                        m[2][2] = c / n;
                    }
                    m
                };
                for mode in 0..4 {
                    let m = rfd2::model::xyzconv::mm4_4_assoc(
                        &f8,
                        &conv2.rt_pub(tok, 3),
                        &mk(a3.0, a3.1, mode, false),
                        &mk(a9.0, a9.1, mode, true),
                        0,
                    );
                    let mut nbad = 0;
                    for r in 0..4 {
                        for c in 0..4 {
                            if (m[r][c] as f32).to_bits()
                                != wf.data[((res * 17 + 4) * 4 + r) * 4 + c].to_bits()
                            {
                                nbad += 1;
                            }
                        }
                    }
                    let name = ["f64 naive", "f32 naive", "f64 hypot", "f32 hypot"][mode];
                    println!("    RTF4 norm={name}: {nbad}/16 differ");
                }
            }

            // where in the 4x4 does it differ, and by how much?
            for r in 0..4 {
                let mut line = String::new();
                for c in 0..4 {
                    let g = frames[((res * 17 + 4) * 4 + r) * 4 + c];
                    let want = wf.data[((res * 17 + 4) * 4 + r) * 4 + c];
                    line.push_str(&format!(
                        "{}{:+.9e}/{:+.9e} ",
                        if g.to_bits() == want.to_bits() { " " } else { "*" },
                        g,
                        want
                    ));
                }
                println!("      F4 row {r}: {line}");
            }
        }
        assert_eq!(got.len(), want.data.len());
        assert_eq!(s.exact, s.n, "compute_all_atom is not bit-exact");
    }
}
