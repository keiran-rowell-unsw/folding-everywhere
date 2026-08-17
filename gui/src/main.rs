//! Folding Everywhere v2 — one local-web GUI for three protein models.
//!
//! Double-click the executable: it binds a loopback port, opens your browser and
//! serves a one-page app with three tabs.
//!
//!   ESMFold        sequence  ->  3D structure          (ESMFold1 or ESMFold2)
//!   ProteinMPNN    backbone  ->  sequences
//!   RFdiffusion2   motif+ligand -> designed backbone
//!
//! All three are pure-Rust fp32 CPU ports; no Python, no PyTorch, no GPU, nothing
//! to install. Each tab's job logic lives in its own module and is lifted
//! unchanged from that model's standalone GUI, so results are identical to the
//! single-model apps this replaces.
//!
//! Design notes:
//! * `tiny_http` only; no async runtime, no JS/CSS from a CDN. The page is one
//!   self-contained string, so the executable works with no network access
//!   (except the one-time model-weight downloads ESMFold and RFdiffusion2 need).
//! * Each tab's job runs on a worker thread and reports progress through its own
//!   `Mutex<State>` that the page polls; the HTTP handler never blocks.
//! * A single global run lock (`RUNNING`) stops two tabs from saturating the CPU
//!   at once — these are all heavyweight models, and interleaving them would make
//!   both slower with no benefit.

mod esmfold;
mod mpnn;
mod rfd2;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tiny_http::{Header, Method, Response, Server};

/// `$HOME` (or `%USERPROFILE%`), the root of every model's weight cache.
pub fn home() -> PathBuf {
    let base = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(base)
}

/// Minimal JSON string escaping, for the hand-rolled status payloads.
pub fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"), c if (c as u32) < 0x20 => o.push(' '), c => o.push(c),
        }
    }
    o.push('"'); o
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap()
}
fn html_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap()
}
fn text_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"text/plain; charset=utf-8"[..]).unwrap()
}
fn pdb_headers(filename: &str) -> (Header, Header) {
    (
        Header::from_bytes(&b"Content-Type"[..], &b"chemical/x-pdb"[..]).unwrap(),
        Header::from_bytes(
            &b"Content-Disposition"[..],
            format!("attachment; filename=\"{filename}\"").as_bytes(),
        ).unwrap(),
    )
}

fn open_browser(url: &str) {
    let _ = if cfg!(target_os = "windows") {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
}

/// Which tab currently holds the run lock, for the "already running" message.
fn busy_msg(who: &str) -> String {
    format!("{who} is already running. These models are CPU-heavy, so only one job \
             runs at a time — wait for it to finish (or cancel it) and try again.")
}

fn main() {
    // Bind ONCE and keep it. Probing with a throwaway Server then re-binding
    // races the OS releasing the socket and fails with AddrInUse.
    let (server, port) = match (8710..8760)
        .find_map(|p| Server::http(("127.0.0.1", p)).ok().map(|s| (s, p)))
    {
        Some(x) => x,
        None => {
            eprintln!("could not bind a local port in 8710-8759");
            std::process::exit(1);
        }
    };
    let url = format!("http://127.0.0.1:{port}/");
    println!("Folding Everywhere v2 — ESMFold · ProteinMPNN · RFdiffusion2");
    println!("Running at {url}");
    println!("ProteinMPNN models embedded in this executable: {}", mpnn::model_names().join(", "));
    println!("Close this window to quit.");
    open_browser(&url);

    let page = INDEX_HTML
        .replace("__EXAMPLE_SEQ__", esmfold::EXAMPLE_SEQ)
        .replace("__EXAMPLE_PDB__", &serde_json::to_string(rfd2::EXAMPLE_PDB).unwrap())
        .replace("__EXAMPLE_BB__", &serde_json::to_string(mpnn::EXAMPLE_PDB).unwrap())
        .replace("__LIBRARY__", rfd2::LIBRARY_INDEX.trim());

    let ef = Arc::new(Mutex::new(esmfold::idle()));
    let mp = Arc::new(Mutex::new(mpnn::State::default()));
    let rf = Arc::new(Mutex::new(rfd2::idle()));
    // One job at a time across all three tabs.
    let running = Arc::new(AtomicBool::new(false));
    let mp_cancel = Arc::new(AtomicBool::new(false));

    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("/").to_string();
        let method = req.method().clone();
        match (method, path.as_str()) {
            // ---------------------------------------------------------- page
            (Method::Get, "/") => {
                let _ = req.respond(Response::from_string(page.clone()).with_header(html_header()));
            }

            // ------------------------------------------------------- ESMFold
            (Method::Get, "/api/esmfold/status") => {
                let body = esmfold::status_json(&ef.lock().unwrap());
                let _ = req.respond(Response::from_string(body).with_header(json_header()));
            }
            (Method::Post, "/api/esmfold/fold") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                // body = "model\nseed\nloops\nsteps\nseq"  (loops/steps: ESMFold2 only)
                let job = esmfold::parse_job(&body);
                if running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
                    *ef.lock().unwrap() = esmfold::starting();
                    esmfold::refuse(&ef, &busy_msg("Another model"));
                    let _ = req.respond(Response::from_string("{\"ok\":false,\"error\":\"busy\"}").with_header(json_header()));
                    continue;
                }
                *ef.lock().unwrap() = esmfold::starting();
                let (st, run) = (ef.clone(), running.clone());
                std::thread::spawn(move || {
                    esmfold::run_fold(st, job);
                    run.store(false, Ordering::SeqCst);
                });
                let _ = req.respond(Response::from_string("{\"ok\":true}").with_header(json_header()));
            }
            (Method::Get, "/api/esmfold/pdb") => {
                match std::fs::read(esmfold::out_pdb()) {
                    Ok(data) => {
                        let (h, d) = pdb_headers("prediction.pdb");
                        let _ = req.respond(Response::from_data(data).with_header(h).with_header(d));
                    }
                    Err(_) => { let _ = req.respond(Response::from_string("not found").with_status_code(404)); }
                }
            }

            // --------------------------------------------------- ProteinMPNN
            (Method::Get, "/api/mpnn/status") => {
                let body = mpnn::state_json(&mp.lock().unwrap());
                let _ = req.respond(Response::from_string(body).with_header(json_header()));
            }
            (Method::Post, "/api/mpnn/cancel") => {
                mp_cancel.store(true, Ordering::Relaxed);
                let _ = req.respond(Response::from_string("{}").with_header(json_header()));
            }
            (Method::Post, "/api/mpnn/run") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let job = mpnn::parse_job(&v);
                if !mpnn::job_has_pdb(&job) {
                    let _ = req.respond(Response::from_string(r#"{"error":"no PDB provided"}"#).with_header(json_header()));
                    continue;
                }
                if running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
                    *mp.lock().unwrap() = mpnn::starting();
                    mpnn::refuse(&mp, &busy_msg("Another model"));
                    let _ = req.respond(Response::from_string(r#"{"error":"busy"}"#).with_header(json_header()));
                    continue;
                }
                mp_cancel.store(false, Ordering::Relaxed);
                *mp.lock().unwrap() = mpnn::starting();
                let (st, c, run) = (mp.clone(), mp_cancel.clone(), running.clone());
                std::thread::spawn(move || {
                    mpnn::run_job(st, job, c);
                    run.store(false, Ordering::SeqCst);
                });
                let _ = req.respond(Response::from_string("{}").with_header(json_header()));
            }
            (Method::Get, "/api/mpnn/fasta") => {
                let body = mpnn::fasta_of(&mp.lock().unwrap());
                let d = Header::from_bytes(
                    &b"Content-Disposition"[..],
                    &b"attachment; filename=\"designs.fasta\""[..],
                ).unwrap();
                let _ = req.respond(Response::from_string(body).with_header(text_header()).with_header(d));
            }

            // -------------------------------------------------- RFdiffusion2
            (Method::Get, "/api/rfd2/status") => {
                let from = req.url().split("from=").nth(1)
                    .and_then(|s| s.split('&').next()).and_then(|s| s.parse().ok()).unwrap_or(0);
                let body = rfd2::json(&rf.lock().unwrap(), from);
                let _ = req.respond(Response::from_string(body).with_header(json_header()));
            }
            (Method::Post, "/api/rfd2/design") => {
                let mut body = String::new();
                let _ = req.as_reader().read_to_string(&mut body);
                let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let job = rfd2::parse_job(&v);
                if running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
                    *rf.lock().unwrap() = rfd2::starting();
                    rfd2::refuse(&rf, &busy_msg("Another model"));
                    let _ = req.respond(Response::from_string("{\"ok\":false,\"error\":\"busy\"}").with_header(json_header()));
                    continue;
                }
                *rf.lock().unwrap() = rfd2::starting();
                let (st, run) = (rf.clone(), running.clone());
                std::thread::spawn(move || {
                    rfd2::run_guarded(st, job);
                    run.store(false, Ordering::SeqCst);
                });
                let _ = req.respond(Response::from_string("{\"ok\":true}").with_header(json_header()));
            }
            (Method::Get, "/api/rfd2/pdb") => {
                // 1-based on the wire, matching the filenames the user sees
                let i = req.url().rsplit('=').next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
                match std::fs::read(rfd2::out_dir().join(format!("design_{i}.pdb"))) {
                    Ok(b) => {
                        let (h, d) = pdb_headers(&format!("design_{i}.pdb"));
                        let _ = req.respond(Response::from_data(b).with_header(h).with_header(d));
                    }
                    Err(e) => { let _ = req.respond(Response::from_string(e.to_string()).with_status_code(404)); }
                }
            }

            _ => { let _ = req.respond(Response::from_string("not found").with_status_code(404)); }
        }
    }
}

const INDEX_HTML: &str = include_str!("index.html");
