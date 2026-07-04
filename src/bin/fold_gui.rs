//! ESMFold (pure-Rust fp32) — minimal local-web GUI.
//!
//! Double-click the exe: it starts a local server, opens your browser to a clean
//! page where you paste a protein sequence and click Fold. On first use it
//! auto-downloads the model weights (via the OS `curl`). Progress (per-layer) is
//! streamed to the page; the result is written to a PDB you can download.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use esmfold::constants::Constants;
use esmfold::pdb::to_pdb;
use esmfold::pipeline::fold_cb;
use esmfold::weights::Weights;
use tiny_http::{Header, Method, Response, Server};

const WEIGHTS_URL: &str = "https://huggingface.co/facebook/esmfold_v1/resolve/main/pytorch_model.bin";
const WEIGHTS_BYTES: u64 = 8_442_062_570; // for the download progress bar
const VALID_AA: &str = "ACDEFGHIKLMNPQRSTVWYXBUZOacdefghiklmnpqrstvwyxbuzo";
const MAX_LEN: usize = 500;

#[derive(Default, Clone)]
struct State {
    busy: bool,
    phase: String,
    progress: f32,
    log: Vec<String>,
    error: Option<String>,
    done: bool,
    plddt: f32,
    ptm: f32,
    pdb_name: Option<String>,
    model_path: String,
}

fn home() -> PathBuf {
    std::env::var("ESMFOLD_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let base = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| ".".into());
            PathBuf::from(base).join(".esmfold")
        })
}

fn weights_path() -> PathBuf {
    if let Ok(p) = std::env::var("ESMFOLD_WEIGHTS") {
        return PathBuf::from(p);
    }
    home().join("pytorch_model.bin")
}

fn set<F: FnOnce(&mut State)>(st: &Arc<Mutex<State>>, f: F) {
    f(&mut st.lock().unwrap());
}
// keep the full log for one fold (~240 messages) so the window can scroll back to the start
const LOG_CAP: usize = 5000;
fn logmsg(st: &Arc<Mutex<State>>, m: &str) {
    let mut s = st.lock().unwrap();
    s.log.push(m.to_string());
    if s.log.len() > LOG_CAP {
        s.log.remove(0);
    }
}

/// Download the weights with `curl`, updating progress by polling the file size.
fn download_weights(st: &Arc<Mutex<State>>, dest: &PathBuf) -> Result<(), String> {
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = dest.with_extension("part");
    let _ = std::fs::remove_file(&part);
    logmsg(st, "Model weights not found — downloading (~8.4 GB, one time)…");
    set(st, |s| { s.phase = "Downloading model weights…".into(); s.progress = 0.0; });

    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut child = Command::new(curl)
        .args(["-L", "--fail", "--silent", "--show-error", "-o"])
        .arg(&part)
        .arg(WEIGHTS_URL)
        .spawn()
        .map_err(|e| format!("could not start curl ({e}). curl ships with Windows 10+/macOS/Linux."))?;

    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                if !status.success() {
                    return Err(format!("download failed (curl exit {:?}). Check your internet connection.", status.code()));
                }
                break;
            }
            None => {
                if let Ok(md) = std::fs::metadata(&part) {
                    let frac = (md.len() as f32 / WEIGHTS_BYTES as f32).min(0.999);
                    set(st, |s| { s.progress = frac; s.phase = format!("Downloading model weights… {:.1} / 8.4 GB", md.len() as f32 / 1e9); });
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    logmsg(st, "Download complete.");
    Ok(())
}

fn sanitize(seq: &str) -> Result<String, String> {
    let s: String = seq.chars().filter(|c| c.is_ascii_alphabetic()).map(|c| c.to_ascii_uppercase()).collect();
    if s.is_empty() {
        return Err("Please enter a protein sequence (one-letter amino-acid codes).".into());
    }
    if s.len() > MAX_LEN {
        return Err(format!("Sequence is {} aa; this build caps at {} aa for reasonable CPU runtime.", s.len(), MAX_LEN));
    }
    if let Some(bad) = s.chars().find(|c| !VALID_AA.contains(*c)) {
        return Err(format!("Invalid amino-acid letter '{bad}'."));
    }
    Ok(s)
}

fn run_fold(st: Arc<Mutex<State>>, seq: String) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        let seq = sanitize(&seq)?;
        logmsg(&st, &format!("Folding sequence of length {}", seq.len()));
        let wp = weights_path();
        if !wp.exists() {
            download_weights(&st, &wp)?;
        }
        set(&st, |s| { s.phase = "Loading model weights…".into(); s.model_path = wp.display().to_string(); s.progress = 0.0; });
        logmsg(&st, &format!("Model weights: {}", wp.display()));
        logmsg(&st, "Loading weights (memory-mapped)…");
        let w = Weights::open(wp.to_str().unwrap()).map_err(|e| format!("could not open weights: {e}"))?;
        let consts = Constants::embedded();

        let stc = st.clone();
        let mut last = String::new();
        let out = fold_cb(&w, &consts, &seq, &mut |msg, frac| {
            let mut s = stc.lock().unwrap();
            s.phase = msg.to_string();
            s.progress = frac;
            // log only when the phase label changes, to keep the log readable
            if msg != last {
                s.log.push(msg.to_string());
                if s.log.len() > LOG_CAP { s.log.remove(0); }
            }
            drop(s);
            last = msg.to_string();
        });

        let plddt = out.plddt.data.iter().sum::<f32>() / out.plddt.data.len() as f32;
        let pdb = to_pdb(&out.atom37.data, &out.plddt.data, &out.aatype, &consts, out.l);
        let outdir = home().join("output");
        std::fs::create_dir_all(&outdir).map_err(|e| e.to_string())?;
        let name = "prediction.pdb";
        std::fs::write(outdir.join(name), pdb).map_err(|e| e.to_string())?;
        set(&st, |s| { s.plddt = plddt; s.ptm = out.ptm; s.pdb_name = Some(name.into()); });
        logmsg(&st, &format!("Done. mean pLDDT {:.3}, pTM {:.3}", plddt, out.ptm));
        Ok(())
    }));

    let mut s = st.lock().unwrap();
    s.busy = false;
    match result {
        Ok(Ok(())) => { s.done = true; s.progress = 1.0; s.phase = "Complete".into(); }
        Ok(Err(e)) => { s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
        Err(_) => { let e = "internal error during folding (panic)".to_string(); s.error = Some(e.clone()); s.phase = "Error".into(); s.log.push(format!("ERROR: {e}")); }
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn status_json(s: &State) -> String {
    // send the full log so the (scrollable) window can scroll back to the very first message
    let log = s.log.iter().map(|l| jstr(l)).collect::<Vec<_>>().join(",");
    format!(
        "{{\"busy\":{},\"phase\":{},\"progress\":{:.4},\"done\":{},\"plddt\":{:.4},\"ptm\":{:.4},\"error\":{},\"pdb\":{},\"model\":{},\"log\":[{}]}}",
        s.busy, jstr(&s.phase), s.progress, s.done, s.plddt, s.ptm,
        s.error.as_deref().map(jstr).unwrap_or_else(|| "null".into()),
        s.pdb_name.as_deref().map(jstr).unwrap_or_else(|| "null".into()),
        jstr(&s.model_path),
        log
    )
}

fn main() {
    let state = Arc::new(Mutex::new(State::default()));
    set(&state, |s| s.phase = "Idle".into());

    let server = Server::http("127.0.0.1:0").expect("could not start local server");
    let addr = server.server_addr();
    let url = format!("http://{}/", addr);
    println!("ESMFold GUI running at {url}");
    // open the browser
    let _ = if cfg!(windows) {
        Command::new("cmd").args(["/C", "start", "", &url]).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(&url).spawn()
    } else {
        Command::new("xdg-open").arg(&url).spawn()
    };

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let method = req.method().clone();
        if url == "/" {
            let html = INDEX_HTML.replace("__EXAMPLE__", EXAMPLE_SEQ);
            let r = Response::from_string(html).with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap());
            let _ = req.respond(r);
        } else if url == "/api/status" {
            let body = status_json(&state.lock().unwrap());
            let r = Response::from_string(body).with_header(Header::from_bytes("Content-Type", "application/json").unwrap());
            let _ = req.respond(r);
        } else if url == "/api/fold" && method == Method::Post {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let mut s = state.lock().unwrap();
            if s.busy {
                drop(s);
                let _ = req.respond(Response::from_string("{\"ok\":false,\"error\":\"already running\"}"));
            } else {
                *s = State { busy: true, phase: "Starting…".into(), ..Default::default() };
                drop(s);
                let st = state.clone();
                std::thread::spawn(move || run_fold(st, body));
                let _ = req.respond(Response::from_string("{\"ok\":true}"));
            }
        } else if url == "/api/pdb" {
            let path = home().join("output").join("prediction.pdb");
            match std::fs::read(&path) {
                Ok(data) => {
                    let r = Response::from_data(data)
                        .with_header(Header::from_bytes("Content-Type", "chemical/x-pdb").unwrap())
                        .with_header(Header::from_bytes("Content-Disposition", "attachment; filename=\"prediction.pdb\"").unwrap());
                    let _ = req.respond(r);
                }
                Err(_) => { let _ = req.respond(Response::from_string("not found").with_status_code(404)); }
            }
        } else {
            let _ = req.respond(Response::from_string("not found").with_status_code(404));
        }
    }
}

const EXAMPLE_SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESMFold — pure-Rust fp32</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--muted:#94a3b8;--accent:#e67e22;--accent2:#2e86c1;--ok:#22c55e;--err:#ef4444}
*{box-sizing:border-box} body{margin:0;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{max-width:820px;margin:0 auto;padding:32px 20px}
.head{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:22px;margin:0} .sub{color:var(--muted);font-size:13px;margin:6px 0 22px}
.links{display:flex;gap:8px}
a.pill{display:inline-block;text-decoration:none;font-size:12px;font-weight:600;color:#dbeafe;background:#1d3a5f;
border:1px solid #2e5a8a;border-radius:999px;padding:5px 12px} a.pill:hover{background:#244b78}
.card{background:var(--card);border:1px solid #334155;border-radius:12px;padding:18px;margin-bottom:16px}
label{font-size:13px;color:var(--muted)} .lab{display:flex;justify-content:space-between;align-items:center}
.example{font-size:12px;color:#7dd3fc;cursor:pointer;text-decoration:underline}
textarea{width:100%;height:120px;margin-top:6px;background:#0b1220;color:var(--ink);
border:1px solid #334155;border-radius:8px;padding:10px;font-family:ui-monospace,Consolas,monospace;font-size:13px;resize:vertical}
.row{display:flex;gap:10px;align-items:center;margin-top:12px}
button{background:var(--accent);color:#111;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer;font-size:14px}
button:disabled{opacity:.5;cursor:default}
.muted{color:var(--muted);font-size:12px}
.model{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--muted);margin:6px 0 2px;word-break:break-all}
.barwrap{height:10px;background:#0b1220;border-radius:6px;overflow:hidden;margin:10px 0}
.bar{height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .3s}
.phase{font-size:14px;margin-bottom:2px}
.log{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--muted);white-space:pre-wrap;
height:230px;overflow-y:scroll;background:#0b1220;border:1px solid #334155;border-radius:8px;padding:10px;margin-top:10px}
.err{color:var(--err);font-size:13px;margin-top:8px}
.ok{color:var(--ok)} a.btn{display:inline-block;background:var(--ok);color:#04210f;text-decoration:none;border-radius:8px;padding:9px 16px;font-weight:600}
.stats{display:flex;gap:24px;margin-top:8px;font-size:14px}
</style></head><body><div class="wrap">
<div class="head">
  <h1>ESMFold <span class="muted">— pure-Rust fp32</span></h1>
  <div class="links">
    <a class="pill" href="https://github.com/lingxusb/folding-everywhere" target="_blank" rel="noopener">Project ↗</a>
    <a class="pill" href="https://lingxusb.github.io" target="_blank" rel="noopener">Author ↗</a>
  </div>
</div>
<div class="sub">Folds a protein sequence to 3D all-atom coordinates. No Python, no GPU. Runs entirely on your CPU.</div>

<div class="card">
  <div class="lab"><label for="seq">Protein sequence (one-letter codes)</label>
    <span class="example" id="ex">Load example (ubiquitin, 76 aa)</span></div>
  <textarea id="seq" placeholder="Paste a protein sequence, or click &quot;Load example&quot; above."></textarea>
  <div class="row">
    <button id="go">Fold protein</button>
    <span class="muted" id="hint">First run downloads the model (~8.4 GB, one time).</span>
  </div>
</div>

<div class="card" id="progcard" style="display:none">
  <div class="phase" id="phase">Starting…</div>
  <div class="barwrap"><div class="bar" id="bar"></div></div>
  <div class="muted" id="pct">0%</div>
  <div class="model" id="model"></div>
  <div class="log" id="log"></div>
  <div class="err" id="err"></div>
</div>

<div class="card" id="result" style="display:none">
  <div class="ok" style="font-weight:600;margin-bottom:6px">✓ Prediction complete</div>
  <div class="stats"><div>mean pLDDT: <b id="plddt">–</b></div><div>pTM: <b id="ptm">–</b></div></div>
  <div class="row"><a class="btn" id="dl" href="/api/pdb">Download PDB</a></div>
</div>

<script>
const $=id=>document.getElementById(id);
const EXAMPLE="__EXAMPLE__";
$('ex').onclick=()=>{$('seq').value=EXAMPLE;};
$('go').onclick=async()=>{
  const seq=$('seq').value.trim();
  if(!seq){alert('Enter a sequence, or click "Load example".');return;}
  $('go').disabled=true; $('result').style.display='none'; $('err').textContent=''; $('model').textContent='';
  $('progcard').style.display='block';
  await fetch('/api/fold',{method:'POST',body:seq});
  poll();
};
function poll(){
  fetch('/api/status').then(r=>r.json()).then(s=>{
    $('phase').textContent=s.phase;
    $('bar').style.width=(s.progress*100).toFixed(1)+'%';
    $('pct').textContent=(s.progress*100).toFixed(0)+'%';
    if(s.model) $('model').textContent='Model file: '+s.model;
    const el=$('log');
    const nearBottom=(el.scrollHeight-el.scrollTop-el.clientHeight)<40;
    el.textContent=s.log.join('\n');
    if(nearBottom) el.scrollTop=el.scrollHeight;   // auto-follow unless the user scrolled up
    if(s.error){$('err').textContent='Error: '+s.error; $('go').disabled=false; return;}
    if(s.done){
      $('plddt').textContent=s.plddt.toFixed(3); $('ptm').textContent=s.ptm.toFixed(3);
      $('result').style.display='block'; $('go').disabled=false; return;
    }
    setTimeout(poll,700);
  }).catch(()=>setTimeout(poll,1000));
}
</script>
</div></body></html>"#;
