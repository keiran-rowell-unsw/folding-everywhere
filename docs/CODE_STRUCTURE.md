# Code structure

A map of the repository. Each model also has its own, deeper structure document:
[`esmfold/docs/CODE_STRUCTURE.md`](../esmfold/docs/CODE_STRUCTURE.md),
[`proteinmpnn/docs/CODE_STRUCTURE.md`](../proteinmpnn/docs/CODE_STRUCTURE.md),
[`rfdiffusion2/docs/`](../rfdiffusion2/docs/).

```
folding-everywhere-v2/
├── gui/                     the app — one binary, three tabs
│   ├── src/main.rs          HTTP router, the global run lock, shared helpers
│   ├── src/esmfold.rs       ESMFold 1+2 job logic and weight download
│   ├── src/mpnn.rs          ProteinMPNN job logic
│   ├── src/rfd2.rs          RFdiffusion2 job logic and checkpoint download
│   ├── src/index.html       the whole page: CSS, markup, JS (no CDN assets)
│   └── data/                IGSO(3) tables, ligand library, the two example structures
├── esmfold/                 ESMFold 1 + 2 subtree      → esmfold/README.md
├── proteinmpnn/             ProteinMPNN subtree        → proteinmpnn/README.md
├── rfdiffusion2/            RFdiffusion2 subtree       → rfdiffusion2/README.md
├── docs/                    GUI.md · BUILD.md · this file
├── dist/                    prebuilt apps, one per platform
├── Cargo.toml               the workspace
└── build_all.sh             three-platform release build
```

## Why one subtree per model

Each model was ported in its own repository, and its tests and embedded data reach *upward*
out of the crate directory with relative paths:

- `proteinmpnn/mpnn/src/embedded.rs` — `include_bytes!("../../weights/v_48_020.pt")`
- `proteinmpnn/mpnn/tests/*.rs` — `{CARGO_MANIFEST_DIR}/../fixtures/…`
- `rfdiffusion2/rfd2/tests/*.rs` — `{CARGO_MANIFEST_DIR}/../fixtures/…`

ProteinMPNN and RFdiffusion2 both have `fixtures/ops/`, `fixtures/rng/` and
`fixtures/weights/`, so a single flat `fixtures/` directory would collide. Giving each model
the directory layout it expects means **not one line of ported model code had to change** to
merge the three repos — which matters, because every one of those lines is validated against
a PyTorch reference.

## The app

`gui/` is the only code written for v2. Its three model modules are the job logic of v1's
three single-model GUIs moved across unchanged — same
weight URLs, same curl flags, same featurisation, same progress callbacks — so a run through
a tab is the same computation the single-model app performed. What is new is only the
plumbing: state ownership, the `/api/<model>/*` route prefixes, and the run lock.

```
main()
 ├── bind 127.0.0.1 : first free port in 8710..8759, open the browser
 ├── build the page: index.html + the two example structures + the ligand index
 ├── ef : Arc<Mutex<esmfold::State>>      \
 ├── mp : Arc<Mutex<mpnn::State>>          }  independent per tab
 ├── rf : Arc<Mutex<rfd2::State>>         /
 ├── running : Arc<AtomicBool>               one job at a time, across all tabs
 └── for req in server.incoming_requests()
       ├── GET  /                          → the page
       ├── */api/esmfold/*                 → esmfold.rs
       ├── */api/mpnn/*                    → mpnn.rs
       └── */api/rfd2/*                    → rfd2.rs
```

A start request takes the run lock with a `compare_exchange`; if it is already held, the
request is refused and the tab shows why. Otherwise the job is spawned on a worker thread,
which releases the lock when it returns. HTTP handlers never block on model work — the page
polls the tab's status endpoint.

Each job runs inside `catch_unwind`, so a panic deep in a model surfaces as an error line
rather than a silently dead worker and a page stuck on "Loading model…".

### The page

`gui/src/index.html` is served as one self-contained string. The stylesheet is the slate
theme the ESMFold and RFdiffusion2 GUIs already shared (`--bg:#0f172a`, `--card:#1e293b`,
`--accent:#e67e22`); the ProteinMPNN panel was restyled into the same card/fieldset idiom.
Element IDs are namespaced `ef_` / `mp_` / `rf_`, because all three original pages used
`#phase`, `#bar`, `#log`, `#err`, `#go`, `#pdb` and `#seed`.

Three template placeholders are filled at startup rather than at build time, so the examples
live next to the code that owns them: `__EXAMPLE_SEQ__` (ubiquitin), `__EXAMPLE_BB__`
(PDB 6EKB), `__EXAMPLE_PDB__` (1LDM motif) and `__LIBRARY__` (the ligand index).

## The model crates

Unchanged from their source repositories. In outline:

| Crate | Path | Shape |
|---|---|---|
| `esmfold1` (lib `esmfold`) | `esmfold/esmfold1/` | `tensor`, `ops/`, `pth` (ZIP64+pickle reader), `weights`, `tokenizer`, `esm2`, `trunk`, `structure`, `rigid`, `heads`, `pdb`, `pipeline` |
| `esmfold2-fp32` (lib `esmfold2`) | `esmfold/esmfold2/` | `tensor`, `ops/`, `weights`, `esmc`, `msa`, `parcae`, `trunk`, `diffusion`, `atom`, `confidence`, `rng`, `featurize`, `pdb`, `pipeline`, `standalone` |
| `proteinmpnn` | `proteinmpnn/mpnn/` | `tensor`, `ops/`, `pth`, `weights`, `embedded`, `pdb`, `featurize`, `features`, `layers`, `model`, `rng` |
| `rfd2` | `rfdiffusion2/rfd2/` | `tensor`, `ops/`, `pth`, `weights`, `chemical`, `openfold`, `ligand`, `prepro`, `featurize`, `sample_init`, `noiser`, `model/` (incl. `rf`, `se3`), `design` |

Their libraries are independent of each other and of the GUI: the app depends on all four,
but no model crate depends on another.
