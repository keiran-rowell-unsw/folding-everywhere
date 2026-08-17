# §1 Reconnaissance — RFdiffusion2

Everything here was read out of the pinned reference or measured, not recalled.
Anything not yet verified is marked **[unverified]** and must not be relied on.

Reference: `RosettaCommons/RFdiffusion2` @ `d365cbf4db3958814a9f8e4f6f94fa309dfebc2b`
(2026-04-03), at `../ref_RFdiffusion2`.
Weights: `RFD_173.pt`, sha256
`590e126057f780afc1249d29545f0f90635562b1a3df5aff013afcfc39c3d3c3`, 1 338 843 322 B.

---

## 1.1 The actual inference path

Training code is ignored. The path a user's design actually takes:

```
rf_diffusion/run_inference.py : main
  └─ get_sampler(conf)
       ├─ seed_all(0)                       # iff inference.deterministic
       └─ model_runners.sampler_selector(conf)
            └─ Sampler.initialize
                 ├─ load_model()
                 │    ├─ du.read_pkl(ckpt)                      -> weights + conf
                 │    ├─ conf = merge(config/training/base.yaml, ckpt.conf, cli conf)
                 │    ├─ noisers.get(conf.diffuser)             -> NormalizingFlow
                 │    │                                            (type=flow_matching)
                 │    ├─ RFScore(conf.rf.model, diffuser, device)
                 │    │    └─ rf2aa LegacyRoseTTAFoldModule
                 │    └─ load_state_dict(ckpt['model_state_dict'])   # EMA weights
                 └─ InferenceDataset(conf, diffuser)
  └─ sample(sampler)
       for i_des in [design_startnum, design_startnum+num_designs):
         ├─ seed_all(i_des + seed_offset)   # iff inference.deterministic
         ├─ sample_one(sampler, i_des)
         │    ├─ sampler.sample_init(i_des)          -> indep, contig_map, atomizer, t_step_input
         │    │     └─ InferenceDataset[i]  : parse PDB, sample contig, atomize,
         │    │                               place guideposts, noise to x_T
         │    ├─ features.init_tXd_inference(...)
         │    └─ for t in T .. final_step:
         │          sampler.sample_step(t, indep, rfo, extra, features_cache)
         │            ├─ features.get_extra_tXd_inference(...)   (RoG / RASA / t-embedding)
         │            ├─ aa_model.Model.prepro(indep, t, is_diffused)   -> RFI
         │            ├─ [str_self_cond] aa_model.self_cond(...)
         │            ├─ RFScore.forward_from_rfi(rfi, t/T)
         │            │     ├─ du.rigid_frames_from_atom_14(rfi.xyz)
         │            │     ├─ LegacyRoseTTAFoldModule(**rfi)     <- the 82.9 M-param network
         │            │     ├─ rigids_from_rfo(...)               quaternion composition
         │            │     ├─ psi_pred = torch.rand((B,I,L,2))   <-- RNG INSIDE THE FORWARD
         │            │     └─ all_atom.compute_backbone(rigids, psi_pred) -> atom37 / atom14
         │            ├─ px0 = model_out['atom37'][0,-1]
         │            └─ diffuser.reverse(...)  n_steps times     -> rigids_{t-1}
         └─ save_outputs(...)
              ├─ guidepost correspondence (greedy matching)
              ├─ idealize sidechains / backbone
              └─ write .pdb (+ .trb pickle)
```

Entry configs: `config/inference/base.yaml` <- `aa.yaml` (the open-source demo
uses `--config-name=aa` with `inference.deterministic=True`,
`inference.seed_offset=43`, `ckpt_path=.../RFD_173.pt`).

**Note the config precedence.** `Sampler.load_model` merges
`config/training/base.yaml` <- `ckpt['conf']` <- the CLI/inference conf. So the
*architecture* comes from the checkpoint, not from `config/inference/base.yaml`.
The inference yaml's `model:` block is largely overridden. The Rust port must
read the architecture from the checkpoint's embedded config, exactly as here.

Measured architecture (from `ckpt['conf'].rf.model`, see
`fixtures/weights/ckpt_conf.json`) — note where it differs from base.yaml:

| key | checkpoint | inference/base.yaml |
|---|---|---|
| `d_pair` | **192** | 128 |
| `n_head_pair` | **6** | 4 |
| `d_t1d` | **114** | (22 via preprocess) |
| `d_hidden_templ` | **64** | 32 |
| `freeze_track_motif` | **True** | False |
| `lj_lin` | **0.75** | 0.6 |
| `n_main_block` | 32 | 32 |
| `n_extra_block` | 4 | 4 |
| `n_ref_block` | 4 | 4 |
| `d_msa` / `d_msa_full` / `d_templ` | 256 / 64 / 64 | same |
| `recycling_type` | `all` | `all` |
| `refiner_topk` | 128 | 128 |

Diffuser (from checkpoint): `type=flow_matching`, `T=200` (the demo's `aa.yaml`
overrides to `T=100`), `trans.sample_schedule=linear`,
`rots.sample_schedule=exp` (`aa.yaml`: `normed_exp`, `exp_rate=10`).

`extra_tXd = ['radius_of_gyration_v2', 'relative_sasa_v2',
'sinusoidal_timestep_embedding']`; `transforms = ['AddConditionalInputs',
'CenterPostTransform']`.

---

## 1.2 Checkpoint inventory (SOP §1.5) — **done**

`python/inventory_checkpoint.py`. Full listings in
`fixtures/weights/inventory_*.json`.

```
RFD_173.pt top-level keys:
  model, epoch, model_state_dict, final_state_dict, optimizer_state_dict,
  scheduler_state_dict, scaler_state_dict, conf, wandb_run_id, wandb_group, rundir
```

**7 208 tensors, 82 911 693 parameters, every one `torch.float32`.**
That count is the first Rust-side test.

`model_state_dict` (EMA, the one `state_dict_to_load` selects) vs
`final_state_dict`: same keys, **only 570 of 7 208 tensors bit-identical** — so
picking the wrong one silently produces a different model. Pin it in a test.

Parameter distribution (all keys are under `model.`, because `RFScore` holds the
network as `self.model`):

| module | tensors | params | share |
|---|---:|---:|---:|
| `model.simulator.main_block.{0..31}` | 6 144 | 77 032 192 | 92.9 % |
| `model.simulator.extra_block.{0..3}` | 768 | 4 445 536 | 5.4 % |
| `model.simulator.str_refiner` (SE3 + sidechain) | 94 | 431 696 | 0.5 % |
| `model.templ_emb` | 164 | 749 632 | 0.9 % |
| `model.latent_emb` | 9 | 113 536 | |
| `model.recycle` | 10 | 48 576 | |
| `model.c6d_pred` / `aa_pred` / `pae_pred` / `pde_pred` / `lddt_pred` / `bind_pred` | 12 | 78 301 | |
| `model.full_emb`, `model.bond_emb` | 5 | 12 224 | |

Practical consequence for the port order: **one main block is 93 % of the
work**. Get `main_block.0` bit-for-bit right and the remaining 31 are the same
code.

---

## 1.3 Randomness inventory (SOP §1.3 / §5.1) — the hard part

RFdiffusion2 consumes **three independent generators**, and the Rust port must
reproduce all three *and their interleaving*.

`run_inference.py:seed_all(seed)` sets all three:

```python
torch.manual_seed(seed)   # at::mt19937
np.random.seed(seed)      # numpy legacy MT19937 (np.random.mtrand._rand)
random.seed(seed)         # CPython's Mersenne Twister
```

and `make_deterministic` additionally sets `torch.use_deterministic_algorithms(True)`.

Seeding happens **per design**, with `seed = i_des + seed_offset`, *after* the
model is built — so unlike ProteinMPNN there is **no parameter-initialisation
burn** to reproduce for the per-design stream. (`get_sampler` does call
`seed_all(0)` before construction, but `sample()` re-seeds before each design.)
**[unverified — to be confirmed empirically with the burn-measurement recipe in
SOP §5.1, because `InferenceDataset` construction happens between the two.]**

### Known consumers, in call order

| # | Site | Generator | What it draws |
|---|---|---|---|
| 1 | `contigs.py:64` `np.random.shuffle(motifs)`; `:184` `random.randint(...)` | numpy + python | contig segment lengths / order |
| 2 | `inference/utils.py:200,202` `np.random.shuffle`; `:254` `np.random.choice` | numpy | decoding schedule, first index |
| 3 | `inference/centering.py:38` `torch.randint(0, N_ori, (1,))` | torch | centering reference residue |
| 4 | `aa_model.py:2630` `torch.rand((natoms)) > p` | torch | atomization / ligand atom masking |
| 5 | `aa_model.py:2655` `random.choice(paths)` | python | path choice |
| 6 | `interpolant.sample_gaussian` → `torch.randn(B,L,3)` | torch | **initial translation noise** for x_T |
| 7 | `interpolant._uniform_so3` → `scipy.spatial.transform.Rotation.random(N)` | **numpy** | **initial rotation noise** for x_T |
| 8 | `rf_score/model.py:282` `psi_pred = torch.rand((B,I,L,2))` | torch | **once per denoising step, inside the forward** |

Two of these deserve emphasis, because they are exactly the SOP §5.1 trap:

**(7) The rotation noise is numpy, not torch.** `_uniform_so3` calls SciPy's
`Rotation.random`, which uses the *global numpy* `RandomState`. So x_T's
translations come from `at::mt19937` and its rotations come from numpy's
MT19937, and the two streams advance independently. Reproducing `--seed N` means
reimplementing **both** engines plus SciPy's random-rotation algorithm.

*Measured* (scipy 1.13.1 — upstream's pin — numpy 1.26.4), bit-exact for
N ∈ {1, 2, 5, 17}:

```python
Rotation.random(N)  ==  Rotation.from_quat(np.random.normal(size=(N, 4)))
```

i.e. exactly `4·N` draws from `np.random.normal` (the legacy `RandomState`
gauss path, which is the **polar/Marsaglia** method with its one-value cache —
so an odd draw count leaves a cached value that the *next* call consumes), then
SciPy's internal quaternion normalisation and quaternion→matrix conversion, all
in **float64**, narrowed to fp32 only by `_uniform_so3`'s final cast.

Note the normalisation must be SciPy's, not a hand-rolled one: pre-normalising
with `np.linalg.norm` before `from_quat` differs in the last bit
(max |Δ| = 3.3e-16 measured). SciPy's quaternion order is **(x, y, z, w)**.

**(8) `psi_pred` is random, and it reaches the output.**
`RFScore.forward_from_rfi` draws `torch.rand((B, I, L, 2))` and feeds it to
`all_atom.compute_backbone`, which uses psi to place the backbone carbonyl **O**.
`px0 = model_out['atom37'][0,-1]`, so **the O coordinates of every px0 are
functions of an RNG draw made inside the forward pass**, once per denoising step,
consuming `2·I·L` uniforms each time. This is not a training artefact left
dangling — it is on the inference path and it changes the written PDB. The port
must draw it at the same point in the same stream with the same count.

### What the Rust side therefore has to implement

1. `at::mt19937` + `uniform_real_distribution<float>` + the `normal_fill`
   `randn` path + `randint` — as in `proteinmpnn/mpnn/src/rng.rs`, which is already
   validated and can be lifted.
2. **numpy's legacy `RandomState`** — a *different* MT19937 (standard 624-word
   state, `random_sample` = `(a>>5, b>>6)` 53-bit construction, `shuffle` via
   Fisher-Yates with `randint` masking, `standard_normal` via the polar/ziggurat
   method used by the legacy generator). This is new work; ProteinMPNN did not
   need it.
3. **CPython's `random`** — same MT19937 core as numpy but different seeding
   (`init_by_array` over the key) and different `randint` (`_randbelow_with_getrandbits`).
4. SciPy's `Rotation.random` = `from_quat(normal(N,4))`, in f64.

Each gets its own fixture set and must be **exactly 0** on rung 2.

### Environment actually pinned for the harness

`.venv/` (isolated; the machine's system Python is untouched):
torch **2.4.0+cpu** (upstream's pin), numpy **1.26.4**, scipy **1.13.1**
(upstream's pin). Fixtures are only meaningful against these versions — record
them in every fixture's metadata.

---

## 1.4 dtype inventory (SOP §1.4 / §5.2)

- Every checkpoint tensor is fp32; `torch.set_default_dtype` is not changed.
- The network runs `msa_latent.dtype`-driven casts (`RoseTTAFoldModel.forward`
  does `.to(dtype)` on msa/pair/state) — so an fp32 input keeps everything fp32.
- **[unverified]** Candidate float64 promotions to check by printing `.dtype` at
  every step (SOP §5.2), because each would change the output bit pattern:
  - `scipy.spatial.transform.Rotation` works in **float64** throughout;
    `_uniform_so3` casts to fp32 only at the end — so the rotation noise is
    computed in double and narrowed.
  - `openfold.utils.rigid_utils` quaternion composition — check whether the
    `_QTR_MAT` tables are float64.
  - `contigs.py` / `features.py` numpy arrays: any bare `np.zeros(...)` is
    float64 and will promote whatever it touches (this is precisely the
    ProteinMPNN `bias_AAs_np` bug).
  - `Akima1DInterpolator` (numpy, float64) — only used for trajectory writing.

This table must be completed before rung 5; it is the single most common source
of "everything matches except the last two digits".

---

## 1.5 Environment / reproducibility notes

- Upstream pins **torch 2.4.0** (`envs/cuda124_env.yml`) and DGL from
  `dglteam/label/th24_cu124`. The reference harness venv here therefore pins
  **torch 2.4.0 CPU** — not the machine's system torch 2.7.1 — so the fixtures
  describe the published model rather than a newer kernel set.
- The structure refiner (`rf2aa/model/layers/SE3_network.py:SE3TransformerWrapper`)
  is **DGL-based**. Running the reference end-to-end requires DGL; the Rust port
  will need a from-scratch SE(3)-transformer over the same top-k graph.
- `run_inference.py` also imports **PyRosetta** (`import_pyrosetta.prepare_pyrosetta`)
  and uses it for `idealize_pose`. That is a licensed dependency and a
  post-processing step, not part of the network. **Scope decision: the Rust port
  reproduces the network and sampler; backbone idealization is ported directly
  (`dev/idealize_backbone.py` is pure torch), and the PyRosetta sidechain
  idealization path (`inference.idealize_sidechain_outputs`) is out of scope
  unless it turns out to be reachable without PyRosetta.** **[unverified]**
- No GPU on this machine; everything is CPU fp32, which matches the SOP's
  requirement to compare like with like.

---

## 1.6 Realistic scope assessment

Lines of Python on the inference path (excluding tests/benchmark/dev):
`rf_diffusion` ≈ 36 k, `rf2aa` ≈ 36 k, `se3_flow_matching` + `openfold` on top.
The *network* itself is comparatively small — `rf2aa/model/` is ~4.2 k LOC and
the SE3 transformer ~1.4 k — but the featurization (`aa_model.prepro`,
`atomize.py`, `contigs.py`, `chemical.py`, `features.py`) is where most of the
surface area and most of the traps live, exactly as SOP §5.6 predicts.

This is a substantially larger port than ProteinMPNN (1.7 M params, 118 tensors)
— 82.9 M params, 7 208 tensors, three RNG streams instead of one, and a
stochastic multi-step sampler. It is tractable, but it is not a one-sitting job,
and the SOP's ladder is what keeps it honest: no rung is claimed without the
number that proves it.
