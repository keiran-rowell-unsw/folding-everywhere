#!/usr/bin/env python3
"""Answer, in one reference run, the questions that decide how much of the
network the Rust port actually has to implement.

Each reference run costs minutes, so this batches every open question rather
than asking them one at a time:

1. Is `calc_lj_grads` reached? It is gated on `use_lj_l1`, which the checkpoint
   sets to True and whose extra features are visible in `str_refiner`'s weight
   shapes (`l0_in = 64 + 2*NTOTALDOFS = 104`) — but it has no
   `@torch.enable_grad()`, so under the sampler's `torch.no_grad()` it should
   raise. One of those two readings is wrong and only a run can say which.
2. Same for `calc_chiral_grads` (which *does* carry `@torch.enable_grad()`).
3. What `p2p_crop` / `topk_crop` actually reach `IterativeSimulator`, i.e.
   whether the striped `PairStr2Pair.subblock` path and the top-k graph are live.
4. The full **dropout trace**: every dropout call in execution order, with its
   module path, mask shape and draw count. This is the order the Rust port has
   to consume the torch stream in; getting it wrong desynchronises `psi_pred`.

Writes `results/forward_probe.txt`.
"""
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

import common  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def patch_nvtx():
    import contextlib

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield

    torch.cuda.nvtx.range = _noop
    torch.cuda.nvtx.range_push = lambda *a, **k: None
    torch.cuda.nvtx.range_pop = lambda *a, **k: None


def main():
    patch_nvtx()
    import pinned
    pinned.enable()
    ref = common.add_ref_to_path()

    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    # ---- instrument before the model is built -----------------------------
    import rf2aa.loss.loss as loss_mod
    counts = {"lj": 0, "chiral": 0}
    orig_lj = loss_mod.calc_lj_grads
    orig_ch = loss_mod.calc_chiral_grads

    def lj_spy(*a, **k):
        counts["lj"] += 1
        out = orig_lj(*a, **k)
        if counts["lj"] == 1:
            say(f"calc_lj_grads: CALLED, returns shapes "
                f"{[tuple(t.shape) for t in out]}")
        return out

    def ch_spy(*a, **k):
        counts["chiral"] += 1
        out = orig_ch(*a, **k)
        if counts["chiral"] == 1:
            say(f"calc_chiral_grads: CALLED, returns shapes "
                f"{[tuple(t.shape) for t in out]}, "
                f"nonzero={[int(t.abs().gt(0).sum()) for t in out]}")
        return out

    loss_mod.calc_lj_grads = lj_spy
    loss_mod.calc_chiral_grads = ch_spy
    import rf2aa.model.Track_module as tm
    tm.calc_lj_grads = lj_spy
    tm.calc_chiral_grads = ch_spy

    # crop values reaching the simulator
    import rf2aa.model.RoseTTAFoldModel as rfm
    orig_sim_fwd = tm.IterativeSimulator.forward

    def sim_spy(self, *a, **k):
        say(f"IterativeSimulator: p2p_crop={k.get('p2p_crop')} "
            f"topk_crop={k.get('topk_crop')} use_checkpoint={k.get('use_checkpoint')} "
            f"use_atom_frames={k.get('use_atom_frames')}")
        say(f"  n_extra_block={self.n_extra_block} n_main_block={self.n_main_block} "
            f"n_ref_block={self.n_ref_block} use_lj_l1={self.use_lj_l1} "
            f"use_chiral_l1={self.use_chiral_l1} refiner_topk={self.refiner_topk}")
        return orig_sim_fwd(self, *a, **k)

    tm.IterativeSimulator.forward = sim_spy
    _ = rfm

    from hydra import compose, initialize_config_dir
    from rf_diffusion.inference import model_runners
    from rf_diffusion import run_inference as ri

    cfg_dir = os.path.join(ref, "rf_diffusion", "config", "inference")
    overrides = [
        f"inference.ckpt_path={common.CKPT_173}",
        f"inference.input_pdb={ref}/rf_diffusion/benchmark/input/mcsa_41/M0584_1ldm.pdb",
        "inference.ligand='NAD,OXM'",
        "contigmap.contigs=['10,A106-106,10']",
        "inference.contig_as_guidepost=False",
        "inference.num_designs=1",
        "inference.deterministic=True",
        "inference.idealize_sidechain_outputs=False",
        "inference.write_trb_indep=False",
        "diffuser.T=2",
    ]
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        conf = compose(config_name="aa", overrides=overrides)

    ri.seed_all(0)
    sampler = model_runners.sampler_selector(conf)
    ri.seed_all(0 + conf.inference.seed_offset)

    # ---- dropout trace ----------------------------------------------------
    import torch.nn as nn
    from rf2aa.util_module import Dropout as RFDropout
    trace = []
    names = {m: n for n, m in sampler.model.named_modules()}

    orig_nn_fwd = nn.Dropout.forward

    def nn_fwd(self, x):
        trace.append(("nn.Dropout", names.get(self, "?"), tuple(x.shape),
                      float(self.p), 1))
        return orig_nn_fwd(self, x)

    orig_rf_fwd = RFDropout.forward

    def rf_fwd(self, x):
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        n = 1
        for s in shape:
            n *= s
        trace.append(("rf.Dropout", names.get(self, "?"), tuple(x.shape),
                      float(self.p_drop), n))
        return orig_rf_fwd(self, x)

    nn.Dropout.forward = nn_fwd
    RFDropout.forward = rf_fwd

    indep, contig_map, atomizer, t_step_input = sampler.sample_init(0)
    import rf_diffusion.features as features
    mconf = sampler._conf
    fc = features.init_tXd_inference(
        indep, getattr(mconf, "extra_tXd", []), mconf.extra_tXd_params,
        mconf.inference.conditions)
    t = int(t_step_input)
    extra = {"rfo_uncond": None, "rfo_cond": None, "n_steps": torch.tensor(1)}
    sampler.sample_step(t, indep, None, extra, fc)

    nn.Dropout.forward = orig_nn_fwd
    RFDropout.forward = orig_rf_fwd

    say(f"calc_lj_grads calls: {counts['lj']}")
    say(f"calc_chiral_grads calls: {counts['chiral']}")
    say(f"dropout calls in one forward: {len(trace)}")
    total_draws = sum(t[4] for t in trace)
    say(f"total RNG draws from dropout: {total_draws} "
        f"({sum(1 for t in trace if t[0]=='nn.Dropout')} of them 1-draw MKL seeds)")
    say("--- first 40 dropout calls, in execution order ---")
    for i, (kind, name, shape, p, n) in enumerate(trace[:40]):
        say(f"  {i:3d} {kind:11s} p={p:<5} draws={n:<8} {name}  x{list(shape)}")

    os.makedirs(common.RESULTS, exist_ok=True)
    with open(os.path.join(common.RESULTS, "forward_probe.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("wrote results/forward_probe.txt")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
