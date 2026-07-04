# NOTICE — official ESMFold2 / ESM-C reference architecture

The Python model definitions in this directory (`esmfold2/`, `esmc/`) are the
**official released ESMFold2 and ESM-C architecture** from Chan Zuckerberg Biohub /
EvolutionaryScale, distributed under the **MIT License**:

> Copyright 2026 Chan Zuckerberg Biohub, Inc. — MIT License.
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software … to deal in the Software without restriction … The above copyright notice
> and this permission notice shall be included in all copies or substantial portions of
> the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND …

They are reproduced here **unmodified**, purely so a reader can see the reference
architecture alongside our pure-Rust port. All credit for the model, its design, and its
weights belongs to the original authors. Please refer to and cite their work:

- ESMFold2 weights & model: <https://huggingface.co/biohub/ESMFold2>
- ESM-C 6B weights: <https://huggingface.co/biohub/ESMC-6B>
- EvolutionaryScale: <https://www.evolutionaryscale.ai/>

This project is an **independent re-implementation for reproducibility** and is **not**
affiliated with or endorsed by the original authors. The **weights** are not included and
are downloaded from the sources above under their own terms.

## fp32 vs the released bf16 path

The released inference path applies a **bfloat16** cast to the sliding-window atom
attention (and uses bf16/autocast on GPU) — a sensible speed and memory optimization that
is part of the *inference recipe*, not the trained weights. Our harness (`../harness/`,
via `common.load_model(fp32=True)` / `esmc_precision="fp32"`) simply loads and runs this
same architecture in **fp32 on CPU** so the computation is deterministic and can be matched
exactly by the Rust port. Running fp32 is therefore a **numerical-precision variant of the
official model**; its outputs may differ from those of the official bf16 release, and it
should not be taken as representing official ESMFold2 results.
