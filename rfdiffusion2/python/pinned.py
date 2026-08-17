"""Bit-exact ("pinned") mode for the reference — see `docs/BITEXACT.md`.

Import this and call `enable()` **before** building the model. Every patched
torch op then computes its result in float64 and rounds to float32 exactly once,
instead of accumulating in float32 in whatever order MKL/oneDNN chooses.

Why this is the route to an end-to-end bit-exact port
----------------------------------------------------
`python/probe_gemm_order.py` measured every plausible fp32 reduction order
against `F.linear` at RFdiffusion2's real shapes. The best candidate agrees on
99.1 % of outputs at small K and ~10 % at K >= 192 — never 100 %. Reproducing
stock PyTorch bit-for-bit would mean reimplementing MKL's blocking.

`python/probe_f64_pinning.py` then measured the alternative: accumulate in f64
and round once. Four deliberately different f64 orders (BLAS-blocked,
sequential, reversed, 8-lane) over 299 200 values produced **zero**
disagreements — because an f64 rounding error (~1e-16 relative) is ~9 orders of
magnitude below an f32 ULP, so the f32 result is the correctly-rounded one and
is therefore order-, blocking-, SIMD- and thread-independent.

Two consequences worth stating plainly:

1. This is a change to the *reference*, not just to the port. A pinned run is
   RFdiffusion2 with the same weights, the same algorithm and the same discrete
   decisions, but with intermediate arithmetic rounded differently — it differs
   from a stock-MKL run by fp32 round-off (~1e-6 relative, measured).
2. Double rounding is safe for the ops patched here. For `+ - * /` and `sqrt`
   with f32 inputs, computing in f64 and rounding once gives exactly the f32
   result (f64's 53-bit significand exceeds 2*24+2), so patching them changes
   nothing; it is the *multi-term reductions* where it bites, which is the
   point.

Usage
-----
    import pinned
    pinned.enable()
    ...build model, run inference...
    print(pinned.report())
"""
import functools

import torch
import torch.nn.functional as F

_ORIGINALS = {}
_ENABLED = False
_COUNTS = {}

F32 = torch.float32
F64 = torch.float64


def _to64(x):
    """Promote fp32 tensors to f64, **recursing into lists and tuples**.

    The recursion is not defensive tidiness, it is load-bearing. `rf2aa` does
    `from opt_einsum import contract as einsum`, and opt_einsum's torch backend
    forwards as `torch.einsum(equation, operands)` — the *sublist* form, with the
    operands in a tuple. A wrapper that only looks for tensors at the top level
    of `args` sees `(str, tuple)`, concludes nothing is fp32, and hands the call
    straight through. Every attention contraction in the network goes through
    that path, so before this fix the pinned reference was running its einsums in
    stock fp32 while `report()` still showed a healthy `torch.einsum` count (the
    410 direct `torch.einsum(eq, a, b)` calls in `util.py` / `util_module.py`,
    which do pass tensors positionally).

    Found by bisection: every stage of `TriangleMultiplication` was bit-identical
    to the Rust port except the einsum, which came back 16.9 % exact with a
    max |Δ| of 3.8e-5 — the signature of an fp32 reduction, not of a bug.
    """
    if torch.is_tensor(x):
        return x.double() if x.dtype == F32 else x
    if isinstance(x, (list, tuple)):
        return type(x)(_to64(v) for v in x)
    return x


def _has_f32(x):
    if torch.is_tensor(x):
        return x.dtype == F32
    if isinstance(x, (list, tuple)):
        return any(_has_f32(v) for v in x)
    return False


def _from64(out):
    """Narrow f64 results back to f32, recursing the same way."""
    if torch.is_tensor(out):
        return out.float() if out.dtype == F64 else out
    if isinstance(out, (list, tuple)):
        return type(out)(_from64(o) for o in out)
    return out


def _wrap(mod, name, tag=None):
    """Patch `mod.name` so fp32 inputs are computed in fp64 and rounded once."""
    orig = getattr(mod, name)
    tag = tag or f"{getattr(mod, '__name__', mod)}.{name}"
    _ORIGINALS[(mod, name)] = orig

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        if not _ENABLED:
            return orig(*args, **kwargs)
        promoted = any(_has_f32(a) for a in list(args) + list(kwargs.values()))
        if not promoted:
            return orig(*args, **kwargs)
        _COUNTS[tag] = _COUNTS.get(tag, 0) + 1
        a64 = [_to64(a) for a in args]
        k64 = {k: _to64(v) for k, v in kwargs.items()}
        return _from64(orig(*a64, **k64))

    setattr(mod, name, wrapper)


def _jacobi_eigh3(a_in):
    """Canonical scalar f64 eigendecomposition for a symmetric 3x3 matrix.

    This is mirrored line-for-line by `rfd2::noiser::jacobi_eigh`; it exists to
    remove LAPACK's singular-vector rounding from the pinned Kabsch path.
    Returns eigenvalues and column eigenvectors without sorting.
    """
    a = [list(map(float, row)) for row in a_in]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(64):
        p, q, off = 0, 1, 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(a[i][j]) > off:
                    p, q, off = i, j, abs(a[i][j])
        if off < 1e-300:
            break
        theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        sign = -1.0 if theta < 0.0 else (1.0 if theta > 0.0 else 0.0)
        t = sign / (abs(theta) + (theta * theta + 1.0) ** 0.5)
        c = 1.0 / (t * t + 1.0) ** 0.5
        s = t * c
        b = [row[:] for row in a]
        for k in range(3):
            b[k][p] = c * a[k][p] - s * a[k][q]
            b[k][q] = s * a[k][p] + c * a[k][q]
        a2 = [row[:] for row in b]
        for k in range(3):
            a2[p][k] = c * b[p][k] - s * b[q][k]
            a2[q][k] = s * b[p][k] + c * b[q][k]
        a2[p][q] = a2[q][p] = 0.0
        a = a2
        v2 = [row[:] for row in v]
        for k in range(3):
            v2[k][p] = c * v[k][p] - s * v[k][q]
            v2[k][q] = s * v[k][p] + c * v[k][q]
        v = v2
    return [a[i][i] for i in range(3)], v


def _canonical_svd3(x):
    """Canonical reduced SVD for one fp32 3x3 tensor, returned as fp32."""
    cmat = [[float(x[i, j].item()) for j in range(3)] for i in range(3)]
    ctc = [[sum(cmat[k][i] * cmat[k][j] for k in range(3))
            for j in range(3)] for i in range(3)]
    evals, evec = _jacobi_eigh3(ctc)
    order = sorted(range(3), key=lambda i: evals[i], reverse=True)
    vv = [[evec[row][order[col]] for col in range(3)] for row in range(3)]
    ss = [max(evals[src], 0.0) ** 0.5 for src in order]
    uu = [[sum(cmat[row][k] * vv[k][col] / ss[col] for k in range(3))
           for col in range(3)] for row in range(3)]
    u = torch.tensor(uu, dtype=torch.float32, device=x.device)
    s = torch.tensor(ss, dtype=torch.float32, device=x.device)
    vh = torch.tensor(vv, dtype=torch.float32, device=x.device).transpose(0, 1)
    return u, s, vh


def _wrap_linalg_svd():
    """Pin SVD, with a canonical algorithm for the live 3x3 Kabsch call."""
    mod, name = torch.linalg, "svd"
    orig = getattr(mod, name)
    _ORIGINALS[(mod, name)] = orig

    @functools.wraps(orig)
    def wrapper(x, *args, **kwargs):
        if not _ENABLED or not torch.is_tensor(x) or x.dtype != F32:
            return orig(x, *args, **kwargs)
        _COUNTS["torch.linalg.svd"] = _COUNTS.get("torch.linalg.svd", 0) + 1
        if x.ndim == 2 and tuple(x.shape) == (3, 3):
            return _canonical_svd3(x)
        return _from64(orig(x.double(), *args, **kwargs))

    setattr(mod, name, wrapper)


# Ops that carry a multi-term reduction (where order actually matters) plus the
# transcendentals whose fp32 versions are SLEEF/Cephes rather than correctly
# rounded (SOP §5.3). Pure elementwise +,-,* are left alone: promoting them
# changes nothing, so patching them would only cost time.
_TARGETS = [
    (F, "linear"),
    (F, "layer_norm"),
    (F, "group_norm"),
    (torch, "group_norm"),
    (F, "softmax"),
    (F, "log_softmax"),
    (F, "elu"),
    (F, "normalize"),
    (F, "scaled_dot_product_attention"),
    (torch, "matmul"),
    (torch, "einsum"),
    (torch, "bmm"),
    (torch, "mm"),
    (torch, "addmm"),
    (torch, "baddbmm"),
    (torch, "tensordot"),
    (torch, "cdist"),
    (torch, "cross"),
    (torch, "norm"),
    (torch, "sqrt"),
    (torch, "rsqrt"),
    (torch, "exp"),
    (torch, "log"),
    (torch, "log1p"),
    (torch, "expm1"),
    (torch, "erf"),
    (torch, "sin"),
    (torch, "cos"),
    (torch, "atan2"),
    (torch, "acos"),
    (torch, "sum"),
    (torch, "mean"),
    (torch, "var"),
    (torch, "std"),
    (torch, "prod"),
    (torch, "logsumexp"),
    # --- closed 2026-08-09: these were holes, all of them on the inference path
    # and every one of them a transcendental whose fp32 kernel is SLEEF/Cephes
    # rather than correctly rounded, i.e. exactly the class of op this module
    # exists for. `sigmoid` alone fires ~7 500 times per denoising step (every
    # gate in every attention block).
    (torch, "sigmoid"),
    (torch, "tanh"),
    (torch, "asinh"),
    (torch, "arcsinh"),
    (torch, "acosh"),
    (torch, "asin"),
    (torch, "atan"),
    (torch, "tan"),
    (torch.linalg, "norm"),
    (torch.linalg, "vector_norm"),
    (F, "sigmoid"),
    # --- closed 2026-08-10: the flow-matching noiser's optimal-transport step
    # Kabsch-aligns the sampled translation noise onto the ground truth, and it
    # gets the rotation from `torch.linalg.svd` of a 3x3 covariance matrix.
    # LAPACK's fp32 SVD is no more reproducible from outside than MKL's fp32
    # GEMM is -- but the same argument rescues it: the Kabsch rotation
    # `R = V U^T` is *mathematically unique* (and invariant to the sign
    # convention, since flipping a paired (u_i, v_i) leaves `v_i u_i^T` alone),
    # so computing the decomposition in f64 and rounding once makes the fp32
    # answer the correctly-rounded one, and therefore independent of which
    # algorithm produced it.
    (torch.linalg, "det"),
    (torch.linalg, "eigh"),
    (torch, "svd"),
    (torch, "det"),
]


# Patching module-level functions is NOT enough, and this is the single easiest
# way to ship a wrong bit-exactness claim. RF2AA is written with tensor methods
# and operators -- `a @ b`, `x.matmul(y)`, `x.sum(-1)`, `x.softmax(-1)` -- and
# those dispatch straight to `torch.Tensor.*`, bypassing `torch.*` entirely.
# Measured before this was added: `x.matmul(w.t())` came back only 15.2 %
# bit-identical to the pinned result while `F.linear` was at 100 %.
_TENSOR_METHODS = [
    "matmul", "__matmul__", "__rmatmul__", "mm", "bmm", "addmm", "baddbmm",
    "sum", "mean", "var", "std", "prod", "logsumexp", "norm", "dot",
    "softmax", "log_softmax", "sqrt", "rsqrt", "exp", "log", "log1p",
    "expm1", "erf", "sin", "cos", "acos", "atan2", "cross", "cdist",
    "tensordot",
    "sigmoid", "tanh", "asinh", "arcsinh", "asin", "atan", "tan",
]


# DGL's graph kernels are the other hole, and a bigger one than the tensor
# methods were: `AttentionSE3` runs its softmax and its neighbour sums through
# `dgl.ops`, which is compiled C++ with its own segmented reduction order over
# each destination node's incoming edges. Nothing in `torch.*` sees those calls,
# so before this the SE(3) refiner's attention was accumulating in fp32 in an
# order set by DGL's CSR layout -- i.e. unpinned, while the audit looked clean.
#
# Same convention as everywhere else: promote the fp32 arguments to f64, let DGL
# reduce in double, round once. The f64 rounding error is ~9 orders of magnitude
# below an f32 ULP, so the result no longer depends on the reduction order.
_DGL_OPS = [
    "copy_e_sum", "copy_e_mean", "copy_e_max", "copy_e_min",
    "copy_u_sum", "copy_u_mean",
    "e_dot_v", "e_dot_u", "u_dot_v",
    "edge_softmax",
    "u_mul_e_sum", "u_mul_e_mean",
]


def enable():
    """Patch the op set. Idempotent.

    Must run **before** `rf2aa` is imported: `se3_transformer`'s attention layer
    does `from dgl.ops import edge_softmax` at module scope, so a patch applied
    afterwards would rebind `dgl.ops.edge_softmax` while the layer kept the
    original.
    """
    global _ENABLED
    if not _ORIGINALS:
        _wrap_linalg_svd()
        for mod, name in _TARGETS:
            if hasattr(mod, name):
                _wrap(mod, name)
        for name in _TENSOR_METHODS:
            if hasattr(torch.Tensor, name):
                try:
                    _wrap(torch.Tensor, name, tag=f"Tensor.{name}")
                except (AttributeError, TypeError) as e:
                    # Record rather than swallow: an unpatchable method is a
                    # hole in the guarantee and must show up in report().
                    _COUNTS[f"UNPATCHABLE Tensor.{name}: {e}"] = -1
        try:
            import dgl.ops as dgl_ops
        except ImportError as e:
            _COUNTS[f"UNPATCHABLE dgl.ops: {e}"] = -1
        else:
            for name in _DGL_OPS:
                if hasattr(dgl_ops, name):
                    _wrap(dgl_ops, name, tag=f"dgl.ops.{name}")
    _ENABLED = True
    return sorted({f"{getattr(m, '__name__', m)}.{n}" for (m, n) in _ORIGINALS})


def disable():
    global _ENABLED
    _ENABLED = False


def restore():
    """Undo the patching entirely."""
    global _ENABLED
    for (mod, name), orig in _ORIGINALS.items():
        setattr(mod, name, orig)
    _ORIGINALS.clear()
    _ENABLED = False


def report():
    """How many times each patched op actually fired -- the audit trail that
    says which parts of the model ran pinned. An op with a count of 0 either is
    not on the inference path or is being reached by a route this module does
    not patch (e.g. `Tensor.matmul` as a method); the latter is a hole and must
    be closed before any bit-exactness claim."""
    return dict(sorted(_COUNTS.items(), key=lambda kv: -kv[1]))


def reset_counts():
    _COUNTS.clear()
