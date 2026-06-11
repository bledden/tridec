"""Single-launch persistent BP / Relay-BP Triton megakernels (issue #2, v0.2).

ONE kernel launch per ``decode_batch`` instead of 2 launches x iterations
x relay legs (thousands of launches; measured launch-bound on Metal:
~31 s launches vs ~1.3 s math for the canonical 2000-shot relay cell).

DESIGN -- "shot-per-program", Plan A of issue #2.
=================================================
One Triton program owns ONE shot and runs the entire decode in-kernel:

  * the BP iteration loop (check pass + bit pass as tile loops over the
    padded-edge-index tables, identical math to bp_triton / relay_triton --
    the same exclude-self two-smallest-magnitude min-sum, the same Eq-2/3/4
    Relay-BP memory updates, the SAME per-(shot,row) operation order, so
    fp results are bit-identical to the two-kernel path),
  * the relay-leg loop with the per-leg disordered gamma vectors,
  * the per-iteration syndrome-convergence check, the lowest-weight
    valid-solution capture, and the nconv stop -- PER SHOT.

Messages live in per-shot-private slices of global buffers, shot-major
(S, E) / (S, N), so each program's working set (~43 KB/shot for the
[[72,12,6]] BB cell) stays cache-resident. Gamma draws are precomputed
host-side ONCE per batch ((n_legs, N), ~380 KB) with the exact RNG of
``RelayBpTriton._gamma_for_leg``, preserving RNG semantics and keeping the
kernel deterministic.

PER-SHOT EARLY EXIT vs the host-loop baseline
---------------------------------------------
``RelayBpTriton``'s host loop stops only when ALL shots reach
``stop_nconv`` converged legs, so already-finished shots keep running legs
(and may still lower their best weight). The megakernel stops EACH shot at
its own ``stop_nconv`` -- the relay_bp Rust oracle's per-shot semantics.
``early_exit=False`` runs all legs for all shots (the bit-identity
validation mode vs the host loop with an unreachable ``stop_nconv``).

TRITON-METAL 3.7 CODEGEN CONSTRAINTS (probed, 2026-06; see
bench/receipts/megakernel_metal_spike.md)
----------------------------------------------------------
  * Cross-lane reductions (``tl.sum``) INSIDE dynamic loops fail Metal
    codegen for BLOCK >= 256 (undeclared ``UNKNOWN_<addr>`` identifiers in
    the emitted .metal source). BLOCK <= 128 compiles and runs correctly,
    hence ``block=128`` is the default here.
  * Assigning to a ``while``-carried variable inside a scalar ``if``
    miscompiles SILENTLY (the assignment is dropped). The kernels therefore
    use NO ``while`` loops and NO scalar-``if`` blocks: the leg loop is a
    ``for`` whose inner iteration loop gets a ZERO trip count once the shot
    converged (early exit as loop-bound algebra), carried scalars update
    via arithmetic / ``tl.where``, and the conditional lowest-weight capture
    is a scalar-bool-masked store plus a zero-trip weight pass.

IN-PROGRAM PASS SYNCHRONIZATION (all platforms, not Metal-specific)
-------------------------------------------------------------------
A Triton program maps to a threadgroup/CTA whose lanes proceed
independently between barriers. The check pass writes ``nu`` that the bit
pass GATHERS CROSS-LANE (and the bit pass scatters ``mu`` that the next
check pass gathers), so every pass boundary with a cross-lane global-memory
dependency carries a ``tl.debug_barrier()`` (threadgroup barrier + device
fence; ``__syncthreads`` on CUDA/ROCm). Without them the kernel is racy and
non-deterministic (observed on Metal: run-to-run posterior agreement ~0.80
at 10 iterations). All barrier sites are in threadgroup-UNIFORM control
flow (scalar loop bounds), as Metal requires.

KNOWN triton-metal GAP (4c42e96, triton 3.7): ``tl.debug_barrier()`` is
SILENTLY DROPPED -- triton 3.7 emits it as ``ttg.barrier all`` in TTGIR and
triton-metal's lowerer only recognizes the pre-3.5 ``tt.debug_barrier``
spelling (unknown ops are skipped without error). Until that one-line
op-name mapping lands in triton-metal, the Metal configuration of these
kernels MUST use ``block=32``: one threadgroup = one 32-wide SIMD-group,
which executes in lockstep under uniform control flow, so the (dropped)
barriers are not needed for correctness. ``_default_block()`` picks 32 on
the triton-metal environment and 128 elsewhere. Verified on Metal at
block=32: bit-identical posteriors vs the two-kernel ``BpTriton`` path and
deterministic across repeated runs (see
bench/receipts/megakernel_metal_spike.md).
"""
import numpy as np
import torch

from .bp_triton import BpTriton, _INF
from .relay_triton import RelayBpTriton

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is an optional extra
    triton = None
    tl = None
    _HAVE_TRITON = False


def _default_block():
    """Lane count per program: 32 on the triton-metal environment (one
    lockstep SIMD-group; triton-metal drops ``tl.debug_barrier`` -- see
    module docstring), 128 elsewhere (<=128 also required by triton-metal's
    in-loop reduction codegen, and a reasonable CTA width on CUDA/ROCm
    pending the cloud autotune pass)."""
    import sys
    if sys.platform == "darwin":
        try:
            import triton_metal  # noqa: F401
            import torch as _torch
            if not _torch.cuda.is_available():
                return 32
        except Exception:
            pass
    return 128


if _HAVE_TRITON:

    @triton.jit
    def _bp_megakernel(
        MU, NU, POST, SYN, LAM,
        CEDGE, BEDGE, EDGE_BIT,
        S, E, N, C,
        n_iter, ms,
        MAXDEG_C: tl.constexpr, MAXDEG_B: tl.constexpr,
        BLOCK: tl.constexpr, INF: tl.constexpr,
    ):
        """Whole plain-BP decode of ONE shot in one program.

        program_id(0) = shot. Tile loops over checks / bits replace the old
        2-D grid; the per-(shot,row) operation order matches bp_triton's
        kernels exactly (same k-order static_range loops), so fp32 results
        are bit-identical to the two-kernel path."""
        s = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mu_b = MU + s * E
        nu_b = NU + s * E
        m_b = POST + s * N
        syn_b = SYN + s * C

        # flooding init: mu_e = prior LLR of bit(e).
        for t0 in range(0, E, BLOCK):
            e_offs = t0 + offs
            em = e_offs < E
            bit = tl.load(EDGE_BIT + e_offs, mask=em, other=0)
            lam_e = tl.load(LAM + bit, mask=em, other=0.0)
            tl.store(mu_b + e_offs, lam_e, mask=em)
        tl.debug_barrier()      # init mu visible to the check pass's gathers

        for _ in range(n_iter):
            # ---- CHECK PASS: exclude-self min-sum, tile over checks. ----
            for c0 in range(0, C, BLOCK):
                c_offs = c0 + offs
                cm = c_offs < C
                neg_par = tl.zeros((BLOCK,), dtype=tl.int32)
                min1 = tl.full((BLOCK,), INF, dtype=tl.float32)
                min2 = tl.full((BLOCK,), INF, dtype=tl.float32)
                for k in tl.static_range(MAXDEG_C):
                    e = tl.load(CEDGE + c_offs * MAXDEG_C + k, mask=cm,
                                other=-1)
                    real = (e >= 0) & cm
                    mu = tl.load(mu_b + tl.where(real, e, 0), mask=real,
                                 other=0.0)
                    a = tl.abs(mu)
                    a = tl.where(real, a, INF)
                    neg_par = neg_par ^ ((mu < 0.0) & real).to(tl.int32)
                    a_lt = a < min1
                    min2 = tl.where(a_lt, min1, tl.minimum(min2, a))
                    min1 = tl.minimum(min1, a)
                syn_c = tl.load(syn_b + c_offs, mask=cm, other=0)
                ssign = tl.where(syn_c == 0, 1.0, -1.0)
                check_sign = tl.where(neg_par == 0, 1.0, -1.0)
                for k in tl.static_range(MAXDEG_C):
                    e = tl.load(CEDGE + c_offs * MAXDEG_C + k, mask=cm,
                                other=-1)
                    real = (e >= 0) & cm
                    e_safe = tl.where(real, e, 0)
                    mu = tl.load(mu_b + e_safe, mask=real, other=0.0)
                    a = tl.abs(mu)
                    self_sign = tl.where(mu >= 0.0, 1.0, -1.0)
                    sign_others = check_sign * self_sign
                    min_others = tl.where(a == min1, min2, min1)
                    min_others = tl.where(min_others >= INF, 0.0, min_others)
                    nu = ms * ssign * sign_others * min_others
                    tl.store(nu_b + e_safe, nu, mask=real)
            tl.debug_barrier()  # nu visible to the bit pass's gathers
            # ---- BIT PASS: posterior + next mu, tile over bits. ---------
            for b0 in range(0, N, BLOCK):
                b_offs = b0 + offs
                bm = b_offs < N
                lam = tl.load(LAM + b_offs, mask=bm, other=0.0)
                incoming = tl.zeros((BLOCK,), dtype=tl.float32)
                for k in tl.static_range(MAXDEG_B):
                    e = tl.load(BEDGE + b_offs * MAXDEG_B + k, mask=bm,
                                other=-1)
                    real = (e >= 0) & bm
                    nu = tl.load(nu_b + tl.where(real, e, 0), mask=real,
                                 other=0.0)
                    incoming += nu
                posterior = lam + incoming
                tl.store(m_b + b_offs, posterior, mask=bm)
                for k in tl.static_range(MAXDEG_B):
                    e = tl.load(BEDGE + b_offs * MAXDEG_B + k, mask=bm,
                                other=-1)
                    real = (e >= 0) & bm
                    e_safe = tl.where(real, e, 0)
                    nu = tl.load(nu_b + e_safe, mask=real, other=0.0)
                    tl.store(mu_b + e_safe, posterior - nu, mask=real)
            tl.debug_barrier()  # mu visible to the next check pass

    @triton.jit
    def _relay_megakernel(
        MU, NU, M, BEST_EH, BEST_W, NCONV_OUT, LEGS_OUT, ITERS_OUT,
        SYN, LAM, WLLR, GAMMA,
        CEDGE, CBIT, BEDGE, EDGE_BIT,
        S, E, N, C,
        n_legs, pre_iter, set_max_iter, stop_nconv,
        ms,
        MAXDEG_C: tl.constexpr, MAXDEG_B: tl.constexpr,
        BLOCK: tl.constexpr, INF: tl.constexpr, DT: tl.constexpr,
        EARLY_EXIT: tl.constexpr,
    ):
        """Whole Relay-BP decode of ONE shot in one program (one launch/batch).

        Pre leg + relay legs + per-iteration first-convergence capture +
        lowest-weight selection + per-shot nconv early exit, all in-kernel.
        ``GAMMA`` is the host-precomputed (n_legs, N) per-leg gamma table.
        Control flow is for-loops only (no while / scalar-if): once
        ``nconv >= stop_nconv`` the remaining legs run with a ZERO iteration
        bound (see module docstring: triton-metal silently miscompiles
        while-carried scalar-if assignment)."""
        s = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mu_b = MU + s * E
        nu_b = NU + s * E
        m_b = M + s * N
        beh_b = BEST_EH + s * N
        syn_b = SYN + s * C

        # init: mu = flooding prior per edge; M(0) = lam; best_eh = 0.
        for t0 in range(0, E, BLOCK):
            e_offs = t0 + offs
            em = e_offs < E
            bit = tl.load(EDGE_BIT + e_offs, mask=em, other=0)
            lam_e = tl.load(LAM + bit, mask=em, other=0.0)
            tl.store(mu_b + e_offs, lam_e, mask=em)
        for t0 in range(0, N, BLOCK):
            b_offs = t0 + offs
            bm = b_offs < N
            lam0 = tl.load(LAM + b_offs, mask=bm, other=0.0)
            tl.store(m_b + b_offs, lam0, mask=bm)
            tl.store(beh_b + b_offs, lam0 * 0.0, mask=bm)
        tl.debug_barrier()      # init mu/M visible to the first check pass

        best_w = INF + 0.0
        nconv = 0
        legs_run = 0
        iters_run = 0
        for leg in range(n_legs):
            n_it = tl.where(leg == 0, pre_iter, set_max_iter)
            if EARLY_EXIT:
                n_it = tl.where(nconv >= stop_nconv, 0, n_it)
            legs_run += tl.where(n_it > 0, 1, 0)
            leg_conv = 0
            for _ in range(n_it):
                iters_run += 1
                # ---- CHECK PASS (identical math to relay_triton). -------
                for c0 in range(0, C, BLOCK):
                    c_offs = c0 + offs
                    cm = c_offs < C
                    neg_par = tl.zeros((BLOCK,), dtype=tl.int32)
                    min1 = tl.full((BLOCK,), INF, dtype=DT)
                    min2 = tl.full((BLOCK,), INF, dtype=DT)
                    for k in tl.static_range(MAXDEG_C):
                        e = tl.load(CEDGE + c_offs * MAXDEG_C + k, mask=cm,
                                    other=-1)
                        real = (e >= 0) & cm
                        mu = tl.load(mu_b + tl.where(real, e, 0), mask=real,
                                     other=0.0)
                        a = tl.abs(mu)
                        a = tl.where(real, a, INF)
                        neg_par = neg_par ^ ((mu < 0.0) & real).to(tl.int32)
                        a_lt = a < min1
                        min2 = tl.where(a_lt, min1, tl.minimum(min2, a))
                        min1 = tl.minimum(min1, a)
                    syn_c = tl.load(syn_b + c_offs, mask=cm, other=0)
                    ssign = tl.where(syn_c == 0, 1.0, -1.0)
                    check_sign = tl.where(neg_par == 0, 1.0, -1.0)
                    for k in tl.static_range(MAXDEG_C):
                        e = tl.load(CEDGE + c_offs * MAXDEG_C + k, mask=cm,
                                    other=-1)
                        real = (e >= 0) & cm
                        e_safe = tl.where(real, e, 0)
                        mu = tl.load(mu_b + e_safe, mask=real, other=0.0)
                        a = tl.abs(mu)
                        self_sign = tl.where(mu >= 0.0, 1.0, -1.0)
                        sign_others = check_sign * self_sign
                        min_others = tl.where(a == min1, min2, min1)
                        min_others = tl.where(min_others >= INF, 0.0,
                                              min_others)
                        nu = ms * ssign * sign_others * min_others
                        tl.store(nu_b + e_safe, nu, mask=real)
                tl.debug_barrier()   # nu visible to the bit pass's gathers
                # ---- BIT PASS (Eq 2/3/4 memory update). -----------------
                for b0 in range(0, N, BLOCK):
                    b_offs = b0 + offs
                    bm = b_offs < N
                    lam = tl.load(LAM + b_offs, mask=bm, other=0.0)
                    gam = tl.load(GAMMA + leg * N + b_offs, mask=bm,
                                  other=0.0)
                    mprev = tl.load(m_b + b_offs, mask=bm, other=0.0)
                    incoming = tl.zeros((BLOCK,), dtype=DT)
                    for k in tl.static_range(MAXDEG_B):
                        e = tl.load(BEDGE + b_offs * MAXDEG_B + k, mask=bm,
                                    other=-1)
                        real = (e >= 0) & bm
                        nu = tl.load(nu_b + tl.where(real, e, 0), mask=real,
                                     other=0.0)
                        incoming += nu
                    lam_bias = (1.0 - gam) * lam + gam * mprev   # Eq 4
                    posterior = lam_bias + incoming              # Eq 3
                    var_tot = lam + incoming                     # Eq 2: PLAIN prior
                    tl.store(m_b + b_offs, posterior, mask=bm)
                    for k in tl.static_range(MAXDEG_B):
                        e = tl.load(BEDGE + b_offs * MAXDEG_B + k, mask=bm,
                                    other=-1)
                        real = (e >= 0) & bm
                        e_safe = tl.where(real, e, 0)
                        nu = tl.load(nu_b + e_safe, mask=real, other=0.0)
                        tl.store(mu_b + e_safe, var_tot - nu, mask=real)
                tl.debug_barrier()   # mu/M visible to convergence gathers
                                     # + the next iteration's check pass
                # ---- CONVERGENCE: GF2 syndrome residual of hard(M). -----
                mism_vec = tl.zeros((BLOCK,), dtype=tl.int32)
                for c0 in range(0, C, BLOCK):
                    c_offs = c0 + offs
                    cm = c_offs < C
                    par = tl.zeros((BLOCK,), dtype=tl.int32)
                    for k in tl.static_range(MAXDEG_C):
                        b = tl.load(CBIT + c_offs * MAXDEG_C + k, mask=cm,
                                    other=-1)
                        real = (b >= 0) & cm
                        pb = tl.load(m_b + tl.where(real, b, 0), mask=real,
                                     other=1.0)
                        par = par ^ ((pb < 0.0) & real).to(tl.int32)
                    syn_c = tl.load(syn_b + c_offs, mask=cm,
                                    other=0).to(tl.int32)
                    mism_vec += tl.where(cm, par ^ syn_c, 0)
                mism = tl.sum(mism_vec)          # BLOCK<=128: metal constraint
                # ---- FIRST-CONVERGENCE / LOWEST-WEIGHT CAPTURE. ---------
                # zero-trip guard: the weight + capture passes only run on
                # valid iterations (no scalar-if; bound is 0 or 1).
                for _ in range(tl.where(mism == 0, 1, 0)):
                    w_vec = tl.zeros((BLOCK,), dtype=DT)
                    for b0 in range(0, N, BLOCK):
                        b_offs = b0 + offs
                        bm = b_offs < N
                        pb = tl.load(m_b + b_offs, mask=bm, other=1.0)
                        wl = tl.load(WLLR + b_offs, mask=bm, other=0.0)
                        w_vec += tl.where((pb < 0.0) & bm, wl, 0.0)
                    w = tl.sum(w_vec)
                    improve = w < best_w
                    for b0 in range(0, N, BLOCK):
                        b_offs = b0 + offs
                        bm = b_offs < N
                        pb = tl.load(m_b + b_offs, mask=bm, other=1.0)
                        eh = (pb < 0.0).to(DT)
                        tl.store(beh_b + b_offs, eh, mask=bm & improve)
                    best_w = tl.where(improve, w, best_w)
                leg_conv = tl.where(mism == 0, 1, leg_conv)
            nconv += leg_conv
        tl.store(BEST_W + s, best_w)
        tl.store(NCONV_OUT + s, nconv)
        tl.store(LEGS_OUT + s, legs_run)
        tl.store(ITERS_OUT + s, iters_run)


class BpMegaTriton(BpTriton):
    """``BpTriton`` with the iteration loop moved in-kernel: ONE launch per
    ``decode_batch`` / ``run_iterations_batch`` call instead of
    2 x n_iter. Same math, same surface; fp32 results bit-identical to
    the two-kernel path (same per-(shot,row) operation order)."""

    def __init__(self, H, priors, max_iter=30, ms_scaling_factor=0.625,
                 block_s=256, block=None):
        super().__init__(H, priors, max_iter=max_iter,
                         ms_scaling_factor=ms_scaling_factor, block_s=block_s)
        self.block = int(block) if block is not None else _default_block()
        self._edge_bit_i32 = torch.as_tensor(
            self._edge_bit_np.astype(np.int32))

    @classmethod
    def from_dem(cls, dem, max_iter=30, ms_scaling_factor=0.625, block_s=256,
                 block=None):
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], max_iter=max_iter,
                  ms_scaling_factor=ms_scaling_factor, block_s=block_s,
                  block=block)
        obj._dem = dem
        obj.dem = dem
        obj._Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._n_obs = ex["n_obs"]
        return obj

    def _run_core(self, syn_t, ssign, lam, cedge, bedge, mu_init,
                  n_iter, dev, S, E, N, C):
        """Single megakernel launch; returns (post, mu, nu) shaped like the
        base class ((N,S)/(E,S) views of the shot-major buffers)."""
        key = "ebit32:" + str(dev)
        ebit = self._dev_cache.get(key)
        if ebit is None:
            ebit = self._edge_bit_i32.to(dev)
            self._dev_cache[key] = ebit
        mu = torch.empty((S, E), dtype=torch.float32, device=dev)
        nu = torch.empty((S, E), dtype=torch.float32, device=dev)
        post = torch.empty((S, N), dtype=torch.float32, device=dev)
        syn_u8 = syn_t.to(torch.uint8).contiguous()
        _bp_megakernel[(S,)](
            mu, nu, post, syn_u8, lam,
            cedge, bedge, ebit,
            S, E, N, C,
            int(n_iter), self.ms,
            MAXDEG_C=self.MAXDEG_C, MAXDEG_B=self.MAXDEG_B,
            BLOCK=self.block, INF=_INF,
        )
        return post.t(), mu.t(), nu.t()


class RelayBpMegaTriton(RelayBpTriton):
    """``RelayBpTriton`` with the ENTIRE relay schedule in-kernel: ONE launch
    per ``decode_batch`` (vs 2 launches x iters x legs). Per-shot nconv
    early exit (the relay_bp oracle's per-shot semantics; the host loop
    only stops when ALL shots are done -- set ``early_exit=False`` for the
    all-legs bit-identity mode). Gamma tensors are precomputed host-side
    with the exact ``_gamma_for_leg`` RNG, so a given (gamma_seed, leg)
    yields the same gamma vector as the host-loop implementation."""

    def __init__(self, H, priors, gamma0=0.1, pre_iter=80, num_sets=60,
                 set_max_iter=60, gamma_dist_interval=(-0.24, 0.66),
                 stop_nconv=5, stopping_criterion="nconv", alpha=1.0,
                 gamma_seed=12345, block_s=256, dtype="float64",
                 block=None, early_exit=True):
        super().__init__(H, priors, gamma0=gamma0, pre_iter=pre_iter,
                         num_sets=num_sets, set_max_iter=set_max_iter,
                         gamma_dist_interval=gamma_dist_interval,
                         stop_nconv=stop_nconv,
                         stopping_criterion=stopping_criterion, alpha=alpha,
                         gamma_seed=gamma_seed, block_s=block_s, dtype=dtype)
        self.block = int(block) if block is not None else _default_block()
        self.early_exit = bool(early_exit)
        self._edge_bit_i32 = torch.as_tensor(
            self._edge_bit_np.astype(np.int32))
        # check -> padded incident BIT indices (for the in-kernel GF2
        # syndrome residual), same row order as cedge.
        cbit = np.where(self._cedge_np >= 0,
                        self._edge_bit_np[np.maximum(self._cedge_np, 0)],
                        -1).astype(np.int32)
        self._cbit_np = cbit
        self.cbit = torch.as_tensor(cbit, dtype=torch.int32)
        self.last_stats = None

    @classmethod
    def from_dem(cls, dem, gamma0=0.1, pre_iter=80, num_sets=60,
                 set_max_iter=60, gamma_dist_interval=(-0.24, 0.66),
                 stop_nconv=5, stopping_criterion="nconv", alpha=1.0,
                 gamma_seed=12345, block_s=256, dtype="float64",
                 block=None, early_exit=True):
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], gamma0=gamma0,
                  pre_iter=pre_iter, num_sets=num_sets,
                  set_max_iter=set_max_iter,
                  gamma_dist_interval=gamma_dist_interval,
                  stop_nconv=stop_nconv,
                  stopping_criterion=stopping_criterion, alpha=alpha,
                  gamma_seed=gamma_seed, block_s=block_s, dtype=dtype,
                  block=block, early_exit=early_exit)
        obj._dem = dem
        obj.dem = dem
        obj._Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._n_obs = ex["n_obs"]
        return obj

    def _mega_tensors(self, dev):
        """Cache + return (ebit32, cbit, gamma_all, wllr_dt) on ``dev``.
        ``gamma_all`` is the (n_legs, N) per-leg gamma table, rows built by
        the inherited ``_gamma_for_leg`` (identical RNG to the host loop)."""
        key = "mega:" + str(dev)
        cached = self._dev_cache.get(key)
        if cached is None:
            ebit = self._edge_bit_i32.to(dev)
            cbit = self.cbit.to(dev)
            n_legs = 1 + self.num_sets
            gamma_all = torch.stack(
                [self._gamma_for_leg(leg, dev) for leg in range(n_legs)]
            ).contiguous()                                   # (n_legs, N)
            wllr_dt = torch.as_tensor(self._wllr_np.astype(
                np.float64 if self.dtype == "float64" else np.float32),
                device=dev)
            cached = (ebit, cbit, gamma_all, wllr_dt)
            self._dev_cache[key] = cached
        return cached

    def _relay_posteriors(self, syn_t, dev):
        """One megakernel launch -> (N, S) best hard error (view)."""
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        lam, cedge, bedge, edge_bit, mu_init, wllr, H_dev = \
            self._device_tensors(dev)
        ebit, cbit, gamma_all, wllr_dt = self._mega_tensors(dev)
        S = int(syn_t.shape[0])
        syn_u8 = syn_t.to(torch.uint8).contiguous()

        mu = torch.empty((S, E), dtype=self._tdt, device=dev)
        nu = torch.empty((S, E), dtype=self._tdt, device=dev)
        M = torch.empty((S, N), dtype=self._tdt, device=dev)
        best_eh = torch.empty((S, N), dtype=self._tdt, device=dev)
        best_w = torch.empty((S,), dtype=self._tdt, device=dev)
        nconv = torch.empty((S,), dtype=torch.int32, device=dev)
        legs = torch.empty((S,), dtype=torch.int32, device=dev)
        iters = torch.empty((S,), dtype=torch.int32, device=dev)

        dt = tl.float64 if self.dtype == "float64" else tl.float32
        n_legs = 1 + self.num_sets
        _relay_megakernel[(S,)](
            mu, nu, M, best_eh, best_w, nconv, legs, iters,
            syn_u8, lam, wllr_dt, gamma_all,
            cedge, cbit, bedge, ebit,
            S, E, N, C,
            n_legs, self.pre_iter, self.set_max_iter, self.stop_nconv,
            self.ms,
            MAXDEG_C=self.MAXDEG_C, MAXDEG_B=self.MAXDEG_B,
            BLOCK=self.block, INF=_INF, DT=dt,
            EARLY_EXIT=1 if self.early_exit else 0,
        )
        self.last_stats = dict(
            nconv=nconv.cpu().numpy(),
            legs=legs.cpu().numpy(),
            iters=iters.cpu().numpy(),
            best_w=best_w.cpu().numpy(),
        )
        return best_eh.t()                                   # (N, S) view
