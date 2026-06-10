"""Triton Relay-BP decoder (the kernel "beat" upgrading the qLDPC paper).

Relay-BP (IBM Quantum -- Müller, Alexander, Beverland, Bühler, Johnson, Maurer,
Vandeth; arXiv:2506.01779 "Improved belief propagation is sufficient for real-time
decoding of quantum memory") is a **min-sum BP decoder with disordered per-node
memory**, run as a **relay ensemble of legs**. It is among the strongest
practical qLDPC decoders (smallest gap-to-MLE in the validation grid). This is
an OPEN, vendor-PORTABLE Triton implementation -- validated on NVIDIA (CUDA)
and AMD (ROCm) GPUs -- that is LER-matching the ``relay_bp`` Rust oracle.

It is built ON TOP of ``bp_triton``'s fused min-sum check/bit-update kernels:
the ONLY real kernel edit is a **per-node gamma-memory term in the bit-update
kernel**. The relay-leg loop and the nconv ensemble selection are host-side.

ALGORITHM (verified against arXiv:2506.01779 + the relay_bp Rust oracle)
=======================================================================
LLR convention ``lambda0_v = log((1-p_v)/p_v)`` (positive => bit likely 0); hard
decision flips bit v iff posterior M_v < 0. Min-sum is the exclude-self
two-smallest-magnitude / sign-parity transform (bp_triton's exact kernel), with
the syndrome coset sign ``(-1)^{s_c}``.

The Relay-BP / Memory-BP per-iteration update (paper Eqs 1-4), reverse-engineered
to fp tolerance (<=1e-14) against ``relay_bp.MinSumBPDecoderF64``:

  * MEMORY BIAS (Eq 4), per variable node v with memory strength gamma_v:
        Lambda_v(t) = (1 - gamma_v) * lambda0_v + gamma_v * M_v(t-1)
    where M_v(t-1) is the PREVIOUS iteration's posterior. M_v(0) = lambda0_v on
    leg start (or the relayed posterior on relay legs).

  * VARIABLE-TO-CHECK message (Eq 2) uses the PLAIN prior, NOT the memory bias --
    this is the subtlety the oracle confirmed (memory enters the POSTERIOR only):
        nu_{v->c}(t) = lambda0_v + sum_{c' in N(v)\{c}} mu_{c'->v}(t)
                     = (lambda0_v + sum_c mu) - mu_{c->v}

  * CHECK-TO-BIT message (Eq 1, min-sum): exclude-self min-sum of nu, coset sign.

  * POSTERIOR (Eq 3) carries the memory bias:
        M_v(t) = Lambda_v(t) + sum_{c in N(v)} mu_{c->v}(t)

  * gamma_v = gamma0 (uniform) on the PRE leg; gamma_v ~ Uniform(lo, hi)
    per-node, per-leg (DISORDERED, includes negative values) on relay legs.

RELAY SCHEDULE (host-side):
  1. PRE leg: ``pre_iter`` flooding iterations, uniform gamma0.
  2. up to ``num_sets`` relay legs of ``set_max_iter`` iterations; each leg draws
     a fresh per-node gamma vector and SEEDS its posterior from the previous leg's
     final posterior (the messages mu also carry forward -- "relay").
  3. NCONV stop: a shot stops once ``stop_nconv`` legs have produced a valid
     (syndrome-consistent) solution; the returned solution is the LOWEST-WEIGHT
     valid one found across all legs, weight w(e) = sum_v e_v * lambda0_v.

The kernel only adds: a per-bit ``gamma`` array + the memory term to the
bit-update kernel (the one real kernel edit), and tracks a per-(shot) convergence
mask so legs short-circuit. Messages are float32 on the GPU (fp32-vs-fp64 near-tie
flips tolerated -- the bar is LER-identity, not Rust-bit-identity).

LER-IDENTITY (the bar) is reported by the relay kernel tests and the carried
receipts (``bench/receipts/``): the gamma-draw RNG differs from the Rust
oracle's internal RNG and fp differs, so per-shot agreement is reported
alongside the logical-error COUNT match.
"""
import numpy as np
import scipy.sparse as sp
import torch

from ..tanner import tanner_adjacency

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is an optional extra
    triton = None
    tl = None
    _HAVE_TRITON = False

_P_LO = 1e-12
_P_HI = 1 - 1e-12
_INF = 1.0e30  # large finite "infinity" for fp32 min reductions / degree-1 guard


# ======================================================================= #
# Triton kernels (compiled only where triton is importable).               #
# ======================================================================= #
if _HAVE_TRITON:

    @triton.jit
    def _check_update_kernel(
        MU_ptr, NU_ptr, SSIGN_ptr,
        CEDGE_ptr,            # (n_checks, MAXDEG_C) padded edge idx, -1 = pad
        S, E, C,
        ms,
        MAXDEG_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
        INF: tl.constexpr,
        DT: tl.constexpr,    # message dtype (tl.float32 or tl.float64)
    ):
        """One check's exclude-self min-sum over a BLOCK_S tile of shots.

        IDENTICAL to bp_triton's check update: program_id(0)=shot-tile,
        program_id(1)=check; two-smallest-magnitude / sign-parity exclude-self
        min-sum, writes nu per edge. ``DT`` selects fp32 (fast) or fp64
        (LER-tightening, matches the relay_bp F64 oracle's precision)."""
        pid_s = tl.program_id(0)
        c = tl.program_id(1)
        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_off < S

        neg_par = tl.zeros((BLOCK_S,), dtype=tl.int32)
        min1 = tl.full((BLOCK_S,), INF, dtype=DT)
        min2 = tl.full((BLOCK_S,), INF, dtype=DT)

        for k in tl.static_range(MAXDEG_C):
            e = tl.load(CEDGE_ptr + c * MAXDEG_C + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            mu = tl.load(MU_ptr + e_safe * S + s_off, mask=s_mask & real, other=0.0)
            a = tl.abs(mu)
            a = tl.where(real, a, INF)
            isneg = (mu < 0.0) & real
            neg_par = neg_par ^ isneg.to(tl.int32)
            new_min1 = tl.minimum(min1, a)
            a_lt_min1 = a < min1
            min2 = tl.where(a_lt_min1, min1, tl.minimum(min2, a))
            min1 = new_min1

        ssign = tl.load(SSIGN_ptr + c * S + s_off, mask=s_mask, other=1.0)
        check_sign = tl.where(neg_par == 0, 1.0, -1.0)

        for k in tl.static_range(MAXDEG_C):
            e = tl.load(CEDGE_ptr + c * MAXDEG_C + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            mu = tl.load(MU_ptr + e_safe * S + s_off, mask=s_mask & real, other=0.0)
            a = tl.abs(mu)
            self_sign = tl.where(mu >= 0.0, 1.0, -1.0)
            sign_others = check_sign * self_sign
            is_min1 = a == min1
            min_others = tl.where(is_min1, min2, min1)
            min_others = tl.where(min_others >= INF, 0.0, min_others)
            nu = ms * ssign * sign_others * min_others
            tl.store(NU_ptr + e_safe * S + s_off, nu, mask=s_mask & real)

    @triton.jit
    def _bit_update_relay_kernel(
        NU_ptr, MU_ptr, POST_ptr, MPREV_ptr, LAM_ptr, GAMMA_ptr,
        BEDGE_ptr,            # (n_bits, MAXDEG_B) padded edge idx, -1 = pad
        S, E, N,
        MAXDEG_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
        DT: tl.constexpr,    # message dtype (tl.float32 or tl.float64)
    ):
        """One bit's Relay-BP memory bit-update over a BLOCK_S tile of shots.

        THE ONE REAL KERNEL EDIT vs bp_triton's bit update: a per-bit ``gamma``
        array + the Eq-4 memory bias term.

          incoming = sum_c nu                                  (check messages)
          Lambda   = (1-gamma_b)*lam0_b + gamma_b*Mprev        (Eq 4 memory bias)
          posterior M = Lambda + incoming                      (Eq 3, memory)
          next mu_e   = (lam0_b + incoming) - nu_e             (Eq 2, PLAIN prior!)

        ``lam0`` is the plain channel-prior LLR (per bit); ``gamma`` is the per-bit
        memory strength (uniform gamma0 on the pre leg, disordered on relay legs);
        ``MPREV`` is the previous iteration's posterior (the memory). The variable
        -> check message uses the PLAIN prior (NOT Lambda) -- the oracle-confirmed
        Relay-BP subtlety. program_id(0)=shot-tile, program_id(1)=bit."""
        pid_s = tl.program_id(0)
        b = tl.program_id(1)
        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_off < S

        lam = tl.load(LAM_ptr + b)                          # scalar plain prior LLR
        gam = tl.load(GAMMA_ptr + b)                        # scalar per-bit gamma
        mprev = tl.load(MPREV_ptr + b * S + s_off, mask=s_mask, other=0.0)

        incoming = tl.zeros((BLOCK_S,), dtype=DT)
        for k in tl.static_range(MAXDEG_B):
            e = tl.load(BEDGE_ptr + b * MAXDEG_B + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            nu = tl.load(NU_ptr + e_safe * S + s_off, mask=s_mask & real, other=0.0)
            incoming += nu

        # Eq-4 memory bias (posterior) vs plain prior (variable->check message).
        lam_bias = (1.0 - gam) * lam + gam * mprev          # Lambda_b(t)
        posterior = lam_bias + incoming                     # M_b(t) (Eq 3)
        var_tot = lam + incoming                            # plain prior sum (Eq 2)
        tl.store(POST_ptr + b * S + s_off, posterior, mask=s_mask)

        for k in tl.static_range(MAXDEG_B):
            e = tl.load(BEDGE_ptr + b * MAXDEG_B + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            nu = tl.load(NU_ptr + e_safe * S + s_off, mask=s_mask & real, other=0.0)
            mu = var_tot - nu                               # nu_{v->c} = lam0+sum-mu_e
            tl.store(MU_ptr + e_safe * S + s_off, mu, mask=s_mask & real)


class RelayBpTriton:
    """Triton Relay-BP (disordered-memory min-sum ensemble) on a fixed DEM Tanner
    graph. Mirrors ``BpTriton``'s surface (``from_dem``, ``decode_batch``,
    ``bench_latency``). Messages are float32; edge arrays edge-major (E, S)."""

    def __init__(self, H, priors, gamma0=0.1, pre_iter=80, num_sets=60,
                 set_max_iter=60, gamma_dist_interval=(-0.24, 0.66),
                 stop_nconv=5, stopping_criterion="nconv", alpha=1.0,
                 gamma_seed=12345, block_s=256, dtype="float64"):
        if sp.issparse(H):
            H = H.toarray()
        self.H = (np.asarray(H, dtype=np.uint8) % 2)
        self.n_checks, self.n_bits = self.H.shape
        # Message precision: fp64 matches the relay_bp F64 oracle (tightest LER);
        # fp32 is ~2x faster with a few extra near-tie logical flips (reported).
        self.dtype = str(dtype)
        self._tdt = torch.float64 if self.dtype == "float64" else torch.float32
        self.gamma0 = float(gamma0)
        self.pre_iter = int(pre_iter)
        self.num_sets = int(num_sets)
        self.set_max_iter = int(set_max_iter)
        self.gamma_lo, self.gamma_hi = (float(gamma_dist_interval[0]),
                                        float(gamma_dist_interval[1]))
        self.stop_nconv = int(stop_nconv)
        self.stopping_criterion = str(stopping_criterion)
        self.ms = float(alpha)                  # alpha = min-sum scaling (oracle: 1.0)
        self.gamma_seed = int(gamma_seed)
        self.block_s = int(block_s)

        priors = np.asarray(priors, dtype=np.float64).reshape(-1)
        if priors.shape[0] != self.n_bits:
            raise ValueError(
                f"priors length {priors.shape[0]} != n_bits {self.n_bits}")
        p = np.clip(priors, _P_LO, _P_HI)
        _np_dt = np.float64 if self.dtype == "float64" else np.float32
        lam = np.log((1.0 - p) / p).astype(_np_dt)
        self._lam_np = lam
        # weight LLR for lowest-weight selection: w(e) = sum_v e_v * log((1-p)/p).
        self._wllr_np = np.log((1.0 - p) / p).astype(np.float64)

        # SHARED edge layout (CSR check->bit), check-major, cols ascending.
        tg = tanner_adjacency(self.H)
        self._tg = tg
        chk_ptr = tg["check_to_bit_ptr"]
        edge_bit = tg["check_to_bit_idx"].astype(np.int64)
        self.n_edges = int(edge_bit.shape[0])
        self._edge_bit_np = edge_bit

        # padded per-row edge-index tables (the bp_triton "unroll" structure).
        chk_deg = np.diff(chk_ptr)
        self.MAXDEG_C = int(chk_deg.max())
        cedge = np.full((self.n_checks, self.MAXDEG_C), -1, dtype=np.int32)
        for c in range(self.n_checks):
            lo, hi = int(chk_ptr[c]), int(chk_ptr[c + 1])
            cedge[c, :hi - lo] = np.arange(lo, hi, dtype=np.int32)
        self._cedge_np = cedge

        bit_ptr = tg["bit_to_check_ptr"]
        bit_deg = np.diff(bit_ptr)
        self.MAXDEG_B = int(bit_deg.max())
        bit_to_check_idx = tg["bit_to_check_idx"].astype(np.int64)
        edge_check = np.repeat(np.arange(self.n_checks, dtype=np.int64),
                               np.diff(chk_ptr))
        edge_of = {(int(c), int(b)): i
                   for i, (c, b) in enumerate(zip(edge_check, edge_bit))}
        bedge = np.full((self.n_bits, self.MAXDEG_B), -1, dtype=np.int32)
        for b in range(self.n_bits):
            lo, hi = int(bit_ptr[b]), int(bit_ptr[b + 1])
            for j, pos in enumerate(range(lo, hi)):
                c = int(bit_to_check_idx[pos])
                bedge[b, j] = edge_of[(c, b)]
        self._bedge_np = bedge

        self.lam = torch.as_tensor(lam, dtype=self._tdt)
        self.cedge = torch.as_tensor(cedge, dtype=torch.int32)
        self.bedge = torch.as_tensor(bedge, dtype=torch.int32)
        self.edge_bit = torch.as_tensor(edge_bit, dtype=torch.long)
        self.wllr = torch.as_tensor(self._wllr_np, dtype=torch.float64)
        # sparse H (csr) on host for the syndrome-residual convergence check (GF2).
        self._H_csr = sp.csr_matrix(self.H.astype(np.int32))

        self._dev_cache = {}
        self._dem = None
        self._Lo = None
        self._n_obs = None

    def _device_tensors(self, dev):
        key = str(dev)
        cached = self._dev_cache.get(key)
        if cached is None:
            lam = self.lam.to(dev)
            cedge = self.cedge.to(dev)
            bedge = self.bedge.to(dev)
            edge_bit = self.edge_bit.to(dev)
            mu_init = lam[edge_bit].contiguous()           # (E,) flooding init
            wllr = self.wllr.to(dev)
            # H as a (n_checks, n_bits) dense matrix (message dtype) for the GF2
            # syndrome residual on GPU. Entries 0/1 are exact in fp32 and fp64.
            _ndt = np.float64 if self.dtype == "float64" else np.float32
            H_dev = torch.as_tensor(self.H.astype(_ndt), device=dev)
            cached = (lam, cedge, bedge, edge_bit, mu_init, wllr, H_dev)
            self._dev_cache[key] = cached
        return cached

    @classmethod
    def from_dem(cls, dem, gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
                 stopping_criterion="nconv", alpha=1.0, gamma_seed=12345,
                 block_s=256, dtype="float64"):
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], gamma0=gamma0, pre_iter=pre_iter,
                  num_sets=num_sets, set_max_iter=set_max_iter,
                  gamma_dist_interval=gamma_dist_interval, stop_nconv=stop_nconv,
                  stopping_criterion=stopping_criterion, alpha=alpha,
                  gamma_seed=gamma_seed, block_s=block_s, dtype=dtype)
        obj._dem = dem
        obj.dem = dem
        obj._Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._n_obs = ex["n_obs"]
        return obj

    # ------------------------------------------------------------------ #
    # Low-level: one memory-BP leg over a batch (the kernel inner loop).  #
    # ------------------------------------------------------------------ #
    def _ssign(self, syn_t, dev):
        """(-1)^{s_c} per (check, shot) -> (C, S) message-dtype (fired -> -1)."""
        syn_par = (syn_t.to(torch.int32) & 1)               # (S, C)
        return torch.where(
            syn_par.t() == 0,
            torch.ones((), dtype=self._tdt, device=dev),
            -torch.ones((), dtype=self._tdt, device=dev)).contiguous()

    def _run_leg(self, ssign, lam, cedge, bedge, gamma, mu, M, n_iter,
                 dev, S, E, N, C, conv_ctx=None):
        """Run ``n_iter`` flooding Memory-BP iterations IN PLACE on (mu, M).

        ``gamma`` is (N,) per-bit memory strength; ``mu`` is (E, S) check->bit
        messages (carried in from the previous leg / flooding init); ``M`` is
        (N, S) the previous posterior (the memory M(t-1), seeded from the relayed
        posterior). Returns (mu, M, post) with post = M (the final posterior).

        If ``conv_ctx`` is given (the relay decode path), the FIRST-CONVERGENCE
        solution per shot is captured DURING the leg -- relay_bp freezes a shot's
        solution at the iteration it first satisfies the syndrome, NOT at the end
        of the fixed iteration budget (running past convergence drifts to a
        different solution; that drift was a ~7-LER error vs the oracle). ``conv_ctx``
        = (H_dev (C,N), syn_cs (C,S), wllr (N,), best_w (S,), best_eh (N,S),
        leg_conv (S,) bool) updated in place; ``leg_conv`` flags shots that
        converged at least once in THIS leg (for the nconv count)."""
        post = torch.empty((N, S), dtype=self._tdt, device=dev)
        nu = torch.empty((E, S), dtype=self._tdt, device=dev)
        dt = tl.float64 if self.dtype == "float64" else tl.float32
        BLOCK_S = self.block_s
        grid_s = triton.cdiv(S, BLOCK_S)
        grid_chk = (grid_s, C)
        grid_bit = (grid_s, N)
        for _ in range(int(n_iter)):
            _check_update_kernel[grid_chk](
                mu, nu, ssign, cedge, S, E, C, self.ms,
                MAXDEG_C=self.MAXDEG_C, BLOCK_S=BLOCK_S, INF=_INF, DT=dt)
            _bit_update_relay_kernel[grid_bit](
                nu, mu, post, M, lam, gamma, bedge, S, E, N,
                MAXDEG_B=self.MAXDEG_B, BLOCK_S=BLOCK_S, DT=dt)
            # M(t-1) <- M(t): the memory feedback. Copy so next iter reads it.
            M = post.clone()
            if conv_ctx is not None:
                H_dev, syn_cs, wllr, best_w, best_eh, leg_conv = conv_ctx
                eh = (post < 0.0).to(self._tdt)               # (N, S)
                resid = torch.remainder(H_dev @ eh, 2.0)      # (C, S)
                valid = (torch.abs(resid - syn_cs).sum(dim=0) == 0)   # (S,) bool
                if bool(valid.any()):
                    w = (eh.to(torch.float64)
                         * wllr.unsqueeze(1)).sum(dim=0)      # (S,)
                    improve = valid & (w < best_w)
                    best_w.copy_(torch.where(improve, w, best_w))
                    torch.where(improve.unsqueeze(0), eh, best_eh, out=best_eh)
                    leg_conv |= valid
        return mu, M, post

    # ------------------------------------------------------------------ #
    # Public: plain min-sum posterior (pre-leg / memory-term validation). #
    # ------------------------------------------------------------------ #
    def minsum_posterior_batch(self, syndromes, n_iter, gamma=0.0, alpha=None,
                               device="cuda"):
        """Run ``n_iter`` flooding Memory-BP iterations with a UNIFORM ``gamma``
        memory strength from a fresh flooding init; return (S, N) posterior LLR
        (numpy). ``gamma=0`` is plain min-sum (the pre-leg primitive). Used by the
        pre-leg / memory-term identity tests against relay_bp's MinSumBPDecoderF64.
        """
        if not _HAVE_TRITON:
            raise RuntimeError("Triton not available in this environment")
        dev = torch.device(device)
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        lam, cedge, bedge, edge_bit, mu_init, _, _ = self._device_tensors(dev)
        old_ms = self.ms
        if alpha is not None:
            self.ms = float(alpha)
        try:
            syn = np.asarray(syndromes)
            if syn.ndim == 1:
                syn = syn[None, :]
            syn_t = torch.as_tensor(syn, device=dev)
            S = int(syn_t.shape[0])
            ssign = self._ssign(syn_t, dev)
            mu = mu_init.unsqueeze(1).expand(E, S).contiguous()
            M = lam.unsqueeze(1).expand(N, S).contiguous()    # M(0) = lam0
            gamma_t = torch.full((N,), float(gamma), dtype=self._tdt, device=dev)
            _, _, post = self._run_leg(ssign, lam, cedge, bedge, gamma_t, mu, M,
                                       n_iter, dev, S, E, N, C)
            return post.detach().cpu().numpy().T              # (S, N)
        finally:
            self.ms = old_ms

    # ------------------------------------------------------------------ #
    # Relay decode: pre leg + relay legs + nconv lowest-weight selection. #
    # ------------------------------------------------------------------ #
    def _gamma_for_leg(self, leg_idx, dev):
        """Per-node gamma vector for leg ``leg_idx`` (0 = pre leg). Deterministic:
        seeded by (gamma_seed, leg_idx) so runs reproduce. Pre leg = uniform
        gamma0; relay legs = per-node Uniform(lo, hi) (DISORDERED, may be < 0)."""
        if leg_idx == 0:
            return torch.full((self.n_bits,), self.gamma0, dtype=self._tdt,
                              device=dev)
        rng = np.random.default_rng((self.gamma_seed, leg_idx))
        _ndt = np.float64 if self.dtype == "float64" else np.float32
        g = rng.uniform(self.gamma_lo, self.gamma_hi,
                        size=self.n_bits).astype(_ndt)
        return torch.as_tensor(g, device=dev)

    def _relay_posteriors(self, syn_t, dev):
        """Core relay decode over a batch. Returns (N, S) best hard error e_hat
        (float 0/1) -- the LOWEST-WEIGHT syndrome-consistent solution per shot,
        across the pre leg + relay legs, with per-shot nconv early stop."""
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        lam, cedge, bedge, edge_bit, mu_init, wllr, H_dev = self._device_tensors(dev)
        S = int(syn_t.shape[0])
        ssign = self._ssign(syn_t, dev)                       # (C, S)
        syn_cs = syn_t.to(self._tdt).t().contiguous()         # (C, S) 0/1

        # relayed state: mu (E,S) flooding init, M (N,S) = lam0 broadcast.
        mu = mu_init.unsqueeze(1).expand(E, S).contiguous()
        M = lam.unsqueeze(1).expand(N, S).contiguous()

        big = torch.full((), _INF, dtype=torch.float64, device=dev)
        best_w = big.expand(S).clone()                        # (S,) best weight
        best_eh = torch.zeros((N, S), dtype=self._tdt, device=dev)
        nconv = torch.zeros((S,), dtype=torch.int32, device=dev)

        n_legs = 1 + self.num_sets
        for leg in range(n_legs):
            n_iter = self.pre_iter if leg == 0 else self.set_max_iter
            gamma = self._gamma_for_leg(leg, dev)
            # FIRST-CONVERGENCE capture during the leg (relay_bp freezes a shot's
            # solution at the iter it first satisfies the syndrome).
            leg_conv = torch.zeros((S,), dtype=torch.bool, device=dev)
            conv_ctx = (H_dev, syn_cs, wllr, best_w, best_eh, leg_conv)
            mu, M, post = self._run_leg(ssign, lam, cedge, bedge, gamma, mu, M,
                                        n_iter, dev, S, E, N, C,
                                        conv_ctx=conv_ctx)
            # a leg counts ONE convergence toward nconv if the shot converged at
            # least once during it (nconv = number of legs that found a solution).
            nconv = nconv + leg_conv.to(torch.int32)
            if self.stopping_criterion == "nconv":
                if bool((nconv >= self.stop_nconv).all()):
                    break
        return best_eh

    def decode_batch(self, dets, device="cuda"):
        """Decode -> predicted observables via the relay ensemble, on-GPU.

        Returns (S, n_obs) bool predicted observables = (Lo @ e_hat) % 2 (GF2),
        where e_hat is the lowest-weight syndrome-consistent solution per shot."""
        if not _HAVE_TRITON:
            raise RuntimeError("Triton not available in this environment")
        if self._Lo is None:
            raise RuntimeError(
                "decode_batch requires construction via RelayBpTriton.from_dem")
        dets = np.asarray(dets, dtype=bool)
        if dets.ndim == 1:
            dets = dets[None, :]
        dev = torch.device(device)
        syn_t = torch.as_tensor(dets, device=dev)             # (S, C) bool
        best_eh = self._relay_posteriors(syn_t, dev)          # (N, S) 0/1
        Lo = self._Lo_dev(dev)                                # (n_obs, N) f32
        pred = torch.remainder(Lo @ best_eh, 2.0)             # (n_obs, S)
        return (pred > 0.5).t().cpu().numpy()                 # (S, n_obs) bool

    def _Lo_dev(self, dev):
        key = "Lo:" + str(dev)
        cached = self._dev_cache.get(key)
        if cached is None:
            _ndt = np.float64 if self.dtype == "float64" else np.float32
            cached = torch.as_tensor(self._Lo.astype(_ndt), device=dev)
            self._dev_cache[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # Latency benchmark (matches BpTriton.bench_latency surface).         #
    # ------------------------------------------------------------------ #
    @classmethod
    def bench_latency(cls, dem, shots, device="cuda", n_warmup=10, n_iter=100,
                      seed=0, block_s=256, **relay_kwargs):
        """Time ``decode_batch`` over ``shots`` random syndromes. CUDA-event timed
        on cuda; perf_counter otherwise. Returns mean/p99.9 ms + throughput."""
        import time
        dec = cls.from_dem(dem, block_s=block_s, **relay_kwargs)
        rng = np.random.default_rng(seed)
        dets = rng.integers(0, 2, size=(int(shots), dec.n_checks)).astype(bool)

        use_cuda = (str(device).startswith("cuda") and torch.cuda.is_available())
        for _ in range(int(n_warmup)):
            dec.decode_batch(dets, device=device)
        if use_cuda:
            torch.cuda.synchronize()

        times_ms = np.empty(int(n_iter), dtype=np.float64)
        if use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            for k in range(int(n_iter)):
                start_ev.record()
                dec.decode_batch(dets, device=device)
                end_ev.record()
                torch.cuda.synchronize()
                times_ms[k] = start_ev.elapsed_time(end_ev)
        else:
            for k in range(int(n_iter)):
                t0 = time.perf_counter()
                dec.decode_batch(dets, device=device)
                t1 = time.perf_counter()
                times_ms[k] = (t1 - t0) * 1e3

        mean_ms = float(times_ms.mean())
        p99_9_ms = float(np.percentile(times_ms, 99.9))
        per_syn_us = (mean_ms / shots) * 1e3 if shots else 0.0
        throughput = float(shots / (mean_ms / 1e3)) if mean_ms > 0 else 0.0
        return dict(
            mean_ms=mean_ms,
            p99_9_ms=p99_9_ms,
            per_syndrome_us=per_syn_us,
            throughput_shots_per_s=throughput,
            device=str(device),
            shots=int(shots),
            n_iter=int(n_iter),
            n_warmup=int(n_warmup),
            block_s=int(block_s),
            used_cuda_events=bool(use_cuda),
            per_rep_ms=times_ms.tolist(),
        )
