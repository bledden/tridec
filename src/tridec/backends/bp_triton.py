"""Triton min-sum BP kernel (the GPU fast path; NVIDIA CUDA and AMD ROCm).

A Triton normalized-min-sum belief-propagation decoder that reproduces
``bp_numpy.BpBaseline`` / ``bp_torch.BpGpu`` for one flooding iteration and is
benchmarked against the torch edge-list baseline (``BpGpu``).

DESIGN -- "bounded-degree unroll", shots-major.
===============================================
The decoding target is the circuit-level **DEM** Tanner graph (from
``dem.extract``), which is irregular (e.g. for a [[72,12,6]] BB memory DEM the
check degrees are 13 or 20 and bit degrees are 2 or 3; n_checks=252,
n_bits=1584, nnz=4536). The kernel precomputes, per check (resp. per bit), the
set of its incident **edge indices padded to the max degree** (MAXDEG_C /
MAXDEG_B) with a sentinel, giving a *compile-time-bounded, regular* neighbor
loop per row with mask-on-load for the padding. This keeps the per-row access
indirection-free *within the kernel's bounded loop* and lets each program tile
over many shots (the large dimension), FP32 accumulate.

Two kernels, both gridded shots-major (the big axis), one BP iteration:

  CHECK-UPDATE (grid: shots-tiles x checks): for one check, load its padded
    incident bit->check messages ``mu`` over a BLOCK_S tile of shots, compute the
    exclude-self min-sum exactly as BpBaseline:
      * sign of OTHERS = (xor of all incident signs) xor sign(self)  [parity],
      * min magnitude of OTHERS = two-smallest-magnitudes (min2 on the edge that
        holds the global min1, else min1; degree-1 "others" -> +inf -> send 0),
      * nu_e = ms * (-1)^{s_c} * sign_others * min_others.
    Writes ``nu`` per (shot, edge) into the global edge array (scatter-free: each
    program owns its check's edges).

  BIT-UPDATE (grid: shots-tiles x bits): for one bit, gather its padded incident
    ``nu``, posterior = prior_llr + sum(nu); next mu_e = posterior - nu_e.

Float32 messages on the GPU (the baseline torch path uses float64; near-tie
flips on the two-smallest-magnitude min are expected and tolerated -- the gate is
>=99.5% hard-decision agreement + LER within noise, NOT bit-identity).

The Python wrapper ``BpTriton`` mirrors ``BpGpu``'s surface (``from_dem``,
``run_iterations_batch``, ``decode_batch``, ``bench_latency``) so the same
benchmark harness applies apples-to-apples (same shots / iters / timing).
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
    ):
        """One check's exclude-self min-sum over a BLOCK_S tile of shots.

        program_id(0) = shot-tile, program_id(1) = check. Loads the check's
        up-to-MAXDEG_C incident mu (mask on pad), computes the two-smallest-
        magnitude / sign-parity exclude-self min-sum, writes nu per edge.
        """
        pid_s = tl.program_id(0)
        c = tl.program_id(1)

        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)   # (BLOCK_S,)
        s_mask = s_off < S

        # First pass: accumulate sign parity (count of negatives, mod 2) and the
        # two smallest magnitudes over the (real) incident edges, per shot.
        neg_par = tl.zeros((BLOCK_S,), dtype=tl.int32)
        min1 = tl.full((BLOCK_S,), INF, dtype=tl.float32)
        min2 = tl.full((BLOCK_S,), INF, dtype=tl.float32)

        for k in tl.static_range(MAXDEG_C):
            e = tl.load(CEDGE_ptr + c * MAXDEG_C + k)     # scalar edge idx or -1
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            mu = tl.load(MU_ptr + e_safe * S + s_off,
                         mask=s_mask & real, other=0.0)    # (BLOCK_S,)
            a = tl.abs(mu)
            a = tl.where(real, a, INF)                     # pad never wins min
            # sign parity: count mu<0 on real edges.
            isneg = (mu < 0.0) & real
            neg_par = neg_par ^ isneg.to(tl.int32)
            # update two smallest magnitudes.
            new_min1 = tl.minimum(min1, a)
            # if a < min1: min2 becomes old min1; else min2 = min(min2, a)
            a_lt_min1 = a < min1
            min2 = tl.where(a_lt_min1, min1, tl.minimum(min2, a))
            min1 = new_min1

        # Per-check coset sign (-1)^{s_c}, broadcast over the shot tile.
        ssign = tl.load(SSIGN_ptr + c * S + s_off, mask=s_mask, other=1.0)
        check_sign = tl.where(neg_par == 0, 1.0, -1.0)     # product of ALL signs

        # Second pass: emit nu_e = ms * ssign * sign_others * min_others, where
        # sign_others = check_sign * sign(self); min_others = (self holds global
        # min) ? min2 : min1, with degree-1 "others" (+inf) -> 0.
        for k in tl.static_range(MAXDEG_C):
            e = tl.load(CEDGE_ptr + c * MAXDEG_C + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            mu = tl.load(MU_ptr + e_safe * S + s_off,
                         mask=s_mask & real, other=0.0)
            a = tl.abs(mu)
            self_sign = tl.where(mu >= 0.0, 1.0, -1.0)     # sign(0):=+1
            sign_others = check_sign * self_sign
            # is this edge (one of) the global min? exact-equality match.
            is_min1 = a == min1
            min_others = tl.where(is_min1, min2, min1)
            min_others = tl.where(min_others >= INF, 0.0, min_others)
            nu = ms * ssign * sign_others * min_others
            tl.store(NU_ptr + e_safe * S + s_off,
                     nu, mask=s_mask & real)

    @triton.jit
    def _bit_update_kernel(
        NU_ptr, MU_ptr, POST_ptr, LAM_ptr,
        BEDGE_ptr,            # (n_bits, MAXDEG_B) padded edge idx, -1 = pad
        S, E, N,
        MAXDEG_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """One bit's scatter+update over a BLOCK_S tile of shots.

        posterior = prior_llr + sum_c nu ; next mu_e = posterior - nu_e.
        program_id(0) = shot-tile, program_id(1) = bit.
        """
        pid_s = tl.program_id(0)
        b = tl.program_id(1)
        s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_off < S

        lam = tl.load(LAM_ptr + b)                          # scalar prior LLR
        incoming = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for k in tl.static_range(MAXDEG_B):
            e = tl.load(BEDGE_ptr + b * MAXDEG_B + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            nu = tl.load(NU_ptr + e_safe * S + s_off,
                         mask=s_mask & real, other=0.0)
            incoming += nu
        posterior = lam + incoming
        tl.store(POST_ptr + b * S + s_off, posterior, mask=s_mask)
        # write next mu on each incident edge: mu_e = posterior - nu_e.
        for k in tl.static_range(MAXDEG_B):
            e = tl.load(BEDGE_ptr + b * MAXDEG_B + k)
            real = e >= 0
            e_safe = tl.where(real, e, 0)
            nu = tl.load(NU_ptr + e_safe * S + s_off,
                         mask=s_mask & real, other=0.0)
            mu = posterior - nu
            tl.store(MU_ptr + e_safe * S + s_off, mu, mask=s_mask & real)


class BpTriton:
    """Triton normalized-min-sum BP on a fixed DEM Tanner graph (flooding).

    Mirrors ``BpGpu``'s surface so the same benchmark harness applies. Messages
    are float32 on the GPU. Edge arrays are stored EDGE-MAJOR (E, S) so the
    shot tile is the contiguous fast axis (coalesced loads across a warp).
    """

    def __init__(self, H, priors, max_iter=30, ms_scaling_factor=0.625,
                 block_s=256, cuda_graph=False):
        if sp.issparse(H):
            H = H.toarray()
        self.H = (np.asarray(H, dtype=np.uint8) % 2)
        self.n_checks, self.n_bits = self.H.shape
        self.max_iter = int(max_iter)
        self.ms = float(ms_scaling_factor)
        self.block_s = int(block_s)
        # #4: opt-in CUDA-graph fast path for the launch-bound small/repeated-batch
        # regime. Captures the fixed n_iter kernel loop + Lo projection per batch
        # shape and replays it (eliminates per-launch overhead); bit-identical to
        # the eager path. Cache keyed by batch size S; disabled automatically if
        # capture ever fails (e.g. uncapturable driver/op).
        self._cuda_graph = bool(cuda_graph)
        self._graphs = {}

        priors = np.asarray(priors, dtype=np.float64).reshape(-1)
        if priors.shape[0] != self.n_bits:
            raise ValueError(
                f"priors length {priors.shape[0]} != n_bits {self.n_bits}")
        p = np.clip(priors, _P_LO, _P_HI)
        lam = np.log((1.0 - p) / p).astype(np.float32)

        # SHARED edge layout (CSR check->bit), check-major, cols ascending.
        tg = tanner_adjacency(self.H)
        self._tg = tg
        chk_ptr = tg["check_to_bit_ptr"]
        edge_bit = tg["check_to_bit_idx"].astype(np.int64)
        self.n_edges = int(edge_bit.shape[0])
        edge_check = np.repeat(np.arange(self.n_checks, dtype=np.int64),
                               np.diff(chk_ptr))
        self._edge_bit_np = edge_bit
        self._edge_check_np = edge_check
        self._lam_np = lam
        self._edge_of = {(int(c), int(b)): i
                         for i, (c, b) in enumerate(zip(edge_check, edge_bit))}

        # ---- padded per-row edge-index tables (the "unroll" structure) -------
        # check -> its incident edge indices, in the SAME edge order as the flat
        # edge array (check-major), padded to MAXDEG_C with -1.
        chk_deg = np.diff(chk_ptr)
        self.MAXDEG_C = int(chk_deg.max())
        cedge = np.full((self.n_checks, self.MAXDEG_C), -1, dtype=np.int32)
        for c in range(self.n_checks):
            lo, hi = int(chk_ptr[c]), int(chk_ptr[c + 1])
            cedge[c, :hi - lo] = np.arange(lo, hi, dtype=np.int32)
        self._cedge_np = cedge

        # bit -> its incident edge indices (CSC order), padded to MAXDEG_B.
        bit_ptr = tg["bit_to_check_ptr"]
        bit_deg = np.diff(bit_ptr)
        self.MAXDEG_B = int(bit_deg.max())
        # Map each bit's incident edges back to flat edge indices: build via the
        # (check, bit)->edge dict using the CSC traversal.
        bit_to_check_idx = tg["bit_to_check_idx"].astype(np.int64)  # checks per bit
        bedge = np.full((self.n_bits, self.MAXDEG_B), -1, dtype=np.int32)
        for b in range(self.n_bits):
            lo, hi = int(bit_ptr[b]), int(bit_ptr[b + 1])
            for j, pos in enumerate(range(lo, hi)):
                c = int(bit_to_check_idx[pos])
                bedge[b, j] = self._edge_of[(c, b)]
        self._bedge_np = bedge

        self.lam = torch.as_tensor(lam, dtype=torch.float32)
        self.cedge = torch.as_tensor(cedge, dtype=torch.int32)
        self.bedge = torch.as_tensor(bedge, dtype=torch.int32)
        self.edge_bit = torch.as_tensor(edge_bit, dtype=torch.long)

        # device-resident static-tensor cache (built lazily per device) so the
        # hot decode path moves index/LLR tensors to the GPU exactly once.
        self._dev_cache = {}

        self._dem = None
        self._Lo = None
        self._n_obs = None

    def _device_tensors(self, dev):
        """Cache + return (lam, cedge, bedge, edge_bit, mu_init) on ``dev``.

        ``mu_init`` is the flooding-init bit->check message per edge (lam gathered
        through edge_bit), shape (E,), broadcast over shots at launch."""
        key = str(dev)
        cached = self._dev_cache.get(key)
        if cached is None:
            lam = self.lam.to(dev)
            cedge = self.cedge.to(dev)
            bedge = self.bedge.to(dev)
            edge_bit = self.edge_bit.to(dev)
            mu_init = lam[edge_bit].contiguous()           # (E,)
            cached = (lam, cedge, bedge, edge_bit, mu_init)
            self._dev_cache[key] = cached
        return cached

    @classmethod
    def from_dem(cls, dem, max_iter=30, ms_scaling_factor=0.625, block_s=256):
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], max_iter=max_iter,
                  ms_scaling_factor=ms_scaling_factor, block_s=block_s)
        obj._dem = dem
        obj.dem = dem
        obj._Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._n_obs = ex["n_obs"]
        return obj

    # ------------------------------------------------------------------ #
    # Batched iteration via the Triton kernels.                          #
    # ------------------------------------------------------------------ #
    def run_iterations_batch(self, syndromes, n_iter=None, device="cuda",
                             return_messages=False):
        """Run ``n_iter`` flooding iterations on a batch. Returns (B, n_bits)
        posterior LLR (numpy). Edge arrays are (E, S) device tensors."""
        if not _HAVE_TRITON:
            raise RuntimeError("Triton not available in this environment")
        if n_iter is None:
            n_iter = self.max_iter
        dev = torch.device(device)
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        lam, cedge, bedge, edge_bit, mu_init = self._device_tensors(dev)

        # Accept numpy or a device tensor; build the (C, S) coset-sign grid
        # entirely on-device to keep the hot path off the host.
        if isinstance(syndromes, torch.Tensor):
            syn_t = syndromes.to(dev)
            if syn_t.ndim == 1:
                syn_t = syn_t[None, :]
        else:
            syn = np.asarray(syndromes)
            if syn.ndim == 1:
                syn = syn[None, :]
            syn_t = torch.as_tensor(syn, device=dev)
        if syn_t.shape[1] != C:
            raise ValueError(
                f"syndrome width {int(syn_t.shape[1])} != n_checks {C}")
        S = int(syn_t.shape[0])
        # (-1)^{s_c} per (check, shot): fired check (odd) -> -1, else +1.
        syn_par = (syn_t.to(torch.int32) & 1)                  # (S, C)
        ssign = torch.where(syn_par.t() == 0,                  # (C, S)
                            torch.ones((), dtype=torch.float32, device=dev),
                            -torch.ones((), dtype=torch.float32, device=dev))
        ssign = ssign.contiguous()

        post, mu, nu = self._run_core(syn_t, ssign, lam, cedge, bedge,
                                      mu_init, n_iter, dev, S, E, N, C)

        post_np = post.detach().cpu().numpy().T  # (S, N)
        if not return_messages:
            return post_np
        hard = (post_np < 0.0).astype(np.uint8)
        return dict(posterior=post_np, hard=hard,
                    mu=mu.detach().cpu().numpy().T,   # (S, E)
                    nu=nu.detach().cpu().numpy().T)

    def _run_core(self, syn_t, ssign, lam, cedge, bedge, mu_init,
                  n_iter, dev, S, E, N, C):
        """Device-resident iteration core: returns (post, mu, nu) on ``dev``.

        ``post`` is (N, S), ``mu``/``nu`` are (E, S). No host transfers -- the
        callers decide whether to copy to host (run_iterations_batch) or keep on
        device (decode_batch's GPU observable projection)."""
        # Edge-major message buffers (E, S): shot is the contiguous fast axis.
        nu = torch.empty((E, S), dtype=torch.float32, device=dev)
        post = torch.empty((N, S), dtype=torch.float32, device=dev)
        # Flooding init: mu_e = prior LLR of bit(e), for all shots.
        mu = mu_init.unsqueeze(1).expand(E, S).contiguous()   # (E, S)

        BLOCK_S = self.block_s
        grid_s = triton.cdiv(S, BLOCK_S)
        grid_chk = (grid_s, C)
        grid_bit = (grid_s, N)

        for _ in range(int(n_iter)):
            _check_update_kernel[grid_chk](
                mu, nu, ssign, cedge,
                S, E, C, self.ms,
                MAXDEG_C=self.MAXDEG_C, BLOCK_S=BLOCK_S, INF=_INF,
            )
            _bit_update_kernel[grid_bit](
                nu, mu, post, lam, bedge,
                S, E, N,
                MAXDEG_B=self.MAXDEG_B, BLOCK_S=BLOCK_S,
            )
        return post, mu, nu

    # ---- #4: CUDA-graph fast path (opt-in via cuda_graph=True) ----------------
    def _build_graph(self, dev, S):
        """Capture the fixed n_iter BP loop + Lo projection for batch size S into
        a CUDA graph over static tensors. Per-decode the only input copied in is
        ``ssign`` (staged through a pinned host buffer); ``pred`` is the captured
        output. Bit-identical to ``_run_core`` + the eager Lo projection."""
        import triton as _triton
        lam, cedge, bedge, _eb, mu_init = self._device_tensors(dev)
        Lo = self._Lo_dev(dev)
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        BLOCK_S = self.block_s
        grid_s = _triton.cdiv(S, BLOCK_S); grid_chk = (grid_s, C); grid_bit = (grid_s, N)
        mu_init_exp = mu_init.unsqueeze(1).expand(E, S).contiguous()
        mu = torch.empty((E, S), dtype=torch.float32, device=dev)
        nu = torch.empty((E, S), dtype=torch.float32, device=dev)
        post = torch.empty((N, S), dtype=torch.float32, device=dev)
        ssign = torch.zeros((C, S), dtype=torch.float32, device=dev)
        pin = torch.empty((S, C), dtype=torch.uint8, pin_memory=True)  # H2D staging

        def core():
            mu.copy_(mu_init_exp)
            for _ in range(int(self.max_iter)):
                _check_update_kernel[grid_chk](mu, nu, ssign, cedge, S, E, C, self.ms,
                                               MAXDEG_C=self.MAXDEG_C, BLOCK_S=BLOCK_S, INF=_INF)
                _bit_update_kernel[grid_bit](nu, mu, post, lam, bedge, S, E, N,
                                             MAXDEG_B=self.MAXDEG_B, BLOCK_S=BLOCK_S)
            return torch.remainder(Lo @ (post < 0.0).to(torch.float32), 2.0)

        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                core()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            pred = core()
        return {"graph": gr, "ssign": ssign, "pred": pred, "pin": pin, "C": C}

    def _decode_graphed(self, dets, dev):
        """dets: (S, C) bool -> predicted observables (n_obs, S) via graph replay."""
        S = int(dets.shape[0])
        g = self._graphs.get(S)
        if g is None:
            g = self._build_graph(dev, S)
            self._graphs[S] = g
        g["pin"].copy_(torch.from_numpy(np.ascontiguousarray(dets, dtype=np.uint8)))
        syn = g["pin"].to(dev, non_blocking=True)              # (S, C) uint8
        one = torch.ones((), dtype=torch.float32, device=dev)
        g["ssign"].copy_(torch.where((syn.to(torch.int32) & 1).t() == 0, one, -one))
        g["graph"].replay()
        return g["pred"]

    def run_iterations(self, syndrome, n_iter=None, device="cuda",
                       return_messages=False):
        out = self.run_iterations_batch(
            syndrome, n_iter=n_iter, device=device, return_messages=True)
        posterior = out["posterior"][0]
        hard = out["hard"][0]
        res = dict(posterior=posterior, hard=hard)
        if return_messages:
            mu = out["mu"][0]
            nu = out["nu"][0]
            res["bit_to_check"] = {key: float(mu[i])
                                   for key, i in self._edge_of.items()}
            res["check_to_bit"] = {key: float(nu[i])
                                   for key, i in self._edge_of.items()}
        return res

    def decode_batch(self, dets, device="cuda"):
        """Decode -> predicted observables, fully on-GPU.

        Runs the Triton BP core on-device, takes the hard decision, then projects
        through the observable map ``Lo`` on the GPU (int32 matmul, mod 2) so the
        ONLY host transfers are the (S, n_checks) syndrome up and the small
        (S, n_obs) prediction down -- the float posterior never leaves the GPU.
        Result equals ``(Lo @ e_hat) % 2`` exactly (GF2)."""
        if self._Lo is None:
            raise RuntimeError(
                "decode_batch requires construction via BpTriton.from_dem")
        dets = np.asarray(dets, dtype=bool)
        if dets.ndim == 1:
            dets = dets[None, :]
        dev = torch.device(device)
        if self._cuda_graph and dev.type == "cuda" and torch.cuda.is_available():
            try:
                pred = self._decode_graphed(dets, dev)          # (n_obs, S)
                return (pred > 0.5).t().cpu().numpy()
            except Exception:
                self._cuda_graph = False  # capture/replay unsupported -> eager path
        E, N, C = self.n_edges, self.n_bits, self.n_checks
        lam, cedge, bedge, edge_bit, mu_init = self._device_tensors(dev)
        Lo = self._Lo_dev(dev)                                  # (n_obs, N) int32

        syn_t = torch.as_tensor(dets, device=dev)              # (S, C) bool
        S = int(syn_t.shape[0])
        syn_par = syn_t.to(torch.int32) & 1
        ssign = torch.where(
            syn_par.t() == 0,
            torch.ones((), dtype=torch.float32, device=dev),
            -torch.ones((), dtype=torch.float32, device=dev)).contiguous()  # (C,S)

        post, _, _ = self._run_core(syn_t, ssign, lam, cedge, bedge, mu_init,
                                    self.max_iter, dev, S, E, N, C)  # post (N,S)
        e_hat = (post < 0.0).to(torch.float32)                 # (N, S) 0/1
        # GF2 observable projection: (Lo @ e_hat) % 2. Entries are exact small
        # integers in float32 (n_bits << 2^24), so fmod(.,2) is exact.
        pred = torch.remainder(Lo @ e_hat, 2.0)                # (n_obs, S)
        return (pred > 0.5).t().cpu().numpy()                  # (S, n_obs) bool

    def _Lo_dev(self, dev):
        """Cache + return the observable map ``Lo`` (n_obs, n_bits) float32 on dev."""
        key = "Lo:" + str(dev)
        cached = self._dev_cache.get(key)
        if cached is None:
            cached = torch.as_tensor(self._Lo.astype(np.float32), device=dev)
            self._dev_cache[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # Latency benchmark (matches BpGpu.bench_latency exactly).            #
    # ------------------------------------------------------------------ #
    @classmethod
    def bench_latency(cls, dem, shots, device="cuda", n_warmup=10, n_iter=100,
                      max_iter=30, ms_scaling_factor=0.625, seed=0,
                      block_s=256):
        import time
        dec = cls.from_dem(dem, max_iter=max_iter,
                           ms_scaling_factor=ms_scaling_factor, block_s=block_s)
        rng = np.random.default_rng(seed)
        dets = rng.integers(0, 2, size=(int(shots), dec.n_checks)).astype(bool)

        use_cuda = (str(device).startswith("cuda")
                    and torch.cuda.is_available())

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
        throughput = float(shots / (mean_ms / 1e3)) if mean_ms > 0 else 0.0
        return dict(
            mean_ms=mean_ms,
            p99_9_ms=p99_9_ms,
            throughput_shots_per_s=throughput,
            device=str(device),
            shots=int(shots),
            n_iter=int(n_iter),
            n_warmup=int(n_warmup),
            max_iter=int(max_iter),
            block_s=int(block_s),
            used_cuda_events=bool(use_cuda),
            # Per-rep CUDA-event sample window (ms), all kept reps -- enables the
            # harness to bootstrap a 95% CI and to drop the first 10% uniformly
            # with the CPU path. The whole array is returned so nothing is
            # fabricated downstream.
            per_rep_ms=times_ms.tolist(),
        )
