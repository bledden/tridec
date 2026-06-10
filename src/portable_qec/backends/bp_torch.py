"""Device-agnostic torch edge-list BP (CPU / CUDA / ROCm via ``device=``).

A torch normalized-min-sum belief-propagation decoder on the SAME CSR/CSC edge
layout ``tanner.tanner_adjacency`` produces and ``bp_numpy.BpBaseline``
consumes -- but **batched over shots** and runnable on CPU or GPU with only
``device=``. It reproduces ``BpBaseline`` BIT-IDENTICALLY for one iteration
(same four-phase normalized-min-sum math, same two-smallest-magnitude exclude-self
min, same sign-product-of-others, same coset sign). This is the honest torch
baseline and the latency reference the Triton kernel is benchmarked against.

The per-iteration update mirrors ``BpBaseline`` exactly, expressed over the flat
edge array (length nnz = number of Tanner edges) and batched along a leading shot
axis so the whole batch decodes in one vectorized pass:

  (1) GATHER       -- ``mu`` is the per-(shot, edge) bit->check message.
  (2) EXCLUDE-SELF -- outgoing message on edge e excludes the message on e:
        * sign of OTHERS = (product of all incident signs) * sign(self), carried
          as parity of negatives per (shot, check) -- exact GF-sign.
        * min magnitude of OTHERS = two-smallest-magnitudes trick: min2 on the
          edge holding the (a) global min, else min1.
  (3) PER-CHECK min-sum transform: nu = ms * (-1)^{s_c} * sign_others * min_others.
  (4) SCATTER + bit update: L_v = lambda_v + sum_c nu ; next mu_e = L_{bit(e)} - nu_e.

LLR convention: ``lambda_v = log(P(x_v=0)/P(x_v=1))`` (positive => bit likely 0);
hard decision flips bit v iff posterior LLR < 0. Schedule: parallel (flooding),
init mu = prior LLR. Defaults match ``BpBaseline``: ``ms_scaling_factor=0.625``, ``max_iter=30``.

torch + numpy only -- no triton/cuda-specifics beyond the optional ``device=``
move and the ``torch.cuda.Event`` timing path (guarded behind ``device=="cuda"``).
"""
import numpy as np
import scipy.sparse as sp
import torch

from ..tanner import tanner_adjacency

# Match BpBaseline's prior-LLR clip exactly (np.clip(p, 1e-12, 1 - 1e-12)).
_P_LO = 1e-12
_P_HI = 1 - 1e-12


class BpGpu:
    """Batched normalized-min-sum BP on a fixed Tanner graph (flooding), in torch.

    Args:
        H: parity-check matrix (n_checks x n_bits), dense array or scipy sparse.
        priors: per-bit (per-error-mechanism) probabilities, length n_bits.
        max_iter: BP iterations for ``decode_batch``.
        ms_scaling_factor: normalized-min-sum scale (0.625 matches BpBaseline).
        dtype: torch float dtype for the messages (float64 to match numpy exactly).
    """

    def __init__(self, H, priors, max_iter=30, ms_scaling_factor=0.625,
                 dtype=torch.float64):
        if sp.issparse(H):
            H = H.toarray()
        self.H = (np.asarray(H, dtype=np.uint8) % 2)
        self.n_checks, self.n_bits = self.H.shape
        self.max_iter = int(max_iter)
        self.ms = float(ms_scaling_factor)
        self.dtype = dtype

        priors = np.asarray(priors, dtype=np.float64).reshape(-1)
        if priors.shape[0] != self.n_bits:
            raise ValueError(
                f"priors length {priors.shape[0]} != n_bits {self.n_bits}")
        p = np.clip(priors, _P_LO, _P_HI)
        lam = np.log((1.0 - p) / p)            # prior LLRs, length n_bits

        # SHARED edge-list layout (CSR check->bit), check-major, cols ascending.
        tg = tanner_adjacency(self.H)
        self._tg = tg
        chk_ptr = tg["check_to_bit_ptr"]
        edge_bit = tg["check_to_bit_idx"]              # bit index per edge
        self.n_edges = int(edge_bit.shape[0])
        edge_check = np.repeat(np.arange(self.n_checks, dtype=np.int64),
                               np.diff(chk_ptr))       # check index per edge

        # Keep numpy copies (for trace dict ordering) + torch index tensors.
        self._edge_bit_np = edge_bit.astype(np.int64)
        self._edge_check_np = edge_check.astype(np.int64)
        self._lam_np = lam
        # (check, bit) -> edge index, for the trace / hand-derived tests.
        self._edge_of = {(int(c), int(b)): i
                         for i, (c, b) in enumerate(zip(edge_check, edge_bit))}

        # torch index tensors built on construction (moved to device on demand).
        self.edge_bit = torch.as_tensor(self._edge_bit_np, dtype=torch.long)
        self.edge_check = torch.as_tensor(self._edge_check_np, dtype=torch.long)
        self.lam = torch.as_tensor(lam, dtype=self.dtype)

        self._dem = None
        self._Lo = None
        self._n_obs = None

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dem(cls, dem, max_iter=30, ms_scaling_factor=0.625,
                 dtype=torch.float64):
        """Build from a stim DEM via ``dem.extract`` (matched protocol).

        Stores ``.dem`` and the observable map ``Lo`` so ``decode_batch`` returns
        predicted observables, mirroring BpBaseline (the shared adapter surface).
        """
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], max_iter=max_iter,
                  ms_scaling_factor=ms_scaling_factor, dtype=dtype)
        obj._dem = dem
        obj.dem = dem
        Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._Lo = Lo
        obj._n_obs = ex["n_obs"]
        return obj

    # ------------------------------------------------------------------ #
    # Device-resident index tensors                                      #
    # ------------------------------------------------------------------ #
    def _on(self, device):
        """Return (edge_bit, edge_check, lam) index/param tensors on ``device``."""
        dev = torch.device(device)
        return (self.edge_bit.to(dev), self.edge_check.to(dev),
                self.lam.to(dev))

    # ------------------------------------------------------------------ #
    # Batched four-phase iteration (the kernel target), shots x edges    #
    # ------------------------------------------------------------------ #
    def _iterate_batch(self, mu, syndrome_sign, edge_bit, edge_check, lam):
        """One flooding iteration on a batch.

        Shapes: mu (B, E) bit->check messages; syndrome_sign (B, C) per-check
        coset sign (-1)^{s_c}; edge_bit/edge_check (E,); lam (n_bits,).
        Returns (mu_next (B,E), nu (B,E), posterior (B, n_bits)) -- mirroring
        BpBaseline._check_update + _bit_update exactly.
        """
        B = mu.shape[0]
        C = self.n_checks
        E = self.n_edges
        N = self.n_bits
        dev = mu.device
        dt = mu.dtype

        absmu = torch.abs(mu)
        # sign(0) := +1 (convention), matching np.where(mu >= 0, 1, -1).
        sgn = torch.where(mu >= 0.0,
                          torch.ones((), dtype=dt, device=dev),
                          -torch.ones((), dtype=dt, device=dev))

        # --- sign product over all incident edges of each check -------------
        # neg parity per (batch, check): scatter-add (mu < 0) into a (B, C) grid.
        neg = (mu < 0.0).to(dt)                                  # (B, E)
        neg_count = torch.zeros((B, C), dtype=dt, device=dev)
        idx_c = edge_check.unsqueeze(0).expand(B, E)             # (B, E)
        neg_count.scatter_add_(1, idx_c, neg)                    # (B, C)
        # check_sign = (-1)^neg_count : +1 if even count, -1 if odd.
        parity = torch.remainder(neg_count, 2.0)                 # 0.0 or 1.0
        check_sign = 1.0 - 2.0 * parity                          # (B, C)
        # sign of OTHERS = check_sign / sign(self) = check_sign * sign(self).
        sign_others = check_sign.gather(1, idx_c) * sgn          # (B, E)

        # --- two smallest magnitudes per check (exclude-self min) -----------
        INF = torch.full((), float("inf"), dtype=dt, device=dev)
        min1 = torch.full((B, C), float("inf"), dtype=dt, device=dev)
        min1.scatter_reduce_(1, idx_c, absmu, reduce="amin",
                             include_self=True)                  # (B, C)
        min1_e = min1.gather(1, idx_c)                           # (B, E)
        # exact-equality mask of the global-min edge(s) (atol=0 in BpBaseline).
        is_min1 = (absmu == min1_e)                              # (B, E) bool
        absmu_masked = torch.where(is_min1, INF, absmu)          # (B, E)
        min2 = torch.full((B, C), float("inf"), dtype=dt, device=dev)
        min2.scatter_reduce_(1, idx_c, absmu_masked, reduce="amin",
                             include_self=True)                  # (B, C)
        min2_e = min2.gather(1, idx_c)                           # (B, E)
        # min of OTHERS: if this edge holds the (a) global min use min2 else min1.
        min_others = torch.where(is_min1, min2_e, min1_e)        # (B, E)
        # Degree-1 checks (no "others") -> +inf -> min-sum sends 0.
        min_others = torch.where(torch.isfinite(min_others), min_others,
                                 torch.zeros((), dtype=dt, device=dev))

        # syndrome_sign per edge = syndrome_sign[:, check(e)].
        ssign_e = syndrome_sign.gather(1, idx_c)                 # (B, E)
        nu = self.ms * ssign_e * sign_others * min_others        # (B, E)

        # --- bit update (phase 4): scatter nu into bits, posterior, next mu --
        incoming = torch.zeros((B, N), dtype=dt, device=dev)
        idx_b = edge_bit.unsqueeze(0).expand(B, E)               # (B, E)
        incoming.scatter_add_(1, idx_b, nu)                      # (B, N)
        posterior = lam.unsqueeze(0) + incoming                  # (B, N)
        mu_next = posterior.gather(1, idx_b) - nu                # (B, E)
        return mu_next, nu, posterior

    def run_iterations_batch(self, syndromes, n_iter=None, device="cpu",
                             return_messages=False):
        """Run ``n_iter`` flooding iterations on a batch of syndromes.

        Args:
            syndromes: (B, n_checks) array of 0/1 syndromes (or 1-D for B=1).
            n_iter: iterations (defaults to ``self.max_iter``).
            device: "cpu" now / "cuda" later -- index + LLR tensors move here.
            return_messages: also return final per-edge mu/nu (numpy, B x E).

        Returns the (B, n_bits) posterior LLR array (numpy). If
        ``return_messages``, returns a dict with posterior/hard/mu/nu.
        """
        if n_iter is None:
            n_iter = self.max_iter
        syn = np.asarray(syndromes, dtype=np.int64)
        if syn.ndim == 1:
            syn = syn[None, :]
        syn = syn % 2
        if syn.shape[1] != self.n_checks:
            raise ValueError(
                f"syndrome width {syn.shape[1]} != n_checks {self.n_checks}")
        B = syn.shape[0]

        edge_bit, edge_check, lam = self._on(device)
        dev = lam.device
        dt = self.dtype

        # Per-check coset sign (-1)^{s_c}: fired check inverts agreement.
        syn_t = torch.as_tensor(syn, dtype=torch.long, device=dev)
        syndrome_sign = torch.where(
            syn_t == 0,
            torch.ones((), dtype=dt, device=dev),
            -torch.ones((), dtype=dt, device=dev))               # (B, C)

        # Flooding init: every bit->check message = the bit's prior LLR.
        mu = lam[edge_bit].unsqueeze(0).expand(B, self.n_edges).clone()  # (B,E)
        nu = torch.zeros((B, self.n_edges), dtype=dt, device=dev)
        posterior = lam.unsqueeze(0).expand(B, self.n_bits).clone()

        for _ in range(int(n_iter)):
            mu, nu, posterior = self._iterate_batch(
                mu, syndrome_sign, edge_bit, edge_check, lam)

        post_np = posterior.detach().cpu().numpy()
        if not return_messages:
            return post_np
        hard = (post_np < 0.0).astype(np.uint8)
        return dict(posterior=post_np, hard=hard,
                    mu=mu.detach().cpu().numpy(),
                    nu=nu.detach().cpu().numpy())

    def run_iterations(self, syndrome, n_iter=None, device="cpu",
                       return_messages=False):
        """Single-syndrome convenience wrapper around ``run_iterations_batch``.

        With ``return_messages`` returns the BpBaseline-shaped trace dict:
        posterior (n_bits,), hard (n_bits,), and per-edge ``check_to_bit`` /
        ``bit_to_check`` dicts keyed by (check, bit). ``n_iter=0`` returns the
        flooding init state (bit->check = prior LLRs)."""
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

    # ------------------------------------------------------------------ #
    # Decode surfaces                                                    #
    # ------------------------------------------------------------------ #
    def decode_batch(self, dets, device="cpu"):
        """Decode a batch of detector syndromes -> predicted observables.

        Same interface as BpBaseline.decode_batch (the shared adapter surface): each shot's
        hard error estimate e_hat maps to observables via ``(Lo @ e_hat) % 2``
        (GF2). Requires construction via ``from_dem`` (so ``Lo`` is available).
        """
        if self._Lo is None:
            raise RuntimeError(
                "decode_batch requires construction via BpGpu.from_dem "
                "(no observable map Lo otherwise)")
        dets = np.asarray(dets, dtype=bool)
        if dets.ndim == 1:
            dets = dets[None, :]
        shots = dets.shape[0]
        syn = dets.astype(np.uint8)
        post = self.run_iterations_batch(syn, n_iter=self.max_iter,
                                         device=device)          # (S, n_bits)
        e_hat = (post < 0.0).astype(np.uint8)                    # (S, n_bits)
        # Predicted observables (GF2): (Lo @ e_hat^T) % 2, then transpose.
        pred = (self._Lo @ e_hat.T) & 1                          # (n_obs, S)
        return pred.T.astype(bool)                               # (S, n_obs)

    # ------------------------------------------------------------------ #
    # Latency benchmark (CUDA-event timed on GPU; CPU path works too)     #
    # ------------------------------------------------------------------ #
    @classmethod
    def bench_latency(cls, dem, shots, device="cuda", n_warmup=10, n_iter=100,
                      max_iter=30, ms_scaling_factor=0.625, seed=0):
        """Time ``decode_batch`` over ``shots`` random syndromes.

        Uses ``torch.cuda.Event`` timing when ``device=="cuda"`` (guarded so it
        imports/runs on CPU, where it falls back to ``perf_counter``). Returns
        mean / p99.9 latency (ms) and throughput (shots/s). Speed is NOT asserted
        by tests; this is a measurement utility.
        """
        import time

        dec = cls.from_dem(dem, max_iter=max_iter,
                           ms_scaling_factor=ms_scaling_factor)
        rng = np.random.default_rng(seed)
        # Random 0/1 detector syndromes of the right width (latency is data-
        # independent for flooding min-sum; correctness lives in the TDD tests).
        dets = rng.integers(0, 2, size=(int(shots), dec.n_checks)).astype(bool)

        use_cuda = (str(device).startswith("cuda")
                    and torch.cuda.is_available())

        # Warmup.
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
                times_ms[k] = start_ev.elapsed_time(end_ev)  # ms
        else:
            for k in range(int(n_iter)):
                t0 = time.perf_counter()
                dec.decode_batch(dets, device=device)
                t1 = time.perf_counter()
                times_ms[k] = (t1 - t0) * 1e3

        mean_ms = float(times_ms.mean())
        p99_9_ms = float(np.percentile(times_ms, 99.9))
        # Throughput from mean per-batch latency.
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
            used_cuda_events=bool(use_cuda),
            # Per-rep CUDA-event sample window (ms) -- see BpTriton.bench_latency.
            per_rep_ms=times_ms.tolist(),
        )
