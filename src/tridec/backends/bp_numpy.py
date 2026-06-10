"""Portable normalized-min-sum BP reference (pure numpy).

A from-scratch, pure-numpy normalized-min-sum belief-propagation decoder on the
Tanner graph, built on the SAME edge-list / CSR-CSC layout that
``tanner.tanner_adjacency`` produces and that the Triton kernel consumes. The
per-iteration update is written as the four GPU-portable phases the kernel
fuses:

  (1) GATHER       -- collect each check's incident bit->check messages,
  (2) EXCLUDE-SELF -- each outgoing message excludes the message on its own edge,
  (3) PER-CHECK TRANSFORM (min-sum) -- outgoing = scale * (sign-product of others)
                       * (min magnitude of others), with the syndrome's coset sign,
  (4) SCATTER      -- write check->bit messages back to the bit-update accumulator;
                       bit-update = prior LLR + sum of incoming check messages.

Everything is vectorized over the flat *edge array* (length nnz = number of Tanner
edges); the exclude-self min uses the standard two-smallest-magnitudes trick, and
exclude-self sum uses (total - self). That edge-indexed gather/scatter is exactly
the shape a GPU kernel needs, so this is the CPU reference the GPU backends
match (bit-identically for one iteration at fp64).

LLR convention: ``lambda_v = log(P(x_v=0)/P(x_v=1))`` (positive => bit likely 0).
The hard decision flips bit v iff its posterior LLR is < 0.

Schedule: parallel (flooding). Validated default config:
``ms_scaling_factor=0.625``, ``max_iter=30``, ``schedule='parallel'``.

Pure numpy -- no triton/torch/cuda.
"""
import numpy as np
import scipy.sparse as sp

from ..tanner import tanner_adjacency


class BpBaseline:
    """Normalized-min-sum BP on a fixed Tanner graph (parallel/flooding schedule).

    Args:
        H: parity-check matrix (ndet x n_err), dense array or scipy sparse.
        priors: per-bit (per-error-mechanism) probabilities, length n_err.
        max_iter: BP iterations for ``decode`` / ``decode_batch``.
        ms_scaling_factor: normalized-min-sum scale (0.625, validated default).
        schedule: only ``"parallel"`` (flooding) is supported here.
    """

    def __init__(self, H, priors, max_iter=30, ms_scaling_factor=0.625,
                 schedule="parallel"):
        if schedule != "parallel":
            raise ValueError("BpBaseline supports only schedule='parallel'")
        if sp.issparse(H):
            H = H.toarray()
        self.H = (np.asarray(H, dtype=np.uint8) % 2)
        self.n_checks, self.n_bits = self.H.shape
        self.max_iter = int(max_iter)
        self.ms = float(ms_scaling_factor)
        self.schedule = schedule

        priors = np.asarray(priors, dtype=np.float64).reshape(-1)
        if priors.shape[0] != self.n_bits:
            raise ValueError(
                f"priors length {priors.shape[0]} != n_bits {self.n_bits}")
        # Clip for numerical stability, then prior LLR lambda = log((1-p)/p).
        p = np.clip(priors, 1e-12, 1 - 1e-12)
        self.lam = np.log((1.0 - p) / p)

        # SHARED edge-list layout (CSR check->bit). The flat edge array is ordered
        # check-major (row by row), columns ascending within a check.
        tg = tanner_adjacency(self.H)
        self._tg = tg
        self.chk_ptr = tg["check_to_bit_ptr"]
        self.edge_bit = tg["check_to_bit_idx"]          # bit index per edge
        self.n_edges = int(self.edge_bit.shape[0])
        # check index per edge (expand CSR row pointers).
        self.edge_check = np.repeat(np.arange(self.n_checks, dtype=np.int64),
                                    np.diff(self.chk_ptr))
        # (check, bit) -> edge index, for the hand-derived trace test.
        self._edge_of = {(int(c), int(b)): i
                         for i, (c, b) in enumerate(zip(self.edge_check,
                                                        self.edge_bit))}

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dem(cls, dem, max_iter=30, ms_scaling_factor=0.625,
                 schedule="parallel"):
        """Build from a stim DEM via ``dem.extract`` (matched protocol).

        Stores ``.dem`` and the observable map ``Lo`` so ``decode_batch`` returns
        predicted observables (the shared adapter surface).
        """
        from ..dem import extract
        ex = extract(dem)
        obj = cls(ex["H"], priors=ex["priors"], max_iter=max_iter,
                  ms_scaling_factor=ms_scaling_factor, schedule=schedule)
        obj.dem = dem
        obj._Lo = ex["Lo"].toarray().astype(np.uint8)
        obj._n_obs = ex["n_obs"]
        return obj

    # ------------------------------------------------------------------ #
    # One-iteration four-phase primitive (the kernel target)            #
    # ------------------------------------------------------------------ #
    def _check_update(self, mu, syndrome_sign):
        """Phases (1)-(3): from bit->check messages ``mu`` (per edge) compute the
        check->bit messages ``nu`` (per edge), excluding self, via min-sum.

        - sign-product of OTHERS = (product of all incident signs) / sign(self),
          carried as parity of negatives (exact in GF-sign), times the per-check
          syndrome coset sign.
        - min magnitude of OTHERS = two-smallest-magnitudes trick: min2 on the
          edge that holds the global min1, else min1.
        """
        ec = self.edge_check
        absmu = np.abs(mu)
        sgn = np.where(mu >= 0.0, 1.0, -1.0)        # sign(0) := +1 (convention)

        # --- sign product over all incident edges of each check -------------
        neg = (mu < 0.0).astype(np.int64)
        neg_count = np.zeros(self.n_checks, dtype=np.int64)
        np.add.at(neg_count, ec, neg)
        check_sign = np.where(neg_count % 2 == 0, 1.0, -1.0)   # prod of all signs
        # sign of OTHERS = check_sign / sign(self) = check_sign * sign(self).
        sign_others = check_sign[ec] * sgn

        # --- two smallest magnitudes per check ------------------------------
        INF = np.inf
        min1 = np.full(self.n_checks, INF)
        np.minimum.at(min1, ec, absmu)
        # mask out the global-min edge(s) and find the second min.
        is_min1 = np.isclose(absmu, min1[ec], rtol=0, atol=0.0)
        absmu_masked = np.where(is_min1, INF, absmu)
        min2 = np.full(self.n_checks, INF)
        np.minimum.at(min2, ec, absmu_masked)
        # min of OTHERS: if this edge holds the (a) global min, use min2 else min1.
        min_others = np.where(is_min1, min2[ec], min1[ec])
        # Degree-1 checks (no "others") -> min_others is +inf; min-sum sends 0.
        min_others = np.where(np.isfinite(min_others), min_others, 0.0)

        nu = self.ms * syndrome_sign[ec] * sign_others * min_others
        return nu

    def _bit_update(self, nu):
        """Phase (4) scatter + bit update: posterior L_v = lambda_v + sum_c nu, and
        the next bit->check message excludes self: mu_e = L_{bit(e)} - nu_e."""
        eb = self.edge_bit
        incoming = np.zeros(self.n_bits, dtype=np.float64)
        np.add.at(incoming, eb, nu)
        posterior = self.lam + incoming
        mu = posterior[eb] - nu
        return mu, posterior

    def run_iterations(self, syndrome, n_iter=None, return_messages=False):
        """Run ``n_iter`` flooding iterations from the given syndrome.

        Returns a dict with ``posterior`` (per-bit LLR), ``hard`` (per-bit GF2
        decision), and -- if ``return_messages`` -- the per-edge ``bit_to_check``
        and ``check_to_bit`` message arrays addressable by ``(check, bit)``.

        ``n_iter=0`` returns the initial state (bit->check = prior LLRs), which the
        tests use to pin the gather phase's starting point.
        """
        if n_iter is None:
            n_iter = self.max_iter
        syndrome = np.asarray(syndrome, dtype=np.int64).reshape(-1) % 2
        if syndrome.shape[0] != self.n_checks:
            raise ValueError(
                f"syndrome length {syndrome.shape[0]} != n_checks {self.n_checks}")
        # Per-check coset sign factor (-1)^{s_c}: a fired check inverts the
        # agreement of its bits (standard min-sum syndrome handling).
        syndrome_sign = np.where(syndrome == 0, 1.0, -1.0)

        # Phase-0 init (flooding): every bit->check message = the bit's prior LLR.
        mu = self.lam[self.edge_bit].copy()
        nu = np.zeros(self.n_edges, dtype=np.float64)
        posterior = self.lam.copy()

        for _ in range(int(n_iter)):
            nu = self._check_update(mu, syndrome_sign)      # (1)-(3)
            mu, posterior = self._bit_update(nu)            # (4)

        hard = (posterior < 0.0).astype(np.uint8)
        out = dict(posterior=posterior, hard=hard)
        if return_messages:
            out["bit_to_check"] = {key: float(mu[i])
                                   for key, i in self._edge_of.items()}
            out["check_to_bit"] = {key: float(nu[i])
                                   for key, i in self._edge_of.items()}
        return out

    # ------------------------------------------------------------------ #
    # Decode surfaces                                                    #
    # ------------------------------------------------------------------ #
    def decode(self, syndrome, max_iter=None):
        """Decode one syndrome to a hard error estimate (length n_bits, uint8)."""
        n_iter = self.max_iter if max_iter is None else int(max_iter)
        res = self.run_iterations(syndrome, n_iter=n_iter)
        return res["hard"]

    def decode_batch(self, dets):
        """Decode a batch of detector syndromes -> predicted observables.

        Same interface as the reference adapters: each shot's error estimate e_hat maps
        to observables via ``(Lo @ e_hat) % 2``. Requires construction via
        ``from_dem`` (so ``Lo`` is available).
        """
        if not hasattr(self, "_Lo"):
            raise RuntimeError(
                "decode_batch requires construction via BpBaseline.from_dem "
                "(no observable map Lo otherwise)")
        dets = np.asarray(dets, dtype=bool)
        shots = dets.shape[0]
        out = np.zeros((shots, self._n_obs), dtype=bool)
        syn = dets.astype(np.uint8)
        Lo = self._Lo
        for i in range(shots):
            e_hat = self.decode(syn[i])
            pred = (Lo @ e_hat.astype(np.uint8)) & 1
            out[i] = pred.astype(bool)
        return out
