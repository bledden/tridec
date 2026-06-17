"""Public decoder API: ``from_dem`` / ``from_matrices`` + backend dispatch.

Backends
--------
  * ``"numpy"``  — pure-numpy normalized-min-sum BP (always available; the
                   CPU reference the GPU paths are validated against).
  * ``"torch"``  — batched torch edge-list BP; bit-identical to numpy at fp64
                   for one iteration; runs on CPU and CUDA/ROCm devices.
  * ``"triton"`` — the Triton kernels (min-sum BP and Relay-BP); requires
                   triton + a CUDA or ROCm GPU. fp32 messages on the BP path
                   (>=99.5% hard-decision agreement vs the fp64 references,
                   LER-validated on H200 and MI300X — see bench/receipts/).
                   In the triton-metal environment (no CUDA/ROCm), a
                   ``"triton"`` request resolves to ``"metal"``.
  * ``"metal"``  — EXPERIMENTAL: the same Triton kernels on Apple silicon via
                   triton-metal (darwin + triton + triton_metal importable, no
                   CUDA/ROCm device). Execution pattern is CPU torch tensors
                   (zero-copy UMA — not mps); fp32 only (Metal has no fp64).
                   Spike-validated on M4 Max — see bench/receipts/metal_spike.md.
  * ``"auto"``   — triton if importable AND a CUDA/ROCm GPU is visible, else
                   metal if the triton-metal environment is detected, else
                   torch if importable, else numpy.

Algorithms per backend (honest availability matrix):

  ===========  =======  =======  ========  ==============
  algorithm     numpy    torch    triton    metal (exp.)
  ===========  =======  =======  ========  ==============
  bp (min-sum)   yes      yes      yes       yes (fp32)
  relay          no       no       yes       yes (fp32)
  ===========  =======  =======  ========  ==============

Relay-BP has no in-package CPU implementation; its CPU reference is IBM's
``relay-bp`` Rust decoder, available through ``tridec.adapters`` (the
``decoders`` extra) and used as the validation oracle for the Triton path.
"""
import sys
import warnings

import numpy as np
import scipy.sparse as sp

from .dem import extract

_BACKENDS = ("auto", "numpy", "torch", "triton", "metal")

# Validated defaults (the configuration the carried receipts were measured at).
_BP_DEFAULTS = dict(max_iter=30, ms_scaling_factor=0.625)
_RELAY_DEFAULTS = dict(
    gamma0=0.1,
    pre_iter=80,
    num_sets=60,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,
    stopping_criterion="nconv",
)


# --------------------------------------------------------------------------- #
# Backend availability / resolution.                                           #
# --------------------------------------------------------------------------- #
def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _triton_gpu_available():
    try:
        import triton  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _metal_available():
    """The EXPERIMENTAL triton-metal environment: darwin, triton AND
    triton_metal importable, torch present, and no CUDA/ROCm device visible.
    Execution pattern is CPU torch tensors (zero-copy UMA — not mps)."""
    if sys.platform != "darwin":
        return False
    try:
        import triton  # noqa: F401
        import triton_metal  # noqa: F401
        import torch
        return not torch.cuda.is_available()
    except Exception:
        return False


def available_backends():
    """The backends usable in THIS environment, best first."""
    out = []
    if _triton_gpu_available():
        out.append("triton")
    if _metal_available():
        out.append("metal")
    if _torch_available():
        out.append("torch")
    out.append("numpy")
    return out


def resolve_backend(backend="auto"):
    """Resolve a backend request to a concrete backend name.

    ``"auto"`` -> triton if importable AND a GPU (CUDA or ROCm) is visible,
    else metal if the triton-metal environment is detected (experimental),
    else torch if importable, else numpy. A ``"triton"`` request in the
    triton-metal environment resolves to ``"metal"`` (the same kernels, CPU
    tensors). Explicitly requesting an unavailable backend raises
    RuntimeError with the reason.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of {_BACKENDS}")
    if backend == "auto":
        if _triton_gpu_available():
            return "triton"
        if _metal_available():
            return "metal"
        if _torch_available():
            return "torch"
        return "numpy"
    if backend == "torch" and not _torch_available():
        raise RuntimeError(
            "torch backend requested but torch is not importable; "
            "install with the [torch] extra: pip install tridec[torch]")
    if backend == "triton" and not _triton_gpu_available():
        if _metal_available():
            return "metal"
        raise RuntimeError(
            "triton backend requested but triton + a CUDA/ROCm GPU are not "
            "available (triton importable: requires the [gpu] extra; GPU "
            "visible: torch.cuda.is_available() must be True). On Apple "
            "silicon, the experimental metal backend requires triton-metal "
            "— see README 'Experimental: Apple silicon (Metal)'.")
    if backend == "metal" and not _metal_available():
        raise RuntimeError(
            "metal backend requested but the triton-metal environment is not "
            "available (requires darwin + triton + triton-metal installed and "
            "no CUDA/ROCm device) — see README 'Experimental: Apple silicon "
            "(Metal)' for the install recipe.")
    return backend


_warned_metal = False


def _warn_metal_once():
    global _warned_metal
    if not _warned_metal:
        _warned_metal = True
        warnings.warn(
            "tridec metal backend is EXPERIMENTAL (triton-metal, fp32, CPU "
            "torch tensors); validated at spike level only — see "
            "bench/receipts/metal_spike.md", UserWarning, stacklevel=3)


def _default_device(backend, device):
    if device is not None:
        return device
    if backend == "triton":
        return "cuda"
    if backend == "metal":
        # triton-metal documented pattern: CPU tensors (zero-copy UMA), NOT mps.
        return "cpu"
    if backend == "torch":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "cpu"
    return "cpu"


def _dense_uint8(M):
    if M is None:
        return None
    if sp.issparse(M):
        M = M.toarray()
    return (np.asarray(M, dtype=np.uint8) % 2)


# --------------------------------------------------------------------------- #
# Decoders.                                                                    #
# --------------------------------------------------------------------------- #
class BpDecoder:
    """Normalized min-sum BP over the numpy / torch / triton backends.

    Construct via ``tridec.from_dem`` / ``tridec.from_matrices``
    (or directly). ``decode_batch(dets)`` returns predicted observables
    (bool[shots, n_obs]) when an observable map is available (always the case
    via ``from_dem``), else hard error estimates (uint8[shots, n_bits]).
    """

    algorithm = "bp"

    def __init__(self, H, priors, observables=None, backend="auto", device=None,
                 max_iter=30, ms_scaling_factor=0.625, block_s=256, dem=None,
                 cuda_graph="auto", cuda_graph_max_s=256, cuda_graph_cache=16):
        self.backend = resolve_backend(backend)
        self.device = _default_device(self.backend, device)
        self.dem = dem
        self.max_iter = int(max_iter)
        self.ms_scaling_factor = float(ms_scaling_factor)

        if self.backend == "numpy":
            from .backends.bp_numpy import BpBaseline
            self._impl = BpBaseline(H, priors, max_iter=max_iter,
                                    ms_scaling_factor=ms_scaling_factor)
        elif self.backend == "torch":
            from .backends.bp_torch import BpGpu
            self._impl = BpGpu(H, priors, max_iter=max_iter,
                               ms_scaling_factor=ms_scaling_factor)
        else:  # triton kernels: "triton" (CUDA/ROCm) or "metal" (triton-metal)
            if self.backend == "metal":
                _warn_metal_once()
            from .backends.bp_triton import BpTriton
            self._impl = BpTriton(H, priors, max_iter=max_iter,
                                  ms_scaling_factor=ms_scaling_factor,
                                  block_s=block_s, cuda_graph=cuda_graph,
                                  cuda_graph_max_s=cuda_graph_max_s,
                                  cuda_graph_cache=cuda_graph_cache)

        Lo = _dense_uint8(observables)
        self._Lo = Lo
        if Lo is not None:
            # Attach the observable map to the backend impl so its validated
            # decode_batch path (e_hat -> (Lo @ e_hat) % 2) applies unchanged.
            self._impl._Lo = Lo
            self._impl._n_obs = int(Lo.shape[0])
        self.n_obs = None if Lo is None else int(Lo.shape[0])
        self.n_bits = self._impl.n_bits
        self.n_checks = self._impl.n_checks

        self.name = f"portable-bp[{self.backend}]"
        self.tie_break = "min_sum_parallel_hard_decision"
        self.config = dict(
            decoder="tridec.BpDecoder", backend=self.backend,
            bp_method="minimum_sum", ms_scaling_factor=self.ms_scaling_factor,
            max_iter=self.max_iter, schedule="parallel")

    @classmethod
    def from_dem(cls, dem, backend="auto", device=None, **opts):
        kw = dict(_BP_DEFAULTS)
        kw.update(opts)
        ex = extract(dem)
        obj = cls(ex["H"], ex["priors"], observables=ex["Lo"], backend=backend,
                  device=device, dem=dem, **kw)
        return obj

    # -- decode surfaces ---------------------------------------------------- #
    def decode_batch(self, detection_events):
        """Decode a batch of detector-event vectors.

        Returns predicted observables (bool[shots, n_obs]) when an observable
        map is present, else hard error estimates (uint8[shots, n_bits]).
        """
        dets = np.asarray(detection_events)
        if dets.ndim == 1:
            dets = dets[None, :]
        if self._Lo is not None:
            if self.backend == "numpy":
                return self._impl.decode_batch(dets.astype(bool))
            return self._impl.decode_batch(dets.astype(bool), device=self.device)
        # No observable map: return hard error estimates.
        syn = dets.astype(np.uint8)
        if self.backend == "numpy":
            out = np.zeros((syn.shape[0], self.n_bits), dtype=np.uint8)
            for i in range(syn.shape[0]):
                out[i] = self._impl.decode(syn[i])
            return out
        post = self._impl.run_iterations_batch(syn, n_iter=self.max_iter,
                                               device=self.device)
        return (post < 0.0).astype(np.uint8)

    def decode(self, detection_events):
        """Single-shot convenience: 1-D in, 1-D out."""
        out = self.decode_batch(np.asarray(detection_events)[None, :])
        return out[0]


class RelayBpDecoder:
    """Relay-BP (disordered-memory min-sum relay ensemble), Triton kernels only
    (backend ``"triton"`` on CUDA/ROCm, or the experimental ``"metal"``).

    Defaults match the ``relay_bp`` Rust oracle configuration the kernels were
    LER-validated against (gamma0=0.1, pre_iter=80, num_sets=60,
    set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=5).

    Message precision: ``dtype=None`` (default) resolves to fp64 on CUDA/ROCm
    (matches the relay_bp F64 oracle) and fp32 on metal — Metal has no fp64,
    so requesting ``dtype="float64"`` there raises.
    """

    algorithm = "relay"

    def __init__(self, H, priors, observables=None, backend="auto", device=None,
                 block_s=256, dtype=None, dem=None, megakernel=True,
                 **relay_params):
        resolved = resolve_backend(backend)
        if resolved not in ("triton", "metal"):
            raise NotImplementedError(
                f"Relay-BP is implemented on the triton kernels only (resolved "
                f"backend: {resolved!r}; needs 'triton' on CUDA/ROCm or the "
                f"experimental 'metal'). There is no in-package CPU Relay-BP; "
                f"for a CPU reference use the relay-bp adapter "
                f"(tridec.adapters.make_relay_bp, [decoders] extra).")
        if resolved == "metal":
            if dtype is None:
                dtype = "float32"
            elif str(dtype) == "float64":
                raise ValueError(
                    "Relay-BP on the metal backend requires dtype='float32': "
                    "Metal has no fp64 — fp64 messages cannot run on Apple "
                    "GPUs (the fp64 oracle-precision path needs CUDA/ROCm).")
            _warn_metal_once()
        elif dtype is None:
            dtype = "float64"
        self.backend = resolved
        self.device = _default_device(resolved, device)
        self.dem = dem

        cfg = dict(_RELAY_DEFAULTS)
        cfg.update(relay_params)
        # v0.2.1 auto-dispatch: the single-launch megakernel is the default
        # Relay-BP impl on GPU (it WINS on relay -- 9-32x on CUDA/ROCm, ~197x
        # on Metal vs the v0.1 two-kernel host loop -- with per-shot early exit
        # and LER matching the relay_bp oracle). It is a drop-in subclass of
        # RelayBpTriton (overrides _relay_posteriors only), so decode_batch and
        # the no-observable path are unchanged. This path is GPU-gated by
        # construction: RelayBpDecoder already rejects non-(triton|metal)
        # backends above, so the megakernel is only ever built on a GPU. Set
        # ``megakernel=False`` for the v0.1 two-kernel host loop. (BP keeps the
        # two-kernel default -- the plain-BP megakernel loses at batch.)
        self.megakernel = bool(megakernel)
        if self.megakernel:
            from .backends.megakernel import RelayBpMegaTriton
            self._impl = RelayBpMegaTriton(H, priors, block_s=block_s,
                                           dtype=dtype, **cfg)
        else:
            from .backends.relay_triton import RelayBpTriton
            self._impl = RelayBpTriton(H, priors, block_s=block_s, dtype=dtype,
                                       **cfg)

        Lo = _dense_uint8(observables)
        self._Lo = Lo
        if Lo is not None:
            self._impl._Lo = Lo
            self._impl._n_obs = int(Lo.shape[0])
        self.n_obs = None if Lo is None else int(Lo.shape[0])
        self.n_bits = self._impl.n_bits
        self.n_checks = self._impl.n_checks

        impl = "megakernel" if self.megakernel else "two-kernel"
        self.name = f"portable-relay-bp[{self.backend},{impl}]"
        self.tie_break = "relay_bp_nconv_disjoint_ensemble"
        self.config = dict(
            decoder="tridec.RelayBpDecoder", backend=self.backend,
            impl=impl, dtype=dtype, **cfg)

    @classmethod
    def from_dem(cls, dem, backend="auto", device=None, **opts):
        ex = extract(dem)
        return cls(ex["H"], ex["priors"], observables=ex["Lo"], backend=backend,
                   device=device, dem=dem, **opts)

    def decode_batch(self, detection_events):
        dets = np.asarray(detection_events)
        if dets.ndim == 1:
            dets = dets[None, :]
        if self._Lo is not None:
            return self._impl.decode_batch(dets.astype(bool), device=self.device)
        # No observable map: return the lowest-weight valid error estimate.
        import torch
        dev = torch.device(self.device)
        syn_t = torch.as_tensor(dets.astype(bool), device=dev)
        best_eh = self._impl._relay_posteriors(syn_t, dev)       # (N, S)
        return best_eh.t().cpu().numpy().astype(np.uint8)

    def decode(self, detection_events):
        out = self.decode_batch(np.asarray(detection_events)[None, :])
        return out[0]


# --------------------------------------------------------------------------- #
# Factories.                                                                   #
# --------------------------------------------------------------------------- #
def from_dem(dem, backend="auto", algorithm="bp", device=None, **opts):
    """Build a decoder from a ``stim.DetectorErrorModel``.

    Args:
        dem: the DEM (build with ``decompose_errors=False`` — the decoders
            consume the raw hyperedge mechanism set).
        backend: "auto" | "numpy" | "torch" | "triton" | "metal"
            (see module docstring; "metal" is experimental).
        algorithm: "bp" (min-sum BP, all backends) or "relay" (Relay-BP,
            triton kernels only: "triton" on CUDA/ROCm or experimental
            "metal").
        device: optional torch device string for the torch/triton backends.
        **opts: decoder hyperparameters (e.g. max_iter, ms_scaling_factor for
            bp; gamma0, pre_iter, num_sets, ... for relay).

    Returns a decoder with ``decode_batch(detection_events) ->
    predicted_observables`` and a single-shot ``decode``.
    """
    if algorithm == "bp":
        return BpDecoder.from_dem(dem, backend=backend, device=device, **opts)
    if algorithm == "relay":
        return RelayBpDecoder.from_dem(dem, backend=backend, device=device,
                                       **opts)
    raise ValueError(f"unknown algorithm {algorithm!r}; expected 'bp' or 'relay'")


def from_matrices(H, priors, observables=None, backend="auto", algorithm="bp",
                  device=None, **opts):
    """Build a decoder from a raw GF2 parity-check matrix + per-bit priors.

    Args:
        H: (n_checks x n_bits) GF2 check matrix (dense or scipy sparse).
        priors: per-bit error probabilities, length n_bits.
        observables: optional (n_obs x n_bits) GF2 observable map. With it,
            ``decode_batch`` returns predicted observables; without it, hard
            error estimates.
        backend, algorithm, device, **opts: as in ``from_dem``.
    """
    if algorithm == "bp":
        return BpDecoder(H, priors, observables=observables, backend=backend,
                         device=device, **opts)
    if algorithm == "relay":
        return RelayBpDecoder(H, priors, observables=observables,
                              backend=backend, device=device, **opts)
    raise ValueError(f"unknown algorithm {algorithm!r}; expected 'bp' or 'relay'")
