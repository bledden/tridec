"""sinter integration: tridec decoders as ``sinter.collect`` custom decoders.

Usage::

    import sinter
    from tridec.sinter import TridecSinterDecoder, sinter_decoders

    stats = sinter.collect(
        num_workers=4,
        tasks=tasks,
        decoders=["tridec_bp", "pymatching"],
        custom_decoders={"tridec_bp": TridecSinterDecoder(algorithm="bp")},
        # or simply: custom_decoders=sinter_decoders(),
        max_shots=1_000_000,
    )

Notes
-----
* sinter hands the compiled decoder LITTLE-endian bit-packed detection events
  (uint8, shape ``(shots, ceil(num_detectors/8))``) and expects little-endian
  bit-packed observable predictions back (uint8, shape
  ``(shots, ceil(num_observables/8))``). The pack/unpack round-trip is pinned
  bit-exactly by ``tests/test_sinter.py``.
* sinter derives the task DEM itself (typically with ``decompose_errors=True``
  so matching decoders work). tridec's extraction consumes decomposed DEMs
  correctly: separators are ignored and repeated detectors cancel mod 2, which
  reconstructs the original hyperedge incidence.
* ``TridecSinterDecoder`` instances hold only plain configuration, so they
  pickle cleanly into sinter's worker processes; the decoder itself is built
  per-task inside ``compile_decoder_for_dem``.

This module requires the optional ``sinter`` dependency
(``pip install "tridec[sinter]"``).
"""
import math

import numpy as np

try:
    import sinter
except ImportError as _e:  # pragma: no cover - exercised only without sinter
    raise ImportError(
        "tridec.sinter requires the optional 'sinter' dependency; "
        "install it with: pip install 'tridec[sinter]'") from _e

from . import api

__all__ = ["TridecSinterDecoder", "sinter_decoders"]


class _CompiledTridecDecoder(sinter.CompiledDecoder):
    """A tridec decoder preconfigured for one DEM, speaking sinter bit-packing."""

    def __init__(self, decoder, num_detectors, num_observables):
        self._decoder = decoder
        self._num_dets = int(num_detectors)
        self._num_obs = int(num_observables)

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        packed = np.asarray(bit_packed_detection_event_data, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != math.ceil(self._num_dets / 8):
            raise ValueError(
                f"expected bit-packed detection events of shape "
                f"(shots, {math.ceil(self._num_dets / 8)}), got {packed.shape}")
        # sinter packs little-endian: detector k lives in byte k//8, bit k%8.
        dets = np.unpackbits(packed, axis=1, bitorder="little",
                             count=self._num_dets).astype(bool)
        pred = np.asarray(self._decoder.decode_batch(dets))
        return np.packbits(pred.astype(np.uint8), axis=1, bitorder="little")


class TridecSinterDecoder(sinter.Decoder):
    """``sinter.Decoder`` exposing tridec's BP / Relay-BP decoders.

    Args:
        algorithm: ``"bp"`` (min-sum BP; numpy/torch/triton backends) or
            ``"relay"`` (Relay-BP; triton backend only).
        backend: tridec backend request (``"auto"`` | ``"numpy"`` | ``"torch"``
            | ``"triton"`` | ``"metal"``). Resolution happens inside the worker
            process, at ``compile_decoder_for_dem`` time.
        device: optional torch device string for the torch/triton backends.
        **opts: decoder hyperparameters forwarded to ``tridec.from_dem``
            (e.g. ``max_iter``, ``ms_scaling_factor`` for bp).
    """

    def __init__(self, algorithm="bp", backend="auto", device=None, **opts):
        if algorithm not in ("bp", "relay"):
            raise ValueError(
                f"unknown algorithm {algorithm!r}; expected 'bp' or 'relay'")
        self.algorithm = algorithm
        self.backend = backend
        self.device = device
        self.opts = dict(opts)

    def compile_decoder_for_dem(self, *, dem):
        decoder = api.from_dem(dem, backend=self.backend,
                               algorithm=self.algorithm, device=self.device,
                               **self.opts)
        return _CompiledTridecDecoder(decoder, dem.num_detectors,
                                      dem.num_observables)


def sinter_decoders(backend="auto", device=None, **opts):
    """The standard ``custom_decoders`` dict for ``sinter.collect``.

    Returns ``{"tridec_bp": ..., "tridec_relay": ...}``. Note that
    ``tridec_relay`` requires the triton backend (CUDA/ROCm GPU, or the
    experimental Metal environment) at collect time.
    """
    return {
        "tridec_bp": TridecSinterDecoder(algorithm="bp", backend=backend,
                                         device=device, **opts),
        "tridec_relay": TridecSinterDecoder(algorithm="relay", backend=backend,
                                            device=device, **opts),
    }
