"""tridec: vendor-portable GPU decoders for quantum LDPC codes.

Triton min-sum BP and Relay-BP decoders that consume any stim
DetectorErrorModel or raw parity-check matrices, with CPU reference
implementations, validated against the standard CPU references (ldpc,
relay-bp), running on NVIDIA (CUDA) and AMD (ROCm) GPUs.

Quickstart::

    import stim, tridec

    circuit = stim.Circuit.from_file("memory.stim")
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = tridec.from_dem(dem, backend="auto")

    dets, obs = circuit.compile_detector_sampler(seed=0).sample(
        10_000, separate_observables=True)
    pred = decoder.decode_batch(dets)            # (shots, n_obs) bool
    ler = (pred != obs).any(axis=1).mean()
"""
from .api import (
    BpDecoder,
    RelayBpDecoder,
    available_backends,
    from_dem,
    from_matrices,
    resolve_backend,
)
from .dem import extract

__version__ = "0.1.0.dev0"

__all__ = [
    "BpDecoder",
    "RelayBpDecoder",
    "available_backends",
    "extract",
    "from_dem",
    "from_matrices",
    "resolve_backend",
    "__version__",
]
