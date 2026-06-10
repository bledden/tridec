"""Public API: from_dem / from_matrices / backend dispatch / Decoder surface."""
import numpy as np
import pytest

from conftest import load_surface_circuit

import tridec
from tridec import BpDecoder, RelayBpDecoder, available_backends, resolve_backend


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
        return torch.cuda.is_available()
    except Exception:
        return False


def _metal_available():
    import sys
    if sys.platform != "darwin":
        return False
    try:
        import triton  # noqa: F401
        import triton_metal  # noqa: F401
        import torch
        return not torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def surface():
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        300, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


# --------------------------------------------------------------------------- #
# Backend resolution.                                                          #
# --------------------------------------------------------------------------- #
def test_resolve_backend_auto_matches_environment():
    resolved = resolve_backend("auto")
    if _triton_gpu_available():
        assert resolved == "triton"
    elif _metal_available():
        assert resolved == "metal"
    elif _torch_available():
        assert resolved == "torch"
    else:
        assert resolved == "numpy"


def test_available_backends_always_includes_numpy():
    avail = available_backends()
    assert "numpy" in avail
    assert set(avail) <= {"numpy", "torch", "triton", "metal"}


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("vulkan")


def test_metal_backend_unavailable_raises_clear_error():
    if _metal_available():
        pytest.skip("triton-metal available here; the unavailable path is moot")
    with pytest.raises(RuntimeError, match="triton-metal"):
        resolve_backend("metal")


def test_triton_backend_unavailable_raises_clear_error(surface):
    if _triton_gpu_available():
        pytest.skip("triton+GPU available here; the unavailable path is moot")
    if _metal_available():
        pytest.skip("metal env: backend='triton' resolves to metal (tested "
                    "in test_metal.py)")
    dem, _, _ = surface
    with pytest.raises((RuntimeError, ImportError), match="[Tt]riton|GPU"):
        tridec.from_dem(dem, backend="triton")


# --------------------------------------------------------------------------- #
# from_dem.                                                                    #
# --------------------------------------------------------------------------- #
def test_from_dem_numpy_decodes(surface):
    dem, dets, obs = surface
    dec = tridec.from_dem(dem, backend="numpy")
    assert isinstance(dec, BpDecoder)
    assert dec.backend == "numpy"
    pred = dec.decode_batch(dets)
    assert pred.shape == (dets.shape[0], dem.num_observables)
    assert pred.dtype == bool
    assert float(np.any(pred != obs, axis=1).mean()) < 0.5


def test_from_dem_auto_backend(surface):
    dem, dets, _ = surface
    dec = tridec.from_dem(dem)  # backend="auto"
    assert dec.backend in ("numpy", "torch", "triton", "metal")
    pred = dec.decode_batch(dets[:32])
    assert pred.shape == (32, dem.num_observables)


def test_decode_single_shot(surface):
    dem, dets, _ = surface
    dec = tridec.from_dem(dem, backend="numpy")
    one = dec.decode(dets[0])
    batch = dec.decode_batch(dets[:1])
    assert one.shape == (dem.num_observables,)
    assert np.array_equal(one, batch[0])


def test_torch_backend_matches_numpy(surface):
    if not _torch_available():
        pytest.skip("torch not installed")
    dem, dets, _ = surface
    dec_np = tridec.from_dem(dem, backend="numpy")
    dec_t = tridec.from_dem(dem, backend="torch", device="cpu")
    a = dec_np.decode_batch(dets[:100])
    b = dec_t.decode_batch(dets[:100])
    # Same fp64 flooding min-sum: expect identical predictions (>=99% on ties).
    frac = float(np.all(a == b, axis=1).mean())
    assert frac >= 0.99


def test_relay_requires_triton_backend(surface):
    dem, _, _ = surface
    if _triton_gpu_available():
        dec = tridec.from_dem(dem, algorithm="relay")
        assert isinstance(dec, RelayBpDecoder)
        assert dec.backend == "triton"
    elif _metal_available():
        pytest.skip("metal env: relay-on-metal covered in test_metal.py")
    else:
        with pytest.raises((RuntimeError, NotImplementedError)):
            tridec.from_dem(dem, algorithm="relay", backend="numpy")
        with pytest.raises((RuntimeError, NotImplementedError, ImportError)):
            tridec.from_dem(dem, algorithm="relay")


def test_decoder_carries_provenance(surface):
    dem, _, _ = surface
    dec = tridec.from_dem(dem, backend="numpy")
    assert dec.dem is dem
    assert isinstance(dec.name, str) and dec.name
    assert isinstance(dec.config, dict) and dec.config.get("backend") == "numpy"
    assert dec.tie_break == "min_sum_parallel_hard_decision"


# --------------------------------------------------------------------------- #
# from_matrices.                                                               #
# --------------------------------------------------------------------------- #
def test_from_matrices_without_observables_returns_error_estimates():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    priors = [0.10, 0.20, 0.05]
    dec = tridec.from_matrices(H, priors, backend="numpy")
    syn = np.array([[1, 0]], dtype=np.uint8)
    e = dec.decode_batch(syn)
    assert e.shape == (1, 3)
    assert set(np.unique(e)) <= {0, 1}


def test_from_matrices_with_observables_returns_observables():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    priors = [0.10, 0.20, 0.05]
    obs_mat = np.array([[1, 1, 1]], dtype=np.uint8)  # one logical = total parity
    dec = tridec.from_matrices(H, priors, observables=obs_mat,
                                     backend="numpy")
    syn = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    pred = dec.decode_batch(syn)
    assert pred.shape == (2, 1)
    assert pred.dtype == bool
    # zero syndrome with these priors -> zero error -> observable not flipped
    assert pred[1, 0] == False  # noqa: E712


def test_from_matrices_dem_is_none():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    dec = tridec.from_matrices(H, [0.1, 0.1, 0.1], backend="numpy")
    assert dec.dem is None
