"""Torch edge-list BP: bit-identity to the numpy reference at fp64 (CPU).

Ported from the source repo's kernel TDD suite. The torch backend is
device-agnostic; these tests run it on CPU and require BIT-IDENTITY (atol 1e-6
on fp64 messages) to the numpy reference for one iteration, plus full-decode
equivalence. Skips cleanly when torch is not installed.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import load_bb_circuit

from tridec.backends.bp_numpy import BpBaseline
from tridec.backends.bp_torch import BpGpu

MS = 0.625
DEVICE = "cpu"

BB_SHOTS = 200
BB_SEED = 0


def _rep_code():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.05])
    syndrome = np.array([1, 0], dtype=np.uint8)
    return H, p, syndrome


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = load_bb_circuit("0.005", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_one_iter_bit_identical_rep_code():
    H, p, syndrome = _rep_code()
    cpu = BpBaseline(H, priors=p, max_iter=1, ms_scaling_factor=MS,
                     schedule="parallel")
    ref = cpu.run_iterations(syndrome, n_iter=1, return_messages=True)
    gpu = BpGpu(H, priors=p, max_iter=1, ms_scaling_factor=MS)
    trace = gpu.run_iterations(syndrome, n_iter=1, device=DEVICE,
                               return_messages=True)
    for key, want in ref["check_to_bit"].items():
        assert np.isclose(trace["check_to_bit"][key], want, rtol=0, atol=1e-6)
    for key, want in ref["bit_to_check"].items():
        assert np.isclose(trace["bit_to_check"][key], want, rtol=0, atol=1e-6)
    assert np.allclose(trace["posterior"], ref["posterior"], rtol=0, atol=1e-6)
    assert np.array_equal(trace["hard"], ref["hard"])


def test_one_iter_bit_identical_bb_dem(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    gpu = BpGpu.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)
    for i in range(min(10, syn.shape[0])):
        ref = cpu.run_iterations(syn[i], n_iter=1, return_messages=True)
        trace = gpu.run_iterations(syn[i], n_iter=1, device=DEVICE,
                                   return_messages=True)
        c2b_ref = np.array([ref["check_to_bit"][k]
                            for k in sorted(ref["check_to_bit"])])
        c2b_got = np.array([trace["check_to_bit"][k]
                            for k in sorted(trace["check_to_bit"])])
        b2c_ref = np.array([ref["bit_to_check"][k]
                            for k in sorted(ref["bit_to_check"])])
        b2c_got = np.array([trace["bit_to_check"][k]
                            for k in sorted(trace["bit_to_check"])])
        assert np.allclose(c2b_got, c2b_ref, rtol=0, atol=1e-6), f"shot {i}"
        assert np.allclose(b2c_got, b2c_ref, rtol=0, atol=1e-6), f"shot {i}"
        assert np.allclose(trace["posterior"], ref["posterior"], rtol=0,
                           atol=1e-6), f"shot {i}"


def test_batched_one_iter_matches_per_shot(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    gpu = BpGpu.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)[:16]
    batch_post = gpu.run_iterations_batch(syn, n_iter=1, device=DEVICE)
    for i in range(syn.shape[0]):
        single = gpu.run_iterations(syn[i], n_iter=1, device=DEVICE)["posterior"]
        assert np.allclose(batch_post[i], single, rtol=0, atol=1e-6), f"row {i}"


def test_full_decode_equivalence(bb_dem_shots):
    """Full 30-iteration decode: torch (fp64) == numpy, per-shot observables.
    Documented fallback for float-order divergence on near-ties: >=99% exact
    AND LER within one shot."""
    dem, dets, obs = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    gpu = BpGpu.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred_cpu = cpu.decode_batch(dets)
    pred_gpu = gpu.decode_batch(dets, device=DEVICE)
    assert pred_gpu.shape == pred_cpu.shape
    assert pred_gpu.dtype == bool

    exact = np.all(pred_gpu == pred_cpu, axis=1)
    frac_exact = float(exact.mean())
    fails_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    fails_gpu = int(np.any(pred_gpu != obs, axis=1).sum())
    if frac_exact < 1.0:
        assert frac_exact >= 0.99, f"per-shot exact match {frac_exact:.4f} < 0.99"
        assert abs(fails_gpu - fails_cpu) <= 1
    else:
        assert fails_gpu == fails_cpu
