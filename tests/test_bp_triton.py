"""Triton min-sum BP kernel gates (GPU-only; skips cleanly without triton/GPU).

Ported unchanged in substance from the source repo's kernel TDD suite, with the
canonical circuit loaded from the in-repo .stim fixture. The kernel computes in
float32; the gate is >=99.5% hard-decision agreement with the fp64 references +
LER within noise, NOT bit-identity (validated on H200/CUDA and MI300X/ROCm —
see bench/receipts/).
"""
import numpy as np
import pytest

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm GPU required for the Triton BP kernel",
                allow_module_level=True)

from conftest import load_bb_circuit

from portable_qec.backends.bp_numpy import BpBaseline
from portable_qec.backends.bp_torch import BpGpu
from portable_qec.backends.bp_triton import BpTriton

MS = 0.625
DEVICE = "cuda"

BB_SHOTS = 2000
BB_SEED = 0


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = load_bb_circuit("0.003", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_one_iter_hard_agreement(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=1, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)
    post_trt = trt.run_iterations_batch(syn, n_iter=1, device=DEVICE)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    hard_cpu = np.empty_like(hard_trt)
    for i in range(syn.shape[0]):
        hard_cpu[i] = cpu.run_iterations(syn[i], n_iter=1)["hard"]
    agree = float((hard_trt == hard_cpu).mean())
    assert agree >= 0.995, f"one-iter hard agreement {agree:.5f} < 0.995"


def test_full_decode_per_bit_agreement(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)
    post_trt = trt.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    hard_cpu = np.empty_like(hard_trt)
    for i in range(syn.shape[0]):
        hard_cpu[i] = cpu.run_iterations(syn[i], n_iter=30)["hard"]
    per_bit = float((hard_trt == hard_cpu).mean())
    assert per_bit >= 0.995, f"30-iter per-bit hard agreement {per_bit:.5f} < 0.995"


def test_full_decode_ler_agreement(bb_dem_shots):
    dem, dets, obs = bb_dem_shots
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred_cpu = cpu.decode_batch(dets)
    pred_trt = trt.decode_batch(dets, device=DEVICE)
    assert pred_trt.shape == pred_cpu.shape
    assert pred_trt.dtype == bool
    fails_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    fails_trt = int(np.any(pred_trt != obs, axis=1).sum())
    assert abs(fails_trt - fails_cpu) <= max(3, int(0.005 * len(dets))), (
        f"LER differs too much: cpu={fails_cpu} triton={fails_trt}")


def test_matches_bp_torch_hard(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    gpu = BpGpu.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    syn = dets.astype(np.uint8)
    post_gpu = gpu.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    post_trt = trt.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    hard_gpu = (post_gpu < 0.0).astype(np.uint8)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    per_bit = float((hard_gpu == hard_trt).mean())
    assert per_bit >= 0.995, f"BpTriton vs BpGpu per-bit agreement {per_bit:.5f}"


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    trt = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred = trt.decode_batch(dets, device=DEVICE)
    assert pred.shape == (BB_SHOTS, dem.num_observables)
    assert pred.dtype == bool
