"""Experimental Metal (Apple silicon, triton-metal) backend gates.

Runs ONLY in the triton-metal environment: darwin + triton + triton_metal
importable + torch with no CUDA/ROCm device. Everywhere else this module
skips cleanly at collection.

These are the spike-level gates (bench/receipts/metal_spike.md) re-run
through the PUBLIC API path (``tridec.from_dem(..., backend="metal")`` /
``backend="auto"`` / ``backend="triton"``) instead of hand-rolled device
args. The triton-metal execution pattern is CPU torch tensors (zero-copy
UMA; "Not mps"), fp32 kernels — the gates are agreement/LER vs the fp64
numpy reference, not bit-identity.
"""
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
pytest.importorskip("triton_metal", reason="triton-metal not installed")
if sys.platform != "darwin":
    pytest.skip("Metal backend is darwin-only", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CUDA/ROCm visible: this is a GPU box, not the Metal env",
                allow_module_level=True)

from conftest import load_bb_circuit, load_surface_circuit

import tridec
from tridec import available_backends, resolve_backend
from tridec.backends.bp_numpy import BpBaseline

MS = 0.625
SHOTS = 2000
SEED = 0


@pytest.fixture(scope="module")
def surface_dem_shots():
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = load_bb_circuit("0.003", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


# --------------------------------------------------------------------------- #
# Backend resolution in the Metal environment.                                 #
# --------------------------------------------------------------------------- #
def test_auto_resolves_to_metal():
    assert resolve_backend("auto") == "metal"


def test_triton_request_resolves_to_metal_without_device_args():
    # The user should NOT have to hand-roll device="cpu": requesting the
    # triton backend in the Metal env lands on the metal path directly.
    assert resolve_backend("triton") == "metal"


def test_metal_alias_resolves():
    assert resolve_backend("metal") == "metal"


def test_available_backends_prefers_metal_over_torch():
    avail = available_backends()
    assert "metal" in avail
    assert avail.index("metal") < avail.index("torch")


def test_metal_decoder_uses_cpu_tensors(surface_dem_shots):
    dem, _, _ = surface_dem_shots
    dec = tridec.from_dem(dem, backend="metal")
    assert dec.backend == "metal"
    assert dec.device == "cpu"     # triton-metal pattern: CPU tensors, not mps
    assert dec.name == "portable-bp[metal]"
    assert dec.config["backend"] == "metal"


# --------------------------------------------------------------------------- #
# BP spike-level gates through the public API.                                 #
# --------------------------------------------------------------------------- #
def _bp_gates(dem, dets, obs, label):
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    dec = tridec.from_dem(dem, backend="metal")
    syn = dets.astype(np.uint8)

    post1 = dec._impl.run_iterations_batch(syn, n_iter=1, device=dec.device)
    hard1 = (post1 < 0.0).astype(np.uint8)
    hard1_cpu = np.empty_like(hard1)
    for i in range(len(syn)):
        hard1_cpu[i] = cpu.run_iterations(syn[i], n_iter=1)["hard"]
    a1 = float((hard1 == hard1_cpu).mean())
    assert a1 >= 0.995, f"{label}: one-iter hard agreement {a1:.5f}"

    post30 = dec._impl.run_iterations_batch(syn, n_iter=30, device=dec.device)
    hard30 = (post30 < 0.0).astype(np.uint8)
    hard30_cpu = np.empty_like(hard30)
    for i in range(len(syn)):
        hard30_cpu[i] = cpu.run_iterations(syn[i], n_iter=30)["hard"]
    a30 = float((hard30 == hard30_cpu).mean())
    assert a30 >= 0.995, f"{label}: 30-iter per-bit agreement {a30:.5f}"

    pred_cpu = cpu.decode_batch(dets)
    pred_mtl = dec.decode_batch(dets)
    assert pred_mtl.shape == pred_cpu.shape and pred_mtl.dtype == bool
    fails_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    fails_mtl = int(np.any(pred_mtl != obs, axis=1).sum())
    print(f"\nMETAL BP [{label}]: one-iter agree {a1:.5f}, 30-iter agree "
          f"{a30:.5f}, LER numpy {fails_cpu}/{SHOTS} vs metal "
          f"{fails_mtl}/{SHOTS}")
    assert abs(fails_mtl - fails_cpu) <= max(3, int(0.005 * len(dets))), (
        f"{label}: LER differs: numpy={fails_cpu} metal={fails_mtl}")


def test_bp_gates_surface(surface_dem_shots):
    dem, dets, obs = surface_dem_shots
    _bp_gates(dem, dets, obs, "surface_d3_r3_p0.003")


def test_bp_gates_bb72(bb_dem_shots):
    dem, dets, obs = bb_dem_shots
    _bp_gates(dem, dets, obs, "bb72_r6_p0.003_Z")


# --------------------------------------------------------------------------- #
# Relay-BP on Metal: fp32-only, enforced.                                      #
# --------------------------------------------------------------------------- #
def test_relay_fp64_raises_clear_error(bb_dem_shots):
    dem, _, _ = bb_dem_shots
    with pytest.raises(ValueError, match="fp64|float64"):
        tridec.from_dem(dem, algorithm="relay", backend="metal",
                        dtype="float64")


def test_relay_defaults_to_fp32_on_metal(bb_dem_shots):
    dem, _, _ = bb_dem_shots
    dec = tridec.from_dem(dem, algorithm="relay", backend="metal")
    assert dec.backend == "metal"
    assert dec.device == "cpu"
    assert dec._impl.dtype == "float32"
    assert dec.config["dtype"] == "float32"


def test_relay_auto_dispatches_to_megakernel(bb_dem_shots):
    """v0.2.1 #5: relay defaults to the single-launch megakernel on GPU;
    ``megakernel=False`` opts back to the two-kernel host loop."""
    from tridec.backends.megakernel import RelayBpMegaTriton
    from tridec.backends.relay_triton import RelayBpTriton
    dem, _, _ = bb_dem_shots
    default = tridec.from_dem(dem, algorithm="relay", backend="metal")
    assert isinstance(default._impl, RelayBpMegaTriton)
    assert default.config["impl"] == "megakernel"
    twok = tridec.from_dem(dem, algorithm="relay", backend="metal",
                           megakernel=False)
    assert isinstance(twok._impl, RelayBpTriton)
    assert not isinstance(twok._impl, RelayBpMegaTriton)
    assert twok.config["impl"] == "two-kernel"


def test_relay_full_ler_vs_oracle(bb_dem_shots):
    """The spike-level relay gate: LER-identity with the relay_bp Rust oracle
    (fp32 Metal kernels; gamma RNGs differ by construction). ~30 s on M4 Max
    (relay's per-iteration host loop is launch-bound on Metal — correct, not
    fast; documented in README/benchmark)."""
    relay_bp = pytest.importorskip("relay_bp")
    from relay_bp.stim import CheckMatrices
    dem, dets, obs = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    cfg = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
               gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
               stopping_criterion="nconv")
    oracle = relay_bp.RelayDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors, **cfg)
    runner = relay_bp.ObservableDecoderRunner(
        oracle, cm.observables_matrix, include_decode_result=False)
    pred_ref = (np.asarray(
        runner.decode_observables_batch(dets.astype(np.uint8))) % 2).astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)

    dec = tridec.from_dem(dem, algorithm="relay", backend="metal")
    pred_mtl = dec.decode_batch(dets)
    assert pred_mtl.shape == pred_ref.shape
    fails_ref = int(np.any(pred_ref != obs, axis=1).sum())
    fails_mtl = int(np.any(pred_mtl != obs, axis=1).sum())
    per_shot = float(np.all(pred_mtl == pred_ref, axis=1).mean())
    print(f"\nMETAL RELAY LER-IDENTITY: oracle={fails_ref} metal={fails_mtl} "
          f"(N={len(dets)}) per-shot-agreement={per_shot:.4f}")
    assert abs(fails_mtl - fails_ref) <= max(5, int(0.01 * len(dets))), (
        f"relay LER differs too much: oracle={fails_ref} metal={fails_mtl}")
    assert per_shot >= 0.99
