"""Megakernel (single-launch persistent BP / Relay-BP) gates on triton-msl.

Runs ONLY in the triton-msl environment (darwin + triton + triton_msl
importable + torch with no CUDA/ROCm device), like test_metal.py. CUDA/ROCm
megakernel gates are the cloud session's job (issue #2).

The megakernel ports the EXISTING kernels' math byte-for-byte (same
per-(shot,row) operation order), so the bars here are STRICTER than the
fp32-vs-fp64 suite gates where an apples-to-apples comparison exists:

  * BP megakernel vs two-kernel ``BpTriton``: BIT-IDENTICAL posteriors
    (measured exactly equal on this env, block=32) + identical predictions.
  * Relay megakernel in ALL-LEGS mode vs the host-loop ``RelayBpTriton``
    driven through the same schedule (unreachable stop_nconv): identical
    per-shot predictions (same gamma tensors by construction).
  * Relay megakernel (real nconv config, per-shot early exit) vs the
    relay_bp Rust oracle: LER-identity bar from test_relay_triton.py (the
    gamma RNGs differ from Rust by construction, so this one is statistical).
  * Determinism across repeated launches (the megakernel is barrier/lockstep
    sensitive -- see tridec/backends/megakernel.py).
"""
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.metal

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
pytest.importorskip("triton_msl", reason="triton-msl not installed")
if sys.platform != "darwin":
    pytest.skip("Metal backend is darwin-only", allow_module_level=True)
if torch.cuda.is_available():
    pytest.skip("CUDA/ROCm visible: this is a GPU box, not the Metal env",
                allow_module_level=True)

from conftest import load_bb_circuit, load_surface_circuit

from tridec.backends.bp_triton import BpTriton
from tridec.backends.megakernel import BpMegaTriton, RelayBpMegaTriton
from tridec.backends.relay_triton import RelayBpTriton
from tridec.validation import wilson_ci, wilson_consistent

MS = 0.625
DEVICE = "cpu"
SHOTS = 2000
SEED = 0

RELAY_CFG = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66),
                 stopping_criterion="nconv", dtype="float32")


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = load_bb_circuit("0.003", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


@pytest.fixture(scope="module")
def surface_dem_shots():
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _bp_identity(dem, dets):
    syn = dets.astype(np.uint8)
    old = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    mega = BpMegaTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    p_old = old.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg2 = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    assert float((p_meg == p_meg2).mean()) == 1.0, "megakernel nondeterminism"
    bit_ident = float((p_old == p_meg).mean())
    assert bit_ident == 1.0, (
        f"BP megakernel posterior not bit-identical to BpTriton "
        f"({bit_ident:.6f}); on this env the op order is identical -- a "
        f"mismatch means a kernel bug or a triton-msl codegen change")
    pred_old = old.decode_batch(dets, device=DEVICE)
    pred_meg = mega.decode_batch(dets, device=DEVICE)
    assert (pred_old == pred_meg).all()


def test_bp_megakernel_bit_identity_bb(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    _bp_identity(dem, dets)


def test_bp_megakernel_bit_identity_surface(surface_dem_shots):
    dem, dets, _ = surface_dem_shots
    _bp_identity(dem, dets)


def test_relay_megakernel_all_legs_identity(bb_dem_shots):
    """Same schedule (all 61 legs), same gamma tensors, same fp32 op order
    -> identical predictions. ~15 s (the host loop dominates)."""
    dem, dets, _ = bb_dem_shots
    sub = dets[:128]
    host = RelayBpTriton.from_dem(dem, stop_nconv=62, **RELAY_CFG)
    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=62, early_exit=False,
                                      **RELAY_CFG)
    p_h = host.decode_batch(sub, device=DEVICE)
    p_m = mega.decode_batch(sub, device=DEVICE)
    ident = float(np.all(p_h == p_m, axis=1).mean())
    assert ident == 1.0, (
        f"relay megakernel (all-legs mode) per-shot identity {ident:.5f}; "
        f"the schedules are identical so any mismatch is a kernel bug (or "
        f"an fp near-tie in the in-kernel fp32 weight selection -- the host "
        f"computes selection weights in fp64; report, don't paper over)")


def test_relay_megakernel_ler_vs_oracle(bb_dem_shots):
    """The real config (stop_nconv=5, per-shot early exit) vs the relay_bp
    Rust oracle: LER-identity bar from test_relay_triton.py."""
    relay_bp = pytest.importorskip("relay_bp")
    from relay_bp.stim import CheckMatrices
    dem, dets, obs = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    oracle = relay_bp.RelayDecoderF64(
        cm.check_matrix, error_priors=cm.error_priors,
        gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
        stopping_criterion="nconv")
    runner = relay_bp.ObservableDecoderRunner(
        oracle, cm.observables_matrix, include_decode_result=False)
    pred_ref = (np.asarray(runner.decode_observables_batch(
        dets.astype(np.uint8))) % 2).astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)

    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=5, early_exit=True,
                                      **RELAY_CFG)
    pred_m = mega.decode_batch(dets, device=DEVICE)
    assert pred_m.shape == pred_ref.shape
    fails_ref = int(np.any(pred_ref != obs, axis=1).sum())
    fails_m = int(np.any(pred_m != obs, axis=1).sum())
    per_shot = float(np.all(pred_m == pred_ref, axis=1).mean())
    print(f"\nMEGA RELAY LER-IDENTITY: oracle={fails_ref} mega={fails_m} "
          f"(N={len(dets)}) per-shot-agreement={per_shot:.4f}")
    assert wilson_consistent(fails_m, len(dets), fails_ref, len(dets)), (
        f"statistical tier: megakernel relay fails={fails_m} "
        f"CI={wilson_ci(fails_m, len(dets))} vs oracle fails={fails_ref} "
        f"CI={wilson_ci(fails_ref, len(dets))} — disjoint 95% Wilson CIs")
    assert per_shot >= 0.99
