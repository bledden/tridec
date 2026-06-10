"""Triton Relay-BP gates (GPU-only + relay-bp oracle; skips cleanly without).

Ported unchanged in substance from the source repo's kernel TDD suite, with the
canonical circuit loaded from the in-repo .stim fixture. The bar is LER-identity
with the relay_bp Rust oracle (logical-error COUNT within fp/ensemble noise) +
the pre-leg / memory-term primitive identities vs MinSumBPDecoderF64 — NOT
Rust-bit-identity (the gamma-draw RNGs differ by construction).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm GPU required for the Triton Relay-BP kernel",
                allow_module_level=True)
relay_bp = pytest.importorskip("relay_bp")

from conftest import load_bb_circuit

from tridec.backends.relay_triton import RelayBpTriton

DEVICE = "cuda"
BB_SHOTS = 2000
BB_SEED = 0

# Relay-BP reference config (matches the relay_bp oracle / the carried receipts).
RELAY_CFG = dict(
    gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66), stop_nconv=5, stopping_criterion="nconv",
)


@pytest.fixture(scope="module")
def bb_dem_shots():
    circ = load_bb_circuit("0.003", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=BB_SEED).sample(
        BB_SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def _relay_oracle(dem):
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    dec = relay_bp.RelayDecoderF64(cm.check_matrix,
                                   error_priors=cm.error_priors, **RELAY_CFG)
    runner = relay_bp.ObservableDecoderRunner(
        dec, cm.observables_matrix, include_decode_result=False)
    return cm, dec, runner


def test_pre_leg_minsum_identity(bb_dem_shots):
    import relay_bp
    from relay_bp.stim import CheckMatrices
    dem, dets, _ = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    syn = dets.astype(np.uint8)
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    post_trt = trt.minsum_posterior_batch(syn, n_iter=1, gamma=0.0,
                                          alpha=1.0, device=DEVICE)
    dec = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors,
                                      max_iter=1, alpha=1.0, gamma0=0.0)
    post_ref = np.stack([np.asarray(dec.decode_detailed(syn[i]).posterior_ratios)
                         for i in range(min(64, len(syn)))])
    maxdiff = float(np.max(np.abs(post_trt[:post_ref.shape[0]] - post_ref)))
    assert maxdiff < 1e-3, f"pre-leg min-sum posterior maxdiff {maxdiff}"


def test_memory_term_identity(bb_dem_shots):
    import relay_bp
    from relay_bp.stim import CheckMatrices
    dem, dets, _ = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    syn = dets[:256].astype(np.uint8)
    n_iter = 30
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    post_trt = trt.minsum_posterior_batch(syn, n_iter=n_iter, gamma=0.1,
                                          alpha=1.0, device=DEVICE)
    hard_trt = (post_trt < 0.0).astype(np.uint8)
    dec = relay_bp.MinSumBPDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors,
                                      max_iter=n_iter, alpha=1.0, gamma0=0.1)
    hard_ref = np.stack([np.asarray(dec.decode_detailed(syn[i]).decoding)
                         for i in range(len(syn))]).astype(np.uint8)
    per_bit = float((hard_trt == hard_ref).mean())
    assert per_bit >= 0.99, f"memory-term per-bit agreement {per_bit:.5f} < 0.99"


def test_full_relay_ler_identity(bb_dem_shots):
    dem, dets, obs = bb_dem_shots
    _, _, runner = _relay_oracle(dem)
    pred_ref = np.asarray(
        runner.decode_observables_batch(dets.astype(np.uint8))) % 2
    pred_ref = pred_ref.astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    pred_trt = trt.decode_batch(dets, device=DEVICE)
    assert pred_trt.shape == pred_ref.shape
    fails_ref = int(np.any(pred_ref != obs, axis=1).sum())
    fails_trt = int(np.any(pred_trt != obs, axis=1).sum())
    per_shot = float(np.all(pred_trt == pred_ref, axis=1).mean())
    print(f"\nRELAY LER-IDENTITY: oracle={fails_ref} triton={fails_trt} "
          f"(N={len(dets)}) per-shot-agreement={per_shot:.4f}")
    assert abs(fails_trt - fails_ref) <= max(5, int(0.01 * len(dets))), (
        f"relay LER differs too much: oracle={fails_ref} triton={fails_trt}")


def test_decode_batch_shape_and_dtype(bb_dem_shots):
    dem, dets, _ = bb_dem_shots
    trt = RelayBpTriton.from_dem(dem, **RELAY_CFG)
    pred = trt.decode_batch(dets[:128], device=DEVICE)
    assert pred.shape == (128, dem.num_observables)
    assert pred.dtype == bool
