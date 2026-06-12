"""Megakernel (single-launch persistent BP / Relay-BP) gates on CUDA/ROCm.

The cloud half of issue #2 (the Metal half is test_megakernel_metal.py).
Unlike triton-metal, CUDA/ROCm honors ``tl.debug_barrier()`` (verified on
H200: bar.sync present in the emitted PTX at every source barrier site, plus
a behavioral load->barrier->store ring repro exact at BLOCK=128/256 whose
no-barrier negative control races to <1% match — see
bench/receipts/megakernel_h200.json, ``barrier_sanity``). So these gates run
at the REAL block sizes (128 and 256), not the Metal lockstep-32 fallback.

Bars (same as the Metal file, with the fp64 additions Metal couldn't run):

  * BP megakernel vs two-kernel ``BpTriton``: BIT-IDENTICAL posteriors +
    identical predictions + exact run-to-run determinism, BB + surface.
  * BP megakernel LER vs the fp64 numpy reference (suite bar: count within
    max(3, 0.5%)).
  * Relay megakernel ALL-LEGS mode vs the host-loop ``RelayBpTriton`` driven
    through the same schedule: identical per-shot error estimates, EXCEPT
    for verified degenerate ties — distinct syndrome-consistent solutions
    with exactly equal fp64 weight, which the kernel (message-dtype tree
    sum) may tie-break differently from the host (fp64 sum). Measured on
    H200: 2/256 shots are exact ties under this schedule. Every mismatching
    shot is VERIFIED to be such a tie (equal fp64 weight + syndrome-
    consistent), anything else fails.
  * Relay megakernel (real nconv config, per-shot early exit) vs the
    relay_bp Rust F64 oracle: LER-identity bar from test_relay_triton.py,
    fp32 AND fp64 (statistical: the gamma RNGs differ by construction).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

triton = pytest.importorskip("triton")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm GPU required for the megakernel gates",
                allow_module_level=True)

from conftest import load_bb_circuit, load_surface_circuit

from tridec.backends.bp_triton import BpTriton
from tridec.backends.megakernel import BpMegaTriton, RelayBpMegaTriton
from tridec.backends.relay_triton import RelayBpTriton

MS = 0.625
DEVICE = "cuda"
SHOTS = 2000
SEED = 0
BLOCKS = [128, 256]

RELAY_CFG = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66),
                 stopping_criterion="nconv")


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


def _bp_identity(dem, dets, block):
    syn = dets.astype(np.uint8)
    old = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    mega = BpMegaTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                                 block=block)
    p_old = old.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg2 = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    assert float((p_meg == p_meg2).mean()) == 1.0, "megakernel nondeterminism"
    bit_ident = float((p_old == p_meg).mean())
    assert bit_ident == 1.0, (
        f"BP megakernel posterior not bit-identical to BpTriton "
        f"({bit_ident:.6f}) at BLOCK={block}; the per-(shot,row) op order is "
        f"identical -- a mismatch means a kernel bug or a barrier failure")
    pred_old = old.decode_batch(dets, device=DEVICE)
    pred_meg = mega.decode_batch(dets, device=DEVICE)
    assert (pred_old == pred_meg).all()


@pytest.mark.parametrize("block", BLOCKS)
def test_bp_megakernel_bit_identity_bb(bb_dem_shots, block):
    dem, dets, _ = bb_dem_shots
    _bp_identity(dem, dets, block)


@pytest.mark.parametrize("block", BLOCKS)
def test_bp_megakernel_bit_identity_surface(surface_dem_shots, block):
    dem, dets, _ = surface_dem_shots
    _bp_identity(dem, dets, block)


@pytest.mark.parametrize("fixture_name", ["bb_dem_shots", "surface_dem_shots"])
def test_bp_megakernel_ler_vs_numpy(fixture_name, request):
    """Suite LER bar vs the fp64 numpy reference (block=default)."""
    from tridec.backends.bp_numpy import BpBaseline
    dem, dets, obs = request.getfixturevalue(fixture_name)
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    mega = BpMegaTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred_cpu = cpu.decode_batch(dets)
    pred_meg = mega.decode_batch(dets, device=DEVICE)
    f_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    f_meg = int(np.any(pred_meg != obs, axis=1).sum())
    assert abs(f_meg - f_cpu) <= max(3, int(0.005 * len(dets))), (
        f"BP megakernel LER {f_meg} vs numpy {f_cpu} (N={len(dets)})")


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_relay_megakernel_all_legs_identity(bb_dem_shots, block, dtype):
    """Same schedule (all 61 legs), same gamma tensors, same op order ->
    identical per-shot error estimates, except VERIFIED degenerate
    lowest-weight ties (see module docstring + megakernel.py docstring);
    fp64 is the gate Metal couldn't run."""
    dem, dets, _ = bb_dem_shots
    sub = dets[:256]
    host = RelayBpTriton.from_dem(dem, stop_nconv=62, dtype=dtype,
                                  **RELAY_CFG)
    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=62, early_exit=False,
                                      dtype=dtype, block=block, **RELAY_CFG)
    dev = torch.device(DEVICE)
    syn_t = torch.as_tensor(sub, device=dev)
    eh_h = host._relay_posteriors(syn_t, dev).cpu().numpy()   # (N, S)
    eh_m = mega._relay_posteriors(syn_t, dev).cpu().numpy()   # (N, S)
    eh_m2 = mega._relay_posteriors(syn_t, dev).cpu().numpy()
    assert float((eh_m == eh_m2).mean()) == 1.0, "megakernel nondeterminism"
    mism = np.where(np.any(eh_h != eh_m, axis=0))[0]
    wllr = host._wllr_np.astype(np.float64)
    H = host.H
    for s in mism:
        a = eh_h[:, s].astype(np.uint8)
        b = eh_m[:, s].astype(np.uint8)
        w_a = float((wllr * a).sum())
        w_b = float((wllr * b).sum())
        assert np.array_equal((H @ b) % 2, sub[s].astype(np.uint8)), (
            f"shot {s}: megakernel solution not syndrome-consistent "
            f"(BLOCK={block}, {dtype}) -- a kernel bug, not a tie")
        assert w_a == w_b, (
            f"shot {s}: prediction mismatch is NOT a degenerate tie: host "
            f"fp64 weight {w_a!r} vs mega {w_b!r} (BLOCK={block}, {dtype}) "
            f"-- a kernel bug, report")
    n_tie = len(mism)
    assert n_tie <= max(3, int(0.02 * sub.shape[0])), (
        f"too many tie mismatches ({n_tie}/{sub.shape[0]}) -- suspicious")
    print(f"\nALL-LEGS IDENTITY [{dtype} BLOCK={block}]: "
          f"{sub.shape[0] - n_tie}/{sub.shape[0]} exact, "
          f"{n_tie} verified equal-weight ties")


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_relay_megakernel_ler_vs_oracle(bb_dem_shots, block, dtype):
    """The real config (stop_nconv=5, per-shot early exit) vs the relay_bp
    Rust F64 oracle: LER-identity bar from test_relay_triton.py. The fp64
    row is the apples-to-apples dtype match with the oracle."""
    relay_bp = pytest.importorskip("relay_bp")
    from relay_bp.stim import CheckMatrices
    dem, dets, obs = bb_dem_shots
    cm = CheckMatrices.from_dem(dem)
    oracle = relay_bp.RelayDecoderF64(
        cm.check_matrix, error_priors=cm.error_priors,
        stop_nconv=5, **RELAY_CFG)
    runner = relay_bp.ObservableDecoderRunner(
        oracle, cm.observables_matrix, include_decode_result=False)
    pred_ref = (np.asarray(runner.decode_observables_batch(
        dets.astype(np.uint8))) % 2).astype(bool)
    if pred_ref.ndim == 1:
        pred_ref = pred_ref.reshape(-1, 1)

    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=5, early_exit=True,
                                      dtype=dtype, block=block, **RELAY_CFG)
    pred_m = mega.decode_batch(dets, device=DEVICE)
    assert pred_m.shape == pred_ref.shape
    fails_ref = int(np.any(pred_ref != obs, axis=1).sum())
    fails_m = int(np.any(pred_m != obs, axis=1).sum())
    per_shot = float(np.all(pred_m == pred_ref, axis=1).mean())
    print(f"\nMEGA RELAY LER-IDENTITY [{dtype} BLOCK={block}]: "
          f"oracle={fails_ref} mega={fails_m} (N={len(dets)}) "
          f"per-shot-agreement={per_shot:.4f}")
    # The GATE is the LER-count bar from test_relay_triton.py (the gamma
    # RNGs differ from Rust by construction, so per-shot agreement is
    # statistical: the v0.1 HOST path measures 0.9900 fp32 / 0.9875 fp64 on
    # this exact cell -- bench/receipts/bench_relay_triton.json -- and the
    # megakernel reproduces those numbers). 0.98 is a sanity floor only.
    assert abs(fails_m - fails_ref) <= max(5, int(0.01 * len(dets))), (
        f"relay LER differs too much: oracle={fails_ref} mega={fails_m}")
    assert per_shot >= 0.98
