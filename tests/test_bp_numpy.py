"""Numpy min-sum BP reference: hand-derived one-iteration identity + ldpc parity.

Ported from the source repo's kernel TDD suite. The one-iteration math is
pinned bit-identically against a hand-derived normalized-min-sum reference on
the 3-bit repetition code; functional LER-equivalence to ldpc.BpDecoder (same
config) is checked on the small surface-code fixture (skip-guarded on ldpc).
"""
import numpy as np
import pytest

from conftest import load_surface_circuit

from tridec.backends.bp_numpy import BpBaseline
from tridec.dem import extract
from tridec.validation import wilson_ci

MS = 0.625  # normalized min-sum scaling factor (the validated default)


# --------------------------------------------------------------------------- #
# 1. HAND-DERIVED one-iteration normalized-min-sum, 3-bit repetition code.     #
# --------------------------------------------------------------------------- #
def _hand_repetition_one_iter():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    p = np.array([0.10, 0.20, 0.05])              # distinct per-bit error probs
    lam = np.log((1.0 - p) / p)                   # prior LLRs (all positive)
    syndrome = np.array([1, 0], dtype=np.uint8)   # c0 fired, c1 did not

    sgn = np.sign
    s0 = (-1.0)  # check c0 fired -> coset sign -1
    m_c0_b0 = MS * s0 * sgn(lam[1]) * abs(lam[1])
    m_c0_b1 = MS * s0 * sgn(lam[0]) * abs(lam[0])
    s1 = (+1.0)
    m_c1_b1 = MS * s1 * sgn(lam[2]) * abs(lam[2])
    m_c1_b2 = MS * s1 * sgn(lam[1]) * abs(lam[1])

    check_to_bit = {(0, 0): m_c0_b0, (0, 1): m_c0_b1,
                    (1, 1): m_c1_b1, (1, 2): m_c1_b2}
    L0 = lam[0] + m_c0_b0
    L1 = lam[1] + m_c0_b1 + m_c1_b1
    L2 = lam[2] + m_c1_b2
    posterior = np.array([L0, L1, L2])
    hard = (posterior < 0).astype(np.uint8)
    return dict(H=H, priors=p, lam=lam, syndrome=syndrome,
                check_to_bit=check_to_bit, posterior=posterior, hard=hard)


def test_one_iteration_check_to_bit_matches_hand_reference():
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=1, return_messages=True)
    got = trace["check_to_bit"]
    for key, want in ref["check_to_bit"].items():
        assert np.isclose(got[key], want, rtol=0, atol=1e-12), (
            f"check_to_bit{key}: got {got[key]} want {want}")


def test_one_iteration_posterior_and_hard_decision_match_hand_reference():
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=1, return_messages=True)
    assert np.allclose(trace["posterior"], ref["posterior"], rtol=0, atol=1e-12)
    e_hat = bp.decode(ref["syndrome"])
    assert np.array_equal(e_hat.astype(np.uint8), ref["hard"])


def test_initial_bit_to_check_messages_are_priors():
    ref = _hand_repetition_one_iter()
    bp = BpBaseline(ref["H"], priors=ref["priors"], max_iter=1,
                    ms_scaling_factor=MS, schedule="parallel")
    trace = bp.run_iterations(ref["syndrome"], n_iter=0, return_messages=True)
    for (c, v), val in trace["bit_to_check"].items():
        assert np.isclose(val, ref["lam"][v], atol=1e-12)


def test_rejects_bad_inputs():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        BpBaseline(H, priors=[0.1, 0.1], max_iter=1)        # wrong priors length
    with pytest.raises(ValueError):
        BpBaseline(H, priors=[0.1, 0.1, 0.1], schedule="serial")
    bp = BpBaseline(H, priors=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        bp.run_iterations(np.array([1, 0, 1]))              # wrong syndrome length


# --------------------------------------------------------------------------- #
# 2. Functional LER-equivalence to ldpc.BpDecoder on the surface fixture.      #
# --------------------------------------------------------------------------- #
SHOTS = 2000
SEED = 0


@pytest.fixture(scope="module")
def surface_dem_shots():
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_decode_batch_shape_and_dtype(surface_dem_shots):
    dem, dets, obs = surface_dem_shots
    bp = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred = bp.decode_batch(dets)
    assert pred.shape == (SHOTS, dem.num_observables)
    assert pred.dtype == bool


def test_ler_matches_ldpc_bp_within_ci(surface_dem_shots):
    """Block LER must agree with ldpc.BpDecoder (identical config) on the SAME
    shots: overlapping Wilson CIs and counts within a handful of shots.
    Bit-identity to ldpc's C++ internals is NOT required (ties/early-stop can
    diverge); the hand-derived one-iteration test pins the math."""
    ldpc = pytest.importorskip("ldpc")
    dem, dets, obs = surface_dem_shots
    ours = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred_ours = ours.decode_batch(dets)
    fails_ours = int(np.any(pred_ours != obs, axis=1).sum())

    ex = extract(dem)
    pri = list(np.clip(ex["priors"], 1e-6, 1 - 1e-6))
    dec = ldpc.BpDecoder(ex["H"], error_channel=pri, max_iter=30,
                         bp_method="minimum_sum", ms_scaling_factor=MS,
                         schedule="parallel")
    Lo = ex["Lo"].toarray().astype(np.uint8)
    syn = dets.astype(np.uint8)
    fails_ldpc = 0
    for i in range(SHOTS):
        e = np.asarray(dec.decode(syn[i]), dtype=np.uint8)
        p = (Lo @ e) & 1
        if not np.array_equal(p.astype(bool), obs[i]):
            fails_ldpc += 1

    lo_o, hi_o = wilson_ci(fails_ours, SHOTS)
    lo_l, hi_l = wilson_ci(fails_ldpc, SHOTS)
    assert lo_o <= hi_l and lo_l <= hi_o, (
        f"CIs disjoint: ours fails={fails_ours} CI=({lo_o:.4f},{hi_o:.4f}) "
        f"vs ldpc fails={fails_ldpc} CI=({lo_l:.4f},{hi_l:.4f})")
    assert abs(fails_ours - fails_ldpc) <= max(5, int(0.05 * max(fails_ldpc, 1))), (
        f"fail counts diverge: ours={fails_ours} ldpc={fails_ldpc}")


def test_better_than_chance(surface_dem_shots):
    dem, dets, obs = surface_dem_shots
    bp = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    pred = bp.decode_batch(dets)
    ler = float(np.any(pred != obs, axis=1).mean())
    assert ler < 0.5
