"""Validation layer: dem_hash, run_matched gates, and the stats primitives."""
import numpy as np
import pytest

from conftest import load_bb_circuit, load_surface_circuit

import tridec
from tridec.validation import (
    bh_fdr,
    dem_hash,
    gap_to_mle_bootstrap,
    gate_a,
    holm_bonferroni,
    per_shot_fails,
    run_matched,
    tost_equivalent,
    wilson_ci,
    wilson_consistent,
)


# --------------------------------------------------------------------------- #
# dem_hash.                                                                    #
# --------------------------------------------------------------------------- #
def test_dem_hash_stable_across_rebuilds():
    c = load_surface_circuit()
    h1 = dem_hash(c.detector_error_model(decompose_errors=False))
    h2 = dem_hash(c.detector_error_model(decompose_errors=False))
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_dem_hash_sensitive_to_model_change():
    h_surface = dem_hash(
        load_surface_circuit().detector_error_model(decompose_errors=False))
    h_bb = dem_hash(
        load_bb_circuit("0.003", "Z").detector_error_model(decompose_errors=False))
    assert h_surface != h_bb


# --------------------------------------------------------------------------- #
# run_matched: gates G1/G2 + manifest schema.                                  #
# --------------------------------------------------------------------------- #
class _FakeDecoder:
    def __init__(self, dem, name="Fake", tie_break="min_sum_parallel_hard_decision"):
        self.dem = dem
        self.name = name
        self.config = {"decoder": "Fake"}
        self.tie_break = tie_break

    def decode_batch(self, dets):
        return np.zeros((dets.shape[0], self.dem.num_observables), dtype=bool)


def test_run_matched_g1_rejects_foreign_dem():
    surface = load_surface_circuit()
    other = load_bb_circuit("0.003", "Z")
    foreign = _FakeDecoder(other.detector_error_model(decompose_errors=False))
    with pytest.raises(ValueError, match="G1"):
        run_matched(surface, [foreign], shots=10, rounds=3, seed=0)


def test_run_matched_g2_rejects_undeclared_tie_break():
    surface = load_surface_circuit()
    dem = surface.detector_error_model(decompose_errors=False)
    bad = _FakeDecoder(dem, tie_break="coin_flip")
    with pytest.raises(ValueError, match="G2"):
        run_matched(surface, [bad], shots=10, rounds=3, seed=0)


def test_run_matched_manifest_schema_and_determinism():
    surface = load_surface_circuit()
    dem = surface.detector_error_model(decompose_errors=False)
    dec = tridec.from_dem(dem, backend="numpy")
    m1 = run_matched(surface, [dec], shots=200, rounds=3, seed=0)
    m2 = run_matched(surface, [dec], shots=200, rounds=3, seed=0)
    assert m1["dem_hash"] == dem_hash(dem)
    assert m1["shots"] == 200 and m1["seed"] == 0 and m1["rounds"] == 3
    rec = m1["decoders"][0]
    for k in ("name", "config", "tie_break", "fails", "ler", "ler_ci",
              "lambda_per_round", "decode_s"):
        assert k in rec
    # fixed seed -> identical fail counts across runs
    assert rec["fails"] == m2["decoders"][0]["fails"]
    assert 0 <= rec["ler"] <= 1


def test_run_matched_keep_per_shot():
    surface = load_surface_circuit()
    dem = surface.detector_error_model(decompose_errors=False)
    dec = tridec.from_dem(dem, backend="numpy")
    m = run_matched(surface, [dec], shots=100, rounds=3, seed=0,
                    keep_per_shot=True)
    rec = m["decoders"][0]
    assert rec["fail_mask"].shape == (100,)
    assert m["obs"].shape == (100, dem.num_observables)
    assert int(rec["fail_mask"].sum()) == rec["fails"]


# --------------------------------------------------------------------------- #
# Stats primitives.                                                            #
# --------------------------------------------------------------------------- #
def test_wilson_ci_basics():
    lo, hi = wilson_ci(0, 1000)
    assert lo == 0.0 or lo < 1e-6
    assert 0 < hi < 0.01
    lo, hi = wilson_ci(500, 1000)
    assert lo < 0.5 < hi
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_consistent():
    # Close rates at the same N: CIs overlap -> consistent (the relay-vs-oracle
    # regime: 39 vs 35 / 2000).
    assert wilson_consistent(39, 2000, 35, 2000) is True
    assert wilson_consistent(168, 2000, 168, 2000) is True
    # Far-apart rates: CIs disjoint -> inconsistent (bare-BP vs a good decoder).
    assert wilson_consistent(700, 2000, 39, 2000) is False
    # Sample-size-aware: the SAME absolute gap is tolerated at small N but not
    # at large N (this is the whole point vs an ad-hoc count bar).
    assert wilson_consistent(8, 200, 2, 200) is True          # small N, overlap
    assert wilson_consistent(800, 20000, 200, 20000) is False  # large N, disjoint
    # Symmetric in argument order.
    assert (wilson_consistent(39, 2000, 35, 2000)
            == wilson_consistent(35, 2000, 39, 2000))


def test_tost_equivalent_identical_rates():
    # n=100k at 5% rate is powered for a 10% relative margin; n=10k is not.
    assert tost_equivalent(5000, 100000, 5000, 100000, margin_rel=0.10) is True
    assert tost_equivalent(100, 10000, 300, 10000, margin_rel=0.10) is False


def test_holm_and_bh():
    pvals = [0.001, 0.04, 0.9]
    holm = holm_bonferroni(pvals)
    bh = bh_fdr(pvals)
    assert holm[0] is True and holm[2] is False
    assert bh[0] is True and bh[2] is False


def test_gap_to_mle_bootstrap_paired():
    rng = np.random.default_rng(0)
    anchor = rng.random(5000) < 0.01
    dec = anchor | (rng.random(5000) < 0.01)  # decoder strictly worse
    out = gap_to_mle_bootstrap(dec, anchor, n_boot=500, seed=1)
    assert out["ratio"] >= 1.0
    assert out["lo"] <= out["ratio"] <= out["hi"]
    # zero conventions
    z = np.zeros(100, dtype=bool)
    assert gap_to_mle_bootstrap(z, z, n_boot=10, seed=0)["ratio"] == 1.0
    assert gap_to_mle_bootstrap(~z, z, n_boot=10, seed=0)["ratio"] == float("inf")


def test_gate_a_counts():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    out = gate_a(a, b)
    assert out["n"] == 4
    assert out["both_wrong"] == 1
    assert out["both_correct"] == 1
    assert out["only_a_right"] == 1
    assert out["only_b_right"] == 1
    assert out["p_tesseract_succeeds_given_bposd_fails"] == 0.5
    assert out["oracle_ler_bound"] == 0.25


def test_per_shot_fails():
    pred = np.array([[True, False], [False, False]])
    obs = np.array([[True, False], [True, False]])
    assert per_shot_fails(pred, obs).tolist() == [False, True]
