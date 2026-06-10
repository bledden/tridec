"""Statistics / verdict layer over matched-run manifests.

Turns matched cross-decoder results into defensible verdicts (ported unchanged
from the validated research analysis module):

  * ``beat`` / ``tie``  -- Wilson-CI separation (with the 300-failure-target
    guard) and TOST equivalence at the declared margin. CI-overlap is NOT a tie.
  * ``gap_to_mle_bootstrap`` (R2, BINDING) -- the LOCKED gap-to-MLE statistic =
    a per-shot PAIRED bootstrap of decoder-vs-Tesseract-MLE failure indicators
    on the SAME sampled shots (the pre-registered ``per_shot_ratio_bootstrap`` statistic). Resamples shot INDICES with
    replacement so the same indices index BOTH arms (respects the shot-level
    correlation between the decoder and the MLE anchor). Seeded / deterministic.
  * ``multiplicity_grid`` -- Holm-Bonferroni (primary, FWER) + BH-FDR (secondary)
    over the FULL grid of (decoder, p, basis) comparisons, INCLUDING losses
    (the full grid INCLUDING losses).
  * ``gate_a`` -- the BP-OSD-vs-Tesseract 2x2 disagreement diagnostic, the
    conditional P(Tesseract succeeds | BP-OSD fails), and the oracle-ensemble LER
    bound (fraction where at least one is correct -> 1 - both_wrong/n).
  * ``per_shot_fails`` -- the per-shot fail-mask helper the bootstrap / Gate-A
    consume (``np.any(pred != obs, axis=1)``).

The gap statistic MUST be the per-shot paired bootstrap (NOT an aggregate ratio
of two marginal counts with a single Wilson CI); it is the binding R2 statistic.
"""
from statistics import NormalDist

import numpy as np

from . import stats


# --- beat / tie -------------------------------------------------------------
def beat(fails_a, n_a, fails_b, n_b, failure_target=300):
    """A beats B iff their Wilson CIs are non-overlapping (A strictly lower) AND
    BOTH cells reached at least ``failure_target`` failures.

    Below the failure target a beat CANNOT be claimed (under-powered cell): the
    statistic is not yet at its pre-committed precision, so even a separated CI
    is not a defensible win.
    """
    if fails_a < failure_target or fails_b < failure_target:
        return False
    lo_a, hi_a = stats.wilson_ci(fails_a, n_a)
    lo_b, hi_b = stats.wilson_ci(fails_b, n_b)
    # A beats B: A's whole CI lies strictly below B's whole CI.
    return bool(hi_a < lo_b)


def tie(fails_a, n_a, fails_b, n_b, margin_rel):
    """A ties B iff TOST declares equivalence at the declared relative margin.

    CI-overlap (i.e. ``not beat``) is NOT a tie: a tie is a POSITIVE claim of
    equivalence, which requires the TOST procedure at the pre-committed margin.
    """
    return stats.tost_equivalent(fails_a, n_a, fails_b, n_b, margin_rel=margin_rel)


# --- gap_to_mle_bootstrap (R2, BINDING) ------------------------------------
def gap_to_mle_bootstrap(decoder_fail_mask, anchor_fail_mask, n_boot=10000, seed=0):
    """Per-shot PAIRED bootstrap of the decoder-vs-MLE failure RATIO (binding R2).

    This is the locked gap-to-MLE statistic (``per_shot_ratio_bootstrap``):
    NOT an aggregate ratio of two marginal failure counts with a single Wilson CI.

    Args:
        decoder_fail_mask: bool[shots] — decoder failed on shot i.
        anchor_fail_mask:  bool[shots] — Tesseract-MLE anchor failed on shot i.
            The two arrays are PAIRED over the SAME sampled shots (identical DEM
            + seed -> identical detectors/observables), so element i of both
            refers to the same shot.
        n_boot: number of bootstrap resamples.
        seed:   seed for a numpy ``default_rng`` Generator (deterministic).

    Returns:
        ``{"ratio", "lo", "hi"}`` where:
          * ratio = mean(decoder_fail) / mean(anchor_fail) (the POINT estimate),
          * (lo, hi) = the 2.5 / 97.5 percentile bootstrap CI over n_boot paired
            resamples of the shot INDICES (same indices index both arms, so the
            decoder/anchor shot-level correlation is preserved).

    Anchor-zero handling: if the anchor never fails on a (resampled) shot set the
    ratio is +inf when the decoder fails there (gap is unbounded -> reported as
    an inf bound) and 1.0 when neither fails (0/0 -> finite sentinel; no decoder
    excess to report). The point ratio uses the same convention on the full set.
    """
    dec = np.asarray(decoder_fail_mask, dtype=bool)
    anch = np.asarray(anchor_fail_mask, dtype=bool)
    if dec.shape != anch.shape:
        raise ValueError(
            f"gap_to_mle_bootstrap: paired masks must have the same shape; got "
            f"decoder {dec.shape} vs anchor {anch.shape}."
        )
    if dec.ndim != 1:
        raise ValueError(
            f"gap_to_mle_bootstrap: masks must be 1-D per-shot arrays; got ndim "
            f"{dec.ndim}."
        )
    n = dec.shape[0]
    if n == 0:
        raise ValueError("gap_to_mle_bootstrap: empty masks (need >= 1 shot).")

    def _ratio(d_mean, a_mean):
        # decoder / anchor with the documented zero conventions.
        if a_mean == 0.0:
            return 1.0 if d_mean == 0.0 else float("inf")
        return d_mean / a_mean

    point = _ratio(dec.mean(), anch.mean())

    rng = np.random.default_rng(seed)
    dec_f = dec.astype(np.float64)
    anch_f = anch.astype(np.float64)
    ratios = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # ONE index set -> both arms (PAIRED)
        ratios[b] = _ratio(dec_f[idx].mean(), anch_f[idx].mean())

    # method="nearest" picks an actual resampled value rather than interpolating
    # between two neighbours — interpolation across +inf entries (anchor-zero
    # resamples) would subtract inf-inf -> NaN. "nearest" keeps inf bounds intact.
    lo = float(np.percentile(ratios, 2.5, method="nearest"))
    hi = float(np.percentile(ratios, 97.5, method="nearest"))
    return {"ratio": float(point), "lo": lo, "hi": hi}


# --- multiplicity_grid ------------------------------------------------------
def _two_prop_pvalue(fails_a, n_a, fails_b, n_b):
    """Two-sided two-proportion z-test p-value (pooled-variance normal approx)."""
    if n_a == 0 or n_b == 0:
        return 1.0
    p1 = fails_a / n_a
    p2 = fails_b / n_b
    p_pool = (fails_a + fails_b) / (n_a + n_b)
    se = (p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b)) ** 0.5
    if se == 0.0:
        return 1.0  # identical degenerate rates -> no evidence of a difference
    z = (p1 - p2) / se
    return float(2.0 * (1.0 - NormalDist().cdf(abs(z))))


def multiplicity_grid(comparisons):
    """Annotate the FULL grid of pairwise comparisons with raw p + multiplicity.

    Args:
        comparisons: list of dicts, each describing one (decoder, p, basis) cell
            comparison with the four counts ``fails_a, n_a, fails_b, n_b``
            (decoder vs its bar). Any extra identity keys (decoder/p/basis/...)
            are passed through untouched.

    Returns:
        a list (same length / order as the input — the FULL grid, INCLUDING
        losses) where each
        comparison gains:
          * ``p``        -- raw two-proportion two-sided p-value,
          * ``holm_sig`` -- Holm-Bonferroni significance (primary, FWER alpha=0.05),
          * ``bh_sig``   -- BH-FDR significance (secondary, q=0.05).
    """
    if not comparisons:
        return []
    pvals = [
        _two_prop_pvalue(c["fails_a"], c["n_a"], c["fails_b"], c["n_b"])
        for c in comparisons
    ]
    holm = stats.holm_bonferroni(pvals)
    bh = stats.bh_fdr(pvals)
    annotated = []
    for c, p, hsig, bsig in zip(comparisons, pvals, holm, bh):
        rec = dict(c)  # passthrough identity + counts, do not mutate the input
        rec["p"] = p
        rec["holm_sig"] = bool(hsig)
        rec["bh_sig"] = bool(bsig)
        annotated.append(rec)
    return annotated


# --- gate_a -----------------------------------------------------------------
def gate_a(bp_osd_fail_mask, tesseract_fail_mask):
    """Gate-A diagnostic: BP-OSD (A) vs Tesseract (B) disagreement over SAME shots.

    Args:
        bp_osd_fail_mask:   bool[shots] — BP-OSD (A) failed on shot i.
        tesseract_fail_mask: bool[shots] — Tesseract (B) failed on shot i.

    Returns a dict with:
      * ``n``             -- number of shots,
      * the 2x2 disagreement counts (over the same shots):
          - ``both_correct`` (neither failed),
          - ``both_wrong``   (both failed),
          - ``only_a_right`` (A correct, B wrong),
          - ``only_b_right`` (A wrong, B correct),
      * ``p_tesseract_succeeds_given_bposd_fails`` -- P(B succeeds | A fails) =
        only_b_right / (both_wrong + only_b_right); ``None`` if A never fails,
      * ``oracle_ler_bound`` -- the oracle-ensemble LER (fraction where AT LEAST
        ONE is correct fails only when BOTH fail) = both_wrong / n.
    """
    a = np.asarray(bp_osd_fail_mask, dtype=bool)
    b = np.asarray(tesseract_fail_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(
            f"gate_a: masks must have the same shape; got A {a.shape} vs B "
            f"{b.shape}."
        )
    n = int(a.shape[0])
    both_wrong = int(np.sum(a & b))
    both_correct = int(np.sum(~a & ~b))
    only_a_right = int(np.sum(~a & b))  # A correct, B wrong
    only_b_right = int(np.sum(a & ~b))  # A wrong,   B correct
    a_fails = both_wrong + only_b_right  # all shots where A (BP-OSD) failed
    if a_fails == 0:
        p_cond = None
    else:
        p_cond = only_b_right / a_fails
    oracle_bound = (both_wrong / n) if n else 0.0
    return {
        "n": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_a_right": only_a_right,
        "only_b_right": only_b_right,
        "p_tesseract_succeeds_given_bposd_fails": p_cond,
        "oracle_ler_bound": oracle_bound,
    }


# --- per_shot_fails ---------------------------------------------------------
def per_shot_fails(pred, obs):
    """Per-shot fail mask = ``np.any(pred != obs, axis=1)`` (one bool per shot).

    ``pred`` / ``obs`` are bool[shots, n_obs]; a shot "fails" iff any predicted
    observable disagrees with the truth. This is the mask the paired bootstrap
    and Gate-A consume.
    """
    pred = np.asarray(pred, dtype=bool)
    obs = np.asarray(obs, dtype=bool)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    return np.any(pred != obs, axis=1)
