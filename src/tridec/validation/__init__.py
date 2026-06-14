"""Validation layer: DEM provenance hashing, the matched-protocol runner, and
the statistics that make decoder comparisons defensible.

This is the credibility substrate of the package: every benchmark number the
decoders claim is producible through ``run_matched`` (one shared DEM, one shot
set, fail-fast provenance gates) and the paired statistics here.
"""
from .analysis import (
    beat,
    gap_to_mle_bootstrap,
    gate_a,
    multiplicity_grid,
    per_shot_fails,
    tie,
)
from .harness import APPROVED_TIE_BREAKS, dem_hash, run_matched
from .stats import (
    bh_fdr,
    failures_to_shots,
    holm_bonferroni,
    tost_equivalent,
    wilson_ci,
    wilson_consistent,
)

__all__ = [
    "APPROVED_TIE_BREAKS",
    "beat",
    "bh_fdr",
    "dem_hash",
    "failures_to_shots",
    "gap_to_mle_bootstrap",
    "gate_a",
    "holm_bonferroni",
    "multiplicity_grid",
    "per_shot_fails",
    "run_matched",
    "tie",
    "tost_equivalent",
    "wilson_ci",
    "wilson_consistent",
]
