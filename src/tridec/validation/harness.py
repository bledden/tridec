"""Matched-protocol cross-decoder runner + pre-committed binding gates.

This is the apples-to-apples LER runner: ONE shared DEM, ONE sampled shot set,
every decoder decoding the BYTE-IDENTICAL DEM. Silent DEM drift across a
decoder zoo is the #1 validation risk, so two gates are checked fail-fast
BEFORE any LER is reported:

  * **G1 (DEM identity).** ``dem_hash(dem)`` is the sha256 of the DEM's
    canonical bytes (``str(dem.flattened()).encode()``). It is stable across
    rebuilds of the same (code, rounds, p, basis, noise) and sensitive to any
    change. ``run_matched`` rebuilds the shared DEM from the circuit, hashes
    it, and asserts every decoder's ``.dem`` hashes to the SAME value. A
    decoder built from a different DEM raises immediately.
  * **G2 (tie-break policy).** Every decoder declares ``.tie_break`` (a short
    deterministic-tie-break string); ``run_matched`` asserts it is in
    ``APPROVED_TIE_BREAKS`` (missing / unknown -> raise).

The manifest produced by ``run_matched`` is a JSON-serializable dict with the
DEM provenance (hash, sizes), the sampling parameters, an optional git HEAD,
and one record per decoder (name, config, tie_break, fails, ler, ler_ci,
lambda_per_round, decode_s).
"""
import hashlib
import os
import subprocess
import time

import numpy as np

from . import stats

# Pre-committed deterministic tie-break policy per decoder (gate G2). Each
# decoder declares its concrete tie-break; ``run_matched`` asserts every
# decoder's ``.tie_break`` is one of these BEFORE any LER is reported, so no
# decoder can silently fall back to an undeclared ordering.
APPROVED_TIE_BREAKS = {
    "min_sum_parallel_hard_decision",   # hard decision off parallel min-sum
    "osd0_reliability_order",           # OSD-0 pivots by BP reliability order
    "osd_cs_order10",                   # OSD combination-sweep, order 10
    "lsd_cs_order10",                   # LSD combination-sweep, order 10
    "astar_beam64_lowest_cost",         # A* beam, lowest-cost coset wins ties
    "relay_bp_nconv_disjoint_ensemble", # Relay-BP nconv stop, lowest weight
    "sliding_window_bposd_cs_commit",   # per-window OSD-cs commit
}


def dem_hash(dem):
    """Gate G1 primitive: sha256 hex of the DEM's canonical bytes.

    Uses ``str(dem.flattened()).encode()`` — the flattened DEM's text form is the
    canonical mechanism listing (detectors, observables, prior per mechanism in
    order). Stable across rebuilds of the same (code, rounds, p, basis, noise);
    sensitive to any change in the mechanism set or priors.
    """
    return hashlib.sha256(str(dem.flattened()).encode()).hexdigest()


def _lambda_per_round(ler, rounds):
    """Per-round logical error rate lambda = 1 - (1 - LER)^(1/rounds).

    Inverts LER = 1 - (1 - lambda)^rounds for a memory of ``rounds`` rounds.
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds!r}")
    return 1.0 - (1.0 - ler) ** (1.0 / rounds)


def _git_head():
    """Best-effort current git HEAD (full SHA) of the working directory; None
    if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _decoder_dem_hash(decoder):
    """The hash of the DEM a decoder was built from. Every decoder must carry
    ``.dem`` (the shared DEM object it was built from); raise loudly if a
    decoder does not expose it (it cannot be provenance-checked and must NOT be
    silently trusted)."""
    dem = getattr(decoder, "dem", None)
    if dem is None:
        raise ValueError(
            f"G1: decoder {getattr(decoder, 'name', decoder)!r} exposes no .dem; "
            f"cannot verify DEM identity (build it from the shared DEM)."
        )
    return dem_hash(dem)


def run_matched(circuit, decoders, shots, rounds, seed=0, label=None,
                keep_per_shot=False):
    """Matched cross-decoder run on ONE shared DEM and ONE sampled shot set.

    Args:
        circuit: stim.Circuit (the memory experiment) — the single source of the
            shared DEM and the sampled detectors/observables.
        decoders: list of decoders built from THIS circuit's DEM. Each must
            carry ``.dem`` (provenance), ``.name``, ``.config`` and a declared
            ``.tie_break``.
        shots: number of shots to sample and decode (all decoders see the SAME
            shots).
        rounds: number of syndrome rounds (for the per-round lambda).
        seed: detector-sampler seed (fixed-seed reproducibility).
        label: optional free-text label recorded in the manifest.
        keep_per_shot: if True, ALSO return the per-decoder per-shot fail mask
            (``fail_mask``: bool[shots], aligned to the shared shots) plus the
            shared truth observables (``obs``: bool[shots, n_obs]), so paired
            gap analyses (``analysis.gap_to_mle_bootstrap`` / ``analysis.gate_a``)
            can consume decoder-vs-anchor masks WITHOUT re-decoding. Default
            False -> the manifest schema is unchanged (no per-shot arrays).
            Gates G1/G2 are enforced identically either way.

    Gates (fail-fast, BEFORE any decoding):
        * G1: rebuild the shared DEM from ``circuit``, hash it, assert every
          decoder's ``.dem`` hashes to the same value. Mismatch -> ValueError.
        * G2: assert every decoder's ``.tie_break`` is in APPROVED_TIE_BREAKS.

    Returns: a JSON-serializable manifest dict. With ``keep_per_shot=True`` the
    per-shot arrays (``obs`` at top level, ``fail_mask`` per decoder) are numpy
    bool arrays (NOT JSON-serializable) — they are an in-process hand-off for
    the gap analysis, deliberately gated off the default manifest.
    """
    if not decoders:
        raise ValueError("run_matched: no decoders supplied")

    # --- canonical hash of THIS circuit's DEM (the drift reference) -------------
    # A FRESH DEM object every call (stim never returns the same object), so the
    # G1 drift check is by canonical HASH, not by object identity.
    circuit_h = dem_hash(circuit.detector_error_model(decompose_errors=False))

    # The matched protocol requires ONE shared DEM object behind ALL decoders.
    # Take the decoders' shared DEM as `dem` and require hash-identity across them.
    dem = getattr(decoders[0], "dem", None)
    if dem is None:
        raise ValueError(
            f"G1: decoder {getattr(decoders[0], 'name', decoders[0])!r} exposes "
            f"no .dem; build decoders from the shared DEM."
        )
    h = dem_hash(dem)

    # --- G1: every decoder consumes the byte-identical, SHARED DEM -------------
    if h != circuit_h:
        raise ValueError(
            f"G1 DEM-identity FAIL: the decoders were built from a DEM hashing to "
            f"{h}, but THIS circuit's DEM hashes to {circuit_h}. Silent DEM drift "
            f"(e.g. a different p/rounds/basis/noise) is forbidden — rebuild ALL "
            f"decoders from THIS circuit's DEM (decompose_errors=False)."
        )
    for d in decoders:
        name = getattr(d, "name", repr(d))
        dh = _decoder_dem_hash(d)
        if dh != h:
            raise ValueError(
                f"G1 DEM-identity FAIL: decoder {name!r} was built from a DEM "
                f"hashing to {dh}, not the shared DEM ({h}). Every decoder must "
                f"consume the byte-identical DEM."
            )
        # Decoders that ingest the DEM object directly (e.g. an MLE anchor like
        # Tesseract) must hold the SAME shared object, so a same-text but
        # distinct DEM object cannot sneak past.
        if name == "Tesseract" and getattr(d, "dem", None) is not dem:
            raise ValueError(
                f"G1 DEM-identity FAIL: Tesseract's .dem is not the shared DEM "
                f"object the other decoders were built from (it ingests the DEM "
                f"directly and must be the same object)."
            )

    # --- G2: every decoder must declare an approved deterministic tie-break -----
    for d in decoders:
        name = getattr(d, "name", repr(d))
        tb = getattr(d, "tie_break", None)
        if tb not in APPROVED_TIE_BREAKS:
            raise ValueError(
                f"G2 tie-break FAIL: decoder {name!r} declares tie_break={tb!r}, "
                f"not in APPROVED_TIE_BREAKS={sorted(APPROVED_TIE_BREAKS)}. "
                f"Every decoder must pre-commit a verified deterministic tie-break."
            )

    # --- sample ONCE; all decoders see the SAME shots --------------------------
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    records = []
    for d in decoders:
        t0 = time.perf_counter()
        pred = d.decode_batch(dets)
        decode_s = time.perf_counter() - t0
        pred = np.asarray(pred, dtype=bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        fail_mask = np.any(pred != obs, axis=1)   # per-shot fail mask
        fails = int(fail_mask.sum())
        ler = fails / shots if shots else 0.0
        lo, hi = stats.wilson_ci(fails, shots)
        rec = {
            "name": d.name,
            "config": dict(d.config),
            "tie_break": d.tie_break,
            "fails": fails,
            "ler": ler,                       # block-LER (per-shot logical failure)
            "ler_ci": [lo, hi],
            "lambda_per_round": _lambda_per_round(ler, rounds),
            "decode_s": decode_s,
        }
        if keep_per_shot:
            rec["fail_mask"] = fail_mask
        records.append(rec)

    manifest = {
        "dem_hash": h,
        "n_det": dem.num_detectors,
        "n_obs": dem.num_observables,
        "shots": shots,
        "seed": seed,
        "rounds": rounds,
        "label": label,
        "git_head": _git_head(),
        "decoders": records,
    }
    if keep_per_shot:
        manifest["obs"] = obs
    return manifest
