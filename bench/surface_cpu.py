"""Generate ``bench/receipts/surface_cpu.json`` — surface-code memory LER
receipts on CPU (second code family; tridec BP vs PyMatching).

Workload: stim-generated rotated surface-code memory-Z circuits
(``surface_code:rotated_memory_z``, all four noise knobs = p), the same
generator family as the d=3 test fixture. Cells: (d=3, r=3) and (d=5, r=5)
at p in {0.003, 0.005}.

Protocol per cell: ONE sampled shot set (fixed seed, recorded). tridec BP
decodes the raw DEM (``decompose_errors=False``); PyMatching decodes the
decomposed DEM (``decompose_errors=True`` — MWPM requires graphlike errors)
on the SAME shots. tridec BP main numbers use the torch CPU backend (fp64,
batched, chunked); a numpy-reference cross-check on the first 2000 shots is
recorded per cell (the two are the same fp64 flooding min-sum; expected
LER-identical up to rare ties).

Honest expectation, stated up front: plain min-sum BP WITHOUT post-processing
(OSD/LSD/relay) is known to lose to matching on surface codes — degenerate
weight-4 loops split BP's beliefs, and the gap WIDENS with distance. These
receipts measure exactly that. The claim is code-agnostic operation with
honest numbers, not beating matching.

Run:  python bench/surface_cpu.py        (writes bench/receipts/surface_cpu.json)
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import stim

import tridec
from tridec.validation.harness import dem_hash
from tridec.validation.stats import wilson_ci

RECEIPT = Path(__file__).resolve().parent / "receipts" / "surface_cpu.json"

CELLS = [(3, 3, 0.003), (3, 3, 0.005), (5, 5, 0.003), (5, 5, 0.005)]
SHOTS = 50_000
SEED = 20260610          # one fixed sampler seed per cell (stim's seeded
                         # sampler is platform-dependent; see docs/benchmark.md §5.1)
CHUNK = 10_000           # decode_batch chunk (results are per-shot independent)
XCHECK_SHOTS = 2_000     # numpy-reference cross-check subset


def gen_circuit(d, r, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def decode_chunked(decoder, dets):
    t0 = time.perf_counter()
    preds = [decoder.decode_batch(dets[i:i + CHUNK])
             for i in range(0, len(dets), CHUNK)]
    dt = time.perf_counter() - t0
    return np.concatenate(preds, axis=0), dt


def run_cell(d, r, p):
    import pymatching
    circ = gen_circuit(d, r, p)
    dem_raw = circ.detector_error_model(decompose_errors=False)
    dem_dec = circ.detector_error_model(decompose_errors=True)
    ex = tridec.extract(dem_raw)

    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    # --- tridec BP (torch CPU, fp64, validated defaults) ------------------- #
    bp = tridec.from_dem(dem_raw, backend="torch", device="cpu")
    pred_bp, t_bp = decode_chunked(bp, dets)
    fails_bp = int(np.any(pred_bp != obs, axis=1).sum())

    # --- numpy-reference cross-check (first XCHECK_SHOTS shots) ------------ #
    bp_np = tridec.from_dem(dem_raw, backend="numpy")
    t0 = time.perf_counter()
    pred_np = bp_np.decode_batch(dets[:XCHECK_SHOTS])
    t_np = time.perf_counter() - t0
    agree = float(np.all(pred_np == pred_bp[:XCHECK_SHOTS], axis=1).mean())
    fails_np = int(np.any(pred_np != obs[:XCHECK_SHOTS], axis=1).sum())
    fails_bp_sub = int(np.any(pred_bp[:XCHECK_SHOTS] != obs[:XCHECK_SHOTS],
                              axis=1).sum())

    # --- PyMatching (MWPM, decomposed DEM, same shots) ---------------------- #
    matching = pymatching.Matching.from_detector_error_model(dem_dec)
    t0 = time.perf_counter()
    pred_mwpm = matching.decode_batch(dets.astype(np.uint8)).astype(bool)
    t_mwpm = time.perf_counter() - t0
    fails_mwpm = int(np.any(pred_mwpm != obs, axis=1).sum())

    cell = {
        "d": d, "rounds": r, "p": p,
        "circuit": "surface_code:rotated_memory_z (stim generated; all four "
                   "noise parameters = p)",
        "shots": SHOTS, "seed": SEED,
        "dem": {"n_det": ex["n_det"], "n_obs": ex["n_obs"],
                "n_err": ex["n_err"],
                "dem_hash_raw_local": dem_hash(dem_raw),
                "dem_hash_note": "sha256 of the locally-built raw DEM; "
                                 "platform-local (see benchmark.md §5.1)"},
        "tridec_bp": {
            "backend": "torch", "device": "cpu", "dtype": "float64",
            "config": bp.config, "dem": "decompose_errors=False",
            "fails": fails_bp, "ler": fails_bp / SHOTS,
            "ler_wilson95": wilson_ci(fails_bp, SHOTS),
            "decode_s": t_bp, "shots_per_s": SHOTS / t_bp,
        },
        "pymatching_mwpm": {
            "dem": "decompose_errors=True",
            "fails": fails_mwpm, "ler": fails_mwpm / SHOTS,
            "ler_wilson95": wilson_ci(fails_mwpm, SHOTS),
            "decode_s": t_mwpm, "shots_per_s": SHOTS / t_mwpm,
        },
        "numpy_crosscheck": {
            "shots": XCHECK_SHOTS,
            "per_shot_prediction_agreement_vs_torch": agree,
            "fails_numpy": fails_np, "fails_torch_same_subset": fails_bp_sub,
            "decode_s": t_np,
        },
        "ler_ratio_bp_over_mwpm": (fails_bp / max(fails_mwpm, 1)),
    }
    print(f"d={d} r={r} p={p}: BP {fails_bp}/{SHOTS} "
          f"(LER {fails_bp/SHOTS:.5f}, {t_bp:.1f}s)  "
          f"MWPM {fails_mwpm}/{SHOTS} (LER {fails_mwpm/SHOTS:.5f}, "
          f"{t_mwpm:.1f}s)  ratio {cell['ler_ratio_bp_over_mwpm']:.2f}  "
          f"numpy-xcheck agree {agree:.4f} ({fails_np} vs {fails_bp_sub})")
    return cell


def main():
    import pymatching
    import torch
    meta = {
        "what": "surface-code memory LER receipts on CPU: tridec min-sum BP "
                "(no post-processing) vs PyMatching MWPM. Second code family "
                "after the BB-code grid.",
        "honest_reading": "Plain BP without post-processing is EXPECTED to "
                          "lose to matching on surface codes (degenerate "
                          "loops split BP's beliefs) and the gap widens with "
                          "distance. The claim is code-agnostic operation "
                          "with measured numbers, not beating matching.",
        "date": time.strftime("%Y-%m-%d"),
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version()},
        "versions": {"stim": stim.__version__,
                     "pymatching": pymatching.__version__,
                     "torch": torch.__version__,
                     "numpy": np.__version__,
                     "tridec": tridec.__version__},
        "protocol": "one sampled shot set per cell (seed recorded); BP "
                    "decodes the raw DEM, MWPM the decomposed DEM, same "
                    "shots; BP main numbers torch CPU fp64 (chunked batch, "
                    "perf_counter), numpy reference cross-checked on the "
                    "first 2000 shots per cell.",
    }
    cells = [run_cell(d, r, p) for d, r, p in CELLS]
    RECEIPT.write_text(json.dumps({"meta": meta, "cells": cells}, indent=1)
                       + "\n")
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
