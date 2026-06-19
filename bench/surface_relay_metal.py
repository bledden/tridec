"""Generate ``bench/receipts/surface_relay_metal_preliminary.json``.

PRELIMINARY / EXPERIMENTAL receipts: Relay-BP on surface-code memory via the
**experimental Metal backend** (triton-msl, fp32, single machine). These
are NOT official Relay-BP surface numbers — those require the CUDA/ROCm
fp32/fp64 paths and are deferred to the Phase-3 GPU session. Collected here
because the Metal env was at hand and the numbers are interesting: Relay-BP's
disordered memory recovers most of the plain-BP-vs-matching gap on surface
codes.

Protocol mirrors bench/surface_cpu.py (same generator family, one shot set
per cell, BP/relay on the raw DEM, MWPM on the decomposed DEM, same shots),
but with PRELIMINARY shot counts (2000/cell — relay on Metal is launch-bound,
~7-16 ms/shot) and the experimental fp32 Metal kernels.

Run (in the triton-msl environment):  python bench/surface_relay_metal.py
"""
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import stim

import tridec
from tridec.validation.stats import wilson_ci

RECEIPT = (Path(__file__).resolve().parent / "receipts"
           / "surface_relay_metal_preliminary.json")

CELLS = [(3, 3, 0.003), (3, 3, 0.005), (5, 5, 0.003), (5, 5, 0.005)]
SHOTS = 2_000          # PRELIMINARY: relay-on-Metal is launch-bound
SEED = 20260610        # same seed family as surface_cpu.py


def gen_circuit(d, r, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def run_cell(d, r, p):
    import pymatching
    circ = gen_circuit(d, r, p)
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    relay = tridec.from_dem(dem, algorithm="relay", backend="metal")
    t0 = time.perf_counter()
    pred_r = relay.decode_batch(dets)
    t_r = time.perf_counter() - t0
    fails_r = int(np.any(pred_r != obs, axis=1).sum())

    bp = tridec.from_dem(dem, backend="metal")
    pred_b = bp.decode_batch(dets)
    fails_b = int(np.any(pred_b != obs, axis=1).sum())

    m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    pred_m = m.decode_batch(dets.astype(np.uint8)).astype(bool)
    fails_m = int(np.any(pred_m != obs, axis=1).sum())

    cell = {
        "d": d, "rounds": r, "p": p, "shots": SHOTS, "seed": SEED,
        "relay_metal_fp32_PRELIMINARY": {
            "config": relay.config, "fails": fails_r, "ler": fails_r / SHOTS,
            "ler_wilson95": wilson_ci(fails_r, SHOTS), "decode_s": t_r,
        },
        "bp_metal_fp32": {"fails": fails_b, "ler": fails_b / SHOTS,
                          "ler_wilson95": wilson_ci(fails_b, SHOTS)},
        "pymatching_mwpm": {"fails": fails_m, "ler": fails_m / SHOTS,
                            "ler_wilson95": wilson_ci(fails_m, SHOTS)},
    }
    print(f"d={d} r={r} p={p} N={SHOTS}: relay {fails_r} ({t_r:.1f}s) | "
          f"bp {fails_b} | mwpm {fails_m}")
    return cell


def main():
    warnings.filterwarnings("ignore", message=".*metal backend is EXPERIMENTAL.*")
    import pymatching
    import torch
    import triton
    meta = {
        "STATUS": "PRELIMINARY / EXPERIMENTAL — fp32 Metal (triton-msl), "
                  "one machine, small shot counts. NOT official Relay-BP "
                  "surface receipts; official numbers are deferred to the "
                  "Phase-3 CUDA/ROCm session (docs/benchmark.md TODO).",
        "date": time.strftime("%Y-%m-%d"),
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version()},
        "versions": {"stim": stim.__version__,
                     "pymatching": pymatching.__version__,
                     "torch": torch.__version__,
                     "triton": triton.__version__,
                     "numpy": np.__version__,
                     "tridec": tridec.__version__},
        "protocol": "one shot set per cell (seed recorded); relay + BP on "
                    "the raw DEM via backend='metal' (fp32, CPU tensors), "
                    "MWPM on the decomposed DEM, same shots. Relay config = "
                    "validated defaults (relay_bp oracle configuration).",
    }
    cells = [run_cell(d, r, p) for d, r, p in CELLS]
    RECEIPT.write_text(json.dumps({"meta": meta, "cells": cells}, indent=1)
                       + "\n")
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
