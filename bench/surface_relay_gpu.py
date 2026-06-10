"""Generate ``bench/receipts/surface_relay_mi300x.json``.

OFFICIAL Relay-BP surface-code receipts (the §3.4 TODO): Relay-BP through the
packaged tridec API (RelayBpDecoder, triton fp32) on a CUDA/ROCm GPU, on the
same four surface-code memory cells as ``bench/receipts/surface_cpu.json``
(rotated memory-Z, (d=3, r=3) and (d=5, r=5) at p ∈ {0.003, 0.005}), 50,000
shots per cell, seed 20260610 — vs PyMatching MWPM **regenerated on this host
for the SAME sampled shots** (stim's seeded sampler is platform-dependent, so
the darwin counts in surface_cpu.json are not shot-comparable; the matched
MWPM column here is). Triton min-sum BP (fp32) decodes the same shots too
(the plain-BP baseline + the Triton-BP surface-throughput TODO).

Protocol mirrors bench/surface_cpu.py / bench/surface_relay_metal.py: one
shot set per cell, BP/relay on the raw DEM, MWPM on the decomposed DEM, same
shots. Relay config = validated defaults (the relay_bp oracle configuration);
GPU decodes chunked at 8192 shots/batch (decode_s = summed wall-clock over
chunks, synchronize-bracketed).

Run (CUDA/ROCm GPU, package installed):  python bench/surface_relay_gpu.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import stim

import tridec
from tridec.validation.stats import wilson_ci

REPO = Path(__file__).resolve().parent.parent
RECEIPT = REPO / "bench" / "receipts" / "surface_relay_mi300x.json"
SURFACE_CPU = REPO / "bench" / "receipts" / "surface_cpu.json"

CELLS = [(3, 3, 0.003), (3, 3, 0.005), (5, 5, 0.003), (5, 5, 0.005)]
SHOTS = 50_000          # same per-cell N as surface_cpu.json
SEED = 20260610         # same seed convention (platform-local stream)
CHUNK = 8192


def gen_circuit(d, r, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=r,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)


def decode_chunked(decoder, dets):
    """Chunked decode_batch; wall-clock summed over synchronize-bracketed
    chunks (the API's device->host copy already syncs; explicit for rigor)."""
    import torch
    preds, total = [], 0.0
    for lo in range(0, dets.shape[0], CHUNK):
        chunk = dets[lo:lo + CHUNK]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds.append(decoder.decode_batch(chunk))
        torch.cuda.synchronize()
        total += time.perf_counter() - t0
    return np.concatenate(preds, axis=0), total


def run_cell(d, r, p, darwin_cell):
    import pymatching
    circ = gen_circuit(d, r, p)
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        SHOTS, separate_observables=True)
    dets = np.asarray(dets, dtype=bool)
    obs = np.asarray(obs, dtype=bool)

    relay = tridec.from_dem(dem, algorithm="relay", backend="triton",
                            dtype="float32")
    pred_r, t_r = decode_chunked(relay, dets)
    fails_r = int(np.any(pred_r != obs, axis=1).sum())

    bp = tridec.from_dem(dem, backend="triton")
    pred_b, t_b = decode_chunked(bp, dets)
    fails_b = int(np.any(pred_b != obs, axis=1).sum())

    # PyMatching MWPM on the decomposed DEM, SAME shots, this host's CPU.
    m = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    t0 = time.perf_counter()
    pred_m = m.decode_batch(dets.astype(np.uint8)).astype(bool)
    t_m = time.perf_counter() - t0
    fails_m = int(np.any(pred_m != obs, axis=1).sum())

    lo_r, hi_r = wilson_ci(fails_r, SHOTS)
    lo_m, hi_m = wilson_ci(fails_m, SHOTS)
    cell = {
        "d": d, "rounds": r, "p": p, "shots": SHOTS, "seed": SEED,
        "n_det": dem.num_detectors, "n_obs": dem.num_observables,
        "relay_triton_fp32": {
            "config": relay.config, "fails": fails_r, "ler": fails_r / SHOTS,
            "ler_wilson95": [lo_r, hi_r], "decode_s": t_r,
            "shots_per_s": round(SHOTS / t_r, 1),
        },
        "bp_triton_fp32": {
            "config": bp.config, "fails": fails_b, "ler": fails_b / SHOTS,
            "ler_wilson95": list(wilson_ci(fails_b, SHOTS)), "decode_s": t_b,
            "shots_per_s": round(SHOTS / t_b, 1),
        },
        "pymatching_mwpm_same_shots": {
            "fails": fails_m, "ler": fails_m / SHOTS,
            "ler_wilson95": [lo_m, hi_m], "decode_s": t_m,
        },
        "relay_vs_mwpm": {
            "ler_ratio": (fails_r / fails_m) if fails_m else None,
            "wilson_ci_overlap": bool(lo_r <= hi_m and lo_m <= hi_r),
        },
        "darwin_surface_cpu_reference": ({
            "note": ("surface_cpu.json counts on the SAME (d, r, p, shots, "
                     "seed) but a DIFFERENT platform-local shot stream "
                     "(darwin/arm64) — context only, not shot-matched"),
            "bp_torch_cpu_fails": darwin_cell["tridec_bp"]["fails"],
            "pymatching_fails": darwin_cell["pymatching_mwpm"]["fails"],
        } if darwin_cell else None),
    }
    print(f"d={d} r={r} p={p} N={SHOTS}: relay {fails_r} ({t_r:.1f}s) | "
          f"bp {fails_b} ({t_b:.1f}s) | mwpm {fails_m} ({t_m:.1f}s) | "
          f"relay/mwpm {cell['relay_vs_mwpm']['ler_ratio']:.2f}", flush=True)
    return cell


def main():
    import importlib.metadata as md

    import pymatching
    import torch
    import triton
    props = torch.cuda.get_device_properties(0)
    darwin = json.loads(SURFACE_CPU.read_text())["cells"]

    def darwin_cell(d, r, p):
        for c in darwin:
            if c["d"] == d and c["rounds"] == r and c["p"] == p:
                return c
        return None

    meta = {
        "STATUS": ("OFFICIAL Relay-BP surface receipts (triton fp32, ROCm "
                   "GPU) — supersedes the PRELIMINARY Metal sample for the "
                   "official table; the Metal receipt remains as the "
                   "Metal-path record."),
        "date": time.strftime("%Y-%m-%d"),
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version(),
                     "gpu_arch": getattr(props, "gcnArchName", None),
                     "gpu_name": props.name,
                     "rocm_hip": torch.version.hip},
        "versions": {"stim": stim.__version__,
                     "pymatching": pymatching.__version__,
                     "torch": torch.__version__,
                     "triton": triton.__version__,
                     "numpy": np.__version__,
                     "tridec": tridec.__version__,
                     "relay_bp_oracle": md.version("relay-bp")
                     if any(True for _ in [0]) else None},
        "protocol": ("one shot set per cell (seed recorded; platform-local "
                     "stream); relay + BP on the raw DEM via the packaged "
                     "API (backend='triton', fp32), MWPM on the decomposed "
                     "DEM, SAME shots, this host. Relay config = validated "
                     "defaults (relay_bp oracle configuration); decodes "
                     f"chunked at {CHUNK}."),
    }
    cells = [run_cell(d, r, p, darwin_cell(d, r, p)) for d, r, p in CELLS]
    RECEIPT.write_text(json.dumps({"meta": meta, "cells": cells}, indent=1)
                       + "\n")
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
