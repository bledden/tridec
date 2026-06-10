"""Generate ``bench/receipts/mi300x_packaged.json``.

Cross-vendor PACKAGED-LIBRARY validation on AMD MI300X (ROCm): unlike the
carried ``bench_relay_mi300x.json`` (produced in the source research repo
before extraction), this receipt is produced THROUGH the installed tridec
package's public API (``tridec.from_dem`` -> ``decode_batch``) on the
canonical BB .dem fixture, plus the package's own test-suite summary on the
same machine.

Contents:
  * suite        — pytest summary (parsed from a suite log run in this env).
  * ler_2000     — BpDecoder (triton fp32) + RelayBpDecoder (triton fp32)
                   logical-failure counts on 2000 sampled shots, with the
                   numpy BP reference and the relay-bp Rust oracle (F64)
                   decoding the SAME shots in-process.
  * throughput   — decode_batch wall-clock (the API call includes the final
                   device->host copy, which synchronizes; an explicit
                   torch.cuda.synchronize() brackets each timing), warmup >= 5,
                   measure >= 20, at batch 2000 and 8192.
  * prior_receipt_comparison — inline vs the carried source-repo MI300X
                   receipt (bench_relay_mi300x.json), same cell/config.

Shots are sampled platform-locally from the provenance .stim circuit (stim's
seeded sampler is platform-dependent — docs/benchmark.md §5.1); the decoders
consume the canonical .dem fixture artifact.

Run (in the ROCm container, package installed):
    python bench/mi300x_packaged.py --suite-log /workspace/suite.log \
        --suite-exit 0 --image rocm/7.0:rocm7.0_pytorch_training_instinct_20250915
"""
import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import stim

import tridec
from tridec.dem import extract
from tridec.validation.stats import wilson_ci

REPO = Path(__file__).resolve().parent.parent
RECEIPT = REPO / "bench" / "receipts" / "mi300x_packaged.json"
PRIOR = REPO / "bench" / "receipts" / "bench_relay_mi300x.json"
DEM_FIXTURE = REPO / "tests" / "fixtures" / "bb72" / "bb72_r6_p0.003_Z.dem"
STIM_FIXTURE = REPO / "tests" / "fixtures" / "bb72" / "bb72_r6_p0.003_Z.stim"

SEED = 0
N_LER = 2000
BATCHES = (2000, 8192)
WARMUP = 5
MEASURE = 20

_SUMMARY = re.compile(
    r"=+ (?P<body>(?:\d+ \w+(?:, )?)+) in (?P<secs>[\d.]+)s(?: \([^)]*\))? =+")


def parse_suite(log_path):
    """Parse the pytest tail summary (counts) from a -v -ra suite log."""
    text = Path(log_path).read_text(errors="replace")
    counts = {}
    secs = None
    for m in _SUMMARY.finditer(text):
        counts = {}
        for part in m.group("body").split(", "):
            n, word = part.split(" ", 1)
            counts[word.strip()] = int(n)
        secs = float(m.group("secs"))
    skip_reasons = sorted({ln.strip() for ln in text.splitlines()
                           if ln.startswith("SKIPPED")})
    return counts, secs, skip_reasons


def env_block(image):
    import importlib.metadata as md

    import torch
    import triton
    props = torch.cuda.get_device_properties(0)
    vers = {}
    for pkg in ("stim", "ldpc", "relay-bp", "pymatching", "numpy", "sinter"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = None
    return {
        "gpu_arch": getattr(props, "gcnArchName", None),
        "gpu_name": props.name,
        "rocm_hip": torch.version.hip,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "platform": f"{sys.platform}/{platform.machine()}",
        "container_image": image,
        "tridec": tridec.__version__,
        **vers,
    }


def bench_decode(decoder, dets_by_batch):
    """Wall-clock decode_batch timing: warmup >= 5, measure >= 20 per batch."""
    import torch
    out = []
    for shots, dets in dets_by_batch:
        for _ in range(WARMUP):
            decoder.decode_batch(dets)
        times = []
        for _ in range(MEASURE):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            decoder.decode_batch(dets)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        mean_s = statistics.mean(times)
        out.append({
            "shots": shots, "warmup": WARMUP, "iters": MEASURE,
            "mean_ms": round(mean_s * 1e3, 3),
            "min_ms": round(min(times) * 1e3, 3),
            "max_ms": round(max(times) * 1e3, 3),
            "stdev_ms": round(statistics.stdev(times) * 1e3, 3),
            "per_syndrome_us": round(mean_s / shots * 1e6, 3),
            "throughput_shots_per_s": round(shots / mean_s, 1),
        })
        print(f"  batch {shots}: mean {mean_s*1e3:.1f} ms "
              f"({shots/mean_s:.0f} shots/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-log", required=True)
    ap.add_argument("--suite-exit", type=int, required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    counts, secs, skip_reasons = parse_suite(args.suite_log)

    dem = stim.DetectorErrorModel.from_file(str(DEM_FIXTURE))
    ex = extract(dem)
    circuit = stim.Circuit.from_file(str(STIM_FIXTURE))
    dets_all, obs_all = circuit.compile_detector_sampler(seed=SEED).sample(
        max(BATCHES), separate_observables=True)
    dets_all = np.asarray(dets_all, dtype=bool)
    obs = np.asarray(obs_all, dtype=bool)[:N_LER]
    dets = dets_all[:N_LER]

    # --- decoders through the PACKAGED public API -------------------------
    bp = tridec.from_dem(dem, backend="triton")                  # fp32 kernel
    relay = tridec.from_dem(dem, algorithm="relay", backend="triton",
                            dtype="float32")
    assert bp.backend == "triton" and relay.backend == "triton"

    print("LER on 2000 sampled shots (seed 0, platform-local stream):",
          flush=True)
    pred_bp = bp.decode_batch(dets)
    fails_bp = int(np.any(pred_bp != obs, axis=1).sum())
    pred_relay = relay.decode_batch(dets)
    fails_relay = int(np.any(pred_relay != obs, axis=1).sum())
    print(f"  bp[triton fp32] fails={fails_bp}  "
          f"relay[triton fp32] fails={fails_relay}", flush=True)

    # numpy BP reference, same shots (in-package CPU reference).
    bp_np = tridec.from_dem(dem, backend="numpy")
    pred_np = bp_np.decode_batch(dets)
    fails_np = int(np.any(pred_np != obs, axis=1).sum())
    bp_agree = float((pred_np == pred_bp).all(axis=1).mean())
    print(f"  bp[numpy fp64]  fails={fails_np} "
          f"(per-shot agreement vs triton {bp_agree:.4f})", flush=True)

    # relay-bp Rust oracle (F64), same shots (the validation oracle).
    oracle = None
    try:
        from tridec.adapters import make_relay_bp, relay_bp_available
        if relay_bp_available():
            ora = make_relay_bp(dem)
            pred_o = ora.decode_batch(dets)
            fails_o = int(np.any(pred_o != obs, axis=1).sum())
            agree = float((pred_o == pred_relay).all(axis=1).mean())
            lo_o, hi_o = wilson_ci(fails_o, N_LER)
            lo_t, hi_t = wilson_ci(fails_relay, N_LER)
            oracle = {
                "fails": fails_o, "ler": fails_o / N_LER,
                "ler_wilson95": [lo_o, hi_o],
                "per_shot_agreement_vs_triton": agree,
                "wilson_ci_overlap_vs_triton": bool(lo_o <= hi_t and lo_t <= hi_o),
            }
            print(f"  relay-bp Rust oracle fails={fails_o} "
                  f"(agreement vs triton {agree:.4f})", flush=True)
    except Exception as e:  # oracle is optional; record why it is absent
        oracle = {"error": repr(e)}

    # --- throughput through decode_batch ----------------------------------
    dets_by_batch = [(n, dets_all[:n]) for n in BATCHES]
    print("BpDecoder[triton] decode_batch throughput:", flush=True)
    bp_thr = bench_decode(bp, dets_by_batch)
    print("RelayBpDecoder[triton fp32] decode_batch throughput:", flush=True)
    relay_thr = bench_decode(relay, dets_by_batch)

    # --- inline comparison vs the carried source-repo receipt -------------
    prior = json.loads(PRIOR.read_text())
    prior_fp32 = {r["shots"]: r for r in prior["bench_triton_real"]
                  if r["dtype"] == "float32"}
    comparison = []
    for r in relay_thr:
        p = prior_fp32.get(r["shots"])
        if not p:
            continue
        delta = (r["per_syndrome_us"] - p["per_syndrome_us"]) \
            / p["per_syndrome_us"] * 100.0
        comparison.append({
            "shots": r["shots"], "dtype": "float32",
            "prior_per_syndrome_us": p["per_syndrome_us"],
            "packaged_per_syndrome_us": r["per_syndrome_us"],
            "delta_pct": round(delta, 2),
            "note": "prior = source-repo harness (bench_relay_mi300x.json); "
                    "packaged = public decode_batch API incl. host<->device "
                    "transfers and observable mapping",
        })

    lo_t, hi_t = wilson_ci(fails_relay, N_LER)
    out = {
        "meta": {
            "what": ("MI300X PACKAGED-library validation: test-suite summary "
                     "+ benches through the installed tridec public API on "
                     "the canonical BB .dem fixture (p=0.003, Z)."),
            "date": time.strftime("%Y-%m-%d"),
            "fixture_dem": str(DEM_FIXTURE.relative_to(REPO)),
            "fixture_dem_sha256": hashlib.sha256(
                DEM_FIXTURE.read_bytes()).hexdigest(),
            "shots_note": ("shots sampled platform-locally from the "
                           "provenance .stim circuit (seed 0); stim's seeded "
                           "sampler is platform-dependent (benchmark.md §5.1) "
                           "so counts are not comparable shot-for-shot with "
                           "darwin/H200 receipts — Wilson CIs are."),
        },
        "env": env_block(args.image),
        "suite": {
            "command": "python3 -m pytest tests/ -v -ra",
            "exit_code": args.suite_exit,
            "counts": counts,
            "duration_s": secs,
            "skip_reasons": skip_reasons,
        },
        "structure": {"n_det": ex["n_det"], "n_obs": ex["n_obs"],
                      "n_err": ex["n_err"]},
        "ler_2000": {
            "shots": N_LER, "seed": SEED,
            "bp_triton_fp32": {
                "config": bp.config, "fails": fails_bp,
                "ler": fails_bp / N_LER,
                "ler_wilson95": list(wilson_ci(fails_bp, N_LER)),
                "per_shot_agreement_vs_numpy": bp_agree,
            },
            "bp_numpy_fp64_reference": {
                "fails": fails_np, "ler": fails_np / N_LER,
                "ler_wilson95": list(wilson_ci(fails_np, N_LER)),
            },
            "relay_triton_fp32": {
                "config": relay.config, "fails": fails_relay,
                "ler": fails_relay / N_LER, "ler_wilson95": [lo_t, hi_t],
            },
            "relay_rust_oracle_f64": oracle,
        },
        "throughput": {
            "protocol": (f"decode_batch wall-clock, torch.cuda.synchronize "
                         f"bracketed, warmup={WARMUP}, measure={MEASURE}, "
                         f"real sampled syndromes"),
            "bp_triton_fp32": bp_thr,
            "relay_triton_fp32": relay_thr,
        },
        "prior_receipt_comparison": {
            "prior_receipt": "bench_relay_mi300x.json",
            "relay_fp32": comparison,
        },
    }
    RECEIPT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
