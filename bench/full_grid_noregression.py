"""FULL-GRID NO-REGRESSION (EXACT tier, release gate for v0.1.0).

All 8 canonical BB [[72,12,6]] grid cells x the ldpc-family adapters present
in zoo_grid.json AND in tridec.adapters (BP, BPOSD-0, BPOSD-10, BPLSD),
decoded through ``tridec.validation.run_matched`` with each cell's EXACT
shots/seed from ``bench/receipts/zoo_grid.json``. The gate: every
(cell, decoder) logical-failure COUNT must equal the pinned grid count
EXACTLY — no tolerance, no statistical fallback.

This binds ONLY in the receipt environment (see tests/test_no_regression_cell
.py for why): stim 1.15.0 + ldpc 2.4.1 on darwin/arm64 — stim's seeded
detector sampler and DEM float rendering are platform-dependent, so anywhere
else exact counts are impossible by construction. The script REFUSES to run
outside that environment rather than producing a receipt that looks exact
but isn't.

Writes ``bench/receipts/full_grid_noregression.json`` (per-cell
ours-vs-pinned) and exits non-zero on ANY mismatch.

Run:  ~receipt-env python bench/full_grid_noregression.py
"""
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import stim

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import tridec  # noqa: E402
from tridec.adapters import make_bp, make_bplsd, make_bposd0, make_bposd10  # noqa: E402
from tridec.validation import dem_hash, run_matched  # noqa: E402

RECEIPT = REPO / "bench" / "receipts" / "full_grid_noregression.json"
ZOO = REPO / "bench" / "receipts" / "zoo_grid.json"
FIXTURES = REPO / "tests" / "fixtures" / "bb72"

# zoo_grid name -> tridec.adapters factory (the ldpc family; Tesseract,
# RelayBP and SlidingWindow rows are NOT in scope — different packages).
FACTORIES = {
    "BP": make_bp,
    "BPOSD-0": make_bposd0,
    "BPOSD-10": make_bposd10,
    "BPLSD": make_bplsd,
}
DECODER_ORDER = ("BP", "BPOSD-0", "BPOSD-10", "BPLSD")


def assert_receipt_env(zoo_env):
    """The EXACT tier binds only in the receipt environment — refuse elsewhere."""
    problems = []
    if stim.__version__ != zoo_env["stim"]:
        problems.append(f"stim {stim.__version__} != receipt {zoo_env['stim']}")
    ldpc_ver = importlib.metadata.version("ldpc")
    if ldpc_ver != zoo_env["ldpc"]:
        problems.append(f"ldpc {ldpc_ver} != receipt {zoo_env['ldpc']}")
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        problems.append(f"platform {sys.platform}/{platform.machine()} is not "
                        f"the receipt platform (darwin/arm64)")
    if problems:
        raise SystemExit(
            "REFUSING to run the EXACT no-regression tier outside the receipt "
            "environment:\n  - " + "\n  - ".join(problems))


def main():
    zoo = json.loads(ZOO.read_text())
    assert_receipt_env(zoo["env"])

    meta = {
        "what": ("full-grid EXACT no-regression: 8 BB cells x ldpc-family "
                 "adapters (BP, BPOSD-0, BPOSD-10, BPLSD) through "
                 "run_matched, each cell at zoo_grid's exact shots/seed; "
                 "per-(cell,decoder) failure counts must EQUAL the pinned "
                 "zoo_grid counts."),
        "date": time.strftime("%Y-%m-%d"),
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version()},
        "versions": {
            "stim": stim.__version__,
            "ldpc": importlib.metadata.version("ldpc"),
            "numpy": importlib.metadata.version("numpy"),
            "tridec": tridec.__version__,
        },
        "zoo_grid_env": zoo["env"],
    }

    cells_out = []
    all_exact = True
    t_total = time.perf_counter()
    for cell in zoo["cells"]:
        p, basis = str(cell["p"]), cell["basis"]
        shots, seed, rounds = cell["shots"], cell["seed"], cell["rounds"]
        circuit = stim.Circuit.from_file(
            str(FIXTURES / f"bb72_r6_p{p}_{basis}.stim"))
        dem = circuit.detector_error_model(decompose_errors=False)
        h = dem_hash(dem)
        hash_ok = (h == cell["dem_hash"])
        if not hash_ok:
            all_exact = False
            print(f"[{p} {basis}] DEM-HASH MISMATCH: {h} != {cell['dem_hash']}",
                  flush=True)

        decoders = [FACTORIES[name](dem) for name in DECODER_ORDER]
        t0 = time.perf_counter()
        manifest = run_matched(circuit, decoders, shots=shots, rounds=rounds,
                               seed=seed,
                               label=f"full-grid no-regression p={p} {basis}")
        dt = time.perf_counter() - t0

        recs = []
        for name in DECODER_ORDER:
            ours = next(r for r in manifest["decoders"] if r["name"] == name)
            pinned = next(r for r in cell["decoders"] if r["name"] == name)
            match = (ours["fails"] == pinned["fails"])
            all_exact &= match
            recs.append({"name": name, "fails_ours": ours["fails"],
                         "fails_pinned": pinned["fails"], "exact_match": match,
                         "decode_s": ours["decode_s"]})
            tag = "OK " if match else "*** MISMATCH ***"
            print(f"[{p} {basis}] {name:9s} ours={ours['fails']:5d} "
                  f"pinned={pinned['fails']:5d} {tag}", flush=True)
        print(f"[{p} {basis}] cell done in {dt:.1f}s (N={shots})", flush=True)

        cells_out.append({
            "p": cell["p"], "basis": basis, "rounds": rounds, "shots": shots,
            "seed": seed, "dem_hash": h, "dem_hash_pinned": cell["dem_hash"],
            "dem_hash_exact_match": hash_ok, "decoders": recs,
            "cell_wall_s": dt,
        })

    out = {"meta": meta, "all_exact": all_exact, "cells": cells_out,
           "total_wall_s": time.perf_counter() - t_total}
    RECEIPT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {RECEIPT}")
    print("RESULT:", "ALL EXACT — release gate PASSED" if all_exact
          else "MISMATCH FOUND — RELEASE GATE FAILED")
    sys.exit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
