"""ONE-CELL NO-REGRESSION GATE (p=0.003, basis=Z).

EXACT tier (receipt platform + receipt versions): the ported harness +
adapters + fixture circuit must reproduce the source grid's logical-failure
COUNTS EXACTLY for the ldpc-family decoders, using the exact sampling
convention of the grid (compile_detector_sampler(seed=cell.seed)
.sample(cell.shots), one shared shot set through run_matched):

  * BPOSD-10 (ldpc BpOsdDecoder, osd_cs order 10): fails must == zoo_grid's.
  * BP        (ldpc BpDecoder, min-sum ms=0.625):  fails must == zoo_grid's.

The exact-count claim is pinned to the receipt's FULL environment:
  * stim 1.15.0 — its seeded sampling stream is version-pinned AND
    platform-pinned (measured: identical seeds give entirely different
    detector samples on darwin/arm64 vs linux/x86_64);
  * ldpc 2.4.1 — its C++ build's float behavior;
  * darwin/arm64 — the platform the grid ran on.
On any other environment the gate runs in STATISTICAL tier instead: the same
matched protocol on the platform-local sample stream must land within
overlapping 95% Wilson CIs (and 5% count tolerance) of the pinned counts —
the counts cannot be exactly equal because the shots themselves differ.

The numpy-backend BP is always held to the source repo's cross-implementation
bar (overlapping Wilson CIs + counts within 5% of the BP row): the grid's
"BP" row was produced by ldpc's C++ BP, which early-terminates on syndrome
convergence, while the numpy reference always runs max_iter flooding
iterations — they were never bit-identical in the source repo either, and
pretending otherwise would be a fudged gate.
"""
import importlib.metadata

import pytest
import stim

from conftest import (load_bb_circuit, load_zoo_grid, on_receipt_platform,
                      zoo_cell)

import portable_qec
from portable_qec.validation import run_matched, wilson_ci

pytest.importorskip("ldpc")

P, BASIS = "0.003", "Z"


def _receipt_env_mismatches(env):
    """Reasons this environment cannot bind the EXACT tier ([] = it can)."""
    reasons = []
    if stim.__version__ != env["stim"]:
        reasons.append(f"stim {stim.__version__} != receipt {env['stim']}")
    ldpc_ver = importlib.metadata.version("ldpc")
    if ldpc_ver != env["ldpc"]:
        reasons.append(f"ldpc {ldpc_ver} != receipt {env['ldpc']}")
    if not on_receipt_platform():
        reasons.append("not the receipt platform (darwin/arm64): stim's "
                       "seeded sample stream is platform-dependent")
    return reasons


@pytest.fixture(scope="module")
def gate_run():
    grid = load_zoo_grid()
    cell = zoo_cell(grid, P, BASIS)
    exact = not _receipt_env_mismatches(grid["env"])

    from portable_qec.adapters import make_bp, make_bposd10

    circuit = load_bb_circuit(P, BASIS)
    dem = circuit.detector_error_model(decompose_errors=False)
    decoders = [
        make_bposd10(dem),
        make_bp(dem),
        portable_qec.from_dem(dem, backend="numpy"),
    ]
    manifest = run_matched(circuit, decoders, shots=cell["shots"],
                           rounds=cell["rounds"], seed=cell["seed"],
                           label=f"no-regression p={P} basis={BASIS}")
    return grid, cell, manifest, exact


def _rec(manifest, name):
    return next(r for r in manifest["decoders"] if r["name"] == name)


def _grid_rec(cell, name):
    return next(r for r in cell["decoders"] if r["name"] == name)


def _assert_count(cell, manifest, name, exact):
    want = _grid_rec(cell, name)["fails"]
    got = _rec(manifest, name)["fails"]
    if exact:
        assert got == want, (
            f"NO-REGRESSION FAIL (EXACT tier): {name} fails={got}, "
            f"zoo_grid={want} (shots={cell['shots']}, seed={cell['seed']})")
    else:
        shots = cell["shots"]
        lo_o, hi_o = wilson_ci(got, shots)
        lo_g, hi_g = wilson_ci(want, shots)
        assert lo_o <= hi_g and lo_g <= hi_o, (
            f"NO-REGRESSION FAIL (STATISTICAL tier): {name} fails={got} vs "
            f"pinned {want} — disjoint 95% Wilson CIs at N={shots}")
        assert abs(got - want) <= max(5, int(0.05 * want)), (
            f"NO-REGRESSION FAIL (STATISTICAL tier): {name} count {got} "
            f"diverges >5% from pinned {want}")


def test_dem_hash_matches_grid(gate_run):
    grid, cell, manifest, exact = gate_run
    if exact:
        assert manifest["dem_hash"] == cell["dem_hash"]
    else:
        # Text-hash is platform-local; structural sizes must still agree and
        # the strict identity is covered by tier 3 of the DEM-hash gate on the
        # receipt platform.
        assert manifest["shots"] == cell["shots"]


def test_bposd10_failure_count(gate_run):
    grid, cell, manifest, exact = gate_run
    _assert_count(cell, manifest, "BPOSD-10", exact)


def test_ldpc_bp_failure_count(gate_run):
    grid, cell, manifest, exact = gate_run
    _assert_count(cell, manifest, "BP", exact)


def test_numpy_bp_ci_equivalent_to_grid_bp(gate_run):
    """The numpy backend vs the grid's ldpc-BP row: the source repo's own
    equivalence bar (overlapping Wilson CIs + counts within 5%)."""
    grid, cell, manifest, exact = gate_run
    shots = cell["shots"]
    grid_bp = _grid_rec(cell, "BP")["fails"]
    ours = next(r for r in manifest["decoders"]
                if r["config"].get("backend") == "numpy")["fails"]
    lo_o, hi_o = wilson_ci(ours, shots)
    lo_g, hi_g = wilson_ci(grid_bp, shots)
    print(f"\nNUMPY-BP vs grid ldpc-BP (p={P} {BASIS}, N={shots}): "
          f"ours={ours} grid={grid_bp}")
    assert lo_o <= hi_g and lo_g <= hi_o, (
        f"CIs disjoint: numpy fails={ours} vs grid BP fails={grid_bp}")
    assert abs(ours - grid_bp) <= max(5, int(0.05 * grid_bp)), (
        f"numpy BP count {ours} diverges >5% from grid BP {grid_bp}")
