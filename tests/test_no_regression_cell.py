"""ONE-CELL NO-REGRESSION GATE (p=0.003, basis=Z).

The ported harness + adapters + fixture circuit must reproduce the source
grid's logical-failure COUNTS EXACTLY for the ldpc-family decoders, using the
exact sampling convention of the grid (compile_detector_sampler(seed=cell.seed)
.sample(cell.shots), one shared shot set through run_matched):

  * BPOSD-10 (ldpc BpOsdDecoder, osd_cs order 10): fails must == zoo_grid's.
  * BP        (ldpc BpDecoder, min-sum ms=0.625):  fails must == zoo_grid's.

The exact-count claim is pinned to the receipt's environment (stim 1.15.0
sampling stream, ldpc 2.4.1); the gate SKIPS (loudly) on other versions rather
than asserting counts another sampler/decoder build never produced.

The numpy-backend BP is run through the SAME matched protocol and held to the
source repo's own cross-implementation bar (overlapping Wilson CIs + counts
within 5%): the grid's "BP" row was produced by ldpc's C++ BP, which
early-terminates on syndrome convergence, while the numpy reference always
runs max_iter flooding iterations — they were never bit-identical in the
source repo either, and pretending otherwise would be a fudged gate.
"""
import importlib.metadata

import numpy as np
import pytest
import stim

from conftest import load_bb_circuit, load_zoo_grid, zoo_cell

import portable_qec
from portable_qec.validation import run_matched, wilson_ci

pytest.importorskip("ldpc")

P, BASIS = "0.003", "Z"


@pytest.fixture(scope="module")
def gate_run():
    grid = load_zoo_grid()
    cell = zoo_cell(grid, P, BASIS)

    env = grid["env"]
    if stim.__version__ != env["stim"]:
        pytest.skip(
            f"exact-count gate is pinned to stim {env['stim']}'s sampling "
            f"stream; running stim {stim.__version__}")
    ldpc_ver = importlib.metadata.version("ldpc")
    if ldpc_ver != env["ldpc"]:
        pytest.skip(
            f"exact-count gate is pinned to ldpc {env['ldpc']}; running "
            f"ldpc {ldpc_ver}")

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
    return grid, cell, manifest


def _rec(manifest, name):
    return next(r for r in manifest["decoders"] if r["name"] == name)


def _grid_rec(cell, name):
    return next(r for r in cell["decoders"] if r["name"] == name)


def test_dem_hash_matches_grid(gate_run):
    _, cell, manifest = gate_run
    assert manifest["dem_hash"] == cell["dem_hash"]


def test_bposd10_failure_count_exact(gate_run):
    _, cell, manifest = gate_run
    want = _grid_rec(cell, "BPOSD-10")["fails"]
    got = _rec(manifest, "BPOSD-10")["fails"]
    assert got == want, (
        f"NO-REGRESSION FAIL: BPOSD-10 fails={got}, zoo_grid={want} "
        f"(shots={cell['shots']}, seed={cell['seed']})")


def test_ldpc_bp_failure_count_exact(gate_run):
    _, cell, manifest = gate_run
    want = _grid_rec(cell, "BP")["fails"]
    got = _rec(manifest, "BP")["fails"]
    assert got == want, (
        f"NO-REGRESSION FAIL: ldpc BP fails={got}, zoo_grid={want} "
        f"(shots={cell['shots']}, seed={cell['seed']})")


def test_numpy_bp_ci_equivalent_to_grid_bp(gate_run):
    """The numpy backend vs the grid's ldpc-BP row: the source repo's own
    equivalence bar (overlapping Wilson CIs + counts within 5%)."""
    _, cell, manifest = gate_run
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
