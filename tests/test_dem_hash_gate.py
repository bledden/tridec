"""DEM-HASH GATE, three tiers (pin artifacts, not generators).

Tier 1 (every platform) — CANONICAL ARTIFACT IDENTITY: the .dem fixture FILES'
bytes must sha256-match the pins in dem_manifest.json. The .dem file is the
decoder-input artifact; its bytes are platform-independent by construction.

Tier 2 (every platform) — REGENERATION TOLERANCE: rebuilding the DEM from the
.stim provenance circuit on THIS platform must give the identical mechanism
structure and priors within rel 1e-12 of the canonical .dem. (stim's
circuit->DEM float computation differs across platforms at the ~ulp level —
measured max rel 5.4e-16 between darwin/arm64 and linux/x86_64 — and its float
text rendering differs too, so exact equality is only guaranteed on the
platform that generated the fixture.)

Tier 3 (receipt platform only) — STRICT SOURCE-GRID IDENTITY: on darwin/arm64
with the receipt's stim, the regenerated DEM's text-hash must EXACTLY match the
per-cell ``dem_hash`` pinned in the carried ``bench/receipts/zoo_grid.json``,
proving the fixture export is byte-faithful to the validated research grid.
"""
import hashlib
import re

import pytest
import stim

from conftest import (BB_CELLS, bb_dem_path, load_bb_circuit, load_bb_dem,
                      load_dem_manifest, load_zoo_grid, on_receipt_platform,
                      zoo_cell)

from portable_qec.dem import extract
from portable_qec.validation import dem_hash

_FLOAT = re.compile(r"error\(([0-9.eE+-]+)\)")


def _mechanisms(dem: stim.DetectorErrorModel):
    """(structure_lines_with_floats_masked, float_values) for a flattened DEM."""
    text = str(dem.flattened())
    values = [float(m) for m in _FLOAT.findall(text)]
    structure = [_FLOAT.sub("error(P)", line) for line in text.splitlines()]
    return structure, values


# --- Tier 1: canonical artifact identity (every platform) ------------------

@pytest.mark.parametrize("p,basis", BB_CELLS)
def test_dem_fixture_bytes_match_manifest(p, basis):
    manifest = load_dem_manifest()
    path = bb_dem_path(p, basis)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    want = manifest["files"][path.name]["sha256"]
    assert sha == want, (
        f"canonical .dem fixture drifted for p={p} {basis}: {sha} != {want}")


# --- Tier 2: regeneration tolerance (every platform) -----------------------

@pytest.mark.parametrize("p,basis", BB_CELLS)
def test_stim_regeneration_matches_dem_fixture(p, basis):
    canon = load_bb_dem(p, basis)
    regen = load_bb_circuit(p, basis).detector_error_model(
        decompose_errors=False)
    s_canon, v_canon = _mechanisms(canon)
    s_regen, v_regen = _mechanisms(regen)
    assert s_regen == s_canon, (
        f"regenerated DEM STRUCTURE differs from canonical .dem for "
        f"p={p} {basis} — this is real drift, not float noise")
    assert len(v_regen) == len(v_canon)
    worst = max((abs(a - b) / b if b else abs(a - b))
                for a, b in zip(v_regen, v_canon))
    assert worst < 1e-12, (
        f"regenerated priors diverge from canonical .dem beyond float noise "
        f"for p={p} {basis}: max rel diff {worst:.3e}")


# --- Tier 3: strict source-grid identity (receipt platform only) -----------

@pytest.mark.parametrize("p,basis", BB_CELLS)
def test_fixture_dem_hash_matches_zoo_grid(p, basis):
    if not on_receipt_platform():
        pytest.skip(
            "strict text-hash identity vs zoo_grid pins is defined on the "
            "receipt platform (darwin/arm64): stim's DEM float computation "
            "and text rendering are platform-dependent (see docs/benchmark.md)")
    grid = load_zoo_grid()
    cell = zoo_cell(grid, p, basis)
    circuit = load_bb_circuit(p, basis)
    dem = circuit.detector_error_model(decompose_errors=False)
    h = dem_hash(dem)
    assert h == cell["dem_hash"], (
        f"DEM-hash gate FAIL for p={p} basis={basis}: fixture hashes to {h}, "
        f"zoo_grid pins {cell['dem_hash']}")


@pytest.mark.parametrize("p,basis", BB_CELLS)
def test_fixture_dem_sizes_match_zoo_grid(p, basis):
    grid = load_zoo_grid()
    cell = zoo_cell(grid, p, basis)
    ex = extract(load_bb_dem(p, basis))
    assert ex["n_det"] == cell["n_det"]
    assert ex["n_obs"] == cell["n_obs"]
    assert ex["H"].shape == (cell["n_det"], ex["n_err"])
    assert ex["Lo"].shape == (cell["n_obs"], ex["n_err"])
    assert ex["priors"].shape == (ex["n_err"],)
