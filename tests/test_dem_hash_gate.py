"""DEM-HASH GATE: the extraction is byte-faithful to the source experiments.

Each of the 8 canonical BB [[72,12,6]] fixture circuits, loaded from the .stim
text files in THIS repo, must rebuild a DetectorErrorModel whose sha256 (via
the ported ``portable_qec.validation.dem_hash``) EXACTLY matches the per-cell
``dem_hash`` pinned in the carried ``bench/receipts/zoo_grid.json`` receipt.

A match proves: same circuit -> same mechanism set, priors, detectors and
observables as the validated research grid, with zero dependency on the source
repo at test time. Any drift in the fixture export, the hash primitive, or the
DEM extraction fails here first.
"""
import pytest

from conftest import BB_CELLS, load_bb_circuit, load_zoo_grid, zoo_cell

from portable_qec.dem import extract
from portable_qec.validation import dem_hash


@pytest.mark.parametrize("p,basis", BB_CELLS)
def test_fixture_dem_hash_matches_zoo_grid(p, basis):
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
    dem = load_bb_circuit(p, basis).detector_error_model(decompose_errors=False)
    ex = extract(dem)
    assert ex["n_det"] == cell["n_det"]
    assert ex["n_obs"] == cell["n_obs"]
    assert ex["H"].shape == (cell["n_det"], ex["n_err"])
    assert ex["Lo"].shape == (cell["n_obs"], ex["n_err"])
    assert ex["priors"].shape == (ex["n_err"],)
