"""Shared test plumbing: fixture/receipt paths and loaders.

The .stim files under tests/fixtures/ are the canonical circuits; the JSONs
under bench/receipts/ are the carried measurement receipts. Tests must depend
ONLY on these in-repo artifacts (never on the research repo they came from).
"""
import json
from pathlib import Path

import pytest
import stim

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
RECEIPTS_DIR = TESTS_DIR.parent / "bench" / "receipts"

# The 8 canonical BB [[72,12,6]] grid cells (p as the exact string used in
# zoo_grid.json's float repr and in the fixture filenames).
BB_CELLS = [(p, b) for p in ("0.001", "0.002", "0.003", "0.005")
            for b in ("X", "Z")]


def load_bb_circuit(p: str, basis: str) -> stim.Circuit:
    return stim.Circuit.from_file(
        str(FIXTURES_DIR / "bb72" / f"bb72_r6_p{p}_{basis}.stim"))


def load_surface_circuit() -> stim.Circuit:
    return stim.Circuit.from_file(
        str(FIXTURES_DIR / "surface" / "surface_d3_r3_p0.003.stim"))


def load_zoo_grid() -> dict:
    return json.loads((RECEIPTS_DIR / "zoo_grid.json").read_text())


def zoo_cell(grid: dict, p: str, basis: str) -> dict:
    for c in grid["cells"]:
        if str(c["p"]) == p and c["basis"] == basis:
            return c
    raise KeyError(f"no zoo_grid cell for p={p} basis={basis}")


@pytest.fixture(scope="session")
def zoo_grid():
    return load_zoo_grid()
