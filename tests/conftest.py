"""Shared test plumbing: fixture/receipt paths and loaders.

Fixture artifacts under tests/fixtures/ come in two tiers:
  * .dem files — the CANONICAL decoder-input artifacts. Their file BYTES are
    platform-independent and sha256-pinned in dem_manifest.json.
  * .stim files — provenance. stim's circuit->DEM computation is platform-
    dependent at the ~ulp level (x86 long-double intermediates vs arm64
    doubles) and its float text rendering differs too, so DEMs REGENERATED
    from .stim are only tolerance-equal across platforms. stim's seeded
    detector sampler is also platform-dependent (measured; see
    docs/benchmark.md, Reproducibility notes). Pin artifacts, not generators.

The JSONs under bench/receipts/ are the carried measurement receipts, byte-
verbatim from the source experiments (generated on the RECEIPT platform,
darwin/arm64). Tests must depend ONLY on in-repo artifacts.
"""
import json
import platform
import sys
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

# The platform the carried receipts (and exact-count pins) were generated on.
RECEIPT_PLATFORM = {"platform": "darwin", "machine": "arm64"}


def on_receipt_platform() -> bool:
    return (sys.platform == RECEIPT_PLATFORM["platform"]
            and platform.machine() == RECEIPT_PLATFORM["machine"])


def load_bb_circuit(p: str, basis: str) -> stim.Circuit:
    return stim.Circuit.from_file(
        str(FIXTURES_DIR / "bb72" / f"bb72_r6_p{p}_{basis}.stim"))


def load_bb_dem(p: str, basis: str) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel.from_file(
        str(FIXTURES_DIR / "bb72" / f"bb72_r6_p{p}_{basis}.dem"))


def bb_dem_path(p: str, basis: str) -> Path:
    return FIXTURES_DIR / "bb72" / f"bb72_r6_p{p}_{basis}.dem"


def load_dem_manifest() -> dict:
    return json.loads(
        (FIXTURES_DIR / "bb72" / "dem_manifest.json").read_text())


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
