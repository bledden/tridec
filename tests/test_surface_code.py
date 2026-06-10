"""Code-agnosticism: the decoders consume ANY stim DEM, not just BB codes.

A rotated surface-code memory (d=3, r=3, p=0.003) is decoded by the numpy BP
backend straight from the stim DEM. LER sanity is cross-checked against
pymatching (MWPM on the decomposed DEM) when available — BP is NOT expected to
match MWPM on surface codes (degeneracy hurts plain BP); the gate is sanity,
not parity.
"""
import numpy as np
import pytest

from conftest import load_surface_circuit

import tridec

SHOTS = 2000


@pytest.fixture(scope="module")
def surface_shots():
    circ = load_surface_circuit()
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        SHOTS, separate_observables=True)
    return circ, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_bp_decodes_surface_code(surface_shots):
    circ, dets, obs = surface_shots
    dem = circ.detector_error_model(decompose_errors=False)
    dec = tridec.from_dem(dem, backend="numpy")
    pred = dec.decode_batch(dets)
    fails = int(np.any(pred != obs, axis=1).sum())
    ler = fails / SHOTS
    # d=3 r=3 p=0.003: a sane decoder sits well below 10% block error.
    assert ler < 0.1, f"surface-code BP LER {ler:.4f} not sane"


def test_bp_ler_sane_vs_pymatching(surface_shots):
    pymatching = pytest.importorskip("pymatching")
    circ, dets, obs = surface_shots
    # MWPM needs graphlike errors -> decomposed DEM; BP uses the raw DEM.
    dem_mwpm = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem_mwpm)
    pred_mwpm = matching.decode_batch(dets.astype(np.uint8))
    fails_mwpm = int(np.any(pred_mwpm.astype(bool) != obs, axis=1).sum())

    dem = circ.detector_error_model(decompose_errors=False)
    dec = tridec.from_dem(dem, backend="numpy")
    pred_bp = dec.decode_batch(dets)
    fails_bp = int(np.any(pred_bp != obs, axis=1).sum())

    print(f"\nSURFACE d=3 r=3 p=0.003: BP fails={fails_bp} "
          f"MWPM fails={fails_mwpm} (N={SHOTS})")
    # Sanity, not parity: plain BP may trail MWPM but must be the same order.
    assert fails_bp <= 10 * max(fails_mwpm, 1) + 5, (
        f"BP ({fails_bp}) implausibly worse than MWPM ({fails_mwpm})")
