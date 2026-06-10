"""sinter integration: tridec decoders as sinter custom decoders.

The classic integration bug is bit-packing: sinter hands the compiled decoder
LITTLE-endian bit-packed detection events ((shots, ceil(n_det/8)) uint8) and
expects little-endian bit-packed observable predictions back. The unit tier
pins that round-trip exactly against the plain decode_batch path (the numpy BP
backend is deterministic, so the match must be bit-exact). The collect tier
runs a real sinter.collect on the d=3 surface-code fixture vs pymatching and
checks the plumbing + LER sanity (BP is NOT expected to beat MWPM here).
"""
import math

import numpy as np
import pytest

sinter = pytest.importorskip("sinter")

from conftest import load_surface_circuit

import tridec
from tridec.sinter import TridecSinterDecoder, sinter_decoders


@pytest.fixture(scope="module")
def surface_task_dem():
    """The fixture circuit + the DEM sinter would hand to a decoder."""
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=True)
    return circ, dem


# --------------------------------------------------------------------------- #
# Interface tier.                                                              #
# --------------------------------------------------------------------------- #
def test_is_a_sinter_decoder():
    dec = TridecSinterDecoder(algorithm="bp", backend="numpy")
    assert isinstance(dec, sinter.Decoder)


def test_compiles_for_dem(surface_task_dem):
    _, dem = surface_task_dem
    compiled = TridecSinterDecoder(
        algorithm="bp", backend="numpy").compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, sinter.CompiledDecoder)


def test_decoder_is_picklable():
    """sinter workers receive custom decoders via multiprocessing pickling."""
    import pickle
    dec = TridecSinterDecoder(algorithm="bp", backend="numpy", max_iter=20)
    clone = pickle.loads(pickle.dumps(dec))
    assert isinstance(clone, TridecSinterDecoder)
    assert clone.algorithm == "bp"
    assert clone.backend == "numpy"
    assert clone.opts == {"max_iter": 20}


def test_sinter_decoders_registry():
    reg = sinter_decoders(backend="numpy")
    assert set(reg) == {"tridec_bp", "tridec_relay"}
    assert all(isinstance(d, sinter.Decoder) for d in reg.values())


# --------------------------------------------------------------------------- #
# Bit-packing tier (the classic integration bug, pinned exactly).              #
# --------------------------------------------------------------------------- #
def test_bit_packed_round_trip_matches_decode_batch(surface_task_dem):
    circ, dem = surface_task_dem
    dets, _ = circ.compile_detector_sampler(seed=7).sample(
        503, separate_observables=True)  # odd shot count: not a multiple of 8
    dets = np.asarray(dets, dtype=bool)

    compiled = TridecSinterDecoder(
        algorithm="bp", backend="numpy").compile_decoder_for_dem(dem=dem)

    packed_in = np.packbits(dets.astype(np.uint8), axis=1, bitorder="little")
    assert packed_in.shape == (503, math.ceil(dem.num_detectors / 8))
    packed_out = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_in)

    # Same DEM, same shots, straight through the public API: must be bit-exact
    # (numpy BP is deterministic).
    ref = tridec.from_dem(dem, backend="numpy").decode_batch(dets)
    expect = np.packbits(ref.astype(np.uint8), axis=1, bitorder="little")

    assert packed_out.dtype == np.uint8
    assert packed_out.shape == (503, math.ceil(dem.num_observables / 8))
    assert np.array_equal(packed_out, expect)


def test_bit_packing_is_little_endian(surface_task_dem):
    """Firing ONLY detector k must round-trip through the packed interface
    identically to the unpacked path — for k both below and above 8, which
    breaks if either unpacking endianness or axis is wrong."""
    _, dem = surface_task_dem
    nd = dem.num_detectors
    compiled = TridecSinterDecoder(
        algorithm="bp", backend="numpy").compile_decoder_for_dem(dem=dem)
    dec = tridec.from_dem(dem, backend="numpy")
    for k in (0, 1, 7, 8, 9, nd - 1):
        dets = np.zeros((1, nd), dtype=bool)
        dets[0, k] = True
        packed = np.packbits(dets.astype(np.uint8), axis=1, bitorder="little")
        out = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=packed)
        expect = np.packbits(dec.decode_batch(dets).astype(np.uint8),
                             axis=1, bitorder="little")
        assert np.array_equal(out, expect), f"mismatch for detector {k}"


# --------------------------------------------------------------------------- #
# collect tier: real sinter.collect, tridec-BP vs pymatching.                  #
# --------------------------------------------------------------------------- #
def test_sinter_collect_surface_code_vs_pymatching():
    pytest.importorskip("pymatching")
    circ = load_surface_circuit()
    shots = 5000
    stats = sinter.collect(
        num_workers=2,
        tasks=[sinter.Task(circuit=circ, json_metadata={"d": 3, "r": 3,
                                                        "p": 0.003})],
        decoders=["tridec_bp", "pymatching"],
        custom_decoders={"tridec_bp": TridecSinterDecoder(algorithm="bp",
                                                          backend="numpy")},
        max_shots=shots,
    )
    by_dec = {s.decoder: s for s in stats}
    assert set(by_dec) == {"tridec_bp", "pymatching"}
    for name, s in by_dec.items():
        assert s.shots == shots, f"{name}: shots={s.shots}"
        ler = s.errors / s.shots
        # d=3 r=3 p=0.003: any sane decoder sits far below 10% block error.
        assert 0.0 < ler < 0.1, f"{name}: LER {ler:.4f} not sane"
    # Honest expectation: plain BP trails MWPM on surface codes (degeneracy),
    # but must be the same order of magnitude.
    bp, mwpm = by_dec["tridec_bp"], by_dec["pymatching"]
    assert bp.errors <= 10 * max(mwpm.errors, 1) + 5, (
        f"BP ({bp.errors}) implausibly worse than MWPM ({mwpm.errors})")
