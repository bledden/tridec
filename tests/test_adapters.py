"""Optional CPU reference adapters (ldpc family + relay-bp), import-guarded."""
import numpy as np
import pytest

from conftest import load_surface_circuit


@pytest.fixture(scope="module")
def surface():
    circ = load_surface_circuit()
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = circ.compile_detector_sampler(seed=0).sample(
        500, separate_observables=True)
    return dem, np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


def test_adapters_module_imports_without_optional_deps():
    """The adapters module itself must import even if ldpc/relay-bp are absent
    (the factories are guarded, not the module)."""
    import portable_qec.adapters as ad
    assert callable(ad.build_decoders)
    assert isinstance(ad.relay_bp_available(), bool)
    assert isinstance(ad.ldpc_available(), bool)


def test_ldpc_adapters_decode_and_carry_provenance(surface):
    pytest.importorskip("ldpc")
    from portable_qec.adapters import make_bp, make_bposd0, make_bposd10
    dem, dets, obs = surface
    for make in (make_bp, make_bposd0, make_bposd10):
        a = make(dem)
        assert a.dem is dem                      # G1 provenance
        assert isinstance(a.config, dict)
        pred = a.decode_batch(dets[:100])
        assert pred.shape == (100, dem.num_observables)
        assert pred.dtype == bool


def test_bposd10_beats_chance_on_surface(surface):
    pytest.importorskip("ldpc")
    from portable_qec.adapters import make_bposd10
    dem, dets, obs = surface
    pred = make_bposd10(dem).decode_batch(dets)
    ler = float(np.any(pred != obs, axis=1).mean())
    assert ler < 0.1


def test_build_decoders_core_selection(surface):
    pytest.importorskip("ldpc")
    from portable_qec.adapters import build_decoders
    dem, _, _ = surface
    decs = build_decoders(dem, which=("BP", "BPOSD-10"))
    assert [d.name for d in decs] == ["BP", "BPOSD-10"]
    for d in decs:
        assert d.dem is dem


def test_relay_bp_adapter(surface):
    from portable_qec.adapters import make_relay_bp, relay_bp_available
    if not relay_bp_available():
        pytest.skip("relay-bp[stim] not installed")
    dem, dets, obs = surface
    a = make_relay_bp(dem)
    assert a.dem is dem
    assert a.tie_break == "relay_bp_nconv_disjoint_ensemble"
    pred = a.decode_batch(dets[:200])
    assert pred.shape == (200, dem.num_observables)
    ler = float(np.any(pred != obs[:200], axis=1).mean())
    assert ler < 0.1
