"""Optional CPU reference-decoder adapters on a SHARED DEM (import-guarded).

These wrap the standard CPU reference implementations — the `ldpc` package's
BP / BP-OSD / BP-LSD and IBM's `relay-bp` Rust decoder — behind the same
``decode_batch(dets) -> predicted_observables`` surface as the native
backends, so a matched harness (``tridec.validation.run_matched``) can
decode the SAME shots with every decoder (apples-to-apples LER). They are the
validation targets the GPU kernels are held against.

Install with the ``decoders`` extra: ``pip install tridec[decoders]``.
The module imports without either package; each factory raises (or the
``*_available()`` probes return False) when its dependency is missing.

Interface (every adapter):
  * ``.name``    -- str identifier (e.g. ``"BPOSD-10"``),
  * ``.config``  -- dict of pinned hyperparameters (provenance),
  * ``.dem``     -- the shared ``stim.DetectorErrorModel`` it was built from,
  * ``.tie_break`` -- declared deterministic tie-break (gate G2),
  * ``.decode_batch(dets: bool[shots, n_det]) -> bool[shots, n_obs]``.

For an ldpc decoder, each shot's detector syndrome is decoded to an error
estimate ``e_hat`` (length n_err); predicted observables = ``(Lo @ e_hat) % 2``.
ldpc 2.4.x exposes only single-shot ``decoder.decode(syndrome)`` (no batched
entry point), so ldpc adapters loop over shots.
"""
import numpy as np

from ..dem import extract

# Pinned min-sum BP hyperparameters shared across the BP-family adapters
# (the provenance constants the validation grid committed to).
_BP_MAX_ITER = 30
_BP_MS_SCALING = 0.625          # standard normalized-min-sum scaling factor
_BP_METHOD = "minimum_sum"      # min-sum BP (the kernel target)
_BP_SCHEDULE = "parallel"


def ldpc_available():
    """True iff the `ldpc` package is importable."""
    try:
        import ldpc  # noqa: F401
    except Exception:
        return False
    return True


def relay_bp_available():
    """True iff relay-bp[stim] is importable (import-guarded membership)."""
    try:
        import relay_bp  # noqa: F401
        from relay_bp.stim import CheckMatrices  # noqa: F401
    except Exception:
        return False
    return True


class _LdpcAdapter:
    """Base for ldpc-family adapters: build H/Lo/priors from the shared DEM,
    decode each shot's syndrome to an error estimate, map to observables."""

    def __init__(self, dem, name, config, decoder, tie_break):
        self.dem = dem
        self.name = name
        self.config = dict(config)
        # Declared deterministic tie-break (gate G2). No silent default: the
        # matched harness asserts this is in APPROVED_TIE_BREAKS.
        self.tie_break = tie_break
        self._decoder = decoder
        ex = extract(dem)
        # Lo: (n_obs x n_err) GF2 map from error mechanisms to observables.
        self._Lo = ex["Lo"].toarray().astype(np.uint8)
        self._n_obs = ex["n_obs"]
        self._n_err = ex["n_err"]
        self._n_det = ex["n_det"]

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        shots = dets.shape[0]
        out = np.zeros((shots, self._n_obs), dtype=bool)
        syn_u8 = dets.astype(np.uint8)
        Lo = self._Lo
        for i in range(shots):
            e_hat = self._decoder.decode(syn_u8[i])
            # predicted observables = (Lo @ e_hat) % 2
            pred = (Lo @ np.asarray(e_hat, dtype=np.uint8)) & 1
            out[i] = pred.astype(bool)
        return out


def _priors(dem):
    """Per-mechanism priors from the shared DEM, clipped for ldpc stability."""
    pri = extract(dem)["priors"]
    return list(np.clip(pri, 1e-6, 1 - 1e-6))


def make_bp(dem):
    """Pure min-sum BP (no post-processing): ldpc.BpDecoder reference."""
    from ldpc import BpDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE)
    dec = BpDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                    bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                    schedule=_BP_SCHEDULE)
    return _LdpcAdapter(dem, "BP", cfg, dec, "min_sum_parallel_hard_decision")


def make_bposd0(dem):
    """BP-OSD order-0 (osd_0): cheapest OSD post-processing."""
    from ldpc import BpOsdDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpOsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, osd_method="osd_0", osd_order=0)
    dec = BpOsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, osd_method="osd_0", osd_order=0)
    return _LdpcAdapter(dem, "BPOSD-0", cfg, dec, "osd0_reliability_order")


def make_bposd10(dem):
    """BP-OSD order-10 combination-sweep (osd_cs): the strong classical bar."""
    from ldpc import BpOsdDecoder

    H = extract(dem)["H"]
    cfg = dict(decoder="BpOsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, osd_method="osd_cs", osd_order=10)
    dec = BpOsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, osd_method="osd_cs", osd_order=10)
    return _LdpcAdapter(dem, "BPOSD-10", cfg, dec, "osd_cs_order10")


def make_bplsd(dem):
    """BP + Localised-Statistics Decoder (lsd_cs, order 10)."""
    from ldpc import BpLsdDecoder

    H = extract(dem)["H"]
    lsd_order = 10
    cfg = dict(decoder="BpLsdDecoder", bp_method=_BP_METHOD,
               ms_scaling_factor=_BP_MS_SCALING, max_iter=_BP_MAX_ITER,
               schedule=_BP_SCHEDULE, lsd_method="lsd_cs", lsd_order=lsd_order)
    dec = BpLsdDecoder(H, error_channel=_priors(dem), max_iter=_BP_MAX_ITER,
                       bp_method=_BP_METHOD, ms_scaling_factor=_BP_MS_SCALING,
                       schedule=_BP_SCHEDULE, lsd_method="lsd_cs",
                       lsd_order=lsd_order)
    return _LdpcAdapter(dem, "BPLSD", cfg, dec, "lsd_cs_order10")


# --------------------------------------------------------------------------- #
# Relay-BP (relay-bp[stim] >= 0.2.2) — IBM's Rust reference decoder.            #
# --------------------------------------------------------------------------- #
# Construct-from-DEM:
#   from relay_bp.stim import CheckMatrices
#   cm = CheckMatrices.from_dem(dem)        # -> .check_matrix (ndet x E csc),
#                                           #    .observables_matrix (nobs x E csc),
#                                           #    .error_priors (E,)
#   dec = relay_bp.RelayDecoderF64(cm.check_matrix, error_priors=cm.error_priors,
#             gamma0=, pre_iter=, num_sets=, set_max_iter=, gamma_dist_interval=,
#             stop_nconv=, stopping_criterion='nconv')   # disjoint-relay ensemble
#   runner = relay_bp.ObservableDecoderRunner(dec, cm.observables_matrix,
#                                             include_decode_result=False)
# Decode:
#   runner.decode_observables_batch(syndromes uint8 [shots, n_det])
#       -> predicted observables uint8 [shots, n_obs]
# This is the path relay_bp.stim.SinterDecoder_RelayBP uses internally, minus
# sinter's bit-packing — the runner is driven directly for a clean decode_batch.
_RELAY_BP_DEFAULTS = dict(
    gamma0=0.1,
    pre_iter=80,
    num_sets=60,
    set_max_iter=60,
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,
    stopping_criterion="nconv",
)


class RelayBPAdapter:
    """Relay-BP adapter (in-process). Builds the relay-BP decoder from the SAME
    shared DEM via ``relay_bp.stim.CheckMatrices.from_dem`` and decodes a batch
    of syndromes straight to observables. G1 holds trivially: ``.dem is dem``."""

    def __init__(self, dem, **params):
        import importlib.metadata as _md

        import relay_bp
        from relay_bp.stim import CheckMatrices

        self.dem = dem
        self.name = "RelayBP"
        try:
            ver = _md.version("relay-bp")
        except Exception:  # pragma: no cover - metadata present once installed
            ver = "unknown"
        cfg = dict(_RELAY_BP_DEFAULTS)
        cfg.update(params)
        self.config = dict(decoder="RelayBP", relay_bp_version=ver, **cfg)
        # Deterministic relay schedule (fixed gamma distribution + nconv stop).
        self.tie_break = "relay_bp_nconv_disjoint_ensemble"

        cm = CheckMatrices.from_dem(dem)
        self._n_obs = cm.observables_matrix.shape[0]
        decoder = relay_bp.RelayDecoderF64(
            cm.check_matrix,
            error_priors=cm.error_priors,
            **cfg,
        )
        self._runner = relay_bp.ObservableDecoderRunner(
            decoder, cm.observables_matrix, include_decode_result=False)

    def decode_batch(self, dets):
        dets = np.asarray(dets, dtype=bool)
        pred = np.asarray(
            self._runner.decode_observables_batch(dets.astype(np.uint8)))
        pred = (pred % 2).astype(bool)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred


def make_relay_bp(dem, **params):
    return RelayBPAdapter(dem, **params)


# Registry: name -> factory(dem).
_FACTORIES = {
    "BPOSD-0": make_bposd0,
    "BPOSD-10": make_bposd10,
    "BPLSD": make_bplsd,
    "BP": make_bp,
}

DEFAULT_DECODERS = ("BPOSD-0", "BPOSD-10", "BPLSD", "BP")


def build_decoders(dem, which=DEFAULT_DECODERS, include_relay=False):
    """Construct all requested adapters from ONE shared DEM object.

    Every returned adapter has ``.dem is dem`` (provenance for the matched
    harness). ``which`` selects/orders the ldpc-family adapters by registry
    name. Relay-BP is OPT-IN via ``include_relay=True`` and is added ONLY when
    its package is available (import-guarded), so the core set always builds.
    """
    decoders = []
    for name in which:
        if name not in _FACTORIES:
            raise KeyError(f"unknown decoder {name!r}; known: {sorted(_FACTORIES)}")
        decoders.append(_FACTORIES[name](dem))

    if include_relay and relay_bp_available():
        decoders.append(make_relay_bp(dem))

    return decoders
