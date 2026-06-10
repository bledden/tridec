"""Canonical DEM extraction — the single source of truth for circuit-level decoding.

Every decoder in this package consumes a ``stim.DetectorErrorModel`` through
``extract`` below (or the equivalent raw matrices via ``from_matrices``). The
extraction is code-agnostic: it walks ``dem.flattened()`` and reads off every
``error`` instruction's detectors, observables and prior, with NO assumptions
about the code family, schedule or noise model that produced the DEM.

Build the DEM with ``circuit.detector_error_model(decompose_errors=False)`` —
the decoders operate on the raw hyperedge mechanism set, not a graphlike
decomposition.

Provides:
  extract(dem) -> dict(H, Lo, priors, n_det, n_obs, n_err, edges)
      H:  (n_det x n_err) scipy CSR uint8 — detector incidence per mechanism,
      Lo: (n_obs x n_err) scipy CSR uint8 — observable incidence per mechanism,
      priors: (n_err,) float — per-mechanism error probability,
      edges: list of (frozenset(detector_ids), tuple(observable_ids)) per
             mechanism, in DEM order.
"""
import numpy as np
import scipy.sparse as sp


def extract(dem):
    """Canonical hyperedge extraction from a stim DEM (decompose_errors=False)."""
    nd, no = dem.num_detectors, dem.num_observables
    rows_d, cols_d, rows_o, cols_o, priors, edges = [], [], [], [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != 'error':
            continue
        pr = inst.args_copy()[0]
        dets, obs = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
        for d in dets:
            rows_d.append(d); cols_d.append(j)
        for o in obs:
            rows_o.append(o); cols_o.append(j)
        priors.append(pr); edges.append((frozenset(dets), tuple(obs)))
        j += 1
    E = j
    H = sp.csr_matrix((np.ones(len(rows_d), np.uint8), (rows_d, cols_d)), shape=(nd, E))
    Lo = sp.csr_matrix((np.ones(len(rows_o), np.uint8), (rows_o, cols_o)), shape=(no, E))
    return dict(H=H, Lo=Lo, priors=np.array(priors), n_det=nd, n_obs=no, n_err=E, edges=edges)
