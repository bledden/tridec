# portable-qec

An open, vendor-portable GPU decoder library for quantum LDPC codes — Triton
min-sum BP and Relay-BP decoders that consume any stim `DetectorErrorModel` or
raw parity-check matrices, with CPU reference implementations, validated
against the standard CPU references (`ldpc`, `relay-bp`), running on NVIDIA
(CUDA) and AMD (ROCm) GPUs.

The same Triton kernels run unmodified on both vendors: the Relay-BP kernel
reproduces its logical-error-rate validation numbers identically on an NVIDIA
H200 (CUDA 12.4, triton 3.0) and an AMD MI300X (ROCm 7.0, triton 3.4) — see
[docs/benchmark.md](docs/benchmark.md) and the raw receipts in
`bench/receipts/`. Scope is NVIDIA + AMD only.

## Install

```bash
pip install portable-qec                # numpy CPU reference only
pip install "portable-qec[torch]"       # + batched torch backend (CPU/GPU)
pip install "portable-qec[gpu]"         # + Triton GPU kernels (CUDA or ROCm)
pip install "portable-qec[decoders]"    # + ldpc / relay-bp reference adapters
```

*(Not yet on PyPI — install from source with `pip install -e .` for now.)*

## Quickstart

```python
import stim
import portable_qec

circuit = stim.Circuit.from_file("memory.stim")
dem = circuit.detector_error_model(decompose_errors=False)

decoder = portable_qec.from_dem(dem, backend="auto")   # triton > torch > numpy

dets, obs = circuit.compile_detector_sampler(seed=0).sample(
    100_000, separate_observables=True)
pred = decoder.decode_batch(dets)                      # (shots, n_obs) bool
print("logical error rate:", (pred != obs).any(axis=1).mean())
```

Raw matrices work too: `portable_qec.from_matrices(H, priors, observables=Lo)`.
Relay-BP: `portable_qec.from_dem(dem, algorithm="relay")` (Triton backend only).

## Backend × algorithm matrix (honest availability)

| Algorithm | `numpy` | `torch` | `triton` |
|---|---|---|---|
| min-sum BP | yes (CPU reference) | yes (CPU + CUDA/ROCm) | yes (CUDA + ROCm) |
| Relay-BP | no | no | yes (CUDA + ROCm) |

There is no in-package CPU Relay-BP; its CPU reference is IBM's `relay-bp`
Rust decoder, wrapped in `portable_qec.adapters` and used as the validation
oracle for the Triton path.

## What's validated where

| Environment | Status |
|---|---|
| CPU (any) | numpy BP reference; torch BP bit-identical to numpy at fp64 (one iteration), LER-identical full decode |
| NVIDIA H200, CUDA 12.4, torch 2.4.1, triton 3.0.0 | Triton BP: ≥99.5% hard-decision agreement vs fp64 references, LER-identical (156 = 156 = 156 fails / 2000 shots vs numpy/torch). Triton Relay-BP: LER-matches the `relay-bp` Rust oracle (31 vs 38 fails / 2000, overlapping Wilson CIs) |
| AMD MI300X, ROCm 7.0.0, torch 2.9, triton 3.4.0 | Same kernels, unmodified: identical primitive-identity numbers (pre-leg posterior max-diff 1.8e-15) and the same oracle-vs-Triton LER identity |
| Apple / Metal | out of scope |

Compatibility floors in `pyproject.toml`; known-good pins: stim 1.15.0,
ldpc 2.4.1, relay-bp 0.2.2, torch 2.4.1 / 2.9, triton 3.0 / 3.4.

## Validation discipline

`portable_qec.validation` ships the matched-protocol harness the numbers were
produced with: `dem_hash` (sha256 of the DEM's canonical bytes), `run_matched`
(one shared DEM, one shot set, fail-fast DEM-identity and tie-break gates),
Wilson/TOST statistics and a paired per-shot gap-to-MLE bootstrap. The test
suite pins the extraction byte-for-byte: 8 canonical BB-code fixture circuits
must hash to the exact DEM sha256s recorded in the carried `zoo_grid.json`
receipt, and a full 16,667-shot cell must reproduce the recorded
logical-failure counts of the `ldpc` reference adapters exactly.

## Status

`v0.1.0-dev` — APIs may move. The kernels and their validation receipts are
stable; the packaging, public API surface and docs are young. GPU paths
require triton + a CUDA/ROCm GPU and skip cleanly in tests where unavailable.

## License

Apache-2.0.
