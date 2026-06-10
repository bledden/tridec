# tridec

[![ci](https://github.com/bledden/tridec/actions/workflows/ci.yml/badge.svg)](https://github.com/bledden/tridec/actions/workflows/ci.yml)

*Badge honesty: CI is CPU-only (ubuntu + macos arm64; the macos lane binds the
strict exact-count receipt gates). There are no GPU runners — the CUDA/ROCm
kernel paths are validated by the carried H200/MI300X receipts in
`bench/receipts/`, and the experimental Metal tier runs on a local machine.*

An open, vendor-portable GPU decoder library for quantum LDPC codes — Triton
min-sum BP and Relay-BP decoders that consume any stim `DetectorErrorModel` or
raw parity-check matrices, with CPU reference implementations, validated
against the standard CPU references (`ldpc`, `relay-bp`), running on NVIDIA
(CUDA) and AMD (ROCm) GPUs.

The same Triton kernels run unmodified on both vendors: the Relay-BP kernel
reproduces its logical-error-rate validation numbers identically on an NVIDIA
H200 (CUDA 12.4, triton 3.0) and an AMD MI300X (ROCm 7.0, triton 3.4) — see
[docs/benchmark.md](docs/benchmark.md) and the raw receipts in
`bench/receipts/`. Validated scope is NVIDIA + AMD; Apple silicon runs the
same kernels through [triton-metal](https://github.com/bledden/triton-metal)
as an **experimental** backend (see below).

## Install

```bash
pip install tridec                # numpy CPU reference only
pip install "tridec[torch]"       # + batched torch backend (CPU/GPU)
pip install "tridec[gpu]"         # + Triton GPU kernels (CUDA or ROCm)
pip install "tridec[decoders]"    # + ldpc / relay-bp reference adapters
pip install "tridec[sinter]"      # + sinter.collect integration
```

## Quickstart

```python
import stim
import tridec

circuit = stim.Circuit.from_file("memory.stim")
dem = circuit.detector_error_model(decompose_errors=False)

decoder = tridec.from_dem(dem, backend="auto")   # triton > torch > numpy

dets, obs = circuit.compile_detector_sampler(seed=0).sample(
    100_000, separate_observables=True)
pred = decoder.decode_batch(dets)                      # (shots, n_obs) bool
print("logical error rate:", (pred != obs).any(axis=1).mean())
```

Raw matrices work too: `tridec.from_matrices(H, priors, observables=Lo)`.
Relay-BP: `tridec.from_dem(dem, algorithm="relay")` (Triton kernels only).

With [sinter](https://pypi.org/project/sinter/) (the `[sinter]` extra):

```python
import sinter
from tridec.sinter import sinter_decoders

stats = sinter.collect(
    num_workers=4, tasks=tasks,
    decoders=["tridec_bp", "pymatching"],
    custom_decoders=sinter_decoders(),
    max_shots=1_000_000)
```

## Backend × algorithm matrix (honest availability)

| Algorithm | `numpy` | `torch` | `triton` | `metal` (experimental) |
|---|---|---|---|---|
| min-sum BP | yes (CPU reference) | yes (CPU + CUDA/ROCm) | yes (CUDA + ROCm) | yes (fp32) |
| Relay-BP | no | no | yes (CUDA + ROCm) | yes (fp32, slow — see below) |

There is no in-package CPU Relay-BP; its CPU reference is IBM's `relay-bp`
Rust decoder, wrapped in `tridec.adapters` and used as the validation
oracle for the Triton path.

## What's validated where

| Environment | Status |
|---|---|
| CPU (any) | numpy BP reference; torch BP bit-identical to numpy at fp64 (one iteration), LER-identical full decode |
| NVIDIA H200, CUDA 12.4, torch 2.4.1, triton 3.0.0 | Triton BP: ≥99.5% hard-decision agreement vs fp64 references, LER-identical (156 = 156 = 156 fails / 2000 shots vs numpy/torch). Triton Relay-BP: LER-matches the `relay-bp` Rust oracle (31 vs 38 fails / 2000, overlapping Wilson CIs) — carried source-repo receipts |
| AMD MI300X, ROCm 7.0.0, torch 2.9, triton 3.4.0 | Same kernels, unmodified: identical primitive-identity numbers (pre-leg posterior max-diff 1.8e-15) and the same oracle-vs-Triton LER identity (carried receipts) — **and validated through the installed package for v0.1.0** (`bench/receipts/mi300x_packaged.json`): full suite 88 passed / 10 skipped on gfx942 (GPU tiers bind, darwin-only strict tiers skip), packaged-API BP 166 = numpy 166 fails / 2000, Relay-BP fp32 34 vs Rust oracle 31 (overlapping CIs), throughput within ±2.2% of the carried receipt |
| Apple silicon (M4 Max), triton-metal | **Experimental, spike-validated only** (`bench/receipts/metal_spike.md`): both kernels pass the same correctness gates at fp32; see the section below |

## Experimental: Apple silicon (Metal)

The same Triton kernels run on Apple-silicon GPUs through
[triton-metal](https://github.com/bledden/triton-metal), with **zero changes
to the kernel source**. This is experimental: validated at spike level on one
machine (M4 Max), fp32 only (Metal has no fp64), and not part of the
official receipt set.

```bash
# triton-metal + a triton >= 3.6 build + torch must be importable, then:
pip install tridec
python -c "import tridec; print(tridec.available_backends())"  # ['metal', ...]
```

`backend="auto"` detects the triton-metal environment (darwin, `triton` +
`triton_metal` importable, no CUDA/ROCm device) and selects `"metal"`;
`backend="triton"` resolves to `"metal"` there too, and `backend="metal"`
asserts the environment is present. The execution pattern is triton-metal's
documented one — **CPU torch tensors** (zero-copy via unified memory; not
`mps`) — so no device arguments are needed.

What the spike measured (2000 canonical shots, seed 0, M4 Max — re-validated
through this API path in `tests/test_metal.py`):

- **min-sum BP (fp32)**: all correctness gates pass — one-iteration hard
  agreement 1.000 vs the fp64 numpy reference on both the surface-code and
  BB-code fixtures; LER 76 = 76 / 2000 (surface) and 167 vs 168 / 2000 (BB).
  Batched decode of 2000 shots in 28 ms (surface) / 167 ms (BB) — **37–56×
  the per-shot numpy baseline on the same machine**.
- **Relay-BP (fp32)**: correct but slow — LER matches the `relay-bp` Rust
  oracle (31 vs 39 fails / 2000, per-shot agreement 99.3%), but
  `decode_batch(2000)` takes **31 s** vs 1.26 s for the Rust CPU oracle:
  relay's per-iteration host loop (~7k small kernel launches) is
  launch-overhead dominated on Metal. Use it for validation, not production.
- Relay-BP on metal enforces fp32: `dtype="float64"` raises with a clear
  error; the default resolves to `float32`.

No claims beyond the spike: no official LER receipts, no cross-machine
validation, no performance tuning. CUDA/ROCm remain the supported GPU paths.

Compatibility floors in `pyproject.toml`; known-good pins: stim 1.15.0,
ldpc 2.4.1, relay-bp 0.2.2, torch 2.4.1 / 2.9, triton 3.0 / 3.4.

## Validation discipline

`tridec.validation` ships the matched-protocol harness the numbers were
produced with: `dem_hash` (sha256 of the DEM's canonical bytes), `run_matched`
(one shared DEM, one shot set, fail-fast DEM-identity and tie-break gates),
Wilson/TOST statistics and a paired per-shot gap-to-MLE bootstrap. The test
suite pins the extraction byte-for-byte: 8 canonical BB-code fixture circuits
must hash to the exact DEM sha256s recorded in the carried `zoo_grid.json`
receipt, and a full 16,667-shot cell must reproduce the recorded
logical-failure counts of the `ldpc` reference adapters exactly.

For v0.1.0 the WHOLE grid was re-decoded in the receipt environment
(`bench/full_grid_noregression.py`): 31 of 32 (cell, decoder) failure counts
reproduce **exactly** — all 24 BP / BP-OSD-0 / BP-OSD-10 counts, and 7 of 8
BPLSD counts. The single deviation (BPLSD, p=0.002/X: 879 vs 880, one shot
in 200,000) is attributed by a same-environment repeat experiment to
run-to-run nondeterminism inside ldpc's `BpLsdDecoder` itself (identical
shots, fresh instances: 879/880/879) — documented in
`bench/receipts/full_grid_noregression.json`.

## Status

`0.1.0` — first release. The kernels and their validation receipts are
stable; the public API surface is young and may still move before 1.0. GPU
paths require triton + a CUDA/ROCm GPU (or the experimental triton-metal
environment); the GPU/metal test tiers skip cleanly where unavailable.
Validated through the installed package on MI300X/ROCm (v0.1.0) and via
carried receipts on H200/CUDA; Metal is experimental.

## License

Apache-2.0.
