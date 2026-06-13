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

## v0.2: the megakernel backend (opt-in)

A single-launch **persistent megakernel** — the entire Relay-BP decode (every
BP iteration, every relay leg, in-kernel syndrome convergence + `nconv` stop +
lowest-weight selection) in **one kernel launch per `decode_batch`**, with
**per-shot early exit**, instead of the v0.1 host loop's thousands of launches.
Validated on all three platforms against the v0.1 two-kernel path and the
`relay-bp` Rust oracle (14/14 gates at BLOCK 128/256 on CUDA and ROCm; barriers
verified honored on both):

| Relay-BP megakernel vs v0.1 two-kernel | speedup |
|---|---|
| Apple M4 Max (Metal, triton-metal) | **~148×** — 30.0 s → 0.202 s / 2000 shots (relay BLOCK=128) |
| NVIDIA H200 (CUDA) | **9–19×** fp32 (fp64 to 37× mid-batch) — batch-1 62.5 → 3.44 ms; 34.6 µs/syn @8192 |
| AMD MI300X (ROCm) | **9–32×** — batch-1 8.48 ms; 46.0 µs/syn @8192 |

*(Speedups are vs **each platform's own** v0.1 two-kernel path and vary with
batch size — min–max across batch 1–16384. Absolute cross-vendor performance,
where H200 leads, is in the limits below.) Receipt stacks: H200 (CUDA 12.4 /
triton 3.0), MI300X (ROCm 6.2 / torch 2.5.1 / triton 3.1, gfx942), M4 Max
(triton-metal, CODEGEN_VERSION 2026.06.13); raw in `bench/receipts/megakernel_*`
— the Metal block-lift re-measure is in `megakernel_metal_lift.{md,json}`. The
Metal 30.0 s baseline and the v0.1 Apple section's 31 s below are independent
measurement runs of the same two-kernel relay (run-to-run jitter), not a
discrepancy.*

Opt-in today via `tridec.backends.megakernel.{RelayBpMegaTriton, BpMegaTriton}`;
**auto-dispatch from `from_dem(..., backend="auto")` lands in v0.2.1** once the
public-API path is gated on a GPU
([#5](https://github.com/bledden/tridec/issues/5)). Receipts:
`bench/receipts/megakernel_{h200,mi300x,metal}*`.

### Megakernel: honest limits + tuning

- **Plain-BP megakernel is a single-shot *latency* tool, not a throughput
  tool.** At batch-1 it is ~1.7× faster than the two-kernel BP path
  (H200 0.61 vs 1.06 ms); at large batch it *loses* (plain BP has no early-exit
  lever) — the two-kernel BP path stays the throughput default. Use
  `BpMegaTriton` for low-latency bare BP, `RelayBpMegaTriton` for the accurate
  latency path.
- **Real-time / single-shot: H200 leads MI300X 2.47× at batch-1** (3.44 vs
  8.48 ms) — wider than v0.1's ~9% two-kernel gap, because the
  single-CTA-per-shot design amplifies per-SM and codegen differences at
  batch-1. Batched, the gap is 1.25–1.33×; correctness is identical across
  vendors. (The pitch is *vendor-portable + performant on both*, never parity.)
- **Per-arch autotuning.** v0.2 ships autotuned `BLOCK`/`num_warps` configs for
  H200, MI300X and M4 Max, pinned in `_CUDA_TUNED` keyed by
  `gcnArchName`/device name. AMD (wavefront-64) wants the *opposite* shape from
  NVIDIA warps — low warps for BP, max BLOCK+warps for relay.
- **Metal: lifted off the old BLOCK=32 pin** (triton-metal CODEGEN_VERSION
  2026.06.13 honors `tl.debug_barrier` and fixed reduction-in-loop for the BP
  shape). Metal now runs **BP at BLOCK=256** (20 → 12 ms, 1.67×) and **relay at
  BLOCK=128** (441 → 202 ms, 2.18× — the ~148× headline above). *Honest
  negative:* the relay megakernel still **refuses at BLOCK≥256** —
  `MetalNonRecoverableError` at compile time, a **loud refusal, never
  silent-wrong** — because its in-loop convergence-reduction + lowest-weight
  pass hits a codegen pattern triton-metal's register-array spine doesn't yet
  cover (reported upstream). BP, using the simpler reported shape, runs at
  256–1024. fp32-only on Metal (no fp64), same as the two-kernel path, and the
  fp32 near-tie-flip caveat below applies to the megakernel unchanged. Receipt:
  `bench/receipts/megakernel_metal_lift.md`.

## Install

Most users want `pip install "tridec[torch,decoders]"` (CPU+GPU torch backend
plus the reference adapters). The bare install is the **numpy CPU reference
only** — correct but slow.

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

*This table covers the **two-kernel** BP/Relay-BP path (v0.1). The v0.2
**megakernel**'s own per-platform validation (14/14 gates on CUDA + ROCm,
~148× on Metal at the lifted BP=256/relay=128 blocks) is in the
[megakernel section](#v02-the-megakernel-backend-opt-in) above, with raw
receipts in `bench/receipts/megakernel_*`.*

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
shots, fresh instances, 5 repeats: 880/880/879/880/880 — the same single shot
flips) — documented in `bench/receipts/full_grid_noregression.json`.

## Status

`0.2.0` — adds the opt-in megakernel backend (tri-platform validated; see
above). v0.1.0 shipped the two-kernel BP/Relay-BP path + the validation
discipline. The kernels and their receipts are stable; the public API surface
is young and may still move before 1.0 — minor `0.x` releases may rename or
remove public API; `1.0` will lock the surface. **Next (`v0.2.1`):
megakernel auto-dispatch from `from_dem(..., backend="auto")`, gated on the
public-API path running on a GPU
([#5](https://github.com/bledden/tridec/issues/5)).** GPU paths require triton
+ a CUDA/ROCm GPU (or the experimental triton-metal environment); the
GPU/metal test tiers skip cleanly where unavailable.

## License

Apache-2.0.
