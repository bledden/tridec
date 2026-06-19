# Megakernel ROCm receipts (issue #2/#3/#4, v0.2): MI300X validation + autotune

**Date:** 2026-06-12
**GPU:** AMD Instinct MI300X OAM (gfx942 / CDNA3, SR-IOV VF "MI3SRIOV"), 304 CUs,
wavefront 64, 192 GiB. Host KFD driver 6.10.5. torch 2.5.1+rocm6.2 (HIP 6.2.41133),
triton 3.1.0, stim 1.15.0, relay-bp 0.2.2, numpy 2.2.6, python 3.11.4.
Bare cloud image (no torch/triton; ROCm userspace supplied by the torch-rocm6.2 wheel).
**JSON receipt:** `megakernel_mi300x.json` · **pytest:** `tests/test_megakernel_cuda.py`
**Status:** the ROCm/MI300X half of the v0.2 cloud session (CUDA/H200 half:
`megakernel_h200.md`; Metal spike: `megakernel_metal_spike.md`). NEW environment
row vs the v0.1 MI300X receipts (`bench_relay_mi300x.json`, torch 2.9-rocm7 /
triton 3.4): this is torch 2.5.1+rocm6.2 / triton 3.1.0 -- expect identical
CORRECTNESS, not identical timings.

## Verdict

PASS. ROCm honors `tl.debug_barrier()` (emitted as `s_barrier` in AMDGCN at every
source site), so the megakernels run at real block sizes (128/256), not the
Metal lockstep-32 fallback. All 14 megakernel gates green at BLOCK=128 AND 256,
including the fp64 relay + Rust-oracle gates. Full tridec suite: 102 passed, 11
skipped (the skips are darwin/arm64 exact-count pins). Autotuned winners on
gfx942: **bp (64, 2)**, **relay (512, 8)** -- all 24 sweep configs passed their
correctness gate before timing.

## 1. Barrier sanity (ROCm, the decisive first gate)

AMD wavefront=64, so BLOCK=128 spans 2 waves and BLOCK=256 spans 4 -- the
threadgroup barrier matters across wave boundaries. Verified honored:

- **Behavioral repro:** a load -> `tl.debug_barrier` -> store ring-rotation
  kernel (integer fp32, 512 CTAs x 400 rounds) is **exact** at BLOCK=128 AND
  256 (frac_match 1.0). The no-barrier negative control races: 0.9550 match at
  BLOCK=128 (2 waves), 0.4349 at BLOCK=256 (4 waves) -- more waves, more racing.
- **ISA inspection:** `s_barrier` present in the emitted AMDGCN of the real
  kernels at every source site: `_bp_megakernel` = 3, `_relay_megakernel` = 8
  (3 source sites + triton reduction barriers) -- IDENTICAL to the H200 PTX
  `bar.sync` counts. Large-BLOCK variants add layout-conversion barriers
  (bp 143 / relay 148), same pattern as H200.

This is the ROCm analogue of the H200 barrier proof and the counter-example to
the triton-msl drop: triton-rocm 3.1 lowers `tl.debug_barrier` correctly.

## 2. Correctness gates (`tests/test_megakernel_cuda.py`, 14 passed)

2000 shots (256 for all-legs), stim seed 0, canonical BB + surface fixtures;
every gate at BLOCK=128 AND 256. torch.cuda is the HIP alias on ROCm.

| gate | result |
|---|---|
| BP posterior bit-identity vs two-kernel `BpTriton` + determinism (bb72 AND surface) | exact at both blocks |
| BP `decode_batch` prediction identity vs `BpTriton` | exact |
| BP LER vs fp64 numpy reference (bar +-max(3, 0.5%)) | pass, both fixtures |
| Relay ALL-LEGS identity vs host loop, **fp32** | pass at both blocks (mismatches verified equal-fp64-weight ties) |
| Relay ALL-LEGS identity vs host loop, **fp64** | pass at both blocks |
| Relay real config (stop_nconv=5) vs relay_bp Rust **F64 oracle**, fp32 | pass (LER bar + per-shot >= 0.98) |
| Relay real config vs F64 oracle, **fp64** | pass |

relay-bp 0.2.2 was built from sdist on the pod (cargo 1.96 via rustup; the
distro cargo 1.75 could not parse its lockfile v4), so the Rust-oracle gates RAN
-- they did not skip. No code changes were needed for ROCm correctness: the
kernels and gates from the H200 half passed as-is.

## 3. Autotune (issue #3)

BLOCK in {64,128,256,512} x num_warps in {2,4,8}, both kernels, BB cell, timing
= `decode_batch(2000)` mean of 10 (3 warmup), each config correctness-gated
before timing. **All 24 configs PASSED.** On AMD, triton num_warps counts
64-lane wavefronts (warps=8 = 512 threads/CTA).

| BLOCK | warps | bp ms | relay ms |
|---|---|---|---|
| 64 | 2 | 6.145 | 370.588 |
| 64 | 4 | 7.778 | 397.371 |
| 64 | 8 | 11.820 | 474.011 |
| 128 | 2 | 6.340 | 225.711 |
| 128 | 4 | 7.723 | 252.771 |
| 128 | 8 | 8.293 | 257.635 |
| 256 | 2 | 8.165 | 263.883 |
| 256 | 4 | 9.344 | 186.048 |
| 256 | 8 | 9.255 | 178.983 |
| 512 | 2 | 9.351 | 288.355 |
| 512 | 4 | 9.447 | 244.315 |
| 512 | 8 | 7.053 | 149.352 |

Winners recorded in `megakernel._CUDA_TUNED["gfx942"]`: **bp (64, 2)**, **relay (512, 8)**.
Unlike H200 (bp 512/8, relay 256/8), MI300X bp wants LOW warps (warps=8
oversubscribes the small per-shot tile -- bp 64/8 is the WORST bp config at
11.8 ms) while relay wants the LARGEST BLOCK+warps (512/8) -- it has 61 legs x
iters of work to hide the wider CTA. torch reports the generic device name on
this VF, so the row is keyed by gcnArchName (gfx942), matched in `_tuned_config`.

## 4. Throughput + latency (same pod, same batches, same syndromes)

`decode_batch` wall-clock, hip-synchronize-bracketed perf_counter, warmup 5 /
reps 20; real sampled syndromes (seed 0). v0.1 baseline = the two-kernel
host-loop path at the SAME batches on the SAME pod.

### Relay-BP fp32 (stop_nconv=5, the real config)

| batch | mega ms | mega us/syn | mega shots/s | v0.1 ms | v0.1 us/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 8.426 | 8426.2 | 119 | 79.223 | 79223.3 | 9.4x |
| 16 | 9.050 | 565.6 | 1768 | 292.395 | 18274.7 | 32.3x |
| 256 | 91.837 | 358.7 | 2788 | 2849.204 | 11129.7 | 31.0x |
| 2000 | 148.745 | 74.4 | 13446 | 3269.073 | 1634.5 | 22.0x |
| 8192 | 376.990 | 46.0 | 21730 | 5100.889 | 622.7 | 13.5x |
| 16384 | 657.619 | 40.1 | 24914 | 7437.320 | 453.9 | 11.3x |

### Relay-BP fp64

| batch | mega ms | mega us/syn | mega shots/s | v0.1 ms | v0.1 us/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 8.870 | 8870.2 | 113 | 71.709 | 71708.7 | 8.1x |
| 16 | 9.034 | 564.6 | 1771 | 839.476 | 52467.3 | 92.9x |
| 256 | 97.131 | 379.4 | 2636 | 3632.733 | 14190.4 | 37.4x |
| 2000 | 185.134 | 92.6 | 10803 | 4210.600 | 2105.3 | 22.7x |
| 8192 | 505.823 | 61.7 | 16195 | 6862.626 | 837.7 | 13.6x |
| 16384 | 915.207 | 55.9 | 17902 | 10276.510 | 627.2 | 11.2x |

### BP-only, 30 iterations -- the megakernel does NOT win (latency design)

| batch | mega ms | mega us/syn | mega shots/s | v0.1 ms | v0.1 us/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 2.600 | 2600.1 | 385 | 1.928 | 1927.8 | 0.7x |
| 16 | 2.469 | 154.3 | 6479 | 0.975 | 60.9 | 0.4x |
| 256 | 2.530 | 9.9 | 101203 | 0.990 | 3.9 | 0.4x |
| 2000 | 6.167 | 3.1 | 324315 | 1.931 | 1.0 | 0.3x |
| 8192 | 49.960 | 6.1 | 163972 | 7.265 | 0.9 | 0.1x |
| 16384 | 102.594 | 6.3 | 159697 | 13.523 | 0.8 | 0.1x |

**Batch-1 floor (issue #4):** relay fp32 `decode_batch(1)` = **8.483 ms** mean.
Decomposition: 8.311 ms is the kernel + `last_stats` readback with a
device-resident syndrome -- the floor is **kernel-bound**, not transfer-bound:
the 1-shot H2D is 15.9 us pageable / 11.6 us pinned, so pinned memory is NOT
the lever (same conclusion as H200). No pinned-path code change was made.

The BP megakernel LOSES to the two-kernel path at every batch on MI300X (0.1-
0.7x), including batch-1 -- more decisively than on H200 (1.7x at batch-1 there).
Plain BP runs a fixed 30 iterations with no early-exit lever, so the two-kernel
path (which spreads shots x rows across all 304 CUs per pass) stays the BP tool
on ROCm. The relay megakernel win comes from per-shot early exit, which the host
loop cannot express -- that is what carries to AMD.

## 5. Cross-vendor: MI300X vs H200 (megakernel relay fp32)

| metric | H200 | MI300X | MI300X/H200 |
|---|---|---|---|
| relay fp32 us/syn at 8192 | 34.6 | 46.0 | 1.33x |
| relay fp32 us/syn at 16384 | 32.2 | 40.1 | 1.25x |
| relay fp32 batch-1 floor ms | 3.44 | 8.48 | 2.47x |

Honest read: the megakernel is ~25-33% slower per-syndrome at large batch on
MI300X and ~2.5x slower at the batch-1 floor. That is a WIDER gap than the v0.1
two-kernel path (MI300X was ~9% behind H200 there). The shot-per-program design
serializes one CTA on a single CU; the batch-1 floor is one CU running the whole
per-shot relay schedule, where H200 single-SM throughput + the more mature
CUDA/triton-3.0 codegen pull ahead. The CORRECTNESS is identical (all gates),
and the speedup-over-v0.1 shape holds on AMD: relay fp32 9-32x.

Cross-platform LER is statistical, not exact (stim sampling + gamma RNG differ);
the oracle gates confirm agreement within the bars on this cell.

