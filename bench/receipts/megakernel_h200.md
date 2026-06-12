# Megakernel CUDA receipts (issue #2/#3/#4, v0.2): H200 validation + autotune

**Date:** 2026-06-11
**GPU:** NVIDIA H200 (driver 580.126.09, app clock 1980 MHz, single-tenant
pod), torch 2.4.1+cu124, triton 3.0.0, CUDA 12.4, python 3.11.10
**JSON receipt:** `megakernel_h200.json` · **pytest:** `tests/test_megakernel_cuda.py`
**Status:** the CUDA half of the cloud session (the Metal spike is
`megakernel_metal_spike.md`; the MI300X/ROCm half is a separate session).

## Verdict

PASS. CUDA honors `tl.debug_barrier()` (the triton-metal showstopper does
not exist here), so the megakernels run at real block sizes. All 14 CUDA
gates green at BLOCK=128 AND 256, including the fp64 relay gates Metal
couldn't run. Autotuned winners: **bp (512, 8)**, **relay (256, 8)** — every
one of the 24 sweep configs passed its correctness gate before timing.

## 1. Barrier sanity (the decisive first gate)

triton-metal silently DROPS `tl.debug_barrier` (megakernel_metal_spike.md),
which forced block=32 there. Verified CUDA does not:

- **Behavioral repro:** a load → `tl.debug_barrier` → store ring-rotation
  kernel (integer-valued fp32, so every add is exact; 512 CTAs × 400
  rounds) is **exact** at BLOCK=128 and 256. The no-barrier negative
  control races to **<1% match** — the test detects the failure mode it is
  designed for.
- **PTX inspection:** `bar.sync` is present in the emitted PTX of the real
  kernels at every source barrier site (`_bp_megakernel`: 3;
  `_relay_megakernel`: 8 = 3 source sites + triton's own reduction
  barriers; large-BLOCK variants add layout-conversion barriers).
- **Real-kernel gate:** BP megakernel at BLOCK=128, 2000 shots, run twice:
  posterior determinism exact, bit-identity vs `BpTriton` exact
  (maxdiff 0.0).

## 2. Correctness gates (`tests/test_megakernel_cuda.py`, 14 passed)

2000 shots (256 for all-legs), stim sampler seed 0, canonical fixtures;
every gate at BLOCK=128 AND BLOCK=256.

| gate | result |
|---|---|
| BP posterior bit-identity vs two-kernel `BpTriton` + determinism (bb72 AND surface d3) | **exact** (1.000000, maxdiff 0.0) at both blocks |
| BP `decode_batch` prediction identity vs `BpTriton` | exact |
| BP LER vs fp64 numpy reference (suite bar ±max(3, 0.5%)) | pass, both fixtures |
| Relay ALL-LEGS identity vs host loop, **fp32** | 254/256 and 255/256 exact; every mismatch VERIFIED as a degenerate tie (below) |
| Relay ALL-LEGS identity vs host loop, **fp64** | **256/256 exact at both blocks, zero ties** |
| Relay real config (stop_nconv=5) vs relay_bp Rust F64 oracle, fp32 | LER oracle 31 vs mega **34** (bar ≤20), per-shot 0.9900 |
| Relay real config vs F64 oracle, **fp64** (deferred from Metal) | LER oracle 31 vs mega **38**, per-shot 0.9875 — **identical to the v0.1 HOST fp64 path** (31/38/0.9875 in `bench_relay_triton.json`) |

### Two findings (fixed/documented on this branch)

1. **fp64 compile bug (latent since the spike):** `best_w` was initialized
   from a python float (fp32) but reassigned to DT inside the leg loop —
   triton 3.0 on CUDA rejects the fp32→fp64 loop-carried type flip. The
   fp32-only Metal spike never compiled this path. Fixed:
   `best_w = tl.full([], INF, DT)`.
2. **Degenerate lowest-weight ties:** the fp32 all-legs mismatches (2/256 at
   BLOCK=128, 1/256 at 256) were diagnosed shot-by-shot: in every case host
   and megakernel return DIFFERENT syndrome-consistent solutions with
   EXACTLY equal fp64 weight (hamming 4–6 apart, |e| equal). The kernel
   sums candidate weights in the message dtype (BLOCK-tree order), the host
   in fp64, so an exact tie can break 1 ulp apart. Not a race — run-to-run
   deterministic, gone entirely in fp64 mode. The gate now verifies any
   mismatch IS such a tie (equal fp64 weight + syndrome-consistent) and
   fails otherwise.

## 3. Autotune (issue #3)

BLOCK ∈ {64,128,256,512} × num_warps ∈ {2,4,8}, both kernels, BB cell,
timing = `decode_batch(2000)` mean of 10 (3 warmup), **each config
correctness-gated before timing** (BP: bit-identity + determinism on 512
shots; relay: determinism + solution identity vs reference config with
verified-tie exemption). All 24 configs PASSED their gates.

| BLOCK | warps | bp ms | relay ms |
|---|---|---|---|
| 64 | 2 | 7.359 | 144.274 |
| 64 | 4 | 6.675 | 140.435 |
| 64 | 8 | 6.268 | 172.405 |
| 128 | 2 | 7.715 | 161.497 |
| 128 | 4 | 6.603 | 109.230 |
| 128 | 8 | 5.746 | 108.702 |
| 256 | 2 | 7.299 | 190.883 |
| 256 | 4 | 5.796 | 117.901 |
| 256 | 8 | **5.553** | **91.608** |
| 512 | 2 | 9.457 | 293.979 |
| 512 | 4 | 6.993 | 178.031 |
| 512 | 8 | **5.495** | 121.106 |

Winners recorded in `megakernel._CUDA_TUNED` ("H200"): **bp (512, 8)**,
**relay (256, 8)**. num_warps=8 matters more than BLOCK (the bp top-3 are
within 5%). Sweep tuned at batch 2000; the per-batch receipts below run the
winners.

## 4. Throughput + latency (same pod, same batches, same syndromes)

`decode_batch` wall-clock, `torch.cuda.synchronize`-bracketed
`perf_counter`, warmup 5 / reps 20; real sampled syndromes (seed 0,
prefixes of one 16384-shot sample). v0.1 baseline = the two-kernel
host-loop path at the SAME batches on the SAME pod.

### Relay-BP fp32 (stop_nconv=5, the real config)

| batch | mega ms | mega µs/syn | mega shots/s | v0.1 host ms | v0.1 host µs/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 3.449 | 3448.8 | 290 | 62.466 | 62466 | **18.1×** |
| 16 | 3.632 | 227.0 | 4,406 | 64.103 | 4006 | **17.7×** |
| 256 | 39.124 | 152.8 | 6,543 | 753.451 | 2943 | **19.3×** |
| 2000 | 92.704 | 46.4 | 21,574 | 966.263 | 483.1 | **10.4×** |
| 8192 | 283.129 | 34.6 | 28,934 | 2679.916 | 327.1 | **9.5×** |
| 16384 | 528.187 | 32.2 | 31,019 | 4845.072 | 295.7 | **9.2×** |

### Relay-BP fp64

| batch | mega ms | mega µs/syn | mega shots/s | v0.1 host ms | v0.1 host µs/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 4.468 | 4468.1 | 224 | 62.234 | 62234 | **13.9×** |
| 16 | 4.777 | 298.5 | 3,350 | 175.267 | 10954 | **36.7×** |
| 256 | 54.143 | 211.5 | 4,728 | 791.255 | 3091 | **14.6×** |
| 2000 | 115.127 | 57.6 | 17,372 | 1361.410 | 680.7 | **11.8×** |
| 8192 | 354.017 | 43.2 | 23,140 | 4330.391 | 528.6 | **12.2×** |
| 16384 | 665.387 | 40.6 | 24,623 | 8347.824 | 509.5 | **12.5×** |

(The v0.1 host fp64 @2000 here, 1361 ms, reproduces the carried receipt's
1359 ms — `bench_relay_triton.json` — same pod class, same protocol.)

### BP-only, 30 iterations — the megakernel does NOT win at large batch

| batch | mega ms | mega µs/syn | mega shots/s | v0.1 2-kernel ms | v0.1 2-kernel µs/syn | speedup |
|---|---|---|---|---|---|---|
| 1 | 0.609 | 609.2 | 1,642 | 1.064 | 1064.1 | **1.7×** |
| 16 | 0.627 | 39.2 | 25,518 | 1.097 | 68.6 | **1.8×** |
| 256 | 0.911 | 3.6 | 280,923 | 1.097 | 4.3 | **1.2×** |
| 2000 | 5.485 | 2.7 | 364,621 | 2.061 | 1.0 | **0.4×** |
| 8192 | 21.438 | 2.6 | 382,127 | 6.790 | 0.8 | **0.3×** |
| 16384 | 42.509 | 2.6 | 385,422 | 12.962 | 0.8 | **0.3×** |

The v0.1 two-kernel BP at 16384 reproduces its carried receipt exactly
(12.96 ms ≈ 13.03 ms / 1.26 M shots/s, `bench_triton_results.json`). The BP
megakernel is a **latency** design (batch-1 floor 1.06 ms → 0.61 ms;
crossover ≈ batch 300–500): shot-per-program serializes the check/bit tile
loops inside one CTA, while the two-kernel path spreads (shots × rows)
across the whole GPU per pass. For plain-BP THROUGHPUT the two-kernel path
remains the right tool on CUDA; there is no early-exit or launch-count
argument to rescue (plain BP runs a fixed 30 iterations either way). The
relay megakernel does not have this trade: its win comes from per-shot
early exit, which the host loop cannot express.

**Batch composition note (honesty):** the carried v0.1 number "fp32 relay
115 µs/syn @8192" (`bench_cudaq_compare.json`) came from the source-repo
harness whose 8192 batch repeated the 2000-shot sample, which caps the
host loop's global-stop tail. With REAL 8192/16384-shot samples (this
receipt; same protocol as `mi300x_packaged.json`, whose host fp32 @8192 =
281 µs/syn on MI300X) the host loop is slower — the megakernel's per-shot
early exit is immune to that tail by construction, which is exactly the
point of issue #2.

**Batch-1 floor (issue #4):** relay fp32 `decode_batch(1)` = **3.44 ms**
mean (min 3.437; vs 62.5 ms v0.1 host loop). Decomposition: 3.37 ms is the
kernel + `last_stats` readback with a device-resident syndrome, i.e. the
floor is **kernel-bound**; the 1-shot H2D upload is 17.0 µs pageable /
12.2 µs pinned — pinned memory is NOT the lever (a ~5 µs saving against a
3.4 ms floor), so no pinned-path code change was made. The BP batch-1
floor drops 1.06 ms → **0.61 ms**. Further relay floor work = splitting
one shot's tile loops across more than one CTA (a different kernel shape),
deferred.

**Vendor context (NOT a head-to-head):** CUDA-Q QEC 0.6.0 measured
~2 µs/syn relay-class at batch 8192 on this GPU class
(`bench_cudaq_compare.json`) — different harness, different relay selection
semantics (FirstConv-style vs lowest-weight nconv=5; its LER measured ~2×
ours on this cell). For scale only: the megakernel's 34.6 µs/syn @8192
closes the v0.1 gap to that number from ~58× to ~17×.

## 5. Did the Metal 65× translate?

No — and it was never expected to. The 65× on Metal was launch-overhead
recovery (~95% of the host-loop wall-clock there was kernel-launch cost).
CUDA launches are ~100× cheaper, so the host loop's baseline is far less
launch-bound, and the megakernel's win on CUDA comes from (a) the per-shot
early exit (the host loop over-decodes: every leg runs until ALL shots
converge, and the tail deepens with batch) and (b) one launch instead of
~7000. Measured: **relay fp32 9.2–19.3×** (batch-dependent), **relay fp64
11.8–36.7×**, and per-syndrome cost at scale 483 → **46 µs @2000** /
296 → **32 µs @16384**. Plain BP: latency 1.7×, large-batch throughput
0.3× (see the BP table — the two-kernel path stays the BP throughput
tool). No-regression elsewhere: CPU suite 88 passed / 5 skipped, and the
Metal megakernel gates re-run green (4 passed) with this branch's kernel
changes (`tl.full` typed scalar + num_warps plumbing) on the spike
machine.

## Deferred to the MI300X half

- ROCm sweep + `_CUDA_TUNED["MI300X"]` entry (the table ships with the
  H200 row + a (128,4) fallback).
- ROCm barrier sanity (s_barrier in GCN ISA) + gate re-run.
- Cross-platform receipt comparison (stim sampler is platform-dependent —
  Wilson CIs, not exact counts).
