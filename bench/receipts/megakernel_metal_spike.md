# Megakernel spike (issue #2, v0.2): single-launch persistent BP / Relay-BP on Metal

> **BLOCK-lift follow-up landed (2026-06-13):** the two triton-metal codegen
> gaps that forced `BLOCK=32` below (dropped `tl.debug_barrier`; reduction-in-
> loop at `BLOCK≥256`) are fixed upstream (CODEGEN_VERSION 2026.06.13). Metal is
> now lifted to **BP=256 / relay=128** — see `megakernel_metal_lift.md` /
> `.json` for the re-measured numbers. This receipt is retained as the
> historical record of the `BLOCK=32` workaround and the triton-metal
> limitation repros. (Relay still refuses at `BLOCK≥256`, loudly — detailed in
> the lift receipt.)

> **Cloud follow-up landed:** the CUDA half of the deferred work (barrier
> verification, BLOCK=128/256 + fp64 gates, autotune, H200 receipts) is in
> `megakernel_h200.md` / `megakernel_h200.json`.

**Date:** 2026-06-10
**Status:** PRELIMINARY / EXPERIMENTAL — Metal (triton-metal) receipts only.
H200/MI300X receipts + the CUDA/ROCm autotune pass are a later cloud session.
**Verdict:** PASS — Plan A (shot-per-program, global per-shot buffers)
survived intact. ONE kernel launch per `decode_batch`, per-shot early exit,
all correctness gates green, and the measured Metal launch overhead is fully
recovered: relay decode_batch(2000) **30.0 s → 0.46 s (65×)**.

## What was built

`src/tridec/backends/megakernel.py` — two new Triton kernels + wrappers:

- `_bp_megakernel` / `BpMegaTriton(BpTriton)`: the whole plain-BP decode of
  one shot in one program (in-kernel iteration loop; check pass + bit pass as
  tile loops over the existing padded-edge-index tables). One launch per
  `decode_batch` instead of 2 × n_iter.
- `_relay_megakernel` / `RelayBpMegaTriton(RelayBpTriton)`: the ENTIRE relay
  schedule in-kernel — pre leg + up to `num_sets` relay legs, per-iteration
  GF2 syndrome-convergence check, first-convergence capture, lowest-weight
  valid-solution selection, and the per-shot `nconv` early exit. One launch
  per `decode_batch` instead of 2 × iters × legs (~7k launches at the
  canonical config).
- Gamma draws are precomputed host-side once per batch ((n_legs, n_bits),
  ~386 KB for the BB cell) with the EXACT RNG of the existing
  `_gamma_for_leg`, so gamma tensors are identical to the host-loop
  implementation by construction.
- Messages are per-shot-private, shot-major (S, E)/(S, N) global buffers
  (~43 KB/shot for [[72,12,6]] BB) — cache-resident; Plan B (SRAM-resident
  messages) was not needed.
- Per-shot early exit follows the relay_bp Rust oracle's PER-SHOT semantics;
  the host loop only stops when ALL shots are done. `early_exit=False` runs
  the all-legs schedule for the identity gate below.

## Environment

- Apple M4 Max, macOS (Darwin 24.6.0), `/tmp/pq-metal-venv` per
  `metal_spike.md` recipe (python 3.14.4)
- triton-metal @ `4c42e9685db5cf58ce856d2816844066a27feb6a` (clean tree),
  triton 3.7.0 (`3.7.0+git4da2e268`, source build), torch 2.9.1,
  stim 1.15.0, relay-bp 0.2.2
- Execution pattern: CPU torch tensors (UMA), `device="cpu"`, fp32,
  **block=32** (one SIMD-group per program — required on Metal, see
  "triton-metal limitations" below)

## Validation gates (script: `bench/megakernel_metal.py`; JSON receipt:
`megakernel_metal_spike.json`; pytest: `tests/test_megakernel_metal.py`)

2000 shots, stim sampler seed 0, the canonical fixtures.

### BP megakernel vs two-kernel `BpTriton` + fp64 numpy reference

| DEM | posterior bit-identity vs BpTriton (30 it) | determinism | numpy hard agree 1 it / 30 it | LER numpy / BpTriton / mega |
|---|---|---|---|---|
| bb72_r6_p0.003_Z | **1.000000** (maxdiff 0.0) | 1.0 | 1.00000 / 0.99998 | 167 / 168 / **168** |
| surface_d3_r3_p0.003 | **1.000000** (maxdiff 0.0) | 1.0 | 1.00000 / 1.00000 | 76 / 76 / **76** |

The megakernel is **bit-for-bit identical** to the existing two-kernel path
(same per-(shot,row) fp32 operation order, deliberately), so it inherits the
existing path's validation against numpy/torch/oracle wholesale.
`decode_batch` predictions identical to `BpTriton` on both DEMs (1.00000).

### Relay megakernel

- **All-legs identity mode** (`early_exit=False` vs host loop with
  unreachable `stop_nconv`; identical 61-leg schedule, identical gamma
  tensors): per-shot prediction identity **1.00000** (N=256), deterministic.
  The in-kernel fp32 weight selection (host computes selection weights in
  fp64) caused zero divergences on this sample.
- **Real config** (`stop_nconv=5`, per-shot early exit), N=2000:
  LER oracle **31** / host RelayBpTriton **39** / megakernel **39** — the
  megakernel exactly reproduces the host implementation's failure count, and
  passes the test_relay_triton.py oracle bar (|39−31| = 8 ≤ 20).
  - per-shot agreement mega vs host: **0.9985** (3/2000 differ — expected
    semantic delta: the host loop keeps running legs for already-converged
    shots and may lower their best weight; the megakernel stops per-shot at
    `stop_nconv`, which is the Rust oracle's semantics)
  - per-shot agreement mega vs oracle: **0.9920** (vs 0.9930 for the host
    loop in metal_spike.md — same ballpark; gamma RNGs differ from Rust by
    construction)
  - early-exit stats: legs min/mean/max = 5 / 5.55 / 61, iterations mean
    353 (of 3680 max), 6/2000 shots never converged (best_eh = 0, same as
    the host path's behavior for unconverged shots).

### No-regression

Full existing suite on the CPU env (`/tmp/tridec-venv`, python 3.14):
**88 passed, 4 skipped** (the 4 skips are the triton/GPU/metal modules, as
on master). No source file of the v0.1.0 release is touched by this branch.

## Bench (wall-clock `decode_batch`, canonical BB cell, 2000 shots, M4 Max)

| workload | two-kernel host loop | megakernel | recovery |
|---|---|---|---|
| Relay-BP (stop_nconv=5), warmup 1, reps 2/3 | 30.0 s (29.93, 29.99) | **0.46 s** (0.455, 0.466, 0.460) | **65×** |
| BP-only 30 it, warmup 2, reps 5 | 178 ms | **26 ms** (25.4–26.9) | 6.9× |

Context: metal_spike.md measured the relay host loop at ~31 s with ~1.3 s of
math — i.e. ~95% launch overhead. The megakernel at 0.46 s is now **2.7×
faster than the relay_bp Rust oracle on the same machine** (1.25 s CPU) and
sits below the old MATH-only time, because the per-shot early exit also
eliminates the host loop's over-decoding (the host runs every leg until ALL
shots converge; mean legs actually needed: 5.55 of 61).

## triton-metal limitations found (probed; minimal repros in /tmp, see below)

1. **`tl.debug_barrier()` is SILENTLY DROPPED** by triton-metal @4c42e96
   with triton 3.7: the op reaches TTGIR as `ttg.barrier all` (renamed
   upstream ~3.5), but the lowerer only recognizes the old
   `tt.debug_barrier` spelling and skips unknown ops without error
   (`triton_metal/codegen/generic_lowerer.py`, op dispatch + the
   `has_barrier_ops` scan). The megakernels NEED these barriers — the check
   pass writes `nu` that the bit pass gathers cross-lane — and without them
   the kernel is racy: measured run-to-run posterior agreement ~0.80 at 10
   iterations, block=128. **Workaround shipped:** on Metal the kernels run
   with `block=32` (one threadgroup = one 32-wide SIMD-group, lockstep under
   uniform control flow), where the missing barriers are benign — verified
   bit-identical + deterministic above. The barriers stay in the kernel
   source (they're required on CUDA/ROCm, where `tl.debug_barrier` works).
   Fix for triton-metal (NOT applied — out of scope for this repo): map
   `ttg.barrier` alongside `tt.debug_barrier` in the lowerer dispatch and in
   the `has_barrier_ops` detection (two one-line edits).
2. **Cross-lane reductions (`tl.sum`) inside dynamic loops miscompile for
   BLOCK ≥ 256**: the emitted .metal source references undeclared
   `UNKNOWN_<addr>` identifiers (Metal shader compilation fails, exit 1).
   BLOCK ≤ 128 is fine. The relay megakernel's convergence/weight reductions
   run inside the iteration loop, so block ≤ 128 on Metal (32 used, per #1).
3. **Scalar-`if` assignment to a `while`-carried variable miscompiles
   SILENTLY** (the assignment is dropped; loop never exits early —
   reproduced with a 10-line kernel). The megakernels therefore use NO
   `while` loops and NO scalar-`if` blocks: the leg loop is a `for` whose
   inner iteration loop gets a ZERO trip count once the shot converged, all
   carried scalars update via arithmetic/`tl.where`, and the conditional
   lowest-weight capture is a scalar-bool-masked store behind a zero-trip
   guard loop. (Verified workable: probe `2b/2c/2d` + the full control
   skeleton, all PASS.)

What DOES work on triton-metal 3.7 (probed PASS): runtime-bound and strided
`for` loops, nested dynamic loops with `tl.where`-computed bounds, zero-trip
dynamic loops, `tl.static_range` unrolls inside dynamic loops (MAXDEG_C up
to 48), gathers/scatters with computed indices, int32 xor parity, scalar
accumulators carried across dynamic loops, scalar-bool & vector store
masks, `tl.sum` in dynamic loops at BLOCK ≤ 128, int64 base-offset
arithmetic.

## Deferred to the cloud session (H200 / MI300X)

- CUDA/ROCm correctness re-run (the barriers compile there; block=128+ and
  the `early_exit` semantics are platform-independent) + LER gates.
- Throughput receipts vs the v0.1 two-kernel path and vendor baselines;
  block-size/num-warps autotune (block=128 default is unprofiled on CUDA).
- Large-batch behavior (occupancy with S programs of 1 CTA each; int32
  offset audit done — bases are int64).
- fp64 relay megakernel gate vs the F64 oracle (Metal has no fp64).

*Spike probe artifacts: /tmp/mk_probe.py, /tmp/mk_probe2.py,
/tmp/mk_probe3.py (the triton-metal limitation repros), /tmp/mk_bar_test.py,
/tmp/mk_bar_test2.py (the dropped-barrier repro). Repo additions on the
`megakernel` branch only: the backend module, this receipt + JSON, the bench
script, and the metal-gated test module.*
