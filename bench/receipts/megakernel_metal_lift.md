# Megakernel Metal BLOCK-lift (follow-up to the 2026-06-10 spike)

> **FINAL UPDATE 2026-06-14 — relay fully lifted to `BLOCK=256`.** The relay
> `BLOCK≥256` refusal (the "honest negative" below) is **resolved**: triton-metal's
> B/C/A branch (`@ b82136b`) fixed the in-loop reduction, and the relay kernel runs
> correctly at `BLOCK=256` with **`num_warps=8`** (so `num_threads = 8×32 = 256 =
> BLOCK` → `n=1` element/thread). New relay default: **`(256, 8)` = 152 ms**
> (1.33× over the 128 below, **2.89× over the BLOCK=32 pin, ~197× over v0.1**),
> deterministic + bit-identical to `BLOCK=128`. The `n>1` footgun (`num_warps=4` at
> 256) now **loudly refuses** (`MetalNonRecoverableError`, never silent-wrong).
> `test_megakernel_metal.py` 4/4 PASS at `(256,8)`. Details: the "Relay full lift"
> section at the bottom + the `relay_full_lift_2026-06-14` block in the JSON.
> Requires triton-metal `≥ b82136b`; on older triton-metal relay@256 loudly refuses.
> *(A silent-wrong regression was caught + fixed mid-flight — see
> `tridec-bug-reports/REPLY_RELAY_BISECTION_2026-06-14.md`.)* The 2026-06-13 body
> below documents the interim relay=128 state.

**Date:** 2026-06-13 (interim) → 2026-06-14 (final relay lift)
**Status:** PASS — the Metal `BLOCK=32` pin is **lifted**. BP → `BLOCK=256`,
relay → `BLOCK=256` (`num_warps=8`). *(Interim 2026-06-13: relay was pinned to
`BLOCK=128` while relay@256 still refused; now resolved — see banner above.)*

This supersedes the **config** in `megakernel_metal_spike.{md,json}` (which ran
everything at `BLOCK=32` as a workaround for two triton-metal codegen gaps).
Those gaps are now fixed upstream, so this receipt re-measures at the lifted
blocks. The spike receipt is retained as the historical record of the
workaround + the triton-metal limitation repros.

## Why the pin could be lifted

`megakernel_metal_spike.md` documented two triton-metal gaps that forced
`BLOCK=32` on Metal:

1. **`tl.debug_barrier()` silently dropped** (Bug 1) → the kernel was racy at
   `BLOCK>32` (run-to-run posterior agreement ~0.80).
2. **Cross-lane reduction (`tl.sum`) inside dynamic loops miscompiled at
   `BLOCK≥256`** (Bug 2) → undeclared `UNKNOWN_<addr>` identifiers.

triton-metal **CODEGEN_VERSION 2026.06.13** (commit `0a1eafb`) fixes Bug 1
(barriers honored) and fixes Bug 2 **for the BP-megakernel shape** via its
register-array spine. Caches (`~/.triton/cache`, `~/.cache/triton_metal`) were
cleared before this run, per the codegen bump.

## Environment

- Apple M4 Max, macOS (Darwin 24.6.0), `/tmp/pq-metal-venv`, python 3.14.4
- triton-metal `0.1.0a1 @ 0a1eafb` (CODEGEN_VERSION 2026.06.13), triton 3.7.0,
  torch 2.9.1, stim 1.15.0, relay-bp 0.2.2, numpy 2.2.6
- fp32, CPU(UMA) tensors (`device="cpu"`), canonical BB `bb72_r6_p0.003_Z`
  fixture (`.stim` sampler seed 0 → shots; pinned `.dem` → decode), 2000 shots

## BLOCK sweep (correctness gated BEFORE timing)

References this run: numpy fp64 BP fails **168/2000**; relay_bp Rust oracle
fails **35/2000** (oracle's own gamma RNG is unpinned, so its count varies
run-to-run — 31 in the spike; the gate is `|Δ|` vs the oracle + per-shot
agreement, both well inside bars).

**BP megakernel** (30 iters) — deterministic + correct at **every** block;
Bug 2 fix confirmed (runs at 256–1024, was capped at 128):

| BLOCK | run-twice det. | hard-agree vs numpy fp64 | fails | time |
|---|---|---|---|---|
| 32 | ✅ | 99.05% | 168 | 20 ms |
| 128 | ✅ | 99.05% | 168 | 15 ms |
| **256** | ✅ | 99.05% | 168 | **13 ms** ← chosen |
| 512 | ✅ | 99.05% | 168 | 14 ms |
| 1024 | ✅ | 99.05% | 168 | 18 ms |

**Relay megakernel** (`stop_nconv=5`, fp32) — deterministic + correct at 32 and
128; **refuses at ≥256**:

| BLOCK | run-twice det. | agree vs oracle | fails | time |
|---|---|---|---|---|
| 32 | ✅ | 99.40% | 39 | 0.44 s |
| **128** | ✅ | 99.40% | 39 | **0.21 s** ← chosen (fastest that compiles) |
| 256 | — | — | — | ❌ `MetalNonRecoverableError` |
| 512 | — | — | — | ❌ `MetalNonRecoverableError` |
| 1024 | — | — | — | ❌ `MetalNonRecoverableError` |

The 99.05% BP / 99.40% relay agreement (not 100%) is the expected fp32 vs fp64
near-tie flips — the *failure counts* match the references exactly (168, and 39
vs the host path's 39).

## Clean bench (warmup 3, reps 5, min/median; 2000 shots)

| workload | BLOCK=32 | lifted | speedup vs 32 |
|---|---|---|---|
| Relay-BP (`stop_nconv=5`) | 441 / 444 ms | **BLOCK=128: 202 / 210 ms** | **2.18×** |
| BP 30-iter | 20 / 20 ms | **BLOCK=256: 12 / 13 ms** | **1.67×** |

**Headline vs the v0.1 two-kernel Metal path:**
- Relay `decode_batch(2000)`: 30.0 s → **0.202 s ≈ 148×** (was 65× at the
  BLOCK=32 spike's 0.46 s; v0.1 baseline carries ±1 s jitter).
- BP 30-iter: 178 ms → **12 ms ≈ 14.8×**.

## Gates at the lifted defaults

`tests/test_megakernel_metal.py` — **4/4 PASS** with no block override
(i.e. BP=256, relay=128 via `_tuned_config`): BP posterior bit-identity vs the
two-kernel `BpTriton` (BB + surface), relay all-legs identity, relay LER vs the
`relay-bp` oracle, all with run-twice determinism.

## Honest negative — relay refuses at BLOCK≥256

The relay megakernel's in-loop **syndrome-convergence reduction +
lowest-weight capture** hit a codegen pattern the register-array spine does
**not yet cover**, so `BLOCK≥256` raises `MetalNonRecoverableError`
("refusing to emit silently-wrong output") **at compile time** — a loud
refusal, never silent-wrong output. The BP megakernel (which uses the simpler
reported reduction-in-loop shape) now runs at 256–1024, confirming the spine
fix landed for *that* shape. This is reported back to the triton-metal thread
for scoping: `tridec-bug-reports/REPLY_BLOCK_LIFT_2026-06-13.md`. Until it's
covered, relay caps at `BLOCK=128` on Metal — which is already the fastest
compiling config and 2.18× over the old pin.

*Receipt JSON: `megakernel_metal_lift.json`. Sweep/measure scripts:
`/tmp/metal_sweep.py`, `/tmp/metal_measure.py`.*

---

## Relay full lift — `BLOCK=256, num_warps=8` (2026-06-14, FINAL)

The interim relay=128 above existed only because relay@256 **refused** (in-loop
reduction unsupported). triton-metal's B/C/A branch (`@ b82136b`, CODEGEN
`2026.06.13.1`) fixed it. Root cause of the residual hazard + the fix:

- At `BLOCK=256` with the default `num_warps=4`, `num_threads = 4×32 = 128`, so
  `n = BLOCK/num_threads = 2` elements/thread. triton-metal's **base path emits
  general element-wise loads/stores one-element-per-thread**, silently
  under-covering at `n>1`. (An intermediate B/C/A build shipped this as
  **silent-wrong + racy**; caught by tridec's repeated-run gate, root-caused, and
  fixed — full trail in `tridec-bug-reports/REPLY_RELAY_BISECTION_2026-06-14.md`.)
- **Fix: `num_warps = BLOCK/32`** → `num_threads == BLOCK` → `n=1`, base path
  covers every element. The `n>1` footgun now raises a **loud
  `MetalNonRecoverableError`** (`b82136b`), never silent-wrong.

Verified on M4 Max, 2000-shot BB cell, vs the triton-metal B/C/A branch
(`PYTHONPATH` to the worktree), 3× decode each — all `n=1` configs are
**deterministic and bit-identical to the known-good `BLOCK=128`**:

| relay config | n | min/med ms | fails | det | ≡ 128 |
|---|---|---|---|---|---|
| (128, 4) | 1 | 202.2 / 204.5 | 39 | ✅ | — (ref) |
| **(256, 8)** | 1 | **152.5 / 153.3** | 39 | ✅ | ✅ ← chosen |
| (512, 16) | 1 | 178.2 / 179.5 | 39 | ✅ | ✅ |
| (1024, 32) | 1 | 247.6 / 248.7 | 39 | ✅ | ✅ |
| (256, **4**) | 2 | — | — | — | **REFUSED (loud)** |

**Headline:** relay `decode_batch(2000)` **30.0 s → 0.1525 s ≈ 197×** vs the v0.1
two-kernel Metal path (was 65× at the BLOCK=32 spike, ~148× at the interim 128).
1.33× over relay@128, 2.89× over the BLOCK=32 pin.

**Gate:** `tests/test_megakernel_metal.py` **4/4 PASS** at `relay=(256,8)` against
`b82136b`. **Chosen Metal defaults:** BP `(256, None)`, relay `(256, 8)`.
**Requires** triton-metal `≥ b82136b`; older → relay@256 loudly refuses (safe).
