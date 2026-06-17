# Follow-ups: #6 grid-flatten (d>=15 ceiling) + #4 cuda_graph default-on

Two follow-ups, validated cross-vendor (NVIDIA H200 / CUDA and AMD MI300X /
ROCm7-hipGraph) via `bench/validate_followups.py`.

## (c) #6 — 1-D grid flatten lifts the surface d>=15 BP-kernel ceiling

`_check_update_kernel` / `_bit_update_kernel` previously launched on a 2-D grid
`(grid_s, units)` where `units` = check count `C` (resp. bit count `N`). Both ride
grid dimension 1, capped at **65535** on CUDA and HIP. Surface BP first crossed it
at **d=15** (`N`=72191 bits > 65535) — the documented wall (`KNOWN_LIMIT_d15_bp_kernel.md`,
issue #6); d=14 (57622 bits) was the last working distance, failing identically on
both vendors → a shared kernel-launch limit, not memory or vendor.

Fix: launch a **1-D grid** `(grid_s * units,)` and recover `(shot-tile, unit)`
inside the kernel (`grid_s = cdiv(S, BLOCK_S); unit = pid // grid_s; pid_s = pid -
unit*grid_s`). Grid dim 0 caps at 2^31-1, so `C`/`N` no longer bind. The per-program
computation is unchanged — every `(shot-tile, unit)` pair is still covered exactly
once — so it is **bit-identical** to the old grid.

**Ceiling lifted (decode now succeeds past d=14), both vendors:**

| d | checks | bits | before | after (H200 / MI300X LER) |
|---|---|---|---|---|
| 13 | 2184 | 45821 | OK | OK (31.0% / 36.1%) |
| 14 | 2729 | 57622 | OK (last working) | OK (35.3% / 37.5%) |
| 15 | 3360 | **72191** | **WALL (`invalid argument` CUDA / `HIP`)** | **OK (40.8% / 39.7%)** |
| 17 | 4896 | **107113** | WALL | **OK (43.7% / 41.1%)** |

`bits` crosses 65535 exactly at d=15 — the grid-dim-1 cap. (LER rises with d
because this is *plain* min-sum BP on surface codes — the documented BP-vs-matching
negative; the point here is the launch ceiling, not LER. Per-vendor LER differs
because the linux stim sampler draws different shots per box, not because the
decoder differs.)

**Bit-identical at d<=14 (no regression):**
- Proven against the old grid on Metal (triton-metal), d=3..11, identical syndromes:
  every prediction bit equal.
- vs the numpy BP baseline on GPU (fp32 kernel vs fp64 reference; <=0.1% near-tie
  disagreement allowed): d=5 **0.000%**, d=9 **0.000%**, d=13 **0.050%** (MI300X) /
  **0.100%** (H200).
- The CUDA/ROCm `test_bp_triton.py` gate (numpy-vs-triton) passes on both: 5/5.

## (a) #4 — cuda_graph default flips False -> "auto" (small-S gated)

`cuda_graph` now defaults to **"auto"**: the CUDA-graph fast path is used only for
small batches (`S <= cuda_graph_max_s`, default 256), where per-launch overhead
dominates and the replay wins; larger one-off batches stay eager because the
capture cost there is pure overhead. The per-shape graph cache is capped
(`cuda_graph_cache`, default 16) to bound captured-tensor memory. `cuda_graph=True`
forces it for any `S`; `=False` disables. Auto-falls-back to eager on any capture
failure (so Metal/CPU and uncapturable drivers are unaffected).

**Bit-identical (graph True / auto vs eager False), MI300X:** S=1/16/256/1024 all ✓.
**Small-S gating:** S=16 captured, S=4096 skipped ✓. **Cache cap:** 8 distinct
shapes with cap=4 -> exactly 4 captured ✓.

**Latency (300-call median, ms/decode, `cuda_graph=True` forced at every S) — this
is why the default gates on small S:**

| S | H200 eager | H200 graph | H200 | MI300X eager | MI300X graph | MI300X |
|---|---|---|---|---|---|---|
| 1 | 1.032 | 0.536 | **1.93x** | 2.007 | 1.361 | 1.47x |
| 16 | 1.063 | 0.774 | 1.37x | 2.200 | 1.282 | 1.72x |
| 64 | 1.047 | 0.766 | 1.37x | 1.995 | 1.347 | 1.48x |
| 256 | 1.062 | 0.794 | 1.34x | 2.040 | 1.567 | 1.30x |
| 512 | 1.202 | 95.702 | **0.01x** | 2.057 | 1.822 | 1.13x |
| 1024 | 1.801 | 95.589 | **0.02x** | 2.348 | 2.256 | 1.04x |
| 4096 | 6.295 | 96.711 | **0.07x** | 4.753 | 4.821 | 0.99x |

Both vendors win at small S (H200 up to 1.93x at batch-1). **But on H200/CUDA the
forced large-batch graph collapses to ~95 ms at S>=512** — a cuBLAS-in-graph
large-workspace pathology (the captured `Lo @ e_hat` matmul); MI300X degrades
gracefully to ~parity instead. This is precisely why `cuda_graph` defaults to
**"auto" with `max_s=256`**: auto never captures above 256, so the **default never
touches the cliff** (the S=1024 bit-identity row used auto and fell back to eager).
Flipping default-on for *all* S would have shipped a 100x H200 regression at S>=512
— the small-S gate is load-bearing, and 256 sits safely below the cliff on both
vendors. `cuda_graph=True` (force-any-S) is a benchmarking/diagnostic knob; prefer
"auto" in production.

## Env
- H200: CUDA, torch 2.4.1+cu124, triton 3.0.0.
- MI300X (VF): ROCm7, torch 2.9.0.dev+rocm7.0, triton 3.4.0+rocm7 (hipGraph).
- `bench/validate_followups.py` exits nonzero on any check failure; both vendors
  printed `ALL FOLLOW-UP VALIDATIONS PASSED`.
