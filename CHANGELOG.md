# Changelog

## 0.2.1 — 2026-06-14

*Validated on all three platforms: Metal (M4 Max), NVIDIA H200 (CUDA), and AMD
MI300X (ROCm/gfx942). The CUDA/ROCm kernels are byte-identical to 0.2.0 (the
megakernel lift is Metal-only and auto-dispatch is routing), so the 0.2.0
performance receipts stand; re-confirmed no regression on H200/MI300X.*

**Statistical-tier gate uses Wilson-CI overlap (#1).** The cross-decoder /
cross-platform / vs-oracle gates (where exact failure counts can't match by
construction) now assert a single sample-size-aware
`validation.wilson_consistent(f1, n1, f2, n2)` (two rates consistent iff their
95% Wilson score intervals overlap), replacing the ad-hoc `abs(diff) <= max(5,
5%)` count bars — too loose at high N/LER, too strict at low counts. New helper
+ unit test; applied across the numpy-vs-ldpc, no-regression statistical tier,
and relay-vs-`relay_bp`-oracle gates (metal/CUDA/ROCm). The fp32-near-tie-flip
bars (same-algo, ≈0 flips) are intentionally left as tight absolute tolerances.

**Relay-BP auto-dispatch (#5).** `tridec.from_dem(..., algorithm="relay")` and
`RelayBpDecoder` now use the single-launch **megakernel by default on GPU** —
it wins decisively on relay (9–32× on CUDA/ROCm, ~197× on Metal vs the v0.1
two-kernel host loop, with per-shot early exit and LER matching the `relay_bp`
oracle). Pass `megakernel=False` for the v0.1 two-kernel path. The dispatch is
**GPU-gated by construction** — `RelayBpDecoder` only accepts the `triton`
(CUDA/ROCm) and `metal` backends, so the megakernel is never built on CPU. The
megakernel is a drop-in `RelayBpTriton` subclass (overrides `_relay_posteriors`
only), so `decode_batch` and the no-observable path are unchanged. **BP keeps
the two-kernel default** (`BpMegaTriton` stays opt-in via
`tridec.backends.megakernel`) — the plain-BP megakernel is a single-shot
*latency* tool that loses at batch throughput. Validated on Metal + CPU (full
suite 99 passed / 6 skipped); CUDA/ROCm dispatch is a pre-publish confirmation
(the megakernel's kernel correctness there is already receipted).

**Metal megakernel fully lifted off the `BLOCK=32` pin — both kernels at
`BLOCK=256`.** The `triton-metal` codegen gaps that forced `BLOCK=32` in 0.2.0
(silently-dropped `tl.debug_barrier`; cross-lane reduction-in-loop) are fixed
upstream. Metal now runs **BP at `(256)`** (20 → 12 ms, 1.67×) and **relay at
`(256, num_warps=8)`** (441 → 152 ms, **2.89×**), lifting the Metal relay
headline from **65× → ~197×** vs the v0.1 two-kernel path (30.0 s → 0.152 s /
2000 shots), relay bit-identical to `BLOCK=128`. The relay `num_warps=8` sets
`num_threads = 256 = BLOCK` so each thread handles one element (`n=1`); at `n>1`
triton-metal's base path under-covers a BLOCK-wide store and now **loudly
refuses** (`MetalNonRecoverableError`, never silent-wrong). All 4 Metal gates
pass at the new defaults; relay validated deterministic + bit-identical to
`BLOCK=128` + LER-vs-oracle over repeated runs. Requires triton-metal with the
in-loop-reduction + `n=1`-store fixes (older → relay@256 loudly refuses).
Receipt: `bench/receipts/megakernel_metal_lift.{md,json}`.

*Process note:* an intermediate upstream build made relay@256 silently-wrong +
racy (base-path `n>1` under-coverage); caught by tridec's repeated-run
determinism gate, root-caused, and resolved with `num_warps=block/32` + a loud
upstream refusal before any lift shipped (the kernel never produced wrong output
through the default path).

## 0.2.0 — 2026-06-11

**Megakernel backend (opt-in), three-platform-validated — two honest negatives
stated.** A single-launch *persistent megakernel* runs the entire Relay-BP
decode (every BP iteration, every relay leg, in-kernel syndrome convergence +
`nconv` stop + lowest-weight selection) in **one kernel launch per
`decode_batch`** with **per-shot early exit**, replacing v0.1's host loop of
~thousands of launches. Same math — validated bit-/LER-identical to the
two-kernel path and the `relay-bp` Rust oracle (14/14 gates at BLOCK 128/256 on
both CUDA and ROCm, incl. fp64-vs-oracle; barriers verified honored on both —
PTX `bar.sync` / AMDGCN `s_barrier`).

Relay-BP speedups vs the v0.1 two-kernel path:
- **Metal** (M4 Max, triton-metal, BLOCK=32): 30.0 s → **0.46 s** / 2000 shots (**65×**).
- **NVIDIA H200** (CUDA, triton 3.0): **9–19×** fp32 (fp64 to 37× at mid-batch); batch-1 62.5 → 3.44 ms.
- **AMD MI300X** (ROCm, torch 2.5.1+rocm6.2 / triton 3.1, gfx942): **9–32×** fp32; batch-1 8.48 ms.

(Speedups are vs each platform's own v0.1 path, min–max across batch 1–16384.)

Per-arch autotuned configs (`_CUDA_TUNED` keyed by `gcnArchName`/device name);
AMD's wavefront-64 wants the opposite shape from NVIDIA warps.

**Two honest negatives (detailed in the receipts + README):**
1. The *plain-BP* megakernel **loses to the two-kernel path at large batch**
   (no early-exit lever) — it is a single-shot *latency* tool (batch-1 ~1.7× the
   two-kernel BP path); the two-kernel path stays the BP throughput default.
2. The cross-vendor latency gap **widened** under the megakernel: H200 leads
   MI300X **2.47× at batch-1** (vs ~9% under v0.1's two-kernel path), 1.25–1.33×
   batched. Correctness is identical; the pitch is portability + performance on
   both, not parity.

**Opt-in, not yet default:** the megakernel ships as
`tridec.backends.megakernel.{RelayBpMegaTriton, BpMegaTriton}`; `from_dem`
auto-dispatch is deferred to **v0.2.1** (#5) — the standalone classes are gated,
but the public-API dispatch path needs its own GPU gating before the default
flips, and that discipline is not bent for the tag.

Receipts: `bench/receipts/megakernel_{h200,mi300x,metal}*`. Metal is BLOCK=32
pending an upstream triton-metal barrier fix (confirmed on its dev branch).

## 0.1.0 — 2026-06-10

First release. Open, vendor-portable Triton min-sum BP and Relay-BP decoders
for stim `DetectorErrorModel`s (or raw parity-check matrices), with numpy and
torch CPU references, `sinter.collect` integration, and a matched-protocol
validation layer — validated on **NVIDIA H200 (CUDA 12.4, triton 3.0)** and
**AMD MI300X (ROCm 7.0, triton 3.4)**, with **experimental Apple-silicon
(Metal) support** via triton-metal.

**No-regression result: 31/32 exact + 1 documented upstream nondeterminism.**
The full source-grid reproduction (8 cells × 4 ldpc-family decoders at exact
shots/seeds, receipt environment) reproduced 31 of 32 logical-failure counts
**exactly**, plus all 8 DEM hashes. The single non-exact cell is `ldpc`'s
`BpLsdDecoder`, which is run-to-run nondeterministic on one borderline shot
(fails across identical repeats: 880/880/879/880/880 — the pinned value was
itself one draw of that coin; probe receipt:
`bench/receipts/full_grid_noregression.json`).

**Three reproducibility findings** (measured during validation, detailed in
`docs/benchmark.md` §5.1) — they affect anyone doing fair cross-decoder
benchmarking:

1. **stim's circuit→DEM computation is platform-dependent at the ~ulp level**
   (and its float text rendering differs) — sha256-of-DEM-text gates are
   platform-local. Pin DEM *artifacts*, not generating circuits.
2. **stim's seeded detector sampler is platform-dependent** — the same seed
   yields different samples on darwin/arm64 vs linux/x86_64; exact
   cross-platform count reproduction is impossible by construction.
3. **`ldpc.BpLsdDecoder` is run-to-run nondeterministic** in a fixed
   environment on borderline shots (above).

**Known limitations** (stated in full in `docs/benchmark.md` §5): plain BP
loses to matching on surface codes (4–25×) and Relay-BP trails MWPM 3.8–4.3×
at surface d=5; throughput claims are batched, not real-time; fp32 GPU
messages produce rare near-tie flips vs fp64 references; the CUDA-Q
comparison receipt is config-asymmetric (their FirstConv stop is not
tunable to ours).

**Changes since `0.1.0a1`:** sinter `CompiledDecoder` adapter (+`[sinter]`
extra); surface-code CPU receipts (50k shots/cell vs PyMatching); official
Relay-BP surface receipts on MI300X; `backend="metal"` (experimental,
fp32-enforced) with auto-detection; platform-aware gate architecture
(canonical `.dem` fixtures + exact/statistical tiers); MI300X packaged-API
validation receipts; CI (ubuntu + macos-arm64 receipt-env lane where the
strict gates bind); `__version__` single-sourced from package metadata;
README/benchmark documentation of all of the above.
