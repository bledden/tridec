# Changelog

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
