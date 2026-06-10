# Benchmark report — tridec v0.1.0-dev

Status: skeleton with the carried validation receipts wired in. Sections
marked **TODO** are planned for the first tagged release; every number below
is traceable to a JSON receipt in [`bench/receipts/`](../bench/receipts/)
(provenance in that directory's README).

---

## 1. Claim

The Triton min-sum BP and Relay-BP decoders in this package are
**vendor-portable** (the same kernel source runs on NVIDIA CUDA and AMD ROCm)
and **LER-faithful** (logical error rates statistically indistinguishable from
the standard CPU references: `ldpc`'s min-sum BP family and IBM's `relay-bp`
Rust decoder) on circuit-level quantum LDPC decoding problems, while
delivering GPU-batch throughput.

No claim of being fastest. No claim of being first. The claim is: open,
portable across the two major GPU vendors, and validated against the
references the field already trusts.

As of June 2026, NVIDIA's nv-qldpc-decoder is closed-source; this library
provides an open, vendor-portable alternative validated against the standard
CPU references.

## 2. Method

**Workload.** Circuit-level memory experiments on the [[72,12,6]]
bivariate-bicycle (BB) code, 6 syndrome rounds, SI1000 superconducting noise,
p ∈ {0.001, 0.002, 0.003, 0.005} × basis ∈ {X, Z}. The canonical circuits ship
as `.stim` fixtures in `tests/fixtures/bb72/`. DEM structure at the reference
cell (p=0.003, Z): 252 detectors, 1584 error mechanisms, 4536 Tanner edges,
check degree 13/20, bit degree 2/3. A rotated surface-code fixture
(d=3, r=3) covers code-agnosticism.

**Matched protocol.** Every comparison decodes ONE shared DEM
(`decompose_errors=False`) and ONE sampled shot set (fixed seed) — the
`run_matched` harness in `tridec.validation` enforces this with
fail-fast gates: G1 (every decoder's DEM hashes to the same sha256) and G2
(every decoder pre-declares a deterministic tie-break). Per-cell DEM sha256s
are pinned in `zoo_grid.json` and re-verified by this package's test suite.

**Statistics.** Wilson 95% CIs on failure counts; gap-to-MLE as a per-shot
PAIRED bootstrap of decoder-vs-anchor failure indicators on the same shots
(10,000 resamples), never an aggregate ratio of marginal counts; Holm/BH for
grid-wide multiplicity.

**Configurations.** BP family: normalized min-sum, ms_scaling_factor=0.625,
max_iter=30, parallel schedule. Relay-BP: gamma0=0.1, pre_iter=80,
num_sets=60, set_max_iter=60, gamma ∈ (-0.24, 0.66), stop_nconv=5
(lowest-weight valid solution) — the `relay-bp` 0.2.2 oracle configuration.

## 3. Results

### 3.1 Dual-vendor LER identity (Relay-BP Triton kernel)

The binding portability result: the SAME kernel source, validated on both
vendors against the `relay-bp` Rust oracle (F64) on 2000 canonical shots
(p=0.003, Z, seed 0). Receipts: `bench_relay_triton.json` (H200),
`bench_relay_mi300x.json` (MI300X).

| | NVIDIA H200 (CUDA 12.4, triton 3.0) | AMD MI300X (ROCm 7.0, triton 3.4) |
|---|---|---|
| Pre-leg posterior max-diff vs oracle | 1.78e-15 | 1.78e-15 |
| Memory-term per-bit agreement | 99.96% | 99.96% |
| Oracle logical errors (n=2000) | 31 | 31 |
| Triton logical errors (n=2000) | 38 | 38 |
| Wilson CIs overlap | yes | yes |

The oracle-vs-Triton gap (31 vs 38) is fp/ensemble noise (the gamma-draw RNG
differs from the Rust oracle's by construction); the two are statistically
indistinguishable, and identical across vendors.

Throughput (Relay-BP `decode_batch`, real canonical syndromes):

| Device | dtype | batch | per-syndrome | shots/s |
|---|---|---|---|---|
| H200 | fp32 | 2000 | 483 µs | 2,070 |
| H200 | fp64 | 2000 | 679 µs | 1,472 |
| MI300X | fp32 | 2000 | 526 µs | 1,903 |
| MI300X | fp32 | 8192 | 278 µs | 3,603 |
| MI300X | fp64 | 8192 | 403 µs | 2,479 |

### 3.2 Triton min-sum BP vs the CPU/torch references (H200)

Receipt: `bench_triton_results.json`. 2000 canonical shots, max_iter=30.

- One-iteration hard-decision agreement vs the fp64 numpy reference: **100%**.
- Full-decode logical failures: numpy **156**, torch **156**, Triton **156**
  (per-shot exact prediction match 99.35%; fp32 near-tie flips cancel in LER).
- `decode_batch` throughput: 229k shots/s at batch 256 rising to **1.26M
  shots/s at batch 16384** (12×–55× over the batched torch fp64 baseline on
  the same device).

### 3.3 Gap to MLE (how good is the algorithm itself?)

From the matched grid (`zoo_grid.json`), reference cell p=0.003, both bases,
16,667 shots each, Tesseract (A*, det_beam=64) as the MLE-proxy anchor.
Paired-bootstrap ratio to anchor [95% CI]:

| Decoder | p=0.003 X | p=0.003 Z |
|---|---|---|
| Tesseract (anchor) | 1.000 | 1.000 |
| Relay-BP | 1.050 [0.963, 1.146] | 1.004 [0.920, 1.090] |
| BP-OSD-10 | 1.058 [0.981, 1.142] | 1.035 [0.957, 1.118] |
| BP-LSD | 1.100 [1.004, 1.203] | 1.053 [0.963, 1.148] |
| BP-OSD-0 | 1.131 [1.032, 1.239] | 1.099 [1.004, 1.199] |
| plain BP (min-sum) | 5.323 [4.777, 5.991] | 4.756 [4.278, 5.313] |

Plain min-sum BP alone is ~5× off MLE on this code — that is exactly why
Relay-BP is the headline algorithm and BP is positioned as the portable
baseline/building block, not a recommendation.

### 3.4 Surface-code memory — second code family (CPU receipts)

Receipt: `surface_cpu.json` (generated by `bench/surface_cpu.py`; 50,000 shots
per cell, seed 20260610, darwin/arm64, stim 1.15.0, pymatching 2.3.1). Rotated
surface-code memory-Z, stim-generated circuits (all four noise knobs = p).
tridec BP decodes the raw DEM (torch CPU backend, fp64, validated defaults
max_iter=30, ms=0.625; numpy reference cross-checked 100% prediction-identical
on 2000-shot subsets per cell); PyMatching decodes the decomposed DEM on the
same shots.

| Cell | tridec BP fails / LER | PyMatching fails / LER | BP/MWPM ratio |
|---|---|---|---|
| d=3, r=3, p=0.003 | 1966 / 0.0393 | 321 / 0.0064 | 6.1× |
| d=3, r=3, p=0.005 | 3725 / 0.0745 | 841 / 0.0168 | 4.4× |
| d=5, r=5, p=0.003 | 4340 / 0.0868 | 173 / 0.0035 | 25.1× |
| d=5, r=5, p=0.005 | 7964 / 0.1593 | 699 / 0.0140 | 11.4× |

**The honest reading, plainly:** plain min-sum BP *without post-processing*
loses badly to matching on surface codes — this is the known landscape, not a
surprise. Degenerate weight-4 loops split BP's beliefs; the BP LER actually
*increases* from d=3 to d=5 at fixed p (no threshold behavior), while
matching improves exactly as it should. These receipts exist to demonstrate
**code-agnostic operation with measured numbers** (the same `from_dem` path
decodes BB codes and surface codes unmodified), not to compete with matching
on matching's home turf. On surface codes, use a matching decoder or a
post-processed BP variant.

Throughput context (same cells): torch-CPU batched BP decoded 50k shots in
3.7–34.7 s (1.4k–13.5k shots/s); PyMatching took ≤0.1 s. **TODO (Phase 3,
GPU session):** official Relay-BP surface-code receipts (triton-only; needs
the CUDA/ROCm box) and Triton-BP surface throughput.

**PRELIMINARY (experimental Metal, fp32, not official):** while the official
Relay-BP surface receipts wait for the GPU session, the experimental
triton-metal path produced a preliminary sample
(`surface_relay_metal_preliminary.json`; 2000 shots/cell, same protocol/seed,
fp32, one M4 Max):

| Cell (N=2000) | Relay-BP (metal fp32) | plain BP | PyMatching |
|---|---|---|---|
| d=3, p=0.003 | 12 | 90 | 7 |
| d=3, p=0.005 | 37 | 164 | 33 |
| d=5, p=0.003 | 33 | 154 | 4 |
| d=5, p=0.005 | 98 | 332 | 25 |

Preliminary reading: Relay-BP's disordered memory recovers most of the
plain-BP-vs-matching gap at d=3 (overlapping Wilson CIs vs MWPM), but a real
gap remains at d=5 in this small fp32 sample (33 vs 4 at p=0.003). Treat as
directional only until the CUDA/ROCm re-run with proper shot counts.

## 4. Comparisons

### 4.1 NVIDIA CUDA-Q QEC 0.6 GPU Relay-BP (H200)

Receipt: `bench_cudaq_compare.json`. Same 2000-shot canonical workload, same
H200. **Config asymmetry, stated up front:** the CUDA-Q decoder was run with
its own configuration surface — its relay selection is FirstConv-style, while
ours matches the `relay-bp` oracle's lowest-weight-over-nconv=5 selection.
The two columns are therefore *different operating points*, not a
parameter-matched kernel duel.

| | CUDA-Q QEC 0.6 | tridec Triton (fp32) | relay-bp Rust (CPU) |
|---|---|---|---|
| LER (n=2000) | 0.031 (62 fails; 0.0295 w/ OSD) | 0.016 (32 fails) | 0.0155 (31 fails) |
| Throughput @ batch 2000 | 113,585 shots/s | 2,112 shots/s | 606 shots/s |
| Best measured throughput | 503,381 shots/s @ 8192 | 8,659 shots/s @ 8192 | — |

Honest reading: CUDA-Q is ~54× faster at matched batch size and ~2× worse in
LER at the configurations measured; the LER gap is a selection-rule
asymmetry, not a kernel-speed artifact, and the throughput gap is real.
CUDA-Q is CUDA-only and partially closed; this package is open and runs the
same source on AMD. **TODO:** parameter-matched re-run if/when CUDA-Q exposes
the equivalent stopping/selection configuration.

### 4.2 CPU latency context

Receipt: `latency_results.json` — bootstrap-CI'd per-syndrome latency for the
CPU zoo (ldpc BP/BP-OSD/BP-LSD, relay-bp, Tesseract) and the GPU paths at
batch 1–16384 on the same H200 host. Headline: GPU batching trades
single-shot latency for throughput; at batch 1 the Triton path is NOT faster
than the Rust CPU decoder. **TODO:** distill the full table here.

## 5. Limitations

- **The matched cross-decoder grid covers one code family.** The zoo/grid
  receipts are the [[72,12,6]] BB code at 6 rounds under SI1000. Surface-code
  memory now carries its own CPU receipts (§3.4: BP vs PyMatching, d=3/d=5,
  two p values, 50k shots each) — and they show plain BP losing to matching
  there, as expected. Relay-BP surface receipts are TODO (Phase 3, GPU).
- **Relay-BP has no in-package CPU implementation** (triton backend only);
  the CPU reference is the external `relay-bp` Rust package.
- **fp32 GPU messages are not bit-identical to fp64 references**; the gates
  are ≥99.5% hard-decision agreement and LER equivalence, and near-tie flips
  are documented in the receipts.
- **Throughput, not real-time latency.** No claim of meeting a per-round
  decode deadline at batch 1; see §4.2.
- **CUDA-Q comparison is config-asymmetric** (§4.1).
- **Exact-count reproduction is environment-pinned**: the no-regression test
  reproduces grid counts exactly under stim 1.15.0 + ldpc 2.4.1 on the receipt
  platform (darwin/arm64) and falls back to a statistical-equivalence tier
  elsewhere — see §5.1.

### 5.1 Reproducibility & platform notes (measured, 2026-06-09)

Two cross-platform facts we measured while validating this package on
darwin/arm64 (Apple M-series) and linux/x86_64 (H200 host), same stim 1.15.0,
same `.stim` circuit file:

1. **stim's circuit→DEM computation is platform-dependent at the ulp level.**
   1512 of 1584 mechanism priors differed between the two platforms, max
   relative difference 5.4e-16 (x86 long-double intermediates vs arm64
   doubles). The float *text* rendering differs too (x86 prints more digits).
   Consequence: a sha256 over the DEM's text — a common "byte-identical DEM"
   gate, including the one this benchmark inherited from its source
   experiments — is **platform-local**. It guarantees every decoder in one
   run consumes the identical DEM; it does not transfer across platforms.
2. **stim's seeded detector sampler is platform-dependent.** The same seed on
   the same circuit produced entirely different detector samples on the two
   platforms, so exact failure-count reproduction across platforms is not
   merely fragile — it is impossible by construction.

Design consequence — **pin artifacts, not generators**: the canonical
fixtures of this package are the `.dem` files themselves (file bytes are
platform-independent and sha256-pinned in `tests/fixtures/bb72/
dem_manifest.json`); the `.stim` circuits are provenance, checked by a
structure-identical + priors-within-1e-12 regeneration test on every
platform, and by strict text-hash identity against the source-grid pins on
the receipt platform only. Matched-decoder comparisons that must be
reproducible across machines should distribute the DEM artifact, not the
generating circuit.

## 6. Reproduction

```bash
git clone <repo> && cd tridec
python -m venv .venv && . .venv/bin/activate
pip install -e ".[decoders,dev]" "stim==1.15.0" "ldpc==2.4.1"
pytest tests/                      # CPU: all gates; GPU tests skip cleanly
```

- DEM-hash gate: `pytest tests/test_dem_hash_gate.py` — 8/8 fixture DEMs must
  hash to the receipts' pinned sha256s.
- No-regression cell: `pytest tests/test_no_regression_cell.py` — re-decodes
  the full 16,667-shot p=0.003/Z cell and requires the ldpc adapters' failure
  counts to equal `zoo_grid.json`'s exactly (293 / 1346).
- GPU gates (on a CUDA or ROCm box with the `[gpu]` extra):
  `pytest tests/test_bp_triton.py tests/test_relay_triton.py`.
- **TODO:** a `bench/` driver script that regenerates the throughput tables
  from scratch on the current machine.
