# #4 — CUDA-graph fast path for the launch-bound small-batch BP regime

Opt-in `cuda_graph=True` (default off). Captures the fixed `n_iter` min-sum BP
kernel loop (`_check_update_kernel` + `_bit_update_kernel`) plus the `Lo` observable
projection into a CUDA graph **per batch shape**, and replays it — eliminating the
per-iteration kernel-launch overhead that dominates small batches. The syndrome is
the only per-decode input (staged through a pinned host buffer into a static
device tensor). Cache keyed by batch size `S`; on any capture failure the decoder
silently falls back to the eager path (so default behavior is never at risk).

## Correctness — bit-identical to the eager path
H200 (CUDA), surface d=5 (rotated_memory_z, p=0.003), `from_dem(..., algorithm="bp")`:
`cuda_graph=True` vs default decode_batch, **bit-identical** at every batch size and
on replay reuse (different shots through the same captured graph):

| batch S | graph == eager |
|---|---|
| 1 | ✓ |
| 16 | ✓ |
| 256 | ✓ |
| 1024 | ✓ |
| batch-1, different shot (replay) | ✓ |

## Latency — the small-batch win
batch-1 decode, 300-call median, H200:

| path | ms/decode | speedup |
|---|---|---|
| eager (`decode_batch`) | 1.026 | 1.0× |
| **cuda_graph** | **0.646** | **1.59×** |

(Standalone prototype showed 1.72×; integrated through the public API is 1.59×.)
This is the launch-bound regime the profile in tridec-serve identified (batch-1 is
~100% fixed/launch overhead); the graph collapses the per-launch cost. Largest win
at small/repeated shapes — exactly the **serving** use-case (continuous batching
reuses a small set of bucket shapes, so the per-shape capture amortizes immediately).

## Scope / notes
- Opt-in (default off) — validated path first, à la the v0.2.0 megakernel; flipping
  the default (or auto-enabling for small `S`) is a follow-up.
- Applies to the min-sum BP two-kernel path (fixed iteration count → capturable).
  Relay-BP's per-shot early-exit is data-dependent → not graph-captured here.
- Env: H200, torch 2.4.1+cu124, triton 3.0.0. Closes #4.
