# Metal spike: tridec Triton kernels on Apple Silicon via triton-msl

**Date:** 2026-06-09
**Verdict:** PASS — both `BpTriton` and `RelayBpTriton` (fp32) pass all existing
correctness gates on the Mac GPU, with **zero source changes to tridec**.

## Environment

- Machine: Apple M4 Max, macOS (Darwin 24.6.0)
- triton-msl: `<triton-msl checkout>` @ `4c42e9685db5cf58ce856d2816844066a27feb6a` (clean tree, editable install)
- triton: 3.7.0 (`3.7.0+git4da2e268`, local source build at `<triton checkout>`, editable)
- torch 2.9.1 / stim 1.15.0 / relay-bp 0.2.2 / Python 3.14.4 (Homebrew)
- tridec: working tree, editable, no GPU extra (triton came via triton-msl's env)

## Venv recipe

```bash
# triton 3.7.0 (source build) + triton-msl + torch already live in the
# Homebrew python3 site; inherit them instead of rebuilding triton from source.
/opt/homebrew/bin/python3 -m venv --system-site-packages /tmp/pq-metal-venv
/tmp/pq-metal-venv/bin/pip install -e <repo> --no-deps
/tmp/pq-metal-venv/bin/pip install "relay-bp[stim]>=0.2.2"   # relay oracle only
```

## Device plumbing

None required. triton-msl's documented usage pattern is **CPU torch tensors**
(zero-copy via UMA / `newBufferWithBytesNoCopy`; ARCHITECTURE.md says explicitly
"Not mps"). Both backends already parameterize `device=` on every entry point,
so the gates were run with `device="cpu"`; for relay, `dtype="float32"` (an
existing constructor parameter — Metal has no fp64). No working-tree edits;
`git status` of tridec is clean.

## Results (gate logic mirrored from tests/test_bp_triton.py + test_relay_triton.py, 2000 shots, seed 0)

### BpTriton fp32 (30 iter, ms=0.625)

| DEM | 1-iter hard agree | 30-iter per-bit agree | LER numpy | LER metal | decode_batch 2000 shots: numpy / metal |
|---|---|---|---|---|---|
| surface_d3_r3_p0.003 | 1.00000 | 1.00000 | 76/2000 (0.0380) | 76/2000 (0.0380) | 1586 ms / **28 ms** |
| bb72_r6_p0.003_Z | 1.00000 | 0.99998 | 167/2000 (0.0835) | 168/2000 (0.0840) | 6243 ms / **167 ms** |

All gates (>=0.995 agreement, LER within max(3, 0.5%)) pass; agreement is
essentially bit-identical, LER differs by at most 1 shot in 2000.

### RelayBpTriton fp32 (gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60, stop_nconv=5)

- pre-leg posterior maxdiff vs `MinSumBPDecoderF64`: **1.41e-06** (gate <1e-3) — PASS
- memory-term per-bit agreement (30 iter, 256 shots): **0.99981** (gate >=0.99) — PASS
- full relay LER vs `RelayDecoderF64` Rust oracle: oracle 31/2000 (0.0155) vs
  metal 39/2000 (0.0195), per-shot agreement 0.9930, |d|=8 <= 20 gate — PASS
  (gamma RNGs differ by construction + fp32 near-tie flips, same caveat as the
  CUDA/ROCm fp32 receipts)
- timing: relay decode_batch(2000) = **31.1 s** on Metal vs 1.26 s for the Rust
  CPU oracle — relay's per-iteration host loop (~7k small kernel launches +
  convergence checks) is launch-overhead dominated on Metal. Correct, not fast.

### Smoke

triton-msl README vector-add on CPU tensors: max error 0.00e+00.

## Honest read

The kernels compile and run on Metal unmodified — both the bounded-degree
unrolled `tl.static_range` loops (MAXDEG_C up to 48 on the surface DEM), the
2-D grids, and all `tl.*` primitives used (load/store with masks, where,
minimum, abs, zeros/full, int32 xor) are inside triton-msl's supported set.
Batched BP is genuinely fast (2000-shot decode in 28–167 ms, 37–56x the
per-shot numpy baseline on the same machine). "Experimental Metal support" is
roughly a half-day of remaining work: a `device="cpu"`-on-macOS selection rule
(or `backend="metal"` alias) in the API/test skip logic, a docs note that relay
requires `dtype="float32"`, and CI/markers — not a project. Relay's launch
overhead on Metal is the only performance caveat worth documenting.

*Spike artifacts: /tmp/smoke_add.py, /tmp/spike_bp.py, /tmp/spike_relay.py.
Nothing committed; this file is the only repo addition (uncommitted).*
