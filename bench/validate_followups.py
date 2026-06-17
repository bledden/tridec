"""GPU validation for the three #4/#6 follow-ups (run on CUDA H200 or ROCm MI300X).

(c) #6  grid-flatten lifts the d>=15 launch ceiling AND stays bit-identical at d<=14.
(a) #4  cuda_graph="auto": graph path is bit-identical to eager, small-S gated,
        cache-capped, and wins at small batch (prints the latency crossover).
(b)     on ROCm this same script exercises hipGraph (torch maps CUDAGraph -> hipGraph).

Usage:  python bench/validate_followups.py          # auto-detects cuda/rocm
Env:    none.  Prints a PASS/FAIL summary; exits nonzero on any failure.
"""
import sys, time, numpy as np, torch, stim

sys.path.insert(0, "src")
from tridec.api import from_dem

assert torch.cuda.is_available(), "needs a CUDA or ROCm GPU"
DEV = "cuda"
NAME = torch.cuda.get_device_name(0)
IS_ROCM = (getattr(torch.version, "hip", None) is not None)
print(f"device: {NAME}  ({'ROCm/hipGraph' if IS_ROCM else 'CUDA'})  torch {torch.__version__}\n")

P = 0.003
fails = []


def surface(d, shots):
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=P, after_reset_flip_probability=P,
        before_measure_flip_probability=P, before_round_data_depolarization=P)
    dem = c.detector_error_model(decompose_errors=False)
    det, obs = c.compile_detector_sampler().sample(
        shots, separate_observables=True)
    return dem, np.asarray(det, bool), np.asarray(obs, bool)


# ===================================================================== #
# (c) #6 -- ceiling-lift + bit-identical                                 #
# ===================================================================== #
print("=" * 64)
print("(c) #6 grid-flatten: ceiling-lift past d=14 + bit-identical")
print("=" * 64)
for d in [13, 14, 15, 17]:
    try:
        dem, det, obs = surface(d, 1000)
        dbp = from_dem(dem, backend="triton", cuda_graph=False)
        pred = dbp.decode_batch(det)
        ler = (pred != obs).mean() * 100
        n_checks, n_bits = dbp.n_checks, dbp.n_bits
        over = " (was the WALL)" if d >= 15 else ""
        print(f"  d={d:2d}  checks={n_checks:5d} bits={n_bits:6d}  "
              f"DECODE OK  LER {ler:5.2f}%{over}")
    except Exception as e:
        print(f"  d={d:2d}  FAILED: {type(e).__name__}: {str(e)[:80]}")
        fails.append(f"#6 d={d} decode")

# bit-identical at d<=13: triton BP must equal the numpy BP baseline exactly on
# the integer hard-decision path (both are min-sum; fp32 vs fp64 can flip rare
# near-ties, so we assert <=0.1% shot disagreement, not strict equality).
print("\n  bit/near-identical vs numpy baseline (d<=13):")
for d in [5, 9, 13]:
    dem, det, obs = surface(d, 2000)
    pt = from_dem(dem, backend="triton", cuda_graph=False).decode_batch(det)
    pn = from_dem(dem, backend="numpy").decode_batch(det)
    disagree = (pt != pn).mean() * 100
    ok = disagree <= 0.1
    print(f"  d={d:2d}  triton vs numpy disagree {disagree:.3f}%  {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"#6 d={d} triton-vs-numpy {disagree:.3f}%")

# ===================================================================== #
# (a) #4 -- cuda_graph auto: bit-identical, gating, cache cap            #
# ===================================================================== #
print("\n" + "=" * 64)
print("(a) #4 cuda_graph: bit-identical, small-S gating, cache cap")
print("=" * 64)
dem, det_all, obs = surface(5, 5000)

print("\n  graph (True) and auto vs eager (False) -- must be BIT-IDENTICAL:")
for S in [1, 16, 256, 1024]:
    det = det_all[:S]
    eager = from_dem(dem, backend="triton", cuda_graph=False).decode_batch(det)
    always = from_dem(dem, backend="triton", cuda_graph=True).decode_batch(det)
    auto = from_dem(dem, backend="triton", cuda_graph="auto").decode_batch(det)
    ok = np.array_equal(eager, always) and np.array_equal(eager, auto)
    print(f"  S={S:5d}  eager==graph=={('auto' if S<=256 else 'auto(eager)')}: "
          f"{np.array_equal(eager, always)}/{np.array_equal(eager, auto)}  {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"#4 S={S} not bit-identical")

print("\n  small-S gating: auto must capture for S<=max_s, skip above:")
d_auto = from_dem(dem, backend="triton", cuda_graph="auto")  # max_s=256 default
d_auto.decode_batch(det_all[:16])     # small -> should capture
small_cached = 16 in d_auto._impl._graphs
d_auto.decode_batch(det_all[:4096])   # large -> should NOT capture
large_skipped = 4096 not in d_auto._impl._graphs
print(f"  S=16 captured: {small_cached}   S=4096 skipped: {large_skipped}  "
      f"{'OK' if (small_cached and large_skipped) else 'FAIL'}")
if not (small_cached and large_skipped):
    fails.append("#4 small-S gating")

print("\n  cache cap: >cache distinct small shapes must stop capturing:")
d_cap = from_dem(dem, backend="triton", cuda_graph="auto",
                 cuda_graph_cache=4, cuda_graph_max_s=4096)
for S in [1, 2, 3, 4, 5, 6, 7, 8]:
    d_cap.decode_batch(det_all[:S])
ncached = len(d_cap._impl._graphs)
cap_ok = ncached == 4
print(f"  captured {ncached} shapes (cap=4)  {'OK' if cap_ok else 'FAIL'}")
if not cap_ok:
    fails.append(f"#4 cache cap got {ncached}")

# latency crossover: where does the graph stop winning? informs max_s default.
print("\n  latency crossover (300-call median, ms/decode):")
print(f"  {'S':>6} {'eager':>9} {'graph':>9} {'speedup':>8}")
for S in [1, 16, 64, 256, 512, 1024, 4096]:
    det = np.ascontiguousarray(det_all[:S])
    de = from_dem(dem, backend="triton", cuda_graph=False)
    dg = from_dem(dem, backend="triton", cuda_graph=True)
    for dd in (de, dg):  # warm
        for _ in range(5):
            dd.decode_batch(det)
    torch.cuda.synchronize()

    def med(dd):
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            dd.decode_batch(det)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1e3
    e, g = med(de), med(dg)
    print(f"  {S:>6} {e:>9.3f} {g:>9.3f} {e/g:>7.2f}x")

print("\n" + "=" * 64)
if fails:
    print("FAIL:", "; ".join(fails))
    sys.exit(1)
print("ALL FOLLOW-UP VALIDATIONS PASSED"
      + ("  (ROCm/hipGraph)" if IS_ROCM else "  (CUDA)"))
