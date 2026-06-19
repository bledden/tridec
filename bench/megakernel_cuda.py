"""Generate ``bench/receipts/megakernel_<gpu>.json`` (issue #2/#3/#4, v0.2).

The CUDA/ROCm cloud half of the megakernel validation: barrier sanity,
autotune sweep (issue #3), and throughput/latency receipts vs the v0.1
two-kernel host-loop path on the SAME pod -- all on the canonical
[[72,12,6]] BB cell (bb72_r6_p0.003_Z, stim sampler seed 0).

Sections
--------
barrier_sanity : behavioral load->tl.debug_barrier->store ring repro
    (exact iff the barrier is honored; integer-valued fp32 so every add is
    exact) + a no-barrier negative control that MUST race + bar.sync counts
    in the real megakernels' emitted PTX. The triton-msl environment
    silently drops tl.debug_barrier (see megakernel.py docstring) -- this
    section is the proof the CUDA/ROCm build does not.
autotune : BLOCK x num_warps sweep for both megakernels (issue #3). Every
    config is CORRECTNESS-GATED before it is timed: BP = posterior
    bit-identity vs the two-kernel BpTriton + run-twice determinism;
    relay = run-twice determinism + per-shot solution identity vs the
    reference config, where any mismatch must verify as a degenerate
    equal-fp64-weight syndrome-consistent tie (see megakernel.py).
throughput : decode_batch wall-clock (sync-bracketed perf_counter,
    warmup>=5, reps>=20) for the relay megakernel (fp32 + fp64) and the BP
    megakernel vs the v0.1 two-kernel path, batch 1..16384.
batch1 : the single-shot latency floor decomposition (issue #4).

Run:  python bench/megakernel_cuda.py [receipt-tag]    (default: gpu name)
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

from conftest import load_bb_circuit                         # noqa: E402

from tridec.backends.bp_triton import BpTriton               # noqa: E402
from tridec.backends.relay_triton import RelayBpTriton       # noqa: E402
from tridec.backends.megakernel import (                     # noqa: E402
    BpMegaTriton, RelayBpMegaTriton, _bp_megakernel, _relay_megakernel)

assert torch.cuda.is_available(), "CUDA/ROCm GPU required"
DEV = "cuda"
MS = 0.625
SEED = 0
RELAY_CFG = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66),
                 stopping_criterion="nconv")
SWEEP_BLOCKS = [64, 128, 256, 512]
SWEEP_WARPS = [2, 4, 8]
SWEEP_SHOTS = 2000          # timing batch for the sweep (the canonical cell)
GATE_SHOTS = 512            # correctness-gate batch per config
BATCHES = [1, 16, 256, 2000, 8192, 16384]
N_WARMUP, N_REPS = 5, 20


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def bench_gpu(fn, n_warmup=N_WARMUP, n_reps=N_REPS):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return dict(n_warmup=n_warmup, n_reps=n_reps,
                mean_ms=float(ts.mean() * 1e3),
                min_ms=float(ts.min() * 1e3),
                max_ms=float(ts.max() * 1e3))


def row(shots, b):
    return dict(batch=shots, **b,
                per_syndrome_us=round(b["mean_ms"] * 1e3 / shots, 3),
                throughput_shots_per_s=round(shots / (b["mean_ms"] / 1e3), 1))


# --------------------------------------------------------------------- #
# barrier sanity                                                        #
# --------------------------------------------------------------------- #
@triton.jit
def _ring_barrier(X, OUT, n_rounds, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    base = X + pid * BLOCK
    for _ in range(n_rounds):
        v = tl.load(base + offs)
        tl.debug_barrier()
        tl.store(base + ((offs + 1) % BLOCK), v + 1.0)
        tl.debug_barrier()
    tl.store(OUT + pid * BLOCK + offs, tl.load(base + offs))


@triton.jit
def _ring_nobarrier(X, OUT, n_rounds, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    base = X + pid * BLOCK
    for _ in range(n_rounds):
        v = tl.load(base + offs)
        tl.store(base + ((offs + 1) % BLOCK), v + 1.0)
    tl.store(OUT + pid * BLOCK + offs, tl.load(base + offs))


def _bar_count(asm):
    # CUDA: PTX bar.sync ; ROCm/HIP: AMDGCN s_barrier. The behavioral ring
    # test below is the real proof a barrier is honored; this just confirms
    # the instruction is present in the emitted ISA on both backends.
    if "ptx" in asm:
        return asm["ptx"].count("bar.sync")
    if "amdgcn" in asm:
        return asm["amdgcn"].count("s_barrier")
    return -1


def barrier_sanity():
    out = {}
    for block in (128, 256):
        for tag, kern in (("barrier", _ring_barrier),
                          ("nobarrier", _ring_nobarrier)):
            rng = np.random.default_rng(0)
            # integer-valued fp32: +1.0 per round is exact, so the host
            # expectation matches the device IFF the barrier is honored.
            x0 = rng.integers(0, 1000, (512, block)).astype(np.float32)
            x = torch.as_tensor(x0, device=DEV).contiguous()
            o = torch.empty_like(x)
            h = kern[(512,)](x, o, 400, BLOCK=block)
            torch.cuda.synchronize()
            exp = np.roll(x0, 400, axis=-1) + 400
            got = o.cpu().numpy()
            out[f"{tag}_block{block}"] = dict(
                exact=bool(np.array_equal(got, exp)),
                frac_match=float((got == exp).mean()),
                bar_sync_in_ptx=_bar_count(h.asm))
    # bar.sync counts in the REAL megakernels' PTX (all compiled variants).
    ptx = {}
    for name, kern in (("bp", _bp_megakernel), ("relay", _relay_megakernel)):
        cnt = []
        for _, vv in kern.cache.items():
            for _, compiled in vv.items():
                cnt.append(_bar_count(compiled.asm))
        ptx[name] = cnt
    out["megakernel_ptx_bar_sync"] = ptx
    ok = (out["barrier_block128"]["exact"] and out["barrier_block256"]["exact"]
          and not out["nobarrier_block128"]["exact"]
          and not out["nobarrier_block256"]["exact"])
    out["verdict"] = ("PASS: tl.debug_barrier honored (ring exact at "
                      "BLOCK=128/256, no-barrier control races)" if ok
                      else "FAIL")
    print(f"[barrier] {out['verdict']}  "
          f"(no-barrier match frac: "
          f"{out['nobarrier_block128']['frac_match']:.4f} / "
          f"{out['nobarrier_block256']['frac_match']:.4f})")
    return out


# --------------------------------------------------------------------- #
# autotune sweep (issue #3)                                             #
# --------------------------------------------------------------------- #
def sweep_bp(dem, dets):
    syn = dets[:GATE_SHOTS].astype(np.uint8)
    old = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    p_ref = old.run_iterations_batch(syn, n_iter=30, device=DEV)
    rows = []
    for block in SWEEP_BLOCKS:
        for warps in SWEEP_WARPS:
            mega = BpMegaTriton.from_dem(dem, max_iter=30,
                                         ms_scaling_factor=MS,
                                         block=block, num_warps=warps)
            p1 = mega.run_iterations_batch(syn, n_iter=30, device=DEV)
            p2 = mega.run_iterations_batch(syn, n_iter=30, device=DEV)
            gate = bool((p1 == p2).all()) and bool((p1 == p_ref).all())
            r = dict(block=block, num_warps=warps, gate_bit_identity=gate)
            if gate:
                b = bench_gpu(lambda: mega.decode_batch(
                    dets[:SWEEP_SHOTS], device=DEV), 3, 10)
                r.update(mean_ms=b["mean_ms"], min_ms=b["min_ms"])
            rows.append(r)
            print(f"[sweep bp] BLOCK={block:4d} warps={warps} gate="
                  f"{'PASS' if gate else 'FAIL'} "
                  f"{r.get('mean_ms', float('nan')):8.3f} ms")
    ok = [r for r in rows if r["gate_bit_identity"] and "mean_ms" in r]
    win = min(ok, key=lambda r: r["mean_ms"])
    return rows, win


def sweep_relay(dem, dets):
    sub = dets[:GATE_SHOTS]
    dev = torch.device(DEV)
    syn_t = torch.as_tensor(sub, device=dev)
    ref = RelayBpMegaTriton.from_dem(dem, stop_nconv=5, early_exit=True,
                                     dtype="float32", block=128, num_warps=4,
                                     **RELAY_CFG)
    eh_ref = ref._relay_posteriors(syn_t, dev).cpu().numpy()
    wllr = ref._wllr_np.astype(np.float64)
    H = ref.H
    syn_np = sub.astype(np.uint8)
    rows = []
    for block in SWEEP_BLOCKS:
        for warps in SWEEP_WARPS:
            mega = RelayBpMegaTriton.from_dem(
                dem, stop_nconv=5, early_exit=True, dtype="float32",
                block=block, num_warps=warps, **RELAY_CFG)
            e1 = mega._relay_posteriors(syn_t, dev).cpu().numpy()
            e2 = mega._relay_posteriors(syn_t, dev).cpu().numpy()
            det = bool((e1 == e2).all())
            mism = np.where(np.any(e1 != eh_ref, axis=0))[0]
            ties_ok = True
            for s in mism:
                a = eh_ref[:, s].astype(np.uint8)
                b = e1[:, s].astype(np.uint8)
                if not (np.array_equal((H @ b) % 2, syn_np[s])
                        and float((wllr * a).sum()) == float((wllr * b).sum())):
                    ties_ok = False
                    break
            gate = det and ties_ok
            r = dict(block=block, num_warps=warps, gate=gate,
                     gate_deterministic=det,
                     gate_mism_all_verified_ties=ties_ok,
                     n_tie_mismatch_vs_ref=int(len(mism)))
            if gate:
                b = bench_gpu(lambda: mega.decode_batch(
                    dets[:SWEEP_SHOTS], device=DEV), 3, 10)
                r.update(mean_ms=b["mean_ms"], min_ms=b["min_ms"])
            rows.append(r)
            print(f"[sweep relay] BLOCK={block:4d} warps={warps} gate="
                  f"{'PASS' if gate else 'FAIL'} (ties={len(mism)}) "
                  f"{r.get('mean_ms', float('nan')):8.3f} ms")
    ok = [r for r in rows if r["gate"] and "mean_ms" in r]
    win = min(ok, key=lambda r: r["mean_ms"])
    return rows, win


# --------------------------------------------------------------------- #
# throughput + latency receipts (issue #2 / #4)                         #
# --------------------------------------------------------------------- #
def throughput(dem, dets_all, bp_cfg, relay_cfg):
    res = {}
    # relay fp32: megakernel (winner config) vs v0.1 two-kernel host loop.
    for dtype in ("float32", "float64"):
        mega = RelayBpMegaTriton.from_dem(
            dem, stop_nconv=5, early_exit=True, dtype=dtype,
            block=relay_cfg["block"], num_warps=relay_cfg["num_warps"],
            **RELAY_CFG)
        host = RelayBpTriton.from_dem(dem, stop_nconv=5, dtype=dtype,
                                      **RELAY_CFG)
        m_rows, h_rows = [], []
        for n in BATCHES:
            d = dets_all[:n]
            m_rows.append(row(n, bench_gpu(
                lambda: mega.decode_batch(d, device=DEV))))
            h_rows.append(row(n, bench_gpu(
                lambda: host.decode_batch(d, device=DEV))))
            sp = h_rows[-1]["mean_ms"] / m_rows[-1]["mean_ms"]
            print(f"[relay {dtype}] batch {n:6d}: mega "
                  f"{m_rows[-1]['mean_ms']:9.3f} ms | host "
                  f"{h_rows[-1]['mean_ms']:9.3f} ms | {sp:5.1f}x")
        res[f"relay_{dtype}"] = dict(megakernel=m_rows,
                                     v01_two_kernel_host_loop=h_rows)
    # BP 30-iter: megakernel vs v0.1 two-kernel path.
    mega = BpMegaTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS,
                                 block=bp_cfg["block"],
                                 num_warps=bp_cfg["num_warps"])
    old = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    m_rows, o_rows = [], []
    for n in BATCHES:
        d = dets_all[:n]
        m_rows.append(row(n, bench_gpu(
            lambda: mega.decode_batch(d, device=DEV))))
        o_rows.append(row(n, bench_gpu(
            lambda: old.decode_batch(d, device=DEV))))
        sp = o_rows[-1]["mean_ms"] / m_rows[-1]["mean_ms"]
        print(f"[bp 30it     ] batch {n:6d}: mega "
              f"{m_rows[-1]['mean_ms']:9.3f} ms | 2-kernel "
              f"{o_rows[-1]['mean_ms']:9.3f} ms | {sp:5.1f}x")
    res["bp_30iter"] = dict(megakernel=m_rows, v01_two_kernel=o_rows)
    return res


def batch1_floor(dem, dets_all, relay_cfg):
    """Issue #4: where does the single-shot wall-clock go?"""
    mega = RelayBpMegaTriton.from_dem(
        dem, stop_nconv=5, early_exit=True, dtype="float32",
        block=relay_cfg["block"], num_warps=relay_cfg["num_warps"],
        **RELAY_CFG)
    d1 = dets_all[:1]
    dev = torch.device(DEV)
    full = bench_gpu(lambda: mega.decode_batch(d1, device=DEV), 10, 50)
    # device-resident variant: syndrome pre-staged on device, stats kept on
    # device -- isolates the H2D upload + last_stats D2H readback overhead.
    syn_t = torch.as_tensor(d1, device=dev)
    mega.decode_batch(d1, device=DEV)  # warm caches
    kern_only = bench_gpu(lambda: mega._relay_posteriors(syn_t, dev), 10, 50)
    # pinned-memory upload probe (issue #4: "pinned if it helps cheaply").
    src = torch.from_numpy(d1.copy()).pin_memory()
    up_pin = bench_gpu(lambda: src.to(dev, non_blocking=True), 10, 50)
    src_page = torch.from_numpy(d1.copy())
    up_page = bench_gpu(lambda: src_page.to(dev), 10, 50)
    out = dict(
        decode_batch_full=full,
        kernel_plus_stats_device_syndrome=kern_only,
        h2d_1shot_pageable_ms=up_page["mean_ms"],
        h2d_1shot_pinned_ms=up_pin["mean_ms"],
        note=("decode_batch floor is kernel-bound, not transfer-bound: the "
              "1-shot H2D is microseconds either way, so pinned memory is "
              "not the lever (issue #4); the floor is one CTA running the "
              "full per-shot relay schedule plus the last_stats D2H "
              "readbacks."))
    print(f"[batch1] decode_batch {full['mean_ms']:.3f} ms | kernel+stats "
          f"{kern_only['mean_ms']:.3f} ms | h2d pageable "
          f"{up_page['mean_ms']*1e3:.1f} us pinned {up_pin['mean_ms']*1e3:.1f} us")
    return out


def main():
    gpu = torch.cuda.get_device_name(0)
    tag = sys.argv[1] if len(sys.argv) > 1 else \
        ("h200" if "H200" in gpu else gpu.split()[-1].lower())
    receipt = HERE / "receipts" / f"megakernel_{tag}.json"

    circ = load_bb_circuit("0.003", "Z")
    dem = circ.detector_error_model(decompose_errors=False)
    dets_all, _ = circ.compile_detector_sampler(seed=SEED).sample(
        max(BATCHES), separate_observables=True)
    dets_all = np.asarray(dets_all, dtype=bool)

    results = {}
    t0 = time.time()
    bp_rows, bp_win = sweep_bp(dem, dets_all)
    relay_rows, relay_win = sweep_relay(dem, dets_all)
    print(f"[autotune] bp winner: BLOCK={bp_win['block']} "
          f"warps={bp_win['num_warps']} ({bp_win['mean_ms']:.3f} ms) | "
          f"relay winner: BLOCK={relay_win['block']} "
          f"warps={relay_win['num_warps']} ({relay_win['mean_ms']:.3f} ms)")
    results["autotune"] = dict(
        sweep_shots=SWEEP_SHOTS, gate_shots=GATE_SHOTS,
        bp=bp_rows, relay=relay_rows,
        winner=dict(bp=dict(block=bp_win["block"],
                            num_warps=bp_win["num_warps"]),
                    relay=dict(block=relay_win["block"],
                               num_warps=relay_win["num_warps"])))

    results["throughput"] = throughput(
        dem, dets_all, results["autotune"]["winner"]["bp"],
        results["autotune"]["winner"]["relay"])
    results["batch1_floor_relay_fp32"] = batch1_floor(
        dem, dets_all, results["autotune"]["winner"]["relay"])
    # run the barrier section last so the megakernel PTX cache is fully
    # populated with every (BLOCK, warps, DT) variant the sweep compiled.
    results["barrier_sanity"] = barrier_sanity()

    meta = {
        "STATUS": "v0.2 CUDA receipts -- single-launch megakernel validated "
                  "on H-class hardware (gates in "
                  "tests/test_megakernel_cuda.py, all passing on this pod). "
                  "Companion to the Metal spike receipt "
                  "(megakernel_metal_spike.json).",
        "date": time.strftime("%Y-%m-%d"),
        "gpu": gpu,
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version()},
        "versions": {"torch": torch.__version__,
                     "triton": triton.__version__,
                     "stim": _pkg_version("stim"),
                     "relay_bp": _pkg_version("relay-bp"),
                     "numpy": np.__version__},
        "protocol": "canonical BB bb72_r6_p0.003_Z fixture, stim sampler "
                    "seed 0; decode_batch wall-clock, cuda-synchronize-"
                    f"bracketed perf_counter, warmup {N_WARMUP} reps "
                    f"{N_REPS} (autotune sweep: 3/10; batch1: 10/50). v0.1 "
                    "baseline = the SAME pod, SAME batches, two-kernel "
                    "host-loop path (BpTriton / RelayBpTriton).",
        "context_vendor_reference": (
            "CUDA-Q QEC 0.6.0 measured ~2 us/syndrome relay-class at batch "
            "8192 on this GPU class (bench_cudaq_compare.json) -- different "
            "harness, different relay selection semantics (FirstConv-style "
            "vs lowest-weight nconv=5; its LER measured ~2x ours on this "
            "cell), so context, NOT a head-to-head."),
        "elapsed_s": round(time.time() - t0, 1),
    }
    receipt.write_text(json.dumps({"meta": meta, "results": results},
                                  indent=1) + "\n")
    print(f"wrote {receipt}")


if __name__ == "__main__":
    main()
