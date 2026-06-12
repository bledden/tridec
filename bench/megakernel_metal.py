"""Generate ``bench/receipts/megakernel_metal_spike.json`` (issue #2 spike).

PRELIMINARY / EXPERIMENTAL: validation gates + launch-overhead bench for the
single-launch persistent BP / Relay-BP megakernels
(``tridec.backends.megakernel``) on the **experimental triton-metal
environment** (Apple silicon, fp32, CPU tensors). H200/MI300X receipts and
the CUDA/ROCm autotune pass are deferred to the cloud session.

Gates (the existing suite's bars, run against the megakernels):
  * BP: bit-identity vs the two-kernel ``BpTriton`` + >=0.995 hard agreement
    vs the fp64 numpy reference + LER within max(3, 0.5%) -- BB AND surface.
  * Relay: ALL-LEGS mode per-shot identity vs the host-loop
    ``RelayBpTriton`` (same gamma tensors, same fp32 math, same schedule);
    nconv mode LER vs the host loop AND the relay_bp Rust oracle
    (LER-indistinguishable bar, per-shot agreement reported).
  * Determinism across repeated runs (the megakernel is barrier-sensitive;
    see the module docstring of tridec.backends.megakernel).

Bench: decode_batch wall-clock on the canonical BB cell, 2000 shots --
relay (baseline measured ~31 s on this machine, launch-bound) and BP-only
30-iter (baseline ~167 ms), warmup + reps documented in the receipt.

Run (in the triton-metal environment):  python bench/megakernel_metal.py
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

from conftest import load_bb_circuit, load_surface_circuit  # noqa: E402

from tridec.backends.bp_numpy import BpBaseline             # noqa: E402
from tridec.backends.bp_triton import BpTriton              # noqa: E402
from tridec.backends.relay_triton import RelayBpTriton      # noqa: E402
from tridec.backends.megakernel import (                    # noqa: E402
    BpMegaTriton, RelayBpMegaTriton)

RECEIPT = HERE / "receipts" / "megakernel_metal_spike.json"

MS = 0.625
SHOTS = 2000
SEED = 0
DEVICE = "cpu"           # triton-metal pattern: CPU tensors (UMA), not mps
RELAY_CFG = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                 gamma_dist_interval=(-0.24, 0.66),
                 stopping_criterion="nconv", dtype="float32")


def sample(circ, shots=SHOTS):
    dets, obs = circ.compile_detector_sampler(seed=SEED).sample(
        shots, separate_observables=True)
    return np.asarray(dets, dtype=bool), np.asarray(obs, dtype=bool)


# --------------------------------------------------------------------- #
# BP gates: bit-identity vs BpTriton, agreement/LER vs numpy.            #
# --------------------------------------------------------------------- #
def bp_gates(label, circ):
    dem = circ.detector_error_model(decompose_errors=False)
    dets, obs = sample(circ)
    syn = dets.astype(np.uint8)

    old = BpTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    mega = BpMegaTriton.from_dem(dem, max_iter=30, ms_scaling_factor=MS)
    cpu = BpBaseline.from_dem(dem, max_iter=30, ms_scaling_factor=MS)

    # bit-identity vs the two-kernel path (30 iters) + determinism.
    p_old = old.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    p_meg2 = mega.run_iterations_batch(syn, n_iter=30, device=DEVICE)
    bit_ident = float((p_old == p_meg).mean())
    determin = float((p_meg == p_meg2).mean())
    maxdiff = float(np.max(np.abs(p_old - p_meg)))

    # numpy-reference gates (1 iter + 30 iter hard agreement, the suite bar).
    p1 = mega.run_iterations_batch(syn, n_iter=1, device=DEVICE)
    h1 = (p1 < 0).astype(np.uint8)
    h30 = (p_meg < 0).astype(np.uint8)
    h1_cpu = np.empty_like(h1)
    h30_cpu = np.empty_like(h30)
    for i in range(len(syn)):
        h1_cpu[i] = cpu.run_iterations(syn[i], n_iter=1)["hard"]
        h30_cpu[i] = cpu.run_iterations(syn[i], n_iter=30)["hard"]
    a1 = float((h1 == h1_cpu).mean())
    a30 = float((h30 == h30_cpu).mean())

    pred_cpu = cpu.decode_batch(dets)
    pred_meg = mega.decode_batch(dets, device=DEVICE)
    pred_old = old.decode_batch(dets, device=DEVICE)
    f_cpu = int(np.any(pred_cpu != obs, axis=1).sum())
    f_meg = int(np.any(pred_meg != obs, axis=1).sum())
    f_old = int(np.any(pred_old != obs, axis=1).sum())
    pred_ident = float(np.all(pred_meg == pred_old, axis=1).mean())

    res = dict(
        label=label, shots=SHOTS, block=mega.block,
        bit_identity_vs_bptriton_30iter=bit_ident,
        posterior_maxdiff_vs_bptriton=maxdiff,
        determinism_exact=determin,
        one_iter_hard_agree_vs_numpy=a1,
        thirty_iter_hard_agree_vs_numpy=a30,
        ler_numpy=f_cpu, ler_bptriton=f_old, ler_mega=f_meg,
        decode_pred_identity_vs_bptriton=pred_ident,
        gates=dict(
            bit_identity=bool(bit_ident == 1.0),
            agree_1iter=bool(a1 >= 0.995),
            agree_30iter=bool(a30 >= 0.995),
            ler=bool(abs(f_meg - f_cpu) <= max(3, int(0.005 * SHOTS))),
            deterministic=bool(determin == 1.0),
        ),
    )
    print(f"[BP {label}] bit-ident {bit_ident:.6f} (maxdiff {maxdiff:.1e}) "
          f"det {determin:.6f} | numpy agree 1it {a1:.5f} 30it {a30:.5f} | "
          f"LER numpy {f_cpu} / old {f_old} / mega {f_meg} | "
          f"pred-ident {pred_ident:.5f}")
    return res


# --------------------------------------------------------------------- #
# Relay gates.                                                          #
# --------------------------------------------------------------------- #
def relay_identity_gate(dem, dets, n_sub=256):
    """ALL-LEGS mode: host loop with unreachable stop_nconv vs megakernel
    with early_exit=False run the SAME schedule -> per-shot identity is the
    bar (same gammas, same fp32 op order)."""
    sub = dets[:n_sub]
    host = RelayBpTriton.from_dem(dem, stop_nconv=62, **RELAY_CFG)
    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=62, early_exit=False,
                                      **RELAY_CFG)
    t0 = time.perf_counter()
    p_h = host.decode_batch(sub, device=DEVICE)
    t1 = time.perf_counter()
    p_m = mega.decode_batch(sub, device=DEVICE)
    t2 = time.perf_counter()
    p_m2 = mega.decode_batch(sub, device=DEVICE)
    ident = float(np.all(p_h == p_m, axis=1).mean())
    det = float((p_m == p_m2).mean())
    print(f"[RELAY all-legs identity] N={n_sub} per-shot identity {ident:.5f}"
          f" det {det:.5f} (host {t1-t0:.1f}s, mega {t2-t1:.1f}s)")
    return dict(n=n_sub, per_shot_identity=ident, determinism=det,
                host_s=t1 - t0, mega_s=t2 - t1,
                gates=dict(identity=bool(ident == 1.0),
                           deterministic=bool(det == 1.0)))


def relay_full_gate(dem, dets, obs):
    """nconv mode (stop_nconv=5, the validated default): LER vs the host
    RelayBpTriton AND the relay_bp Rust oracle; per-shot agreements."""
    import relay_bp
    from relay_bp.stim import CheckMatrices
    cm = CheckMatrices.from_dem(dem)
    oracle_cfg = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60,
                      gamma_dist_interval=(-0.24, 0.66), stop_nconv=5,
                      stopping_criterion="nconv")
    oracle = relay_bp.RelayDecoderF64(cm.check_matrix,
                                      error_priors=cm.error_priors,
                                      **oracle_cfg)
    runner = relay_bp.ObservableDecoderRunner(
        oracle, cm.observables_matrix, include_decode_result=False)
    t0 = time.perf_counter()
    p_ref = (np.asarray(runner.decode_observables_batch(
        dets.astype(np.uint8))) % 2).astype(bool)
    t_oracle = time.perf_counter() - t0
    if p_ref.ndim == 1:
        p_ref = p_ref.reshape(-1, 1)

    host = RelayBpTriton.from_dem(dem, stop_nconv=5, **RELAY_CFG)
    mega = RelayBpMegaTriton.from_dem(dem, stop_nconv=5, early_exit=True,
                                      **RELAY_CFG)
    t0 = time.perf_counter()
    p_h = host.decode_batch(dets, device=DEVICE)
    t_host = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_m = mega.decode_batch(dets, device=DEVICE)
    t_mega = time.perf_counter() - t0

    n = len(dets)
    f_ref = int(np.any(p_ref != obs, axis=1).sum())
    f_h = int(np.any(p_h != obs, axis=1).sum())
    f_m = int(np.any(p_m != obs, axis=1).sum())
    ag_mh = float(np.all(p_m == p_h, axis=1).mean())
    ag_mo = float(np.all(p_m == p_ref, axis=1).mean())
    st = mega.last_stats
    print(f"[RELAY nconv5 full] N={n} LER oracle {f_ref} / host {f_h} / "
          f"mega {f_m} | mega-vs-host per-shot {ag_mh:.4f} | mega-vs-oracle "
          f"{ag_mo:.4f}")
    print(f"  mega legs min/mean/max {int(st['legs'].min())}/"
          f"{float(st['legs'].mean()):.2f}/{int(st['legs'].max())} | iters "
          f"mean {float(st['iters'].mean()):.1f} | times: oracle "
          f"{t_oracle:.2f}s host {t_host:.1f}s mega {t_mega:.2f}s")
    return dict(
        n=n, ler_oracle=f_ref, ler_host=f_h, ler_mega=f_m,
        per_shot_agree_mega_vs_host=ag_mh,
        per_shot_agree_mega_vs_oracle=ag_mo,
        oracle_s=t_oracle, host_s=t_host, mega_s=t_mega,
        mega_legs_mean=float(st["legs"].mean()),
        mega_legs_max=int(st["legs"].max()),
        mega_iters_mean=float(st["iters"].mean()),
        mega_unconverged=int((st["nconv"] == 0).sum()),
        gates=dict(
            ler_vs_oracle=bool(abs(f_m - f_ref) <= max(5, int(0.01 * n))),
            ler_vs_host=bool(abs(f_m - f_h) <= max(5, int(0.01 * n))),
        ),
    )


# --------------------------------------------------------------------- #
# Bench: wall-clock decode_batch on the canonical BB cell.               #
# --------------------------------------------------------------------- #
def bench(fn, n_warmup, n_reps):
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return dict(n_warmup=n_warmup, n_reps=n_reps, mean_s=float(np.mean(ts)),
                min_s=float(np.min(ts)), per_rep_s=[float(t) for t in ts])


def main():
    import torch
    import triton
    import relay_bp
    import triton_metal  # the env gate: this script is metal-env-only

    circ_bb = load_bb_circuit("0.003", "Z")
    circ_sf = load_surface_circuit()
    dem_bb = circ_bb.detector_error_model(decompose_errors=False)
    dets_bb, obs_bb = sample(circ_bb)

    results = {}
    results["bp_bb72"] = bp_gates("bb72_r6_p0.003_Z", circ_bb)
    results["bp_surface"] = bp_gates("surface_d3_r3_p0.003", circ_sf)
    results["relay_identity_all_legs"] = relay_identity_gate(dem_bb, dets_bb)
    results["relay_full_nconv5"] = relay_full_gate(dem_bb, dets_bb, obs_bb)

    # ---- bench: BP-only 30-iter decode_batch(2000) ----------------------
    bp_old = BpTriton.from_dem(dem_bb, max_iter=30, ms_scaling_factor=MS)
    bp_meg = BpMegaTriton.from_dem(dem_bb, max_iter=30, ms_scaling_factor=MS)
    b_old = bench(lambda: bp_old.decode_batch(dets_bb, device=DEVICE), 2, 5)
    b_meg = bench(lambda: bp_meg.decode_batch(dets_bb, device=DEVICE), 2, 5)
    print(f"[BENCH bp 2000x30it] two-kernel {1e3*b_old['mean_s']:.0f} ms | "
          f"mega {1e3*b_meg['mean_s']:.0f} ms")
    results["bench_bp_2000"] = dict(two_kernel=b_old, megakernel=b_meg)

    # ---- bench: full relay decode_batch(2000) ---------------------------
    r_host = RelayBpTriton.from_dem(dem_bb, stop_nconv=5, **RELAY_CFG)
    r_meg = RelayBpMegaTriton.from_dem(dem_bb, stop_nconv=5, early_exit=True,
                                       **RELAY_CFG)
    b_host = bench(lambda: r_host.decode_batch(dets_bb, device=DEVICE), 1, 2)
    b_megr = bench(lambda: r_meg.decode_batch(dets_bb, device=DEVICE), 1, 3)
    print(f"[BENCH relay 2000] host-loop {b_host['mean_s']:.1f} s | "
          f"mega {b_megr['mean_s']:.2f} s | speedup "
          f"{b_host['mean_s']/b_megr['mean_s']:.1f}x")
    results["bench_relay_2000"] = dict(host_loop=b_host, megakernel=b_megr,
                                       speedup=b_host["mean_s"]
                                       / b_megr["mean_s"])

    meta = {
        "STATUS": "PRELIMINARY / EXPERIMENTAL -- megakernel spike on the "
                  "triton-metal environment (fp32, CPU tensors, block=32; "
                  "see tridec/backends/megakernel.py docstring for the "
                  "triton-metal codegen constraints). H200/MI300X receipts "
                  "+ autotune deferred to the cloud session (issue #2).",
        "date": time.strftime("%Y-%m-%d"),
        "platform": {"system": sys.platform, "machine": platform.machine(),
                     "python": platform.python_version()},
        "versions": {"stim": stim.__version__, "torch": torch.__version__,
                     "triton": triton.__version__,
                     "relay_bp": _pkg_version("relay-bp"),
                     "numpy": np.__version__},
        "protocol": "canonical BB bb72_r6_p0.003_Z + surface_d3_r3_p0.003 "
                    "fixtures, 2000 shots (stim sampler seed 0). Gates "
                    "mirror tests/test_bp_triton.py, test_relay_triton.py, "
                    "test_metal.py. Bench = wall-clock decode_batch, "
                    "warmup/reps recorded per entry.",
    }
    RECEIPT.write_text(json.dumps({"meta": meta, "results": results},
                                  indent=1) + "\n")
    print(f"wrote {RECEIPT}")


if __name__ == "__main__":
    main()
