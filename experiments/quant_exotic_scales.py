#!/misc/envs/quant/bin/python
"""FP4 quantization with exotic FP8 scale grids.

Compares three FP8 scale formats for FP4 (E2M1) block-scaled quantization:
  - E4M3   (NVFP4 baseline, 126 positive values, signed)
  - UE4M4  (unsigned 4-exp 4-mantissa, 255 positive values)
  - UE5M3  (unsigned 5-exp 3-mantissa, 255 positive values)

For each scale type and block size {16, 32}, runs Naive / SSE-Optimal /
H-Optimal scale search using the same FP4 codebook {0, 0.5, 1, 1.5, 2, 3, 4, 6}.

Usage: /misc/envs/quant/bin/python experiments/quant_exotic_scales.py
"""

import time
import torch

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap,
    build_fp8_e4m3_scales,
    fp4_quantize,
    fp4_dequantize,
    compute_block_sse,
    Q_MAX,
    D_0,
)
from qwantize.exotic_scales import (
    build_ue4m4_scales,
    build_ue5m3_scales,
    snap_to_table,
)
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# Parameterized FP4 quantization (scale grid as argument)
# ===================================================================

def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def fp4_naive(W, scale_snap, block_size):
    """Naive FP4 with caller-supplied scale snap function."""
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    amax = x.abs().amax(dim=-1)
    s = scale_snap((amax / Q_MAX).clamp(min=1e-12))
    dq = _qd(x, s)
    return dq.reshape(M, K)


def fp4_optimal(W, scale_table, scale_snap, block_size):
    """SSE-optimal FP4 with caller-supplied scale grid + snap."""
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    bs = block_size
    amax = x.abs().amax(-1)
    s0 = scale_snap((amax / Q_MAX).clamp(min=1e-12))

    E0 = compute_block_sse(x, s0)
    best_s, best_E = s0.clone(), E0.clone()

    noise = x.pow(2).sum(-1) <= E0
    s_min = ((amax - E0.sqrt()) / Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / D_0

    act = ~noise
    if act.any():
        xa, smn, smx = x[act], s_min[act], s_max[act]
        bE, bS = best_E[act].clone(), best_s[act].clone()
        dev = x.device
        for sv in scale_table:
            sf = sv.item()
            ok = (sf >= smn) & (sf <= smx)
            if not ok.any():
                continue
            clip = (xa.abs() - Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE)
            if not ev.any():
                continue
            Es = compute_block_sse(xa, torch.full((xa.shape[0],), sf, device=dev))
            imp = ev & (Es < bE)
            bE[imp], bS[imp] = Es[imp], sf
        best_E[act], best_s[act] = bE, bS

    dq = _qd(x, best_s)
    return dq.reshape(M, K)


def fp4_hoptimal(W, X, scale_table, scale_snap, block_size):
    """H-optimal FP4 with caller-supplied scale grid + snap.

    Block Hessians H_j = X_j^T @ X_j computed once.
    """
    M, K = W.shape
    bs = block_size
    nblk = K // bs
    x = W.float().reshape(-1, bs)             # (N, bs), N = M * nblk
    N = x.shape[0]
    dev = x.device

    H_blocks = torch.empty(nblk, bs, bs, device=dev)
    batch_t = 8192
    for j in range(nblk):
        acc = torch.zeros(bs, bs, device=dev)
        for t0 in range(0, X.shape[0], batch_t):
            Xj = X[t0:t0 + batch_t, j * bs:(j + 1) * bs].float()
            acc.addmm_(Xj.T, Xj)
        H_blocks[j] = acc

    M_dim = N // nblk
    assert N == M_dim * nblk

    amax = x.abs().amax(-1)
    s0 = scale_snap((amax / Q_MAX).clamp(min=1e-12))

    E0_sse = compute_block_sse(x, s0)
    # Hessian-weighted baseline error
    quants0 = fp4_quantize(x, s0.unsqueeze(-1))
    dq0 = fp4_dequantize(quants0, s0.unsqueeze(-1))
    r0 = x - dq0
    r0_3d = r0.reshape(M_dim, nblk, bs)
    Hr0 = torch.einsum("jab,mjb->mja", H_blocks, r0_3d)
    E0_H = (r0_3d * Hr0).sum(-1).reshape(-1)

    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / D_0

    act = ~noise
    if act.any():
        bE, bS = best_E[act].clone(), best_s[act].clone()
        smn, smx = s_min[act], s_max[act]
        xa = x[act]
        active_idx = act.nonzero(as_tuple=True)[0]
        active_j = active_idx % nblk

        for sv in scale_table:
            sf = sv.item()
            ok = (sf >= smn) & (sf <= smx)
            if not ok.any():
                continue
            clip = (xa.abs() - Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.tensor(sf, device=dev)
            quants_s = fp4_quantize(xa, sf_t)
            dq_s = fp4_dequantize(quants_s, sf_t)
            r = xa - dq_s
            # chunk to bound memory for big H gather
            EH = torch.empty(xa.shape[0], device=dev)
            chunk = 4096
            for c0 in range(0, xa.shape[0], chunk):
                c1 = min(c0 + chunk, xa.shape[0])
                Hc = H_blocks[active_j[c0:c1]]
                Hr = torch.bmm(Hc, r[c0:c1].unsqueeze(-1)).squeeze(-1)
                EH[c0:c1] = (r[c0:c1] * Hr).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_E[act], best_s[act] = bE, bS

    dq = _qd(x, best_s)
    return dq.reshape(M, K)


# ===================================================================
# Benchmark
# ===================================================================

def _fmt_time(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    print("FP4 with Exotic FP8 Scale Grids")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}\n")

    e4m3 = build_fp8_e4m3_scales(device=DEVICE)
    ue4m4 = build_ue4m4_scales(device=DEVICE)
    ue5m3 = build_ue5m3_scales(device=DEVICE)

    SCALE_CFG = {
        "E4M3":  (e4m3,  lambda x: _fp8_e4m3_snap(x)),
        "UE4M4": (ue4m4, lambda x: snap_to_table(x, ue4m4)),
        "UE5M3": (ue5m3, lambda x: snap_to_table(x, ue5m3)),
    }

    results = []  # (bs, scale_name, approach, metrics, time)

    for bs in [16, 32]:
        print(f"{'=' * 80}")
        print(f"Block size: {bs}")
        print(f"{'=' * 80}")

        for scale_name, (table, snap) in SCALE_CFG.items():
            for approach in ["Naive", "Optimal", "H-Optimal"]:
                torch.cuda.synchronize()
                t0 = time.time()
                if approach == "Naive":
                    Q = fp4_naive(W, snap, bs)
                elif approach == "Optimal":
                    Q = fp4_optimal(W, table, snap, bs)
                else:
                    Q = fp4_hoptimal(W, X, table, snap, bs)
                torch.cuda.synchronize()
                t = time.time() - t0
                m = compute_metrics(W, Q, X)
                del Q
                torch.cuda.empty_cache()

                results.append((bs, scale_name, approach, m, t))
                print(f"  FP4-{scale_name:<5} {approach:<10}  "
                      f"W={m['weight_error_pct']:7.4f}%  "
                      f"O={m['output_error_pct']:7.4f}%  "
                      f"{_fmt_time(t):>8}")
        print()

    print("=" * 80)
    print("Markdown for docs/source/results.md")
    print("=" * 80 + "\n")

    for bs_f in [16, 32]:
        print(f"### Block Size {bs_f}\n")
        print("| Scale | Approach | # Values | Weight Error | Output Error | Time |")
        print("|:--|:--|:--:|:--:|:--:|--:|")
        n_values = {"E4M3": 126, "UE4M4": 255, "UE5M3": 255}
        for bs, scale_name, approach, m, t in results:
            if bs != bs_f:
                continue
            print(f"| {scale_name} | {approach} | {n_values[scale_name]} "
                  f"| {m['weight_error_pct']:.2f}% "
                  f"| {m['output_error_pct']:.2f}% "
                  f"| {_fmt_time(t)} |")
        print()


if __name__ == "__main__":
    main()
