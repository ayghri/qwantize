#!/usr/bin/env python
"""FP4 and INT4 quantization at block size 64 with FP8 E4M3 scales.

Parameterized by codebook: FP4 uses E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6} (Q_MAX=6,
D_0=0.25); INT4 uses symmetric {-7..7} (Q_MAX=7, D_0=0.5).

Mirrors the structure of experiments/quant_exotic_scales.py — bounded search
over the 126 positive E4M3 scales, Naive / Optimal / H-Optimal variants.
"""

import time

import torch

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap,
    build_fp8_e4m3_scales,
    fp4_quantize as _fp4_quant,
    fp4_dequantize as _fp4_dequant,
)
from qwantize.nvint4.reference import (
    int4_quantize as _int4_quant,
    int4_dequantize_block as _int4_dequant,
)
from qwantize.metrics import compute_metrics


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# Codebook abstractions
# ===================================================================

class FP4Codebook:
    name = "FP4"
    Q_MAX = 6.0
    D_0 = 0.25
    quant = staticmethod(_fp4_quant)
    dequant = staticmethod(_fp4_dequant)


class INT4Codebook:
    name = "INT4"
    Q_MAX = 7.0
    D_0 = 0.5
    quant = staticmethod(_int4_quant)
    dequant = staticmethod(_int4_dequant)


def _qd(cb, x, s):
    su = s.unsqueeze(-1)
    return cb.dequant(cb.quant(x, su), su)


def _block_sse(cb, x, s):
    su = s.unsqueeze(-1) if s.dim() == 1 else s
    quants = cb.quant(x, su)
    dq = cb.dequant(quants, su)
    return (x - dq).pow(2).sum(dim=-1)


# ===================================================================
# Naive / Optimal / H-Optimal (parameterized)
# ===================================================================

def quant_naive(cb, W, block_size):
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    amax = x.abs().amax(dim=-1)
    s = _fp8_e4m3_snap((amax / cb.Q_MAX).clamp(min=1e-12))
    dq = _qd(cb, x, s)
    return dq.reshape(M, K)


def quant_optimal(cb, W, scale_table, block_size):
    M, K = W.shape
    bs = block_size
    x = W.float().reshape(-1, bs)
    amax = x.abs().amax(-1)
    s0 = _fp8_e4m3_snap((amax / cb.Q_MAX).clamp(min=1e-12))

    E0 = _block_sse(cb, x, s0)
    best_s, best_E = s0.clone(), E0.clone()

    noise = x.pow(2).sum(-1) <= E0
    s_min = ((amax - E0.sqrt()) / cb.Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / cb.D_0

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
            clip = (xa.abs() - cb.Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE)
            if not ev.any():
                continue
            Es = _block_sse(cb, xa, torch.full((xa.shape[0],), sf, device=dev))
            imp = ev & (Es < bE)
            bE[imp], bS[imp] = Es[imp], sf
        best_E[act], best_s[act] = bE, bS

    dq = _qd(cb, x, best_s)
    return dq.reshape(M, K)


def quant_hoptimal(cb, W, X_cpu, scale_table, block_size):
    M, K = W.shape
    bs = block_size
    nblk = K // bs
    x = W.float().reshape(-1, bs)
    N = x.shape[0]
    dev = x.device

    # Block Hessians H_j = X_j^T @ X_j (streamed from CPU)
    H_blocks = torch.empty(nblk, bs, bs, device=dev)
    batch_t = 4096
    for j in range(nblk):
        acc = torch.zeros(bs, bs, device=dev)
        for t0 in range(0, X_cpu.shape[0], batch_t):
            Xj = X_cpu[t0:t0 + batch_t, j * bs:(j + 1) * bs].to(
                dev, non_blocking=True).float()
            acc.addmm_(Xj.T, Xj)
            del Xj
        H_blocks[j] = acc

    M_dim = N // nblk
    assert N == M_dim * nblk

    amax = x.abs().amax(-1)
    s0 = _fp8_e4m3_snap((amax / cb.Q_MAX).clamp(min=1e-12))

    E0_sse = _block_sse(cb, x, s0)

    # Hessian-weighted baseline error
    q0 = cb.quant(x, s0.unsqueeze(-1))
    dq0 = cb.dequant(q0, s0.unsqueeze(-1))
    r0 = x - dq0
    r0_3d = r0.reshape(M_dim, nblk, bs)
    Hr0 = torch.einsum("jab,mjb->mja", H_blocks, r0_3d)
    E0_H = (r0_3d * Hr0).sum(-1).reshape(-1)

    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / cb.Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / cb.D_0

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
            clip = (xa.abs() - cb.Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.tensor(sf, device=dev)
            quants_s = cb.quant(xa, sf_t)
            dq_s = cb.dequant(quants_s, sf_t)
            r = xa - dq_s
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

    dq = _qd(cb, x, best_s)
    return dq.reshape(M, K)


# ===================================================================
# Streamed metrics
# ===================================================================

def compute_metrics_streamed(W, W_dq, X_cpu, batch=4096):
    m = {}
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    m["weight_error"] = w_err.item()
    m["weight_error_pct"] = (w_err / w_norm * 100).item()

    W_f = W.float()
    W_dq_f = W_dq.float()
    sse = ref_sse = 0.0
    for i in range(0, X_cpu.shape[0], batch):
        Xb = X_cpu[i:i + batch].to(W.device, non_blocking=True).float()
        out_ref = Xb @ W_f.T
        out_dq = Xb @ W_dq_f.T
        sse += (out_dq - out_ref).pow(2).sum().item()
        ref_sse += out_ref.pow(2).sum().item()
        del Xb, out_ref, out_dq
    m["output_error"] = sse ** 0.5
    m["output_error_pct"] = (sse ** 0.5) / (ref_sse ** 0.5) * 100
    return m


def _fmt_time(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


# ===================================================================
# Main
# ===================================================================

BS = 64


def main():
    print("FP4 and INT4 with FP8 E4M3 scales, block size 64")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K}  X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]} ({X_cpu.dtype})")
    print(f"||W||_F = {W.norm().item():.4e}\n")

    e4m3 = build_fp8_e4m3_scales(device=DEVICE)
    results = []  # (codebook_name, approach, m, t)

    for cb in [FP4Codebook, INT4Codebook]:
        print("=" * 80)
        print(f"Codebook: {cb.name}   (block size {BS}, E4M3 scales)")
        print("=" * 80)

        for approach in ["Naive", "Optimal", "H-Optimal"]:
            torch.cuda.synchronize()
            t0 = time.time()
            if approach == "Naive":
                Q = quant_naive(cb, W, BS)
            elif approach == "Optimal":
                Q = quant_optimal(cb, W, e4m3, BS)
            else:
                Q = quant_hoptimal(cb, W, X_cpu, e4m3, BS)
            torch.cuda.synchronize()
            t = time.time() - t0
            m = compute_metrics_streamed(W, Q, X_cpu)
            del Q
            torch.cuda.empty_cache()

            results.append((cb.name, approach, m, t))
            print(f"  {cb.name}-E4M3 BS{BS} {approach:<10}  "
                  f"W={m['weight_error_pct']:7.4f}%  "
                  f"O={m['output_error_pct']:7.4f}%  "
                  f"||Wq-W||_F={m['weight_error']:.4e}  "
                  f"||X(Wq-W)^T||_F={m['output_error']:.4e}  "
                  f"{_fmt_time(t):>8}")
        print()

    print("=" * 80)
    print("Markdown")
    print("=" * 80 + "\n")

    print(f"### Block Size {BS}, E4M3 scales\n")
    print("| Codebook | Approach | Weight Error | Output Error | Time |")
    print("|:--|:--|:--:|:--:|--:|")
    for name, approach, m, t in results:
        print(f"| {name} | {approach} "
              f"| {m['weight_error_pct']:.2f}% "
              f"| {m['output_error_pct']:.2f}% "
              f"| {_fmt_time(t)} |")


if __name__ == "__main__":
    main()
