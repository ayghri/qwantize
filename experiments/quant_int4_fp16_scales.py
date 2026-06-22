#!/usr/bin/env python
"""INT4 quantization with FP16 (E5M10) scales — analog of quant_fp16_scales.py.

Symmetric INT4 codebook {-7, ..., 7}, Q_MAX = 7, D_0 = 0.5.
Per-block scale is FP16, optimization is iterative alternation between
quantization assignment q and continuous-then-snapped scale s.
"""

import time

import torch

from qwantize.nvint4.reference import (
    int4_quantize, int4_dequantize_block, compute_block_sse, Q_MAX,
)
from qwantize.metrics import compute_metrics


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def fp16_snap(x):
    return x.to(torch.float16).to(torch.float32)


def _qd(x, s):
    su = s.unsqueeze(-1)
    return int4_dequantize_block(int4_quantize(x, su), su)


def _int4_signed(x, s):
    return int4_quantize(x, s.unsqueeze(-1))


def int4_fp16_naive(W, block_size):
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))
    dq = _qd(x, s)
    return dq.reshape(M, K)


def int4_fp16_optimal(W, block_size, max_iter=20):
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))

    for _ in range(max_iter):
        q = _int4_signed(x, s)
        denom = (q * q).sum(-1)
        numer = (x * q).sum(-1)
        valid = denom > 1e-12
        s_cont = torch.where(valid, numer / denom.clamp(min=1e-12), s)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    dq = _qd(x, s)
    return dq.reshape(M, K)


def _block_hessians(X_cpu, block_size, dev):
    """Streamed X-on-CPU Hessian builder."""
    K = X_cpu.shape[1]
    bs = block_size
    nblk = K // bs
    H = torch.empty(nblk, bs, bs, device=dev)
    batch_t = 4096
    for j in range(nblk):
        acc = torch.zeros(bs, bs, device=dev)
        for t0 in range(0, X_cpu.shape[0], batch_t):
            Xj = X_cpu[t0:t0 + batch_t, j * bs:(j + 1) * bs].to(
                dev, non_blocking=True).float()
            acc.addmm_(Xj.T, Xj)
            del Xj
        H[j] = acc
    return H


def int4_fp16_hoptimal(W, X_cpu, block_size, max_iter=20):
    M, K = W.shape
    bs = block_size
    nblk = K // bs
    dev = W.device
    x = W.float().reshape(-1, bs)
    N = x.shape[0]
    M_dim = N // nblk
    assert N == M_dim * nblk

    H = _block_hessians(X_cpu, bs, dev)

    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))

    x_3d = x.reshape(M_dim, nblk, bs)

    for _ in range(max_iter):
        q = _int4_signed(x, s)
        q_3d = q.reshape(M_dim, nblk, bs)
        Hq_3d = torch.einsum("jab,mjb->mja", H, q_3d)
        numer = (x_3d * Hq_3d).sum(-1)
        denom = (q_3d * Hq_3d).sum(-1)
        valid = (numer > 0) & (denom > 1e-12)
        s_cont = torch.where(valid,
                             numer / denom.clamp(min=1e-12),
                             s.reshape(M_dim, nblk)).reshape(-1)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    dq = _qd(x, s)
    return dq.reshape(M, K)


def compute_metrics_streamed(W, W_dq, X_cpu, batch=4096):
    metrics = {}
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    metrics["weight_error"] = w_err.item()
    metrics["weight_error_pct"] = (w_err / w_norm * 100).item()

    W_f = W.float()
    W_dq_f = W_dq.float()
    sse = ref_sse = 0.0
    T = X_cpu.shape[0]
    for i in range(0, T, batch):
        Xb = X_cpu[i:i + batch].to(W.device, non_blocking=True).float()
        out_ref = Xb @ W_f.T
        out_dq = Xb @ W_dq_f.T
        sse += (out_dq - out_ref).pow(2).sum().item()
        ref_sse += out_ref.pow(2).sum().item()
        del Xb, out_ref, out_dq
    metrics["output_error"] = sse ** 0.5
    metrics["output_error_pct"] = (sse ** 0.5) / (ref_sse ** 0.5) * 100
    return metrics


def _fmt_time(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    print("INT4 with FP16 (E5M10) scales")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K}  X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]} ({X_cpu.dtype})")
    print(f"||W||_F = {W.norm().item():.4e}\n")

    results = []

    for bs in [16, 32, 128]:
        print("=" * 80)
        print(f"Block size {bs}   (scale overhead = {16 / bs:.3f} bits/weight)")
        print("=" * 80)

        for approach in ["Naive", "Optimal", "H-Optimal"]:
            torch.cuda.synchronize()
            t0 = time.time()
            if approach == "Naive":
                Q = int4_fp16_naive(W, bs)
            elif approach == "Optimal":
                Q = int4_fp16_optimal(W, bs)
            else:
                Q = int4_fp16_hoptimal(W, X_cpu, bs)
            torch.cuda.synchronize()
            t = time.time() - t0

            m = compute_metrics_streamed(W, Q, X_cpu)
            del Q
            torch.cuda.empty_cache()

            results.append((bs, approach, m, t))
            print(f"  INT4 / FP16-scale  {approach:<10}  "
                  f"W={m['weight_error_pct']:7.4f}%  "
                  f"O={m['output_error_pct']:7.4f}%  "
                  f"||Wq-W||_F={m['weight_error']:.4e}  "
                  f"||X(Wq-W)^T||_F={m['output_error']:.4e}  "
                  f"{_fmt_time(t):>8}")
        print()

    print("=" * 80)
    print("Markdown")
    print("=" * 80 + "\n")

    print("### INT4 with FP16 (E5M10) scales\n")
    print("| Block | Scale b/w | Approach | Weight Error | Output Error | Time |")
    print("|:--:|:--:|:--|:--:|:--:|--:|")
    for bs, approach, m, t in results:
        bw = 16 / bs
        print(f"| {bs} | {bw:.3f} | {approach} "
              f"| {m['weight_error_pct']:.2f}% "
              f"| {m['output_error_pct']:.2f}% "
              f"| {_fmt_time(t)} |")


if __name__ == "__main__":
    main()
