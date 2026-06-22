#!/usr/bin/env python
"""FP4 quantization with FP16 (E5M10) scales.

Drops the FP8 scale grid and uses IEEE-754 half-precision (1 sign + 5 exp +
10 mantissa, ~1024 mantissa codes per binade) as the per-block scale. Since
FP16 is much finer than any FP8 grid, the snapped continuous optimum is
essentially the true SSE / H-optimal minimum, so we solve scales by
iterative alternation between FP4 assignment q and continuous-then-snapped s:

  q  = fp4_quantize(x, s)               # signed codebook values
  s* = <x, q> / <q, q>                  # SSE-optimal s given q
  s* = <x, H q> / <q, H q>              # H-optimal s given q
  s  = fp16_snap(s*)

This converges in <10 iterations on layer_0.

The point of the experiment is that with FP16 scales we can afford a much
larger block size (128 here) and still have low per-weight scale overhead:
  bs=16, FP8:  8/16  = 0.500 b/w  (baseline)
  bs=32, FP8:  8/32  = 0.250 b/w
  bs=128, FP16: 16/128 = 0.125 b/w   ← lowest overhead

The question we're answering: does the finer scale precision compensate for
the larger block size (which spans more of the weight distribution per scale)?
"""

import time

import torch

from qwantize.nvfp4.reference import (
    fp4_quantize,
    fp4_dequantize,
    compute_block_sse,
    Q_MAX,
)
from qwantize.metrics import compute_metrics


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# -------------------------------------------------------------------
# FP16 scale snap
# -------------------------------------------------------------------

def fp16_snap(x):
    """Round to nearest FP16 (E5M10). Input/output stay in float32."""
    return x.to(torch.float16).to(torch.float32)


# -------------------------------------------------------------------
# Quant primitives
# -------------------------------------------------------------------

def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def _fp4_signed(x, s):
    """Return signed FP4 codebook values for x / s."""
    return fp4_quantize(x, s.unsqueeze(-1))


# -------------------------------------------------------------------
# Naive: s = fp16(max|x| / 6)
# -------------------------------------------------------------------

def fp4_fp16_naive(W, block_size):
    M, K = W.shape
    x = W.float().reshape(-1, block_size)
    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))
    dq = _qd(x, s)
    return dq.reshape(M, K)


# -------------------------------------------------------------------
# SSE-Optimal: iterative alternation, FP16-snapped s update
# -------------------------------------------------------------------

def fp4_fp16_optimal(W, block_size, max_iter=20):
    M, K = W.shape
    x = W.float().reshape(-1, block_size)            # (N, bs)
    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))   # (N,)

    for _ in range(max_iter):
        q = _fp4_signed(x, s)                        # (N, bs)
        denom = (q * q).sum(-1)                      # (N,)
        numer = (x * q).sum(-1)
        valid = denom > 1e-12
        s_cont = torch.where(valid, numer / denom.clamp(min=1e-12), s)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    dq = _qd(x, s)
    return dq.reshape(M, K)


# -------------------------------------------------------------------
# H-Optimal: iterative alternation using per-column-block Hessian
# -------------------------------------------------------------------

def _block_hessians(X, block_size):
    """Return (nblk, bs, bs) H_j = X_j^T @ X_j (chunked over rows)."""
    K = X.shape[1]
    bs = block_size
    nblk = K // bs
    dev = X.device
    H = torch.empty(nblk, bs, bs, device=dev)
    batch_t = 8192
    for j in range(nblk):
        acc = torch.zeros(bs, bs, device=dev)
        for t0 in range(0, X.shape[0], batch_t):
            Xj = X[t0:t0 + batch_t, j * bs:(j + 1) * bs].float()
            acc.addmm_(Xj.T, Xj)
        H[j] = acc
    return H


def fp4_fp16_hoptimal(W, X, block_size, max_iter=20):
    M, K = W.shape
    bs = block_size
    nblk = K // bs
    x = W.float().reshape(-1, bs)                    # (N, bs)
    N = x.shape[0]
    M_dim = N // nblk
    assert N == M_dim * nblk
    dev = x.device

    H = _block_hessians(X, bs)                       # (nblk, bs, bs)

    amax = x.abs().amax(dim=-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))   # (N,)

    # 3-D view for batched Hessian ops
    x_3d = x.reshape(M_dim, nblk, bs)

    for _ in range(max_iter):
        q = _fp4_signed(x, s)                        # (N, bs)
        q_3d = q.reshape(M_dim, nblk, bs)
        Hq_3d = torch.einsum("jab,mjb->mja", H, q_3d)  # (M, nblk, bs)
        numer = (x_3d * Hq_3d).sum(-1)                # (M, nblk)
        denom = (q_3d * Hq_3d).sum(-1)                # (M, nblk)
        valid = (numer > 0) & (denom > 1e-12)
        s_cont = torch.where(
            valid,
            numer / denom.clamp(min=1e-12),
            s.reshape(M_dim, nblk),
        ).reshape(-1)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    dq = _qd(x, s)
    return dq.reshape(M, K)


# -------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------

def _fmt_time(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    print("FP4 with FP16 (E5M10) scales")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    # Load X in bfloat16 to fit alongside other GPU users (then cast as needed)
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}  X dtype: {X.dtype}")
    print(f"||W||_F = {W.float().norm().item():.4e}\n")

    results = []  # (bs, approach, m, t)

    for bs in [16, 32, 128]:
        print(f"{'=' * 80}")
        print(f"Block size {bs}   (scale overhead = {16 / bs:.3f} bits/weight)")
        print(f"{'=' * 80}")

        for approach in ["Naive", "Optimal", "H-Optimal"]:
            torch.cuda.synchronize()
            t0 = time.time()
            if approach == "Naive":
                Q = fp4_fp16_naive(W, bs)
            elif approach == "Optimal":
                Q = fp4_fp16_optimal(W, bs)
            else:
                Q = fp4_fp16_hoptimal(W, X, bs)
            torch.cuda.synchronize()
            t = time.time() - t0

            m = compute_metrics(W, Q, X)
            del Q
            torch.cuda.empty_cache()

            results.append((bs, approach, m, t))
            print(f"  FP4 / FP16-scale  {approach:<10}  "
                  f"W={m['weight_error_pct']:7.4f}%  "
                  f"O={m['output_error_pct']:7.4f}%  "
                  f"||Wq-W||_F={m['weight_error']:.4e}  "
                  f"||X(Wq-W)^T||_F={m['output_error']:.4e}  "
                  f"{_fmt_time(t):>8}")
        print()

    print("=" * 80)
    print("Markdown")
    print("=" * 80 + "\n")

    print("### FP4 with FP16 (E5M10) scales\n")
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
