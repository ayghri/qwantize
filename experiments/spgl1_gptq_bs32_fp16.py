#!/usr/bin/env python
"""GPTQ-Ord + SPGL1 for FP4 at block size 32 with FP16 (E5M10) scales.

For FP16 scales the grid is too fine to enumerate (unlike FP8 E4M3), so the
per-block scale is found by iterative SSE alternation:

    q  = fp4_quantize(w, s)           # signed codebook values
    s* = <w, q> / <q, q>              # SSE-optimal given q
    s  = snap_fp16(s*)                # snap to nearest FP16

This converges in <10 iterations. H is used for block ordering (saliency)
and SPGL1 compensation, but NOT for the per-block scale search — the SSE
formula is sufficient since FP16 snapping is near-lossless.

Bit budget at bs=32, FP16 scales:
    4 b/w (FP4) + 16/32 = 0.5 b/w (scale) = 4.5 b/w total

Configurations tested:
  1. GPTQ-Seq  + Naive snap        (sequential, s = snap_fp16(max|w|/6))
  2. GPTQ-Ord  + SSE-Opt snap      (H-saliency ordering, unconstrained comp.)
  3. GPTQ-Ord  + SSE-Opt + SPGL1   (H-saliency ordering, L1-constrained comp.)

Usage:
    PYTHONPATH=. python experiments/spgl1_gptq_bs32_fp16.py
    PYTHONPATH=. python experiments/spgl1_gptq_bs32_fp16.py --skip-baseline
    PYTHONPATH=. python experiments/spgl1_gptq_bs32_fp16.py --skip-spgl1
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import (
    fp4_quantize, fp4_dequantize, Q_MAX,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

BS = 32


def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# FP16 snap + FP4 primitives
# ---------------------------------------------------------------------------

def fp16_snap(x):
    return x.to(torch.float16).to(torch.float32)


def _fp4_signed(x, s):
    """Signed FP4 codebook values for x / s."""
    return fp4_quantize(x, s.unsqueeze(-1))


def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


# ---------------------------------------------------------------------------
# Per-block snap functions
# ---------------------------------------------------------------------------

def naive_block_snap_fp16(x):
    """s = snap_fp16(max|x| / 6).  No iteration."""
    amax = x.abs().amax(-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))
    return _qd(x, s)


def sseopt_block_snap_fp16(x, max_iter=12):
    """SSE-iterative FP4 snap with FP16 scale.

    s = snap_fp16(<w, dequant(q, 1)> / ||dequant(q, 1)||^2)
      = snap_fp16(<w, q_signed> / <q_signed, q_signed>)

    Converges in <10 iterations for layer_0 weights.
    """
    amax = x.abs().amax(-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))

    for _ in range(max_iter):
        q = _fp4_signed(x, s)                       # (M, bs) signed codes
        numer = (x * q).sum(-1)
        denom = (q * q).sum(-1)
        valid = denom > 1e-12
        s_cont = torch.where(valid, numer / denom.clamp(min=1e-12), s)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    return _qd(x, s)


# ---------------------------------------------------------------------------
# Metrics (X streamed from CPU)
# ---------------------------------------------------------------------------

def compute_metrics_streamed(W, W_dq, X_cpu, batch=4096):
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    W_f, W_dq_f = W.float(), W_dq.float()
    sse = ref_sse = 0.0
    for i in range(0, X_cpu.shape[0], batch):
        Xb = X_cpu[i:i + batch].to(W.device, non_blocking=True).float()
        out_ref = Xb @ W_f.T
        out_dq = Xb @ W_dq_f.T
        diff = out_dq - out_ref
        sse += diff.pow(2).sum().item()
        ref_sse += out_ref.pow(2).sum().item()
        del Xb, out_ref, out_dq, diff
    return {
        "weight_error": w_err.item(),
        "weight_error_pct": (w_err / w_norm * 100).item(),
        "output_error": sse ** 0.5,
        "output_error_pct": (sse ** 0.5) / (ref_sse ** 0.5) * 100,
    }


# ---------------------------------------------------------------------------
# 1. GPTQ-Seq + Naive snap (sequential ordering, unconstrained H^-1 comp.)
# ---------------------------------------------------------------------------

def gptq_strided_seq_naive(W_orig, H, damp=0.01):
    M, K = W_orig.shape
    nblk = K // BS

    W_perm = W_orig.contiguous().clone()
    Hi = H.clone()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    L = torch.linalg.cholesky(Hi)
    Hi = torch.cholesky_inverse(L).contiguous()
    del L

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs, ce = j * BS, (j + 1) * BS
        rem = K - ce

        w_blk = W_perm[:, cs:ce].clone()
        q_blk = naive_block_snap_fp16(w_blk)
        Q[:, cs:ce] = q_blk

        h_diag = Hi.diagonal()[cs:ce].clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = Hi[cs:ce, ce:]
            W_perm[:, ce:] -= err @ h_cross

    return Q


# ---------------------------------------------------------------------------
# 2. GPTQ-Ord + SSE-Opt (H-saliency ordering, unconstrained compensation)
# ---------------------------------------------------------------------------

def gptq_strided_baseline(W_orig, H, H_blocks, damp=0.01):
    M, K = W_orig.shape
    nblk = K // BS

    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * BS:(j + 1) * BS]
        q = sseopt_block_snap_fp16(w_blk)
        r = w_blk - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * BS +
        torch.arange(BS, device=W_orig.device).unsqueeze(0)
    ).reshape(-1)

    W_perm = W_orig[:, col_perm].contiguous().clone()
    H_perm = H[col_perm][:, col_perm].contiguous()
    H_blocks_perm = torch.stack([
        H_perm[j * BS:(j + 1) * BS, j * BS:(j + 1) * BS] for j in range(nblk)
    ])

    Hi = H_perm.clone()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    L = torch.linalg.cholesky(Hi)
    Hi = torch.cholesky_inverse(L).contiguous()
    del L, H_blocks_perm

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs, ce = j * BS, (j + 1) * BS
        rem = K - ce

        w_blk = W_perm[:, cs:ce].clone()
        q_blk = sseopt_block_snap_fp16(w_blk)
        Q[:, cs:ce] = q_blk

        h_diag = Hi.diagonal()[cs:ce].clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = Hi[cs:ce, ce:]
            W_perm[:, ce:] -= err @ h_cross

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=col_perm.device)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# 3. GPTQ-Ord + SSE-Opt + SPGL1 (L1-constrained compensation)
# ---------------------------------------------------------------------------

def gptq_strided_spgl1(W_orig, H, H_blocks,
                       tau_frac=1.0, spgl1_iters=10, max_line_iters=10,
                       matvec_dtype=torch.float16, verbose=True):
    M, K = W_orig.shape
    nblk = K // BS
    dev = W_orig.device

    # Block ordering: H-weighted saliency with SSE-opt snap residual
    losses = torch.empty(nblk, device=dev)
    for j in range(nblk):
        w_blk = W_orig[:, j * BS:(j + 1) * BS]
        q = sseopt_block_snap_fp16(w_blk)
        r = w_blk - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * BS +
        torch.arange(BS, device=dev).unsqueeze(0)
    ).reshape(-1)

    W_orig_perm = W_orig[:, col_perm].contiguous()
    W_perm = W_orig_perm.clone()
    H_perm = H[col_perm][:, col_perm].contiguous()

    Q = torch.zeros_like(W_perm)
    Delta_eff = torch.zeros_like(W_perm)

    if verbose:
        print(f"  spgl1_iters={spgl1_iters}  tau_frac={tau_frac}", flush=True)
    t_snap = t_grad = t_spgl1 = t_apply = 0.0
    total_line_iters = 0

    for j in range(nblk):
        cs, ce = j * BS, (j + 1) * BS
        rem_size = K - ce

        # Snap block j via SSE-iterative FP16 scale
        t0 = time.time()
        w_blk = W_perm[:, cs:ce].clone()
        q_blk = sseopt_block_snap_fp16(w_blk)
        Q[:, cs:ce] = q_blk
        Delta_eff[:, cs:ce] = q_blk - W_orig_perm[:, cs:ce]
        torch.cuda.synchronize()
        t_snap += time.time() - t0

        if rem_size == 0:
            break

        # SPGL1 LASSO compensation over remaining columns
        t0 = time.time()
        H_red = H_perm[ce:, ce:]
        ATb = -(Delta_eff @ H_perm[:, ce:])
        Delta_H = Delta_eff @ H_perm
        b_norm_sq = (Delta_eff * Delta_H).sum(-1).clamp(min=0)
        torch.cuda.synchronize()
        t_grad += time.time() - t0

        diag_scale = H_red.diagonal().mean().clamp(min=1e-12)
        tau_vec = tau_frac * ATb.abs().sum(-1) / diag_scale

        t0 = time.time()
        delta, info = spgl1_lasso_reduced_batched(
            H_red, ATb, b_norm_sq, tau=tau_vec,
            max_iter=spgl1_iters, max_line_iters=max_line_iters,
            verbose=False, matvec_dtype=matvec_dtype,
        )
        torch.cuda.synchronize()
        t_spgl1 += time.time() - t0
        total_line_iters += info["n_line_iters"]

        t0 = time.time()
        W_perm[:, ce:] += delta
        Delta_eff[:, ce:] += delta
        torch.cuda.synchronize()
        t_apply += time.time() - t0

        if verbose and (j < 3 or j % 30 == 0 or j == nblk - 1):
            atb_med = ATb.abs().sum(-1).median().item()
            delta_med = delta.abs().sum(-1).median().item()
            del ATb, delta, Delta_H, b_norm_sq
            print(f"    blk {j:4d}/{nblk}  rem={rem_size:>5}  "
                  f"||ATb||_1 med={atb_med:.3e}  "
                  f"||δ||_1 med={delta_med:.3e}  "
                  f"snap={_fmt(t_snap)}  grad={_fmt(t_grad)}  "
                  f"spgl1={_fmt(t_spgl1)}", flush=True)
        else:
            del ATb, delta, Delta_H, b_norm_sq

    if verbose:
        print(f"\n  total — snap:{_fmt(t_snap)}  grad:{_fmt(t_grad)}  "
              f"spgl1:{_fmt(t_spgl1)}  apply:{_fmt(t_apply)}  "
              f"line_iters:{total_line_iters}", flush=True)

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau-frac", type=float, default=1.0)
    p.add_argument("--spgl1-iters", type=int, default=10)
    p.add_argument("--matvec-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp16")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip GPTQ-Seq+Naive and GPTQ-Ord+SSE-Opt")
    p.add_argument("--max-line-iters", type=int, default=10,
                   help="line search backtrack cap per SPGL1 outer iteration")
    p.add_argument("--skip-spgl1", action="store_true",
                   help="run only baselines, no SPGL1")
    args = p.parse_args()

    matvec_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    matvec_dtype = matvec_map[args.matvec_dtype]

    print("GPTQ-Ord + SSE-Opt + SPGL1   |   FP4  bs=32  FP16 (E5M10) scales")
    print(f"  scale b/w={16/BS:.3f}  total b/w={4 + 16/BS:.3f}  "
          f"tau_frac={args.tau_frac}  spgl1_iters={args.spgl1_iters}  "
          f"max_line_iters={args.max_line_iters}  matvec={args.matvec_dtype}",
          flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    assert K % BS == 0, f"K={K} not divisible by BS={BS}"
    nblk = K // BS
    print(f"W: {M}x{K}  X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]} ({X_cpu.dtype})",
          flush=True)
    print(f"||W||_F = {W.norm().item():.4e}  nblk={nblk}", flush=True)

    print("Building Hessian (CPU -> GPU streaming)...", end=" ", flush=True)
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    chunk = 4096
    for t0 in range(0, X_cpu.shape[0], chunk):
        Xc = X_cpu[t0:t0 + chunk].to(DEVICE, non_blocking=True).float()
        H.addmm_(Xc.T, Xc)
        del Xc
    H /= X_cpu.shape[0]
    H_blocks = torch.stack([
        H[j * BS:(j + 1) * BS, j * BS:(j + 1) * BS] for j in range(nblk)
    ])
    print("done", flush=True)

    results = []

    if not args.skip_baseline:
        print("\n[seq] GPTQ-Seq + Naive snap...", flush=True)
        t0 = time.time()
        Q_seq = gptq_strided_seq_naive(W, H)
        t_seq = time.time() - t0
        m_seq = compute_metrics_streamed(W, Q_seq, X_cpu)
        del Q_seq; torch.cuda.empty_cache()
        results.append(("GPTQ-Seq+Naive", m_seq, t_seq))
        print(f"  W%={m_seq['weight_error_pct']:.4f}  "
              f"O%={m_seq['output_error_pct']:.4f}  time={_fmt(t_seq)}", flush=True)

        print("\n[base] GPTQ-Ord + SSE-Opt (unconstrained)...", flush=True)
        t0 = time.time()
        Q_base = gptq_strided_baseline(W, H, H_blocks)
        t_base = time.time() - t0
        m_base = compute_metrics_streamed(W, Q_base, X_cpu)
        del Q_base; torch.cuda.empty_cache()
        results.append(("GPTQ-Ord+SSE-Opt", m_base, t_base))
        print(f"  W%={m_base['weight_error_pct']:.4f}  "
              f"O%={m_base['output_error_pct']:.4f}  time={_fmt(t_base)}", flush=True)

    if not args.skip_spgl1:
        print(f"\n[spgl1] GPTQ-Ord + SSE-Opt + SPGL1...", flush=True)
        t0 = time.time()
        Q_spgl1 = gptq_strided_spgl1(
            W, H, H_blocks,
            tau_frac=args.tau_frac,
            spgl1_iters=args.spgl1_iters,
            max_line_iters=args.max_line_iters,
            matvec_dtype=matvec_dtype,
            verbose=True,
        )
        t_spgl1_total = time.time() - t0
        m_spgl1 = compute_metrics_streamed(W, Q_spgl1, X_cpu)
        del Q_spgl1; torch.cuda.empty_cache()
        results.append(("GPTQ-Ord+SSE-Opt+SPGL1", m_spgl1, t_spgl1_total))
        print(f"  W%={m_spgl1['weight_error_pct']:.4f}  "
              f"O%={m_spgl1['output_error_pct']:.4f}  time={_fmt(t_spgl1_total)}",
              flush=True)

    print("\n" + "=" * 72)
    print(f"{'Config':<28}  {'W%':>8}  {'O%':>8}  {'Time':>8}")
    print("-" * 72)
    for name, m, t in results:
        print(f"  {name:<26}  {m['weight_error_pct']:>8.4f}  "
              f"{m['output_error_pct']:>8.4f}  {_fmt(t):>8}")
    print("=" * 72)

    print("\n### FP4 bs=32 FP16 scales — GPTQ variants\n")
    print("| Config | W% | O% | Time |")
    print("|:--|:--:|:--:|--:|")
    for name, m, t in results:
        print(f"| {name} | {m['weight_error_pct']:.2f}% "
              f"| {m['output_error_pct']:.2f}% | {_fmt(t)} |")


if __name__ == "__main__":
    main()
