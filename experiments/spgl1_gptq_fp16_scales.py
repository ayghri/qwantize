#!/usr/bin/env python
"""GPTQ-Ord + H-Opt + SPGL1 compensation, with FP16 (E5M10) scales.

Same algorithm as experiments/spgl1_gptq_compensation.py, but the per-block
scale is FP16 (1024 mantissa codes / binade) instead of an FP8 grid. Since
FP16 is fine enough that the snapped continuous optimum is essentially the
true minimum, the per-block H-optimal scale is found by iterative
alternation:

    q  = fp4_quantize(x, s)
    s* = <x, H q> / <q, H q>
    s  = fp16_snap(s*)

repeated until stable. No grid search.

Block size 128 is the headline configuration:
    bs=16, FP8 scale  -> 0.500 b/w  (NVFP4 baseline)
    bs=32, FP8 scale  -> 0.250 b/w
    bs=128, FP16 scale -> 0.125 b/w  (this script)

Usage:
    PYTHONPATH=. python \
        experiments/spgl1_gptq_fp16_scales.py --bs 128
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import (
    fp4_quantize, fp4_dequantize, compute_block_sse, Q_MAX,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def compute_metrics_streamed(W, W_dq, X_cpu, batch=4096):
    """Like qwantize.metrics.compute_metrics but pulls X chunks from CPU."""
    metrics = {}
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    metrics["weight_error"] = w_err.item()
    metrics["weight_error_pct"] = (w_err / w_norm * 100).item()

    if X_cpu is not None:
        W_f = W.float()
        W_dq_f = W_dq.float()
        sse = ref_sse = 0.0
        T = X_cpu.shape[0]
        for i in range(0, T, batch):
            Xb = X_cpu[i:i + batch].to(W.device, non_blocking=True).float()
            out_ref = Xb @ W_f.T
            out_dq = Xb @ W_dq_f.T
            diff = out_dq - out_ref
            sse += diff.pow(2).sum().item()
            ref_sse += out_ref.pow(2).sum().item()
            del Xb, out_ref, out_dq, diff
        metrics["output_error"] = sse ** 0.5
        metrics["output_error_pct"] = (sse ** 0.5) / (ref_sse ** 0.5) * 100
    return metrics


# ---------------------------------------------------------------------------
# FP16 scale snap + helpers
# ---------------------------------------------------------------------------

def fp16_snap(x):
    return x.to(torch.float16).to(torch.float32)


def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def _fp4_signed(x, s):
    return fp4_quantize(x, s.unsqueeze(-1))


# ---------------------------------------------------------------------------
# Per-block H-optimal snap, FP16 scales, iterative
# ---------------------------------------------------------------------------

def naive_block_snap_fp16(x):
    """Naive per-row snap: s = fp16(max|x| / 6).  No Hessian, no iteration."""
    amax = x.abs().amax(-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))
    return _qd(x, s)


def hopt_block_snap_fp16(x, H_blk, bs, max_iter=12):
    """Per-row H-optimal FP4 snap with FP16 scale, alternating refinement.

    Args:
        x: (M, bs) block of weights for this column block.
        H_blk: (bs, bs) block Hessian.
        max_iter: alternation cap.

    Returns:
        (M, bs) dequantized block.
    """
    amax = x.abs().amax(-1)
    s = fp16_snap((amax / Q_MAX).clamp(min=1e-12))    # (M,)

    for _ in range(max_iter):
        q = _fp4_signed(x, s)                          # (M, bs) signed codebook
        Hq = q @ H_blk                                 # (M, bs)
        numer = (x * Hq).sum(-1)
        denom = (q * Hq).sum(-1)
        valid = (numer > 0) & (denom > 1e-12)
        s_cont = torch.where(valid, numer / denom.clamp(min=1e-12), s)
        s_new = fp16_snap(s_cont.clamp(min=1e-12))
        if torch.equal(s_new, s):
            break
        s = s_new

    return _qd(x, s)


# ---------------------------------------------------------------------------
# Plain GPTQ-Seq + Naive snap (no ordering, no H-optimal), FP16 scales
# ---------------------------------------------------------------------------

def gptq_strided_seq_naive(W_orig, H, bs, damp=0.01):
    """Sequential GPTQ with naive snap (s = fp16(amax/6)). Standard H_inv
    error propagation across all not-yet-quantized columns."""
    M, K = W_orig.shape
    nblk = K // bs

    W_perm = W_orig.contiguous().clone()
    Hi = H.clone()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    L = torch.linalg.cholesky(Hi)
    Hi = torch.cholesky_inverse(L).contiguous()
    del L

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem = K - ce

        w_blk = W_perm[:, cs:ce].clone()
        q_blk = naive_block_snap_fp16(w_blk)
        Q[:, cs:ce] = q_blk

        h_diag = torch.as_strided(
            Hi, size=(bs,), stride=(K + 1,),
            storage_offset=cs * K + cs,
        ).clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = torch.as_strided(
                Hi, size=(bs, rem), stride=(K, 1),
                storage_offset=cs * K + ce,
            )
            w_rem = torch.as_strided(
                W_perm, size=(M, rem), stride=(K, 1),
                storage_offset=ce,
            )
            w_rem.sub_(err @ h_cross)

    return Q


# ---------------------------------------------------------------------------
# Baseline GPTQ-Ord (unconstrained compensation), FP16 scales
# ---------------------------------------------------------------------------

def gptq_strided_baseline(W_orig, H, bs, H_blocks, damp=0.01):
    M, K = W_orig.shape
    nblk = K // bs

    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap_fp16(w_blk, H_blocks[j], bs)
        r = w_blk - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * bs +
        torch.arange(bs, device=W_orig.device).unsqueeze(0)
    ).reshape(-1)

    W_perm = W_orig[:, col_perm].contiguous().clone()
    H_perm = H[col_perm][:, col_perm].contiguous()
    H_blocks_perm = torch.stack([
        H_perm[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
    ])

    Hi = H_perm.clone()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    L = torch.linalg.cholesky(Hi)
    Hi = torch.cholesky_inverse(L).contiguous()
    del L

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem = K - ce

        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap_fp16(w_blk, H_blocks_perm[j], bs)
        Q[:, cs:ce] = q_blk

        h_diag = torch.as_strided(
            Hi, size=(bs,), stride=(K + 1,),
            storage_offset=cs * K + cs,
        ).clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = torch.as_strided(
                Hi, size=(bs, rem), stride=(K, 1),
                storage_offset=cs * K + ce,
            )
            w_rem = torch.as_strided(
                W_perm, size=(M, rem), stride=(K, 1),
                storage_offset=ce,
            )
            w_rem.sub_(err @ h_cross)

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=col_perm.device)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# Block saliencies, FP16 scales
# ---------------------------------------------------------------------------

def _compute_block_saliencies(W_orig, H, bs, H_blocks,
                              ordering="block_h", damp=0.01):
    M, K = W_orig.shape
    nblk = K // bs

    if ordering == "block_h":
        H_eff = H_blocks
    elif ordering == "strict_obs":
        Hi = H.clone()
        dmu = damp * Hi.diagonal().mean()
        Hi.diagonal().add_(dmu)
        L = torch.linalg.cholesky(Hi)
        Hi = torch.cholesky_inverse(L)
        del L
        H_inv_blocks = torch.stack([
            Hi[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
        ])
        H_eff = torch.linalg.inv(H_inv_blocks)
    else:
        raise ValueError(f"unknown ordering '{ordering}'")

    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap_fp16(w_blk, H_blocks[j], bs)
        r = w_blk - q
        losses[j] = (r * (r @ H_eff[j])).sum()
    return losses


# ---------------------------------------------------------------------------
# GPTQ-Ord + H-Opt + SPGL1 compensation, FP16 scales
# ---------------------------------------------------------------------------

def gptq_strided_spgl1(W_orig, H, bs, H_blocks,
                       tau_frac=1.0, spgl1_iters=20,
                       ordering="block_h", matvec_dtype=None, verbose=True):
    M, K = W_orig.shape
    nblk = K // bs
    dev = W_orig.device

    if verbose:
        print(f"  ordering={ordering!r}", flush=True)
    losses = _compute_block_saliencies(W_orig, H, bs, H_blocks,
                                       ordering=ordering)
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * bs +
        torch.arange(bs, device=dev).unsqueeze(0)
    ).reshape(-1)

    W_orig_perm = W_orig[:, col_perm].contiguous()
    W_perm = W_orig_perm.clone()
    H_perm = H[col_perm][:, col_perm].contiguous()
    H_blocks_perm = torch.stack([
        H_perm[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
    ])

    Q = torch.zeros_like(W_perm)
    Delta_eff = torch.zeros_like(W_perm)

    if verbose:
        print(f"  spgl1_iters={spgl1_iters}  tau_frac={tau_frac}", flush=True)
    t_snap = t_spgl1 = t_grad = t_apply = 0.0
    total_line_iters = 0

    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem_size = K - ce

        # Step 1: snap block j (H-opt iterative FP16)
        t0 = time.time()
        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap_fp16(w_blk, H_blocks_perm[j], bs)
        Q[:, cs:ce] = q_blk
        Delta_eff[:, cs:ce] = q_blk - W_orig_perm[:, cs:ce]
        torch.cuda.synchronize()
        t_snap += time.time() - t0

        if rem_size == 0:
            break

        # Step 2: SPGL1 LASSO compensation across remaining columns
        t0 = time.time()
        H_red = H_perm[ce:, ce:]
        ATb = -(Delta_eff @ H_perm[:, ce:])
        Delta_H = Delta_eff @ H_perm
        b_norm_sq = (Delta_eff * Delta_H).sum(-1).clamp(min=0)
        torch.cuda.synchronize()
        t_grad += time.time() - t0

        diag_scale = H_red.diagonal().mean().clamp(min=1e-12)
        ref_l1 = ATb.abs().sum(-1) / diag_scale
        tau_vec = tau_frac * ref_l1

        t0 = time.time()
        delta, info = spgl1_lasso_reduced_batched(
            H_red, ATb, b_norm_sq, tau=tau_vec,
            max_iter=spgl1_iters, verbose=False,
            matvec_dtype=matvec_dtype,
        )
        torch.cuda.synchronize()
        t_spgl1 += time.time() - t0
        total_line_iters += info["n_line_iters"]

        t0 = time.time()
        W_perm[:, ce:] += delta
        Delta_eff[:, ce:] += delta
        torch.cuda.synchronize()
        t_apply += time.time() - t0

        if verbose and (j < 3 or j % 10 == 0 or j == nblk - 1):
            print(f"    block {j:4d}/{nblk}  K_rem={rem_size:>5}  "
                  f"||ATb||_1 med={ATb.abs().sum(-1).median():.3e}  "
                  f"||δ||_1 med={delta.abs().sum(-1).median():.3e}  "
                  f"snap={_fmt(t_snap):>6}  grad={_fmt(t_grad):>6}  "
                  f"spgl1={_fmt(t_spgl1):>6}", flush=True)

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
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--ordering", choices=["block_h", "strict_obs"],
                   default="block_h")
    p.add_argument("--matvec-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp16")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-spgl1", action="store_true",
                   help="only run baselines (GPTQ-Seq+Naive, GPTQ-Ord+H-Opt)")
    args = p.parse_args()

    print("GPTQ-Ord + H-Opt + SPGL1 compensation   |   FP16 (E5M10) scales")
    print(f"  bs={args.bs}  scale b/w = {16/args.bs:.3f}  "
          f"total b/w = {4 + 16/args.bs:.3f}", flush=True)
    print(f"  tau_frac={args.tau_frac}  spgl1_iters={args.spgl1_iters}  "
          f"ordering={args.ordering}  matvec={args.matvec_dtype}", flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    # Load X on CPU and stream chunks to DEVICE — fits when GPU memory is tight.
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K} ({W.dtype})  "
          f"X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]} ({X_cpu.dtype})",
          flush=True)
    print(f"||W||_F = {W.norm().item():.4e}", flush=True)

    bs = args.bs
    nblk = K // bs
    assert K % bs == 0, f"K={K} not divisible by bs={bs}"

    print("Building Hessian (chunked, CPU->GPU stream)...", end=" ", flush=True)
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    chunk = 4096
    for t0 in range(0, X_cpu.shape[0], chunk):
        Xc = X_cpu[t0:t0 + chunk].to(DEVICE, non_blocking=True).float()
        H.addmm_(Xc.T, Xc)
        del Xc
    H /= X_cpu.shape[0]
    print("done", flush=True)

    H_blocks = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
    ])

    if not args.skip_baseline:
        print("\n[plain] GPTQ-Seq + Naive snap (no ordering, no H-opt)...",
              flush=True)
        t0 = time.time()
        Q_seq = gptq_strided_seq_naive(W, H, bs)
        t_seq = time.time() - t0
        m_seq = compute_metrics_streamed(W, Q_seq, X_cpu)
        print(f"  W%={m_seq['weight_error_pct']:.4f}  "
              f"O%={m_seq['output_error_pct']:.4f}  time={_fmt(t_seq)}",
              flush=True)
        print(f"  ||Wq-W||_F={m_seq['weight_error']:.4e}  "
              f"||X(Wq-W)^T||_F={m_seq['output_error']:.4e}", flush=True)
        del Q_seq; torch.cuda.empty_cache()

        print("\n[baseline] GPTQ-Ord + H-Opt (unconstrained, FP16 scales)...",
              flush=True)
        t0 = time.time()
        Q_base = gptq_strided_baseline(W, H, bs, H_blocks)
        t_base = time.time() - t0
        m_base = compute_metrics_streamed(W, Q_base, X_cpu)
        print(f"  W%={m_base['weight_error_pct']:.4f}  "
              f"O%={m_base['output_error_pct']:.4f}  time={_fmt(t_base)}",
              flush=True)
        print(f"  ||Wq-W||_F={m_base['weight_error']:.4e}  "
              f"||X(Wq-W)^T||_F={m_base['output_error']:.4e}", flush=True)
        del Q_base; torch.cuda.empty_cache()

    if args.skip_spgl1:
        return

    print(f"\n[new] GPTQ-Ord + SPGL1 compensation (FP16 scales)...", flush=True)
    matvec_dtype_map = {"fp32": None,
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16}
    t0 = time.time()
    Q_spgl1 = gptq_strided_spgl1(
        W, H, bs, H_blocks,
        tau_frac=args.tau_frac, spgl1_iters=args.spgl1_iters,
        ordering=args.ordering,
        matvec_dtype=matvec_dtype_map[args.matvec_dtype], verbose=True,
    )
    t_spgl1_total = time.time() - t0
    m_spgl1 = compute_metrics_streamed(W, Q_spgl1, X_cpu)
    print(f"  W%={m_spgl1['weight_error_pct']:.4f}  "
          f"O%={m_spgl1['output_error_pct']:.4f}  time={_fmt(t_spgl1_total)}",
          flush=True)
    print(f"  ||Wq-W||_F={m_spgl1['weight_error']:.4e}  "
          f"||X(Wq-W)^T||_F={m_spgl1['output_error']:.4e}", flush=True)

    if not args.skip_baseline:
        delta = m_spgl1["output_error_pct"] - m_base["output_error_pct"]
        print(f"\n  ΔO = {delta:+.4f}pp  ({'BETTER' if delta < 0 else 'WORSE'})")


if __name__ == "__main__":
    main()
