#!/usr/bin/env python
"""SPGL1 → FP4 snap quantization pipeline.

Pipeline (per row, batched on GPU):
  1. Q = H-optimal FP4 snap of W_0           (per-block FP8 E4M3 scale chosen
                                              to minimize ||X(W-W_0)|| Frobenius
                                              within each FP4 block)
  2. d_init = w_0 - q   (deviation from grid)
  3. LASSO via SPGL1:   d* = argmin ||X d - X d_init||_2  s.t.  ||d||_1 <= tau
     where tau = tau_frac * ||d_init||_1 per row.
  4. w_inter = q + d*   (intermediate point on the Pareto curve)
  5. Re-snap w_inter to FP4 via H-optimal — different basin may give
     a better grid choice than naive(w_inter) or h-optimal(w_0).
  6. Report output error, weight error vs baseline H-Optimal.

Short schedule: one tau value, max_iter=50, single sweep. Just to
check if there's any signal that SPGL1 + snap beats H-Optimal alone.

Usage:
    PYTHONPATH=. python experiments/spgl1_quant_pipeline.py
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import nvfp4_optimal_hessian
from qwantize.spgl1 import spgl1_lasso_batched, make_dense_op
from qwantize.metrics import compute_metrics


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def run_spgl1_then_snap(W, X, Q, tau_frac, max_iter, row_chunk):
    """Pipeline: SPGL1 LASSO per row chunk, then re-snap to FP4.

    Args:
        W, X, Q: (M, K), (T, K), (M, K).
        tau_frac: per-row tau = tau_frac * ||w_0 - q||_1.
        max_iter: SPGL1 iter cap.
        row_chunk: rows per GPU batch.

    Returns:
        W_inter: (M, K) intermediate (NOT yet snapped to grid).
    """
    M, K = W.shape
    matvec, rmatvec = make_dense_op(X)

    W_inter = torch.empty_like(W)
    total_iters = 0
    total_line_iters = 0

    for r0 in range(0, M, row_chunk):
        r1 = min(r0 + row_chunk, M)
        W_chunk = W[r0:r1]                                # (b, K)
        Q_chunk = Q[r0:r1]
        d_init = W_chunk - Q_chunk                         # (b, K)
        l1_init = d_init.abs().sum(dim=-1)                 # (b,)
        tau_vec = tau_frac * l1_init                       # (b,)

        # b = X @ d_init for each row in the chunk
        b_chunk = d_init @ X.T                              # (b, T)

        d_star, _, info = spgl1_lasso_batched(
            matvec, rmatvec, b_chunk, tau=tau_vec, n=K,
            max_iter=max_iter, verbose=False,
        )
        total_iters += info.get("exit_iter", max_iter) or max_iter
        total_line_iters += info["n_line_iters"]

        W_inter[r0:r1] = Q_chunk + d_star

    print(f"    SPGL1: avg iters/chunk={total_iters/max(1, (M+row_chunk-1)//row_chunk):.1f}  "
          f"total line iters={total_line_iters}", flush=True)
    return W_inter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau-frac", type=float, default=0.5,
                   help="per-row tau = tau_frac * ||w0 - q||_1")
    p.add_argument("--max-iter", type=int, default=50,
                   help="SPGL1 inner iteration cap")
    p.add_argument("--row-chunk", type=int, default=64,
                   help="rows per SPGL1 batch on GPU")
    p.add_argument("--bs", type=int, default=16,
                   help="FP4 block size")
    args = p.parse_args()

    print("SPGL1 → FP4 snap quantization pipeline")
    print(f"  device={DEVICE}  tau_frac={args.tau_frac}  "
          f"max_iter={args.max_iter}  row_chunk={args.row_chunk}\n", flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}\n", flush=True)

    bs = args.bs
    ncb = K // bs

    # ------------------------------------------------------------------
    # Baseline 1: naive snap
    # ------------------------------------------------------------------
    print("[baseline] Naive FP4 snap of W_0...", flush=True)
    from qwantize.nvfp4.reference import nvfp4_naive
    t0 = time.time()
    _, _, W_naive = nvfp4_naive(W.view(M, ncb, bs), return_dequant=True)
    W_naive = W_naive.view(M, K)
    t_naive = time.time() - t0
    m_naive = compute_metrics(W, W_naive, X)
    print(f"  naive             W%={m_naive['weight_error_pct']:.4f}  "
          f"O%={m_naive['output_error_pct']:.4f}  time={_fmt(t_naive)}\n",
          flush=True)

    # ------------------------------------------------------------------
    # Baseline 2: H-optimal snap of W_0 (= Q used by SPGL1)
    # ------------------------------------------------------------------
    print("[baseline] H-optimal FP4 snap of W_0 (= SPGL1 target Q)...",
          flush=True)
    t0 = time.time()
    _, _, Q_blk = nvfp4_optimal_hessian(
        W.view(M, ncb, bs), return_dequant=True, X=X,
    )
    Q = Q_blk.view(M, K)
    t_hopt = time.time() - t0
    m_hopt = compute_metrics(W, Q, X)
    print(f"  H-optimal         W%={m_hopt['weight_error_pct']:.4f}  "
          f"O%={m_hopt['output_error_pct']:.4f}  time={_fmt(t_hopt)}\n",
          flush=True)

    # ------------------------------------------------------------------
    # SPGL1 step
    # ------------------------------------------------------------------
    print(f"[spgl1] Running batched LASSO (tau_frac={args.tau_frac})...",
          flush=True)
    t0 = time.time()
    W_inter = run_spgl1_then_snap(
        W, X, Q, args.tau_frac, args.max_iter, args.row_chunk,
    )
    t_spgl1 = time.time() - t0
    print(f"  SPGL1 wall time: {_fmt(t_spgl1)}\n", flush=True)

    # Report the intermediate (off-grid) point — useful diagnostic
    m_inter = compute_metrics(W, W_inter, X)
    print(f"[diag] Off-grid intermediate (w_inter = Q + d*):")
    print(f"  W%={m_inter['weight_error_pct']:.4f}  "
          f"O%={m_inter['output_error_pct']:.4f}  "
          f"(this is *not* a quantization — w_inter is not on the FP4 grid)\n",
          flush=True)

    # ------------------------------------------------------------------
    # Re-snap W_inter to FP4 grid
    # ------------------------------------------------------------------
    print("[snap] Re-snap W_inter using H-optimal...", flush=True)
    t0 = time.time()
    _, _, W_resnap_blk = nvfp4_optimal_hessian(
        W_inter.view(M, ncb, bs), return_dequant=True, X=X,
    )
    W_resnap_hopt = W_resnap_blk.view(M, K)
    t_resnap_hopt = time.time() - t0
    m_resnap_hopt = compute_metrics(W, W_resnap_hopt, X)
    print(f"  spgl1 → H-opt snap  W%={m_resnap_hopt['weight_error_pct']:.4f}  "
          f"O%={m_resnap_hopt['output_error_pct']:.4f}  "
          f"time={_fmt(t_resnap_hopt)}", flush=True)

    print("\n[snap] Re-snap W_inter using naive...", flush=True)
    t0 = time.time()
    _, _, W_resnap_naive_blk = nvfp4_naive(
        W_inter.view(M, ncb, bs), return_dequant=True,
    )
    W_resnap_naive = W_resnap_naive_blk.view(M, K)
    t_resnap_naive = time.time() - t0
    m_resnap_naive = compute_metrics(W, W_resnap_naive, X)
    print(f"  spgl1 → naive snap  W%={m_resnap_naive['weight_error_pct']:.4f}  "
          f"O%={m_resnap_naive['output_error_pct']:.4f}  "
          f"time={_fmt(t_resnap_naive)}\n", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY (layer_0 down_proj, BS=16, FP4 + E4M3 scales)")
    print("=" * 72)
    print(f"  {'method':<30}  {'W err':>8}  {'O err':>8}  {'time':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Naive (baseline)':<30}  "
          f"{m_naive['weight_error_pct']:7.4f}%  "
          f"{m_naive['output_error_pct']:7.4f}%  {_fmt(t_naive):>8}")
    print(f"  {'H-Optimal (baseline)':<30}  "
          f"{m_hopt['weight_error_pct']:7.4f}%  "
          f"{m_hopt['output_error_pct']:7.4f}%  {_fmt(t_hopt):>8}")
    print(f"  {'SPGL1 intermediate (off-grid)':<30}  "
          f"{m_inter['weight_error_pct']:7.4f}%  "
          f"{m_inter['output_error_pct']:7.4f}%  {_fmt(t_spgl1):>8}")
    print(f"  {'SPGL1 → H-opt snap':<30}  "
          f"{m_resnap_hopt['weight_error_pct']:7.4f}%  "
          f"{m_resnap_hopt['output_error_pct']:7.4f}%  "
          f"{_fmt(t_spgl1 + t_resnap_hopt):>8}")
    print(f"  {'SPGL1 → naive snap':<30}  "
          f"{m_resnap_naive['weight_error_pct']:7.4f}%  "
          f"{m_resnap_naive['output_error_pct']:7.4f}%  "
          f"{_fmt(t_spgl1 + t_resnap_naive):>8}")
    print(f"  {'GPTQ-Ord + H-Opt (docs)':<30}  "
          f"{'11.53':>7}%  {'4.21':>7}%  {'15.5s':>8}")

    delta = m_resnap_hopt['output_error_pct'] - m_hopt['output_error_pct']
    print(f"\n  SPGL1+H-opt vs H-opt baseline: ΔO = {delta:+.4f}pp"
          f"  ({'BETTER' if delta < 0 else 'WORSE'})")


if __name__ == "__main__":
    main()
