#!/usr/bin/env python
"""GPTQ-Ord with SPGL1 compensation across remaining columns.

After snapping each block in descending-loss order, instead of GPTQ's
unconstrained H_inv error propagation, we solve

    min_{delta}  ||X (Δ_eff + delta · 1[rem])^T||_2   s.t.   ||delta||_1 <= tau

where Δ_eff is the cumulative effective shift (q-w_orig on locked blocks,
running compensation on rem). The minimizer delta is the L1-CONSTRAINED
compensation that the still-active columns can absorb.

Reduced (Gram) form used throughout — never materializes the (M, T)
output residual:
    A    = X[:, rem]
    A^T A = H[rem, rem]                       (slice of precomputed H)
    A^T b = -(Δ_eff @ H[:, rem])               (batched M x K_rem)
    b_norm_sq = Δ_eff^T H Δ_eff per row

Mode `gptq-ord-spgl1` is the new variant.
Mode `gptq-ord-baseline` runs the standard GPTQ-Ord (unconstrained) for
direct comparison.

Usage:
    PYTHONPATH=. python experiments/spgl1_gptq_compensation.py
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap, build_fp8_e4m3_scales,
    fp4_quantize, fp4_dequantize, compute_block_sse,
    Q_MAX, D_0,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# Per-block H-optimal FP4 snap
# ---------------------------------------------------------------------------

def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def _base_nvfp4(x):
    return _fp8_e4m3_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))


def hopt_block_snap(x, H_blk, all_scales, bs):
    """Per-row H-optimal FP4 snap, (M, bs) -> (M, bs) dequantized."""
    s0 = _base_nvfp4(x)
    E0_sse = compute_block_sse(x, s0)
    amax = x.abs().amax(-1)

    r0 = x - _qd(x, s0)
    E0_H = (r0 * (r0 @ H_blk)).sum(-1)
    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / D_0

    act = ~noise
    if act.any():
        xa, smn, smx = x[act], s_min[act], s_max[act]
        bE, bS = best_E[act].clone(), best_s[act].clone()
        dev = x.device
        for sv in all_scales:
            sf = sv.item()
            ok = (sf >= smn) & (sf <= smx)
            if not ok.any():
                continue
            clip = (xa.abs() - Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.full((xa.shape[0],), sf, device=dev)
            r = xa - _qd(xa, sf_t)
            EH = (r * (r @ H_blk)).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_s[act] = bS

    return _qd(x, best_s)


# ---------------------------------------------------------------------------
# GPTQ-Ord with SPGL1-LASSO compensation
# ---------------------------------------------------------------------------

def gptq_strided_baseline(W_orig, H, bs, all_scales, H_blocks, damp=0.01):
    """Plain GPTQ-Ord + H-opt block snap (baseline reference)."""
    M, K = W_orig.shape
    nblk = K // bs

    # Order blocks by descending H-opt loss
    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap(w_blk, H_blocks[j], all_scales, bs)
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

    # Cholesky inverse of damped H (standard GPTQ)
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
        q_blk = hopt_block_snap(w_blk, H_blocks_perm[j], all_scales, bs)
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


def _compute_block_saliencies(W_orig, H, bs, all_scales, H_blocks,
                               ordering="block_h", damp=0.01):
    """Per-block saliency for ordering.

    ordering="block_h"   : score = r^T H_block r   (cheap, no H_inv)
    ordering="strict_obs": score = r^T ([H_inv]_block)^-1 r  (uses H_inv)
    """
    M, K = W_orig.shape
    nblk = K // bs

    if ordering == "block_h":
        H_eff = H_blocks                                  # (nblk, bs, bs)

    elif ordering == "strict_obs":
        # Single Cholesky inversion of H, with mild damping for stability
        Hi = H.clone()
        dmu = damp * Hi.diagonal().mean()
        Hi.diagonal().add_(dmu)
        L = torch.linalg.cholesky(Hi)
        Hi = torch.cholesky_inverse(L)
        del L
        # Block-diagonal of H_inv
        H_inv_blocks = torch.stack([
            Hi[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
        ])                                                # (nblk, bs, bs)
        # Strict-OBS effective Hessian per block = invert each (bs, bs) slab
        H_eff = torch.linalg.inv(H_inv_blocks)            # (nblk, bs, bs)
    else:
        raise ValueError(f"unknown ordering '{ordering}'")

    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap(w_blk, H_blocks[j], all_scales, bs)
        r = w_blk - q
        losses[j] = (r * (r @ H_eff[j])).sum()
    return losses


def gptq_strided_spgl1(W_orig, H, bs, all_scales, H_blocks,
                       tau_frac=1.0, spgl1_iters=20,
                       ordering="block_h", matvec_dtype=None, verbose=True):
    """GPTQ-Ord with SPGL1 LASSO compensation across remaining columns."""
    M, K = W_orig.shape
    nblk = K // bs
    dev = W_orig.device

    # Block-saliency ordering
    if verbose:
        print(f"  ordering={ordering!r}", flush=True)
    losses = _compute_block_saliencies(W_orig, H, bs, all_scales, H_blocks,
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
    Delta_eff = torch.zeros_like(W_perm)             # = effective W - W_orig

    if verbose:
        print(f"  spgl1_iters={spgl1_iters}  tau_frac={tau_frac}", flush=True)
    t_snap = 0.0
    t_spgl1 = 0.0
    t_grad = 0.0
    t_apply = 0.0
    total_line_iters = 0

    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem_size = K - ce

        # ---- Step 1: snap block j with H-opt ----
        t0 = time.time()
        w_blk = W_perm[:, cs:ce].clone()                # current iterate (post-comp)
        q_blk = hopt_block_snap(w_blk, H_blocks_perm[j], all_scales, bs)
        Q[:, cs:ce] = q_blk
        Delta_eff[:, cs:ce] = q_blk - W_orig_perm[:, cs:ce]
        torch.cuda.synchronize()
        t_snap += time.time() - t0

        if rem_size == 0:
            break

        # ---- Step 2: SPGL1 LASSO compensation across remaining cols ----
        t0 = time.time()
        # ATb = -(Δ_eff @ H_perm[:, ce:])       (M, K_rem)
        # b_norm_sq = Δ_eff^T H Δ_eff           (M,)
        H_red = H_perm[ce:, ce:]                          # view (K_rem, K_rem)
        ATb = -(Delta_eff @ H_perm[:, ce:])               # (M, K_rem)
        Delta_H = Delta_eff @ H_perm                       # (M, K), only needed for b_norm_sq
        b_norm_sq = (Delta_eff * Delta_H).sum(-1).clamp(min=0)  # (M,)
        torch.cuda.synchronize()
        t_grad += time.time() - t0

        # Pick tau heuristically:
        #   ATb has magnitude ~ X_rem^T u (the gradient at delta=0).
        #   Unconstrained LS magnitude ~ |ATb| / diag(H_red).
        diag_scale = H_red.diagonal().mean().clamp(min=1e-12)
        ref_l1 = ATb.abs().sum(-1) / diag_scale            # per-row LS magnitude proxy
        tau_vec = tau_frac * ref_l1                         # (M,)

        t0 = time.time()
        delta, info = spgl1_lasso_reduced_batched(
            H_red, ATb, b_norm_sq, tau=tau_vec,
            max_iter=spgl1_iters, verbose=False,
            matvec_dtype=matvec_dtype,
        )
        torch.cuda.synchronize()
        t_spgl1 += time.time() - t0
        total_line_iters += info["n_line_iters"]

        # ---- Step 3: apply compensation ----
        t0 = time.time()
        W_perm[:, ce:] += delta
        Delta_eff[:, ce:] += delta
        torch.cuda.synchronize()
        t_apply += time.time() - t0

        if verbose and (j < 3 or j % 50 == 0 or j == nblk - 1):
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
    p.add_argument("--tau-frac", type=float, default=1.0,
                   help="tau = tau_frac * (||ATb||_1 / diag-scale of H_red); "
                        "tau_frac=large → unconstrained-like, small → restrictive")
    p.add_argument("--spgl1-iters", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--ordering", choices=["block_h", "strict_obs"],
                   default="block_h",
                   help="block_h: r^T H_block r (no H_inv); "
                        "strict_obs: r^T ([H_inv]_block)^-1 r (uses H_inv)")
    p.add_argument("--matvec-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="dtype for the inner H @ x matvec in SPGL1")
    args = p.parse_args()

    print("GPTQ-Ord with SPGL1-LASSO compensation across remaining columns")
    print(f"  tau_frac={args.tau_frac}  spgl1_iters={args.spgl1_iters}",
          flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    # Load X in bf16 to fit in shared GPU memory; convert per-chunk for metrics
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True)
    if X.dtype == torch.float32:
        X = X.to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K} ({W.dtype})  X: {X.shape[0]}x{X.shape[1]} ({X.dtype})",
          flush=True)

    bs = args.bs
    nblk = K // bs

    print("Building Hessian (chunked)...", end=" ", flush=True)
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    chunk = 8192
    for t0 in range(0, X.shape[0], chunk):
        Xc = X[t0:t0 + chunk].float()
        H.addmm_(Xc.T, Xc)
    H /= X.shape[0]
    print(f"done", flush=True)

    H_blocks = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
    ])
    all_scales = build_fp8_e4m3_scales(device=DEVICE)

    print("\n[baseline] GPTQ-Ord + H-opt (unconstrained compensation)...",
          flush=True)
    t0 = time.time()
    Q_base = gptq_strided_baseline(W, H, bs, all_scales, H_blocks)
    t_base = time.time() - t0
    m_base = compute_metrics(W, Q_base, X)
    print(f"  W%={m_base['weight_error_pct']:.4f}  "
          f"O%={m_base['output_error_pct']:.4f}  time={_fmt(t_base)}",
          flush=True)

    print(f"\n[new] GPTQ-Ord + SPGL1 compensation (tau_frac={args.tau_frac})...",
          flush=True)
    matvec_dtype_map = {"fp32": None,
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16}
    t0 = time.time()
    Q_spgl1 = gptq_strided_spgl1(
        W, H, bs, all_scales, H_blocks,
        tau_frac=args.tau_frac, spgl1_iters=args.spgl1_iters,
        ordering=args.ordering,
        matvec_dtype=matvec_dtype_map[args.matvec_dtype], verbose=True,
    )
    t_spgl1_total = time.time() - t0
    m_spgl1 = compute_metrics(W, Q_spgl1, X)
    print(f"  W%={m_spgl1['weight_error_pct']:.4f}  "
          f"O%={m_spgl1['output_error_pct']:.4f}  time={_fmt(t_spgl1_total)}",
          flush=True)

    delta = m_spgl1["output_error_pct"] - m_base["output_error_pct"]
    print(f"\n  ΔO = {delta:+.4f}pp  ({'BETTER' if delta < 0 else 'WORSE'})")


if __name__ == "__main__":
    main()
