#!/usr/bin/env python
"""GPTQ-Ord with SPGL1+resnap as the per-block snap operator.

Each time GPTQ-Ord asks the snap function to quantize one column-block
of 16 weights (per row), we:
  1. Compute initial H-opt grid Q_blk from current w_blk + H_blk.
  2. Run reduced-form SPGL1 LASSO per row:
        min_d  ||X_blk d - X_blk (w_blk - Q_blk)||_2   s.t. ||d||_1 <= tau
     where tau = tau_frac * ||w_blk - Q_blk||_1 per row.
  3. w_inter = Q_blk + d*
  4. Re-snap w_inter with H-opt → final block quantization.

This combines:
  - GPTQ-Ord cross-block compensation (descending-loss order + H_inv prop)
  - SPGL1 within-block perturbation that may cross basin boundaries

Reduced-form SPGL1 uses H_blk (16x16) — never materializes (M, T) residuals.

Usage:
    PYTHONPATH=. python experiments/spgl1_in_gptq.py
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap, build_fp8_e4m3_scales,
    fp4_quantize, fp4_dequantize, compute_block_sse,
    Q_MAX, D_0,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched, l1_norm_batched
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# GPTQ as_strided helpers — copied verbatim from quant_gptq_strided.py
# ===================================================================

def gptq_strided(W, H, quantize_block_fn, block_size=16, damp=0.01):
    W = W.clone().float().contiguous()
    M, K = W.shape
    assert K % block_size == 0
    nblk = K // block_size

    Hi = H.clone().float()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    try:
        L = torch.linalg.cholesky(Hi)
        Hi = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        Hi.diagonal().add_(1e-5 * dmu)
        L = torch.linalg.cholesky(Hi)
        Hi = torch.cholesky_inverse(L)
    del L
    Hi = Hi.contiguous()

    Q = torch.zeros_like(W)
    total_loss = 0.0

    for b in range(nblk):
        cs = b * block_size
        ce = cs + block_size
        rem = K - ce

        w_blk = W[:, cs:ce].clone()

        h_diag = torch.as_strided(
            Hi, size=(block_size,), stride=(K + 1,),
            storage_offset=cs * K + cs,
        ).clone()

        w_q = quantize_block_fn(w_blk, b)
        Q[:, cs:ce] = w_q

        err = (w_blk - w_q) / h_diag.unsqueeze(0)
        total_loss += ((w_blk - w_q) ** 2 / h_diag.unsqueeze(0)).sum().item()

        if rem > 0:
            h_cross = torch.as_strided(
                Hi, size=(block_size, rem), stride=(K, 1),
                storage_offset=cs * K + ce,
            )
            w_rem = torch.as_strided(
                W, size=(M, rem), stride=(K, 1),
                storage_offset=ce,
            )
            w_rem.sub_(err @ h_cross)

    return Q, total_loss / M


def gptq_strided_ordered(W, H, loss_block_fn, make_quant_fn, block_size=16, damp=0.01):
    W_f = W.float()
    M, K = W_f.shape
    bs = block_size
    nblk = K // bs
    dev = W.device

    H_blocks_orig = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs].float()
        for j in range(nblk)
    ])
    losses = torch.empty(nblk, device=dev)
    for j in range(nblk):
        w_blk = W_f[:, j * bs:(j + 1) * bs]
        w_q = loss_block_fn(w_blk, j)
        r = w_blk - w_q
        Hr = r @ H_blocks_orig[j]
        losses[j] = (r * Hr).sum()

    _, blk_perm = losses.sort(descending=True)

    col_perm = (
        blk_perm.unsqueeze(1) * bs
        + torch.arange(bs, device=dev).unsqueeze(0)
    ).reshape(-1)

    W_perm = W_f[:, col_perm].contiguous()
    H_perm = H.float()[col_perm][:, col_perm].contiguous()

    H_blocks_perm = torch.stack([
        H_perm[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs]
        for j in range(nblk)
    ])

    quant_fn = make_quant_fn(H_blocks_perm)

    Q_perm, avg_loss = gptq_strided(W_perm, H_perm, quant_fn,
                                     block_size=bs, damp=damp)

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    Q = Q_perm[:, inv_perm]

    return Q, avg_loss


# ===================================================================
# H-optimal per-block FP4 snap (uses 16x16 H_blk)
# ===================================================================

def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def _base_nvfp4(x):
    return _fp8_e4m3_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))


def _hoptimal_block_snap(x, H_blk, all_scales, bs):
    """Per-row H-optimal FP4 snap of (M, bs) weights using a single (bs, bs)
    block Hessian. Returns (M, bs) dequantized."""
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


# ===================================================================
# Block snap factories
# ===================================================================

def make_block_fn_hopt(bs, all_scales, H_blocks):
    """Plain H-optimal block snap (baseline)."""
    def fn(w, idx):
        x = w.float()
        return _hoptimal_block_snap(x, H_blocks[idx], all_scales, bs)
    return fn


def make_block_fn_spgl1_resnap(bs, all_scales, H_blocks,
                                tau_frac=0.5, max_iter=20):
    """SPGL1 + H-opt re-snap block snap.

    For each block j:
        1. Compute initial H-opt target Q_blk_init.
        2. SPGL1 LASSO (reduced form, 16-dim) per row.
        3. w_inter = Q_blk_init + d*.
        4. Re-snap w_inter via H-opt.
    """
    def fn(w, idx):
        x = w.float()                                       # (M, bs)
        H_blk = H_blocks[idx]                               # (bs, bs)

        # Step 1: initial H-opt target
        Q_init = _hoptimal_block_snap(x, H_blk, all_scales, bs)   # (M, bs)
        d_init = x - Q_init                                  # (M, bs)

        l1 = d_init.abs().sum(-1)                            # (M,)
        # tau per row; rows with zero d_init get tau=0 (no perturbation)
        tau_vec = tau_frac * l1

        # Skip SPGL1 if all rows are already on grid (no L1 budget)
        if (tau_vec > 1e-12).any():
            ATb = d_init @ H_blk                              # (M, bs)
            b_norm_sq = (d_init * ATb).sum(-1)                # (M,)
            d_star, _ = spgl1_lasso_reduced_batched(
                H_blk, ATb, b_norm_sq, tau=tau_vec,
                max_iter=max_iter, verbose=False,
            )
            w_inter = Q_init + d_star
        else:
            w_inter = Q_init

        # Step 4: re-snap
        return _hoptimal_block_snap(w_inter, H_blk, all_scales, bs)

    return fn


def make_factory_hopt(bs, all_scales):
    def factory(H_blocks_perm):
        return make_block_fn_hopt(bs, all_scales, H_blocks_perm)
    return factory


def make_factory_spgl1(bs, all_scales, tau_frac, max_iter):
    def factory(H_blocks_perm):
        return make_block_fn_spgl1_resnap(
            bs, all_scales, H_blocks_perm, tau_frac, max_iter,
        )
    return factory


def quantize_no_gptq(W, block_fn, block_size):
    M, K = W.shape
    Q = torch.empty_like(W)
    for j in range(K // block_size):
        cs = j * block_size
        Q[:, cs:cs + block_size] = block_fn(W[:, cs:cs + block_size], j)
    return Q


# ===================================================================
# Main
# ===================================================================

def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau-frac", type=float, default=0.5)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--mode", choices=["base", "seq", "ord", "all"],
                   default="all",
                   help="which GPTQ mode to run")
    args = p.parse_args()

    print("GPTQ-Ord  with SPGL1+resnap block snap")
    print(f"  tau_frac={args.tau_frac}  spgl1_max_iter={args.max_iter}",
          flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}", flush=True)

    bs = args.bs
    nblk = K // bs

    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done", flush=True)

    H_blocks = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
    ])
    all_scales = build_fp8_e4m3_scales(device=DEVICE)

    snap_hopt = make_block_fn_hopt(bs, all_scales, H_blocks)
    snap_spgl1 = make_block_fn_spgl1_resnap(
        bs, all_scales, H_blocks, args.tau_frac, args.max_iter,
    )

    results = []

    def _record(label, Q, t):
        m = compute_metrics(W, Q, X)
        results.append((label, m["weight_error_pct"],
                       m["output_error_pct"], t))
        print(f"  {label:<35}  W%={m['weight_error_pct']:7.4f}  "
              f"O%={m['output_error_pct']:7.4f}  time={_fmt(t)}", flush=True)

    if args.mode in ("base", "all"):
        print("\n[no GPTQ] H-opt baseline...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        Q = quantize_no_gptq(W, snap_hopt, bs)
        torch.cuda.synchronize()
        _record("H-opt (no GPTQ)", Q, time.time() - t0)

        print("[no GPTQ] SPGL1+resnap...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        Q = quantize_no_gptq(W, snap_spgl1, bs)
        torch.cuda.synchronize()
        _record("SPGL1+resnap (no GPTQ)", Q, time.time() - t0)

    if args.mode in ("seq", "all"):
        print("\n[GPTQ-Seq] H-opt baseline...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        fn = make_block_fn_hopt(bs, all_scales, H_blocks)
        Q, _ = gptq_strided(W, H, fn, block_size=bs)
        torch.cuda.synchronize()
        _record("GPTQ-Seq + H-opt", Q, time.time() - t0)

        print("[GPTQ-Seq] SPGL1+resnap...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        fn = make_block_fn_spgl1_resnap(
            bs, all_scales, H_blocks, args.tau_frac, args.max_iter,
        )
        Q, _ = gptq_strided(W, H, fn, block_size=bs)
        torch.cuda.synchronize()
        _record("GPTQ-Seq + SPGL1+resnap", Q, time.time() - t0)

    if args.mode in ("ord", "all"):
        print("\n[GPTQ-Ord] H-opt baseline...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        loss_fn = make_block_fn_hopt(bs, all_scales, H_blocks)
        factory = make_factory_hopt(bs, all_scales)
        Q, _ = gptq_strided_ordered(W, H, loss_fn, factory, block_size=bs)
        torch.cuda.synchronize()
        _record("GPTQ-Ord + H-opt", Q, time.time() - t0)

        print("[GPTQ-Ord] SPGL1+resnap...", flush=True)
        torch.cuda.synchronize(); t0 = time.time()
        loss_fn = make_block_fn_spgl1_resnap(
            bs, all_scales, H_blocks, args.tau_frac, args.max_iter,
        )
        factory = make_factory_spgl1(bs, all_scales,
                                      args.tau_frac, args.max_iter)
        Q, _ = gptq_strided_ordered(W, H, loss_fn, factory, block_size=bs)
        torch.cuda.synchronize()
        _record("GPTQ-Ord + SPGL1+resnap", Q, time.time() - t0)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {'method':<35}  {'W err':>8}  {'O err':>8}  {'time':>8}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, we, oe, t in results:
        print(f"  {label:<35}  {we:7.4f}%  {oe:7.4f}%  {_fmt(t):>8}")


if __name__ == "__main__":
    main()
