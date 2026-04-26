#!/misc/envs/quant/bin/python
"""Row-by-row ordered GPTQ benchmark (batched).

Each row gets its own column block ordering based on that row's
per-block Hessian-weighted quantization loss. Rows are processed
in batches for GPU efficiency: gather-based permutation, batched
H_inv cross-term extraction, and batched matmul for error propagation.

Compares: baseline, sequential GPTQ, column-ordered GPTQ, row-ordered GPTQ.

Usage: /misc/envs/quant/bin/python experiments/quant_gptq_row_ordered.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from quant_gptq_strided import (
    gptq_strided,
    gptq_strided_ordered,
    make_block_fn,
    _make_quant_factory,
    quantize_no_gptq,
    _base_nvfp4,
    _base_mxfp4,
    _qd,
    _optimal_scale,
    APPROACH_NAMES,
)
from qwantize.nvfp4.reference import (
    build_fp8_e4m3_scales,
    fp4_quantize,
    fp4_dequantize,
    compute_block_sse,
    Q_MAX,
    D_0,
)
from qwantize.mxfp4.reference import build_ue8m0_scales
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# Per-row H-optimal scale search (bmm for per-row H_blocks)
# ===================================================================

def _hoptimal_scale_per_row(x, s0, H_blks, all_scales, bs):
    """H-optimal scale with per-row block Hessians.

    Like _hoptimal_scale but H_blks has shape (B, bs, bs) — one
    block Hessian per row in the batch.

    Args:
        x: (B, bs) block weights.
        s0: (B,) baseline scales.
        H_blks: (B, bs, bs) per-row block Hessians.
        all_scales: candidate scale tensor.
        bs: block size.
    """
    E0_sse = compute_block_sse(x, s0)
    amax = x.abs().amax(-1)

    r0 = x - _qd(x, s0)
    Hr0 = torch.bmm(r0.unsqueeze(1), H_blks).squeeze(1)
    E0_H = (r0 * Hr0).sum(-1)

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
        H_act = H_blks[act]
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
            Hr = torch.bmm(r.unsqueeze(1), H_act).squeeze(1)
            EH = (r * Hr).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_E[act], best_s[act] = bE, bS
    return best_s


# ===================================================================
# Batch quantizer factory
# ===================================================================

def _make_batch_quant_fn(approach, all_scales, base_fn, bs):
    """Create a batch quantizer: fn(w_blk, H_blk_batch) -> dequantized.

    Args:
        approach: "naive", "optimal", or "hoptimal".
        all_scales: candidate scale tensor.
        base_fn: baseline scale function.
        bs: block size.

    Returns:
        fn(w_blk, H_blk_batch) -> (B, bs) dequantized.
        H_blk_batch is (B, bs, bs) for hoptimal, ignored otherwise.
    """
    if approach == "naive":
        def fn(w, H_blk=None):
            x = w.float()
            return _qd(x, base_fn(x))
        return fn

    elif approach == "optimal":
        def fn(w, H_blk=None):
            x = w.float()
            return _qd(x, _optimal_scale(x, base_fn(x), all_scales, bs))
        return fn

    elif approach == "hoptimal":
        def fn(w, H_blk):
            x = w.float()
            return _qd(x, _hoptimal_scale_per_row(
                x, base_fn(x), H_blk, all_scales, bs))
        return fn

    raise ValueError(approach)


# ===================================================================
# Batched row-ordered GPTQ
# ===================================================================

def gptq_row_ordered(W, H, H_blocks_orig, loss_block_fn, batch_quant_fn,
                     block_size=16, damp=0.01, row_batch=16):
    """GPTQ with per-row column block ordering, processed in batches.

    Each row independently sorts column blocks by descending Hessian-
    weighted quantization loss. Rows are processed in batches of
    ``row_batch`` using:
      - ``torch.gather`` for batched permutation
      - Batched H_inv cross-term extraction
      - ``torch.bmm`` for batched error propagation
      - ``torch.scatter_`` for batched inverse permutation

    Args:
        W: (M, K) weight matrix.
        H: (K, K) Hessian.
        H_blocks_orig: (nblk, bs, bs) block-diagonal Hessians from H.
        loss_block_fn: fn(w_blk, idx) -> (M, bs) dequantized, for loss.
        batch_quant_fn: fn(w_blk, H_blk_batch) -> (B, bs) dequantized.
        block_size: Column block size.
        damp: Hessian damping.
        row_batch: Rows per batch.

    Returns:
        Q: (M, K) quantized (original column order).
        avg_loss: Average per-row GPTQ loss.
    """
    W_f = W.float()
    M, K = W_f.shape
    bs = block_size
    nblk = K // bs
    dev = W.device

    # --- Precompute H_inv ---
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

    # --- Batch-compute per-row per-block losses ---
    print(f"    Computing per-row block losses...", end=" ", flush=True)
    W_blocked = W_f.reshape(M, nblk, bs)
    all_losses = torch.empty(M, nblk, device=dev)
    for j in range(nblk):
        w_blk = W_blocked[:, j, :]
        w_q = loss_block_fn(w_blk, j)
        r = w_blk - w_q
        Hr = r @ H_blocks_orig[j]
        all_losses[:, j] = (r * Hr).sum(-1)

    _, all_blk_perms = all_losses.sort(dim=1, descending=True)
    del all_losses

    arange_bs = torch.arange(bs, device=dev)
    all_col_perms = (
        all_blk_perms.unsqueeze(2) * bs + arange_bs.unsqueeze(0)
    ).reshape(M, K)
    print("done", flush=True)

    # --- Batched per-row GPTQ ---
    Q = torch.zeros_like(W_f)
    total_loss = 0.0
    arange_K = torch.arange(K, device=dev)
    t_start = time.time()

    for m0 in range(0, M, row_batch):
        B = min(row_batch, M - m0)
        batch_perms = all_col_perms[m0:m0 + B]       # (B, K)
        batch_blk_perms = all_blk_perms[m0:m0 + B]   # (B, nblk)

        # Permute batch of rows via gather
        W_batch = W_f[m0:m0 + B].gather(1, batch_perms)  # (B, K)

        Q_batch = torch.empty_like(W_batch)

        for b in range(nblk):
            cs = b * bs
            ce = cs + bs
            rem = K - ce

            w_blk = W_batch[:, cs:ce].clone()  # (B, bs)

            # H_inv diagonal: Hi[perm[i,cs+j], perm[i,cs+j]] for each (i,j)
            perm_block = batch_perms[:, cs:ce]                  # (B, bs)
            flat = perm_block.reshape(-1)                       # (B*bs,)
            h_diag = Hi[flat, flat].reshape(B, bs)              # (B, bs)

            # Per-row block Hessians for this GPTQ step
            orig_blk = batch_blk_perms[:, b]                    # (B,)
            H_blk_batch = H_blocks_orig[orig_blk]               # (B, bs, bs)

            # Quantize
            w_q = batch_quant_fn(w_blk, H_blk_batch)           # (B, bs)
            Q_batch[:, cs:ce] = w_q

            # Error
            err = (w_blk - w_q) / h_diag                       # (B, bs)
            total_loss += ((w_blk - w_q) ** 2 / h_diag).sum().item()

            # Error propagation via batched gather + bmm
            if rem > 0:
                perm_rest = batch_perms[:, ce:]                 # (B, rem)

                # Gather cross-term rows then columns:
                #   Hi_rows[i,j,:] = Hi[perm_block[i,j], :]
                Hi_rows = Hi[flat].reshape(B, bs, K)            # (B, bs, K)
                #   h_cross[i,j,k] = Hi_rows[i,j,perm_rest[i,k]]
                h_cross = Hi_rows.gather(
                    2, perm_rest.unsqueeze(1).expand(-1, bs, -1)
                )                                               # (B, bs, rem)
                del Hi_rows

                # (B, 1, bs) @ (B, bs, rem) -> (B, 1, rem) -> (B, rem)
                update = torch.bmm(err.unsqueeze(1), h_cross).squeeze(1)
                W_batch[:, ce:] -= update
                del h_cross, update

        # Inverse permute via scatter: Q[m0+i, perm[i,j]] = Q_batch[i,j]
        Q[m0:m0 + B].scatter_(1, batch_perms, Q_batch)
        del W_batch, Q_batch

        done = m0 + B
        if done % (row_batch * 8) == 0 or done == M:
            elapsed = time.time() - t_start
            eta = elapsed / done * (M - done) if done < M else 0
            print(f"    rows {done}/{M}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)

    return Q, total_loss / M


# ===================================================================
# Benchmark
# ===================================================================

def _ts(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    print("Row-Ordered GPTQ Benchmark (Batched)")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}")

    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done ({K}x{K})\n")

    nvfp4_scales = build_fp8_e4m3_scales(device=DEVICE)
    mxfp4_scales = build_ue8m0_scales(device=DEVICE)
    FORMAT_CFG = {
        "nvfp4": (nvfp4_scales, _base_nvfp4),
        "mxfp4": (mxfp4_scales, _base_mxfp4),
    }

    results = []
    approach = "hoptimal"

    for bs in [16, 32]:
        nblk = K // bs
        H_blocks = torch.stack([
            H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs]
            for j in range(nblk)
        ])

        print(f"{'=' * 90}")
        print(f"Block size: {bs}  ({nblk} blocks)")
        print(f"{'=' * 90}")

        for fmt in ["nvfp4", "mxfp4"]:
            scales, base_fn = FORMAT_CFG[fmt]
            label = f"{fmt.upper()} {APPROACH_NAMES[approach]}"
            factory = _make_quant_factory(fmt, approach, bs, scales, base_fn)

            # --- Baseline ---
            fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
            torch.cuda.synchronize(); t0 = time.time()
            Qb = quantize_no_gptq(W, fn, bs)
            torch.cuda.synchronize(); tb = time.time() - t0
            mb = compute_metrics(W, Qb, X); del Qb

            # --- Sequential GPTQ ---
            fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
            torch.cuda.synchronize(); t0 = time.time()
            Qs, _ = gptq_strided(W, H, fn, block_size=bs)
            torch.cuda.synchronize(); ts = time.time() - t0
            ms = compute_metrics(W, Qs, X); del Qs

            # --- Column-ordered GPTQ ---
            fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
            torch.cuda.synchronize(); t0 = time.time()
            Qc, _ = gptq_strided_ordered(W, H, fn, factory, block_size=bs)
            torch.cuda.synchronize(); tc = time.time() - t0
            mc = compute_metrics(W, Qc, X); del Qc

            torch.cuda.empty_cache()

            # --- Row-ordered GPTQ (batched) ---
            print(f"  [{label}] Row-ordered GPTQ...", flush=True)
            fn_loss = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
            bq_fn = _make_batch_quant_fn(approach, scales, base_fn, bs)
            torch.cuda.synchronize(); t0 = time.time()
            Qr, _ = gptq_row_ordered(
                W, H, H_blocks, fn_loss, bq_fn,
                block_size=bs, row_batch=16,
            )
            torch.cuda.synchronize(); tr = time.time() - t0
            mr = compute_metrics(W, Qr, X); del Qr

            torch.cuda.empty_cache()

            for mode, m, t in [("base", mb, tb), ("seq", ms, ts),
                                ("col-ord", mc, tc), ("row-ord", mr, tr)]:
                results.append((bs, fmt.upper(), APPROACH_NAMES[approach],
                                mode, m, t))

            d_s = ms["output_error_pct"] - mb["output_error_pct"]
            d_c = mc["output_error_pct"] - mb["output_error_pct"]
            d_r = mr["output_error_pct"] - mb["output_error_pct"]
            d_rc = mr["output_error_pct"] - mc["output_error_pct"]

            print(f"  {label}")
            print(f"    {'Baseline':<14}  "
                  f"W={mb['weight_error_pct']:7.4f}%  "
                  f"O={mb['output_error_pct']:7.4f}%  {_ts(tb):>8}")
            print(f"    {'Seq GPTQ':<14}  "
                  f"W={ms['weight_error_pct']:7.4f}%  "
                  f"O={ms['output_error_pct']:7.4f}%  "
                  f"dO={d_s:+.2f}pp  {_ts(ts):>8}")
            print(f"    {'Col-Ord GPTQ':<14}  "
                  f"W={mc['weight_error_pct']:7.4f}%  "
                  f"O={mc['output_error_pct']:7.4f}%  "
                  f"dO={d_c:+.2f}pp  {_ts(tc):>8}")
            print(f"    {'Row-Ord GPTQ':<14}  "
                  f"W={mr['weight_error_pct']:7.4f}%  "
                  f"O={mr['output_error_pct']:7.4f}%  "
                  f"dO={d_r:+.2f}pp  "
                  f"vs col={d_rc:+.3f}pp  {_ts(tr):>8}")
            print()

    # --- Summary ---
    MODE_NAMES = {"base": "Baseline", "seq": "GPTQ-Seq",
                  "col-ord": "GPTQ-ColOrd", "row-ord": "GPTQ-RowOrd"}

    print(f"\n{'=' * 100}")
    print(f"{'Method':<40} {'BS':>3} {'Weight%':>10} {'Output%':>10} {'Time':>10}")
    print(f"{'-' * 100}")
    for bs, fmt, approach, mode, m, t in results:
        name = f"{MODE_NAMES[mode]}+{fmt} {approach}" if mode != "base" \
               else f"{fmt} {approach}"
        print(f"{name:<40} {bs:>3} "
              f"{m['weight_error_pct']:>9.4f}% "
              f"{m['output_error_pct']:>9.4f}% "
              f"{_ts(t):>10}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
