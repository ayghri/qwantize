#!/misc/envs/quant/bin/python
"""GPTQ with as_strided sequential column-block quantization.

Quantizes weight columns block-by-block using GPTQ error compensation,
with torch.as_strided for zero-copy sub-matrix views during error
propagation.

After quantizing each column block, the remaining weight sub-matrix
and Hessian-inverse cross-term are accessed via as_strided into the
original contiguous storage -- no copies, no slicing overhead.

Benchmarks all combinations of:
  Formats:     NVFP4 (FP8 E4M3 scales), MXFP4 (UE8M0 power-of-2 scales)
  Block sizes: 16, 32
  Approaches:  Naive, SSE-Optimal, H-Optimal
  +/- GPTQ error compensation
  Sequential vs Ordered (descending loss) column processing

Usage: /misc/envs/quant/bin/python experiments/quant_gptq_strided.py
"""

import sys
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
from qwantize.mxfp4.reference import build_ue8m0_scales
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

APPROACH_NAMES = {"naive": "Naive", "optimal": "Optimal", "hoptimal": "H-Optimal"}


# ===================================================================
# GPTQ with as_strided
# ===================================================================

def gptq_strided(W, H, quantize_block_fn, block_size=16, damp=0.01):
    """GPTQ with torch.as_strided for sequential column-block processing.

    After each column block is quantized, uses as_strided to create
    zero-copy views of the remaining weight sub-matrix and the
    Hessian-inverse cross-term for in-place error propagation.

    Args:
        W: (M, K) weight matrix.
        H: (K, K) Hessian (X^T X / N).
        quantize_block_fn: fn(w_block, col_idx) -> (M, bs) dequantized.
        block_size: Column block size (must divide K).
        damp: Hessian diagonal damping fraction.

    Returns:
        Q: (M, K) quantized weight matrix.
        avg_loss: Average per-row GPTQ loss.
    """
    W = W.clone().float().contiguous()
    M, K = W.shape
    assert K % block_size == 0
    nblk = K // block_size

    # Cholesky inverse of damped Hessian
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
        cs = b * block_size          # column start
        ce = cs + block_size         # column end
        rem = K - ce                 # remaining columns

        # Current block weights (copy -- will be compared against quantized)
        w_blk = W[:, cs:ce].clone()

        # H_inv diagonal for this block via as_strided (diagonal stride = K+1)
        h_diag = torch.as_strided(
            Hi, size=(block_size,), stride=(K + 1,),
            storage_offset=cs * K + cs,
        ).clone()

        # Quantize this column block
        w_q = quantize_block_fn(w_blk, b)
        Q[:, cs:ce] = w_q

        # Per-column error scaled by inverse Hessian diagonal
        err = (w_blk - w_q) / h_diag.unsqueeze(0)          # (M, bs)
        total_loss += ((w_blk - w_q) ** 2 / h_diag.unsqueeze(0)).sum().item()

        # Propagate error to all remaining columns via as_strided views
        if rem > 0:
            # H_inv[cs:ce, ce:] -- cross-block inverse Hessian
            #   storage layout: H_inv is (K, K) contiguous
            #   element (cs+i, ce+j) at offset (cs+i)*K + (ce+j)
            #   -> offset=cs*K+ce, stride=(K, 1), size=(bs, rem)
            h_cross = torch.as_strided(
                Hi, size=(block_size, rem), stride=(K, 1),
                storage_offset=cs * K + ce,
            )

            # W[:, ce:] -- remaining weight columns
            #   element (i, ce+j) at offset i*K + (ce+j)
            #   -> offset=ce, stride=(K, 1), size=(M, rem)
            w_rem = torch.as_strided(
                W, size=(M, rem), stride=(K, 1),
                storage_offset=ce,
            )

            # In-place GPTQ error compensation:
            #   W[:, ce:] -= err @ H_inv[cs:ce, ce:]
            w_rem.sub_(err @ h_cross)

    return Q, total_loss / M


def gptq_strided_ordered(W, H, loss_block_fn, make_quant_fn, block_size=16, damp=0.01):
    """GPTQ with column blocks reordered by descending quantization loss.

    Quantizes high-loss blocks first so their error is compensated
    across more remaining columns, maximizing GPTQ benefit.

    Steps:
      1. Quantize each block independently → Hessian-weighted loss per block
      2. Sort blocks descending by loss (worst first)
      3. Permute W columns and H rows+cols by block
      4. Run gptq_strided on the permuted problem
      5. Inverse-permute Q back to original column order

    Args:
        W: (M, K) weight matrix.
        H: (K, K) Hessian (X^T X / N).
        loss_block_fn: fn(w_block, idx) -> dequantized, used to estimate
            per-block loss for ordering (same signature as quantize_block_fn).
        make_quant_fn: fn(H_blocks_perm) -> quantize_block_fn, factory
            that builds the quantizer for the permuted column order.
            Receives the block-diagonal Hessians extracted from the
            permuted H so H-optimal methods use the correct blocks.
        block_size: Column block size (must divide K).
        damp: Hessian diagonal damping fraction.

    Returns:
        Q: (M, K) quantized weight matrix (original column order).
        avg_loss: Average per-row GPTQ loss.
    """
    W_f = W.float()
    M, K = W_f.shape
    bs = block_size
    nblk = K // bs
    dev = W.device

    # 1. Estimate per-block Hessian-weighted quantization loss
    H_blocks_orig = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs].float()
        for j in range(nblk)
    ])
    losses = torch.empty(nblk, device=dev)
    for j in range(nblk):
        w_blk = W_f[:, j * bs:(j + 1) * bs]
        w_q = loss_block_fn(w_blk, j)
        r = w_blk - w_q                          # (M, bs)
        Hr = r @ H_blocks_orig[j]                 # (M, bs)
        losses[j] = (r * Hr).sum()

    # 2. Sort descending (highest loss quantized first)
    _, blk_perm = losses.sort(descending=True)

    # 3. Build full column permutation from block permutation
    #    block i in permuted order contains original columns
    #    [blk_perm[i]*bs .. blk_perm[i]*bs + bs)
    col_perm = (
        blk_perm.unsqueeze(1) * bs
        + torch.arange(bs, device=dev).unsqueeze(0)
    ).reshape(-1)

    # 4. Permute W and H (rows & cols of H move together)
    W_perm = W_f[:, col_perm].contiguous()
    H_perm = H.float()[col_perm][:, col_perm].contiguous()

    # 5. Extract block-diagonal Hessians from permuted H
    H_blocks_perm = torch.stack([
        H_perm[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs]
        for j in range(nblk)
    ])

    # 6. Build quantizer for the permuted column order
    quant_fn = make_quant_fn(H_blocks_perm)

    # 7. Run standard sequential GPTQ on permuted data
    Q_perm, avg_loss = gptq_strided(W_perm, H_perm, quant_fn,
                                     block_size=bs, damp=damp)

    # 8. Inverse permute back to original column order
    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    Q = Q_perm[:, inv_perm]

    return Q, avg_loss


# ===================================================================
# Block quantizer primitives (block-size agnostic)
# ===================================================================

def _qd(x, s):
    """FP4 E2M1 quantize + dequantize in one step."""
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


# -- Baseline scale functions (format-specific) ---------------------

def _base_nvfp4(x):
    """NVFP4 naive scale: s = FP8_E4M3(max|x| / 6)."""
    return _fp8_e4m3_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))


def _base_mxfp4(x):
    """MXFP4 naive scale: s = 2^(floor(log2(amax)) - 2)."""
    a = x.abs().amax(-1).clamp(min=1e-30)
    e = (a.log2() - 2 + 127).floor().clamp(1, 254)
    return torch.pow(2.0, e - 127.0)


# -- Scale search functions (format-agnostic) -----------------------

def _optimal_scale(x, s0, all_scales, bs):
    """SSE-optimal scale via bounded search over candidate scales."""
    E0 = compute_block_sse(x, s0)
    best_s, best_E = s0.clone(), E0.clone()
    amax = x.abs().amax(-1)

    # Bounding
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
        for sv in all_scales:
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
    return best_s


def _hoptimal_scale(x, s0, H_blk, all_scales, bs):
    """H-optimal scale via bounded search with Hessian-weighted error.

    Uses SSE bounds for pruning but selects the scale minimizing
    r^T H r instead of ||r||^2.

    Args:
        x: (N, bs) block weights.
        s0: (N,) baseline scales.
        H_blk: (bs, bs) block Hessian for this column block.
        all_scales: candidate scale values.
        bs: block size.
    """
    E0_sse = compute_block_sse(x, s0)
    amax = x.abs().amax(-1)

    # Hessian-weighted baseline error
    r0 = x - _qd(x, s0)
    E0_H = (r0 * (r0 @ H_blk)).sum(-1)

    best_s, best_E = s0.clone(), E0_H.clone()

    # SSE-based bounding (still valid for pruning)
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
            ev = ok & (clip < bE * 10)  # looser fast-fail for Hessian metric
            if not ev.any():
                continue
            sf_t = torch.full((xa.shape[0],), sf, device=dev)
            r = xa - _qd(xa, sf_t)
            EH = (r * (r @ H_blk)).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_E[act], best_s[act] = bE, bS
    return best_s


# -- Block quantizer factory ---------------------------------------

def make_block_fn(fmt, approach, bs, all_scales, base_fn, H_blocks=None):
    """Create a quantize_block_fn(w_block, col_idx) -> dequantized.

    Args:
        fmt: "nvfp4" or "mxfp4" (for labeling only).
        approach: "naive", "optimal", or "hoptimal".
        bs: block size.
        all_scales: tensor of candidate scale values.
        base_fn: baseline scale function (_base_nvfp4 or _base_mxfp4).
        H_blocks: (num_blocks, bs, bs) block Hessians (for hoptimal).
    """
    if approach == "naive":
        def fn(w, idx):
            x = w.float()
            return _qd(x, base_fn(x))
        return fn

    elif approach == "optimal":
        def fn(w, idx):
            x = w.float()
            s = _optimal_scale(x, base_fn(x), all_scales, bs)
            return _qd(x, s)
        return fn

    elif approach == "hoptimal":
        assert H_blocks is not None
        def fn(w, idx):
            x = w.float()
            s = _hoptimal_scale(x, base_fn(x), H_blocks[idx], all_scales, bs)
            return _qd(x, s)
        return fn

    raise ValueError(f"Unknown approach: {approach}")


# -- Direct block quantization (no GPTQ) ---------------------------

def quantize_no_gptq(W, block_fn, block_size):
    """Block-wise quantization without GPTQ error compensation."""
    M, K = W.shape
    Q = torch.empty_like(W)
    for j in range(K // block_size):
        cs = j * block_size
        Q[:, cs:cs + block_size] = block_fn(W[:, cs:cs + block_size], j)
    return Q


# ===================================================================
# Benchmark
# ===================================================================

def _fmt_time(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def _make_quant_factory(fmt, approach, bs, all_scales, base_fn):
    """Return a factory: fn(H_blocks_perm) -> quantize_block_fn.

    Used by gptq_strided_ordered to rebuild the quantizer after
    column permutation (H_blocks change with the new block order).
    """
    def factory(H_blocks_perm):
        return make_block_fn(fmt, approach, bs, all_scales, base_fn, H_blocks_perm)
    return factory


def main():
    print("GPTQ + as_strided Benchmark (Sequential & Ordered)")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}")

    # Full Hessian H = X^T X / N
    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done ({K}x{K})\n")

    # Pre-build scale candidate lists
    nvfp4_scales = build_fp8_e4m3_scales(device=DEVICE)
    mxfp4_scales = build_ue8m0_scales(device=DEVICE)

    FORMAT_CFG = {
        "nvfp4": (nvfp4_scales, _base_nvfp4),
        "mxfp4": (mxfp4_scales, _base_mxfp4),
    }

    # results: (bs, fmt, approach, mode, metrics, time)
    #   mode: "base", "seq", "ord"
    results = []

    for bs in [16, 32]:
        assert K % bs == 0
        nblk = K // bs

        # Extract block-diagonal Hessians for H-optimal
        H_blocks = torch.stack([
            H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
        ])  # (nblk, bs, bs)

        print(f"{'=' * 80}")
        print(f"Block size: {bs}  ({nblk} blocks)")
        print(f"{'=' * 80}")

        for fmt in ["nvfp4", "mxfp4"]:
            scales, base_fn = FORMAT_CFG[fmt]

            for approach in ["naive", "optimal", "hoptimal"]:
                label = f"{fmt.upper()} {APPROACH_NAMES[approach]}"

                # --- Baseline (no GPTQ) ---
                fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
                torch.cuda.synchronize()
                t0 = time.time()
                Q_base = quantize_no_gptq(W, fn, bs)
                torch.cuda.synchronize()
                t_base = time.time() - t0
                m_base = compute_metrics(W, Q_base, X)
                del Q_base

                # --- Sequential GPTQ ---
                fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
                torch.cuda.synchronize()
                t0 = time.time()
                Q_seq, _ = gptq_strided(W, H, fn, block_size=bs)
                torch.cuda.synchronize()
                t_seq = time.time() - t0
                m_seq = compute_metrics(W, Q_seq, X)
                del Q_seq

                # --- Ordered GPTQ (descending loss) ---
                loss_fn = make_block_fn(fmt, approach, bs, scales, base_fn, H_blocks)
                factory = _make_quant_factory(fmt, approach, bs, scales, base_fn)
                torch.cuda.synchronize()
                t0 = time.time()
                Q_ord, _ = gptq_strided_ordered(W, H, loss_fn, factory,
                                                 block_size=bs)
                torch.cuda.synchronize()
                t_ord = time.time() - t0
                m_ord = compute_metrics(W, Q_ord, X)
                del Q_ord

                torch.cuda.empty_cache()

                d_seq = m_seq["output_error_pct"] - m_base["output_error_pct"]
                d_ord = m_ord["output_error_pct"] - m_base["output_error_pct"]
                d_ov = m_ord["output_error_pct"] - m_seq["output_error_pct"]

                results.append((bs, fmt.upper(), APPROACH_NAMES[approach],
                                "base", m_base, t_base))
                results.append((bs, fmt.upper(), APPROACH_NAMES[approach],
                                "seq", m_seq, t_seq))
                results.append((bs, fmt.upper(), APPROACH_NAMES[approach],
                                "ord", m_ord, t_ord))

                print(f"  {label:<22}  "
                      f"W={m_base['weight_error_pct']:7.4f}%  "
                      f"O={m_base['output_error_pct']:7.4f}%  "
                      f"{_fmt_time(t_base):>8}")
                print(f"  GPTQ+{label:<17}  "
                      f"W={m_seq['weight_error_pct']:7.4f}%  "
                      f"O={m_seq['output_error_pct']:7.4f}%  "
                      f"dO={d_seq:+.2f}pp  "
                      f"{_fmt_time(t_seq):>8}")
                print(f"  GPTQord+{label:<14}  "
                      f"W={m_ord['weight_error_pct']:7.4f}%  "
                      f"O={m_ord['output_error_pct']:7.4f}%  "
                      f"dO={d_ord:+.2f}pp  "
                      f"vs seq={d_ov:+.3f}pp  "
                      f"{_fmt_time(t_ord):>8}")

        print()

    # --- Markdown tables for docs ---
    MODE_LABELS = {"base": "", "seq": "GPTQ+", "ord": "GPTQ-Ord+"}
    MODE_MARKS = {"base": " ", "seq": "+", "ord": "Ord"}

    print("\n" + "=" * 80)
    print("Markdown for docs/source/results.md")
    print("=" * 80 + "\n")

    for bs_f in [16, 32]:
        print(f"### Block Size {bs_f}\n")
        print("| Format | Approach | GPTQ | Weight Error | Output Error | Time |")
        print("|:--|:--|:--:|:--:|:--:|--:|")
        for bs, fmt, approach, mode, m, t in results:
            if bs != bs_f:
                continue
            name = f"{MODE_LABELS[mode]}{approach}"
            print(f"| {fmt} | {name} | {MODE_MARKS[mode]} "
                  f"| {m['weight_error_pct']:.2f}% "
                  f"| {m['output_error_pct']:.2f}% "
                  f"| {_fmt_time(t)} |")
        print()


if __name__ == "__main__":
    main()
