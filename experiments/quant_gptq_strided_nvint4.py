#!/misc/envs/quant/bin/python
"""GPTQ + as_strided benchmark for NVINT4 (symmetric INT4, FP8 E4M3 scales).

Mirrors experiments/quant_gptq_strided.py but for the NVINT4 format
(Q_MAX=7, D_0=0.5, codebook {-7..7}).

Combinations:
  Block sizes: 16, 32
  Approaches:  Naive, SSE-Optimal, H-Optimal
  Modes:       no-GPTQ, GPTQ-Seq, GPTQ-Ord (descending loss)

Usage: /misc/envs/quant/bin/python experiments/quant_gptq_strided_nvint4.py
"""

import time
import torch

from qwantize.nvint4.reference import (
    _fp8_e4m3_snap,
    build_fp8_e4m3_scales,
    int4_quantize,
    int4_dequantize_block,
    compute_block_sse,
    Q_MAX,
    D_0,
)
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

APPROACH_NAMES = {"naive": "Naive", "optimal": "Optimal", "hoptimal": "H-Optimal"}


# ===================================================================
# GPTQ with as_strided (identical structure to quant_gptq_strided.py)
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
# NVINT4 block primitives
# ===================================================================

def _qd(x, s):
    """INT4 quantize + dequantize."""
    su = s.unsqueeze(-1)
    return int4_dequantize_block(int4_quantize(x, su), su)


def _base_nvint4(x):
    """NVINT4 naive scale: s = FP8_E4M3(max|x| / 7)."""
    return _fp8_e4m3_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))


def _optimal_scale(x, s0, all_scales, bs):
    E0 = compute_block_sse(x, s0)
    best_s, best_E = s0.clone(), E0.clone()
    amax = x.abs().amax(-1)

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
        best_E[act], best_s[act] = bE, bS
    return best_s


def make_block_fn(approach, bs, all_scales, H_blocks=None):
    if approach == "naive":
        def fn(w, idx):
            x = w.float()
            return _qd(x, _base_nvint4(x))
        return fn

    if approach == "optimal":
        def fn(w, idx):
            x = w.float()
            s = _optimal_scale(x, _base_nvint4(x), all_scales, bs)
            return _qd(x, s)
        return fn

    if approach == "hoptimal":
        assert H_blocks is not None
        def fn(w, idx):
            x = w.float()
            s = _hoptimal_scale(x, _base_nvint4(x), H_blocks[idx], all_scales, bs)
            return _qd(x, s)
        return fn

    raise ValueError(f"Unknown approach: {approach}")


def quantize_no_gptq(W, block_fn, block_size):
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


def _make_quant_factory(approach, bs, all_scales):
    def factory(H_blocks_perm):
        return make_block_fn(approach, bs, all_scales, H_blocks_perm)
    return factory


def main():
    print("NVINT4 GPTQ + as_strided Benchmark (Sequential & Ordered)")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}")

    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done ({K}x{K})\n")

    nvint4_scales = build_fp8_e4m3_scales(device=DEVICE)

    results = []

    for bs in [16, 32]:
        assert K % bs == 0
        nblk = K // bs

        H_blocks = torch.stack([
            H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
        ])

        print(f"{'=' * 80}")
        print(f"Block size: {bs}  ({nblk} blocks)")
        print(f"{'=' * 80}")

        for approach in ["naive", "optimal", "hoptimal"]:
            label = f"NVINT4 {APPROACH_NAMES[approach]}"

            fn = make_block_fn(approach, bs, nvint4_scales, H_blocks)
            torch.cuda.synchronize()
            t0 = time.time()
            Q_base = quantize_no_gptq(W, fn, bs)
            torch.cuda.synchronize()
            t_base = time.time() - t0
            m_base = compute_metrics(W, Q_base, X)
            del Q_base

            fn = make_block_fn(approach, bs, nvint4_scales, H_blocks)
            torch.cuda.synchronize()
            t0 = time.time()
            Q_seq, _ = gptq_strided(W, H, fn, block_size=bs)
            torch.cuda.synchronize()
            t_seq = time.time() - t0
            m_seq = compute_metrics(W, Q_seq, X)
            del Q_seq

            loss_fn = make_block_fn(approach, bs, nvint4_scales, H_blocks)
            factory = _make_quant_factory(approach, bs, nvint4_scales)
            torch.cuda.synchronize()
            t0 = time.time()
            Q_ord, _ = gptq_strided_ordered(W, H, loss_fn, factory, block_size=bs)
            torch.cuda.synchronize()
            t_ord = time.time() - t0
            m_ord = compute_metrics(W, Q_ord, X)
            del Q_ord

            torch.cuda.empty_cache()

            d_seq = m_seq["output_error_pct"] - m_base["output_error_pct"]
            d_ord = m_ord["output_error_pct"] - m_base["output_error_pct"]
            d_ov = m_ord["output_error_pct"] - m_seq["output_error_pct"]

            results.append((bs, "NVINT4", APPROACH_NAMES[approach], "base", m_base, t_base))
            results.append((bs, "NVINT4", APPROACH_NAMES[approach], "seq", m_seq, t_seq))
            results.append((bs, "NVINT4", APPROACH_NAMES[approach], "ord", m_ord, t_ord))

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
