#!/misc/envs/quant/bin/python
"""INT4 H-Optimal GPTQ benchmark: seq, col-ord, row-ord for BS 16/32.

INT4: uniform 8-level codebook [0, 1/7, 2/7, ..., 1] with FP8 E4M3 scales.
Same GPTQ framework as NVFP4/MXFP4 benchmarks.

Usage: /misc/envs/quant/bin/python experiments/quant_gptq_int4.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from quant_gptq_strided import (
    gptq_strided,
    gptq_strided_ordered,
    quantize_no_gptq,
)
from quant_gptq_row_ordered import gptq_row_ordered
from custom_codebook import custom_quantize, custom_dequantize, compute_custom_block_sse
from qwantize.nvfp4.reference import _fp8_e4m3_snap, build_fp8_e4m3_scales
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# INT4 codebook + quantizer primitives
# ===================================================================

def _build_int4_codebook(device):
    cb = torch.linspace(0, 1, 8, device=device)
    bd = torch.empty(7, device=device)
    bd[0] = cb[1] / 2
    for k in range(1, 7):
        bd[k] = (cb[k] + cb[k + 1]) / 2
    return cb, bd

INT4_QMAX = 1.0
INT4_D0 = 1.0 / 14  # cb[1] / 2


def _int4_qd(x, s, cb, bd):
    su = s.unsqueeze(-1)
    return custom_dequantize(custom_quantize(x, su, cb, bd), su)


def _base_int4(x):
    return _fp8_e4m3_snap((x.abs().amax(-1) / INT4_QMAX).clamp(min=1e-12))


def _int4_block_sse(x, s, cb, bd):
    return compute_custom_block_sse(x, s, cb, bd)


def _optimal_scale_int4(x, s0, all_scales, bs, cb, bd):
    """SSE-optimal scale for INT4 via bounded search."""
    E0 = _int4_block_sse(x, s0, cb, bd)
    best_s, best_E = s0.clone(), E0.clone()
    amax = x.abs().amax(-1)

    noise = x.pow(2).sum(-1) <= E0
    s_min = ((amax - E0.sqrt()) / INT4_QMAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / INT4_D0

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
            clip = (xa.abs() - INT4_QMAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE)
            if not ev.any():
                continue
            Es = _int4_block_sse(xa, torch.full((xa.shape[0],), sf, device=dev), cb, bd)
            imp = ev & (Es < bE)
            bE[imp], bS[imp] = Es[imp], sf
        best_E[act], best_s[act] = bE, bS
    return best_s


def _hoptimal_scale_int4(x, s0, H_blk, all_scales, bs, cb, bd):
    """H-optimal scale for INT4, single H_blk (bs, bs)."""
    E0_sse = _int4_block_sse(x, s0, cb, bd)
    amax = x.abs().amax(-1)

    r0 = x - _int4_qd(x, s0, cb, bd)
    E0_H = (r0 * (r0 @ H_blk)).sum(-1)

    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / INT4_QMAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / INT4_D0

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
            clip = (xa.abs() - INT4_QMAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.full((xa.shape[0],), sf, device=dev)
            r = xa - _int4_qd(xa, sf_t, cb, bd)
            EH = (r * (r @ H_blk)).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_E[act], best_s[act] = bE, bS
    return best_s


def _hoptimal_scale_int4_per_row(x, s0, H_blks, all_scales, bs, cb, bd):
    """H-optimal scale for INT4, per-row H_blks (B, bs, bs)."""
    E0_sse = _int4_block_sse(x, s0, cb, bd)
    amax = x.abs().amax(-1)

    r0 = x - _int4_qd(x, s0, cb, bd)
    Hr0 = torch.bmm(r0.unsqueeze(1), H_blks).squeeze(1)
    E0_H = (r0 * Hr0).sum(-1)

    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / INT4_QMAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / INT4_D0

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
            clip = (xa.abs() - INT4_QMAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.full((xa.shape[0],), sf, device=dev)
            r = xa - _int4_qd(xa, sf_t, cb, bd)
            Hr = torch.bmm(r.unsqueeze(1), H_act).squeeze(1)
            EH = (r * Hr).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_E[act], best_s[act] = bE, bS
    return best_s


# ===================================================================
# Block fn factories for INT4
# ===================================================================

def make_int4_block_fn(bs, all_scales, cb, bd, H_blocks=None):
    """H-Optimal INT4 block fn: fn(w_block, col_idx) -> dequantized."""
    if H_blocks is not None:
        def fn(w, idx):
            x = w.float()
            s = _hoptimal_scale_int4(x, _base_int4(x), H_blocks[idx],
                                      all_scales, bs, cb, bd)
            return _int4_qd(x, s, cb, bd)
        return fn
    else:
        def fn(w, idx):
            x = w.float()
            s = _optimal_scale_int4(x, _base_int4(x), all_scales, bs, cb, bd)
            return _int4_qd(x, s, cb, bd)
        return fn


def make_int4_quant_factory(bs, all_scales, cb, bd):
    """Factory for col-ordered GPTQ: fn(H_blocks_perm) -> block_fn."""
    def factory(H_blocks_perm):
        return make_int4_block_fn(bs, all_scales, cb, bd, H_blocks_perm)
    return factory


def make_int4_batch_quant_fn(all_scales, bs, cb, bd):
    """Batch quantizer for row-ordered GPTQ: fn(w, H_blk_batch) -> dq."""
    def fn(w, H_blk):
        x = w.float()
        s = _hoptimal_scale_int4_per_row(x, _base_int4(x), H_blk,
                                          all_scales, bs, cb, bd)
        return _int4_qd(x, s, cb, bd)
    return fn


# ===================================================================
# Benchmark
# ===================================================================

def _ts(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def main():
    print("INT4 H-Optimal GPTQ Benchmark")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}")

    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done ({K}x{K})")

    all_scales = build_fp8_e4m3_scales(device=DEVICE)
    cb, bd = _build_int4_codebook(DEVICE)
    print(f"INT4 codebook: [{', '.join(f'{v:.4f}' for v in cb.tolist())}]\n")

    results = []

    for bs in [16, 32]:
        nblk = K // bs
        H_blocks = torch.stack([
            H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs]
            for j in range(nblk)
        ])

        print(f"{'=' * 90}")
        print(f"Block size: {bs}  ({nblk} blocks)")
        print(f"{'=' * 90}")

        label = f"INT4 H-Optimal BS={bs}"
        factory = make_int4_quant_factory(bs, all_scales, cb, bd)

        # --- Baseline ---
        fn = make_int4_block_fn(bs, all_scales, cb, bd, H_blocks)
        torch.cuda.synchronize(); t0 = time.time()
        Qb = quantize_no_gptq(W, fn, bs)
        torch.cuda.synchronize(); tb = time.time() - t0
        mb = compute_metrics(W, Qb, X); del Qb

        # --- Sequential GPTQ ---
        fn = make_int4_block_fn(bs, all_scales, cb, bd, H_blocks)
        torch.cuda.synchronize(); t0 = time.time()
        Qs, _ = gptq_strided(W, H, fn, block_size=bs)
        torch.cuda.synchronize(); ts = time.time() - t0
        ms = compute_metrics(W, Qs, X); del Qs

        # --- Column-ordered GPTQ ---
        fn = make_int4_block_fn(bs, all_scales, cb, bd, H_blocks)
        torch.cuda.synchronize(); t0 = time.time()
        Qc, _ = gptq_strided_ordered(W, H, fn, factory, block_size=bs)
        torch.cuda.synchronize(); tc = time.time() - t0
        mc = compute_metrics(W, Qc, X); del Qc

        torch.cuda.empty_cache()

        # --- Row-ordered GPTQ (batched) ---
        print(f"  [{label}] Row-ordered GPTQ...", flush=True)
        fn_loss = make_int4_block_fn(bs, all_scales, cb, bd, H_blocks)
        bq_fn = make_int4_batch_quant_fn(all_scales, bs, cb, bd)
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
            results.append((bs, mode, m, t))

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

    print(f"\n{'=' * 90}")
    print(f"{'Method':<35} {'BS':>3} {'Weight%':>10} {'Output%':>10} {'Time':>10}")
    print(f"{'-' * 90}")
    for bs, mode, m, t in results:
        name = f"{MODE_NAMES[mode]} INT4-H"
        print(f"{name:<35} {bs:>3} "
              f"{m['weight_error_pct']:>9.4f}% "
              f"{m['output_error_pct']:>9.4f}% "
              f"{_ts(t):>10}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
