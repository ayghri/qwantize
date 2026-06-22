#!/usr/bin/env python
"""GPTQ-Ord + H-Opt + SPGL1 at block size 64 with E4M3 scales.

Parameterized by codebook (FP4 or INT4). Per-block H-optimal scale is found
by bounded search over the 126 positive FP8 E4M3 values (the same algorithm
used in nvfp4_optimal_hessian / nvint4_optimal_hessian), and the
post-snap compensation across remaining columns is solved by SPGL1 LASSO
in reduced (Gram) form.

X is streamed from CPU to keep GPU memory usage low.
"""

import argparse
import time

import torch

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap, build_fp8_e4m3_scales,
    fp4_quantize as _fp4_quant,
    fp4_dequantize as _fp4_dequant,
)
from qwantize.nvint4.reference import (
    int4_quantize as _int4_quant,
    int4_dequantize_block as _int4_dequant,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ---------------------------------------------------------------------------
# Codebook abstraction
# ---------------------------------------------------------------------------

class FP4Codebook:
    name = "FP4"
    Q_MAX = 6.0
    D_0 = 0.25
    quant = staticmethod(_fp4_quant)
    dequant = staticmethod(_fp4_dequant)


class INT4Codebook:
    name = "INT4"
    Q_MAX = 7.0
    D_0 = 0.5
    quant = staticmethod(_int4_quant)
    dequant = staticmethod(_int4_dequant)


def get_codebook(name):
    return {"fp4": FP4Codebook, "int4": INT4Codebook}[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def _qd(cb, x, s):
    su = s.unsqueeze(-1)
    return cb.dequant(cb.quant(x, su), su)


def _block_sse(cb, x, s):
    su = s.unsqueeze(-1) if s.dim() == 1 else s
    quants = cb.quant(x, su)
    dq = cb.dequant(quants, su)
    return (x - dq).pow(2).sum(dim=-1)


def compute_metrics_streamed(W, W_dq, X_cpu, batch=4096):
    m = {}
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    m["weight_error"] = w_err.item()
    m["weight_error_pct"] = (w_err / w_norm * 100).item()
    W_f = W.float()
    W_dq_f = W_dq.float()
    sse = ref_sse = 0.0
    for i in range(0, X_cpu.shape[0], batch):
        Xb = X_cpu[i:i + batch].to(W.device, non_blocking=True).float()
        out_ref = Xb @ W_f.T
        out_dq = Xb @ W_dq_f.T
        sse += (out_dq - out_ref).pow(2).sum().item()
        ref_sse += out_ref.pow(2).sum().item()
        del Xb, out_ref, out_dq
    m["output_error"] = sse ** 0.5
    m["output_error_pct"] = (sse ** 0.5) / (ref_sse ** 0.5) * 100
    return m


# ---------------------------------------------------------------------------
# Per-block H-optimal snap (E4M3 grid search)
# ---------------------------------------------------------------------------

def hopt_block_snap(cb, x, H_blk, scale_table, bs):
    """Per-row H-optimal block snap over E4M3 candidates."""
    s0 = _fp8_e4m3_snap((x.abs().amax(-1) / cb.Q_MAX).clamp(min=1e-12))
    E0_sse = _block_sse(cb, x, s0)
    amax = x.abs().amax(-1)

    r0 = x - _qd(cb, x, s0)
    E0_H = (r0 * (r0 @ H_blk)).sum(-1)
    best_s, best_E = s0.clone(), E0_H.clone()

    noise = x.pow(2).sum(-1) <= E0_sse
    s_min = ((amax - E0_sse.sqrt()) / cb.Q_MAX).clamp(min=0)
    sa, _ = x.abs().sort(-1)
    ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
    noise |= ks >= bs
    s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / cb.D_0

    act = ~noise
    if act.any():
        xa, smn, smx = x[act], s_min[act], s_max[act]
        bE, bS = best_E[act].clone(), best_s[act].clone()
        dev = x.device
        for sv in scale_table:
            sf = sv.item()
            ok = (sf >= smn) & (sf <= smx)
            if not ok.any():
                continue
            clip = (xa.abs() - cb.Q_MAX * sf).clamp(min=0).pow(2).sum(-1)
            ev = ok & (clip < bE * 10)
            if not ev.any():
                continue
            sf_t = torch.full((xa.shape[0],), sf, device=dev)
            r = xa - _qd(cb, xa, sf_t)
            EH = (r * (r @ H_blk)).sum(-1)
            imp = ev & (EH < bE)
            bE[imp], bS[imp] = EH[imp], sf
        best_s[act] = bS

    return _qd(cb, x, best_s)


# ---------------------------------------------------------------------------
# GPTQ-Ord + H-Opt baseline (no SPGL1)
# ---------------------------------------------------------------------------

def gptq_strided_baseline(cb, W_orig, H, bs, scale_table, H_blocks,
                          damp=0.01, cpu_chol=False):
    M, K = W_orig.shape
    nblk = K // bs
    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap(cb, w_blk, H_blocks[j], scale_table, bs)
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

    if cpu_chol:
        H_cpu = H_perm.cpu().double()
        H_cpu.diagonal().add_(damp * H_cpu.diagonal().mean())
        L = torch.linalg.cholesky(H_cpu)
        Hi = torch.cholesky_inverse(L).to(W_orig.device, dtype=torch.float32).contiguous()
        del L, H_cpu
    else:
        Hi = H_perm.clone()
        Hi.diagonal().add_(damp * Hi.diagonal().mean())
        L = torch.linalg.cholesky(Hi)
        Hi = torch.cholesky_inverse(L).contiguous()
        del L
    H_perm.untyped_storage().resize_(0)
    torch.cuda.empty_cache()

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem = K - ce
        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap(cb, w_blk, H_blocks_perm[j], scale_table, bs)
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
# Block saliencies (block-H ordering)
# ---------------------------------------------------------------------------

def _compute_block_saliencies(cb, W_orig, bs, scale_table, H_blocks):
    M, K = W_orig.shape
    nblk = K // bs
    losses = torch.empty(nblk, device=W_orig.device)
    for j in range(nblk):
        w_blk = W_orig[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap(cb, w_blk, H_blocks[j], scale_table, bs)
        r = w_blk - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    return losses


# ---------------------------------------------------------------------------
# GPTQ-Ord + H-Opt + SPGL1 compensation
# ---------------------------------------------------------------------------

def gptq_strided_spgl1(cb, W_orig, H, bs, scale_table, H_blocks,
                       tau_frac=1.0, spgl1_iters=10,
                       matvec_dtype=None, verbose=True, m_chunk=None):
    M, K = W_orig.shape
    nblk = K // bs
    dev = W_orig.device

    if verbose:
        print(f"  ordering=block_h", flush=True)
    losses = _compute_block_saliencies(cb, W_orig, bs, scale_table, H_blocks)
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
    H.untyped_storage().resize_(0)
    H_blocks.untyped_storage().resize_(0)
    torch.cuda.empty_cache()

    Q = torch.zeros_like(W_perm)
    Delta_eff = torch.zeros_like(W_perm)

    if verbose:
        print(f"  spgl1_iters={spgl1_iters}  tau_frac={tau_frac}  "
              f"m_chunk={m_chunk}", flush=True)
    t_snap = t_spgl1 = t_grad = t_apply = 0.0
    total_line_iters = 0

    for j in range(nblk):
        cs = j * bs
        ce = cs + bs
        rem_size = K - ce

        t0 = time.time()
        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap(cb, w_blk, H_blocks_perm[j], scale_table, bs)
        Q[:, cs:ce] = q_blk
        Delta_eff[:, cs:ce] = q_blk - W_orig_perm[:, cs:ce]
        torch.cuda.synchronize()
        t_snap += time.time() - t0

        if rem_size == 0:
            break

        H_red = H_perm[ce:, ce:]
        diag_scale = H_red.diagonal().mean().clamp(min=1e-12)

        t0 = time.time()
        # Compute ATb and b_norm_sq in M-chunks (memory hygiene)
        if m_chunk is None or m_chunk >= M:
            ATb = -(Delta_eff @ H_perm[:, ce:])
            Delta_H = Delta_eff @ H_perm
            b_norm_sq = (Delta_eff * Delta_H).sum(-1).clamp(min=0)
            del Delta_H
        else:
            ATb = torch.empty(M, rem_size, device=dev, dtype=Delta_eff.dtype)
            b_norm_sq = torch.empty(M, device=dev, dtype=Delta_eff.dtype)
            for r0 in range(0, M, m_chunk):
                r1 = min(r0 + m_chunk, M)
                d_chunk = Delta_eff[r0:r1]
                ATb[r0:r1] = -(d_chunk @ H_perm[:, ce:])
                dH = d_chunk @ H_perm
                b_norm_sq[r0:r1] = (d_chunk * dH).sum(-1).clamp(min=0)
                del dH
        torch.cuda.synchronize()
        t_grad += time.time() - t0

        ref_l1 = ATb.abs().sum(-1) / diag_scale
        tau_vec = tau_frac * ref_l1

        t0 = time.time()
        if m_chunk is None or m_chunk >= M:
            delta, info = spgl1_lasso_reduced_batched(
                H_red, ATb, b_norm_sq, tau=tau_vec,
                max_iter=spgl1_iters, verbose=False,
                matvec_dtype=matvec_dtype,
            )
            total_line_iters += info["n_line_iters"]
        else:
            delta = torch.empty_like(ATb)
            for r0 in range(0, M, m_chunk):
                r1 = min(r0 + m_chunk, M)
                d, info = spgl1_lasso_reduced_batched(
                    H_red, ATb[r0:r1], b_norm_sq[r0:r1], tau=tau_vec[r0:r1],
                    max_iter=spgl1_iters, verbose=False,
                    matvec_dtype=matvec_dtype,
                )
                delta[r0:r1] = d
                total_line_iters += info["n_line_iters"]
                del d
        torch.cuda.synchronize()
        t_spgl1 += time.time() - t0

        t0 = time.time()
        W_perm[:, ce:] += delta
        Delta_eff[:, ce:] += delta
        atb_med = ATb.abs().sum(-1).median().item() if verbose else 0.0
        delta_med = delta.abs().sum(-1).median().item() if verbose else 0.0
        del delta, ATb, b_norm_sq, tau_vec
        torch.cuda.synchronize()
        t_apply += time.time() - t0

        if verbose and (j < 3 or j % 20 == 0 or j == nblk - 1):
            print(f"    block {j:4d}/{nblk}  K_rem={rem_size:>5}  "
                  f"||ATb||_1 med={atb_med:.3e}  "
                  f"||δ||_1 med={delta_med:.3e}  "
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
    p.add_argument("--codebook", choices=["fp4", "int4"], default="fp4")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--tau-frac", type=float, default=1.0)
    p.add_argument("--spgl1-iters", type=int, default=10)
    p.add_argument("--matvec-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp16")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-spgl1", action="store_true")
    p.add_argument("--cpu-chol", action="store_true")
    p.add_argument("--m-chunk", type=int, default=None)
    args = p.parse_args()

    cb = get_codebook(args.codebook)

    print(f"GPTQ-Ord + H-Opt + SPGL1   |   {cb.name} + FP8 E4M3 scales")
    print(f"  bs={args.bs}  scale b/w = {8/args.bs:.3f}  "
          f"total b/w = {4 + 8/args.bs:.3f}", flush=True)
    print(f"  tau_frac={args.tau_frac}  spgl1_iters={args.spgl1_iters}  "
          f"matvec={args.matvec_dtype}", flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    print(f"W: {M}x{K}  X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]} ({X_cpu.dtype})",
          flush=True)
    print(f"||W||_F = {W.norm().item():.4e}", flush=True)

    bs = args.bs
    nblk = K // bs
    assert K % bs == 0

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
    scale_table = build_fp8_e4m3_scales(device=DEVICE)

    if not args.skip_baseline:
        print(f"\n[baseline] GPTQ-Ord + H-Opt ({cb.name} E4M3)...", flush=True)
        t0 = time.time()
        Q_base = gptq_strided_baseline(cb, W, H, bs, scale_table, H_blocks,
                                       cpu_chol=args.cpu_chol)
        t_base = time.time() - t0
        m_base = compute_metrics_streamed(W, Q_base, X_cpu)
        print(f"  W%={m_base['weight_error_pct']:.4f}  "
              f"O%={m_base['output_error_pct']:.4f}  time={_fmt(t_base)}",
              flush=True)
        print(f"  ||Wq-W||_F={m_base['weight_error']:.4e}  "
              f"||X(Wq-W)^T||_F={m_base['output_error']:.4e}", flush=True)
        del Q_base; torch.cuda.empty_cache()

        # Re-build H since baseline freed it (resize_)
        print("Rebuilding Hessian after baseline...", end=" ", flush=True)
        H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
        for t0 in range(0, X_cpu.shape[0], chunk):
            Xc = X_cpu[t0:t0 + chunk].to(DEVICE, non_blocking=True).float()
            H.addmm_(Xc.T, Xc)
            del Xc
        H /= X_cpu.shape[0]
        H_blocks = torch.stack([
            H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(nblk)
        ])
        print("done", flush=True)

    if args.skip_spgl1:
        return

    print(f"\n[new] GPTQ-Ord + H-Opt + SPGL1 ({cb.name} E4M3)...", flush=True)
    matvec_dtype_map = {"fp32": None,
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16}
    t0 = time.time()
    Q_spgl1 = gptq_strided_spgl1(
        cb, W, H, bs, scale_table, H_blocks,
        tau_frac=args.tau_frac, spgl1_iters=args.spgl1_iters,
        matvec_dtype=matvec_dtype_map[args.matvec_dtype], verbose=True,
        m_chunk=args.m_chunk,
    )
    t_spgl1_total = time.time() - t0
    m_spgl1 = compute_metrics_streamed(W, Q_spgl1, X_cpu)
    print(f"  W%={m_spgl1['weight_error_pct']:.4f}  "
          f"O%={m_spgl1['output_error_pct']:.4f}  time={_fmt(t_spgl1_total)}",
          flush=True)
    print(f"  ||Wq-W||_F={m_spgl1['weight_error']:.4e}  "
          f"||X(Wq-W)^T||_F={m_spgl1['output_error']:.4e}", flush=True)


if __name__ == "__main__":
    main()
