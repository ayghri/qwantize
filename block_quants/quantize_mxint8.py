#!/usr/bin/env python
"""INT8 quantization with FP16 (E5M10) scales.

Symmetric INT8 codebook: q = round(w/s).clamp(-127, 127), dequant = q * s.

Two per-block scale policies:
  SSE-Opt:  s = fp16(<w, q> / <q, q>)           — no Hessian needed
  H-Opt:    s = fp16(<w, H q> / <q, H q>)        — minimizes output error per block

H-Opt is the right target for SPGL1: it minimises each block's contribution
to output error before SPGL1 compensates the remaining columns.

Bit budget: 8 (INT8) + 16/bs (scale)  →  bs=128: 8.125 b/w, bs=256: 8.0625 b/w

Usage:
    PYTHONPATH=. python experiments/quantize_mxint8.py --bs 128
    PYTHONPATH=. python experiments/quantize_mxint8.py --bs 256
"""

import argparse
import time

import torch

from qwantize.int8.reference import int8_quantize, int8_dequantize_block, Q_MAX
from qwantize.spgl1 import spgl1_lasso_reduced_batched

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def fp16_snap(x):
    return x.to(torch.float16).to(torch.float32)


def _signed(x, s):
    return int8_quantize(x, s.unsqueeze(-1))


def _dequant(x, s):
    su = s.unsqueeze(-1)
    return int8_dequantize_block(int8_quantize(x, su), su)


def naive_snap(x):
    """s = fp16(max|x| / 127)."""
    s = fp16_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))
    return _dequant(x, s), s


def sseopt_snap(x, max_iter=12):
    """SSE-iterative: s = fp16(<w, q> / <q, q>)."""
    s = fp16_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))
    for _ in range(max_iter):
        q = _signed(x, s)
        numer = (x * q).sum(-1)
        denom = (q * q).sum(-1)
        s_new = fp16_snap((numer / denom.clamp(min=1e-12)).clamp(min=1e-12))
        s_new = torch.where(denom > 1e-12, s_new, s)
        if torch.equal(s_new, s):
            break
        s = s_new
    return _dequant(x, s), s


def hopt_snap(x, H_blk, max_iter=12):
    """H-optimal iterative: s = fp16(<w, Hq> / <q, Hq>).

    Minimises (w - sq)^T H (w - sq) — the block's contribution to
    total output error ||X(W_q - W)^T||_F^2.
    """
    s = fp16_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))
    for _ in range(max_iter):
        q = _signed(x, s)          # (M, bs)
        Hq = q @ H_blk             # (M, bs)
        numer = (x * Hq).sum(-1)
        denom = (q * Hq).sum(-1)
        valid = (numer > 0) & (denom > 1e-12)
        s_new = fp16_snap(
            torch.where(valid, numer / denom.clamp(min=1e-12), s).clamp(min=1e-12)
        )
        if torch.equal(s_new, s):
            break
        s = s_new
    return _dequant(x, s), s


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
        diff = Xb @ W_dq_f.T - out_ref
        sse += diff.pow(2).sum().item()
        ref_sse += out_ref.pow(2).sum().item()
        del Xb, out_ref, diff
    return {
        "weight_error": w_err.item(),
        "weight_error_pct": (w_err / w_norm * 100).item(),
        "output_error": sse ** 0.5,
        "output_error_pct": (sse ** 0.5) / (ref_sse ** 0.5) * 100,
    }


# ---------------------------------------------------------------------------
# GPTQ-Seq + Naive
# ---------------------------------------------------------------------------

def gptq_seq_naive(W_orig, H, bs, damp=0.01):
    M, K = W_orig.shape
    nblk = K // bs
    W = W_orig.contiguous().clone()
    Hi = H.clone()
    Hi.diagonal().add_(damp * Hi.diagonal().mean())
    Hi = torch.cholesky_inverse(torch.linalg.cholesky(Hi)).contiguous()
    Q = torch.zeros_like(W)
    for j in range(nblk):
        cs, ce = j * bs, (j + 1) * bs
        w = W[:, cs:ce].clone()
        q, _ = naive_snap(w)
        Q[:, cs:ce] = q
        err = (w - q) / Hi.diagonal()[cs:ce].clone().unsqueeze(0)
        if ce < K:
            W[:, ce:] -= err @ Hi[cs:ce, ce:]
    return Q


# ---------------------------------------------------------------------------
# GPTQ-Ord + H-Opt (unconstrained H^-1 compensation)
# ---------------------------------------------------------------------------

def gptq_ord_hopt(W_orig, H, H_blocks, bs, damp=0.01):
    M, K = W_orig.shape
    nblk = K // bs
    dev = W_orig.device

    losses = torch.empty(nblk, device=dev)
    for j in range(nblk):
        w = W_orig[:, j*bs:(j+1)*bs]
        q, _ = hopt_snap(w, H_blocks[j])
        r = w - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    _, blk_perm = losses.sort(descending=True)
    col_perm = (blk_perm.unsqueeze(1) * bs + torch.arange(bs, device=dev).unsqueeze(0)).reshape(-1)

    W = W_orig[:, col_perm].contiguous().clone()
    H_p = H[col_perm][:, col_perm].contiguous()
    H_blk_p = torch.stack([H_p[j*bs:(j+1)*bs, j*bs:(j+1)*bs] for j in range(nblk)])
    Hi = H_p.clone()
    Hi.diagonal().add_(damp * Hi.diagonal().mean())
    Hi = torch.cholesky_inverse(torch.linalg.cholesky(Hi)).contiguous()

    Q = torch.zeros_like(W)
    for j in range(nblk):
        cs, ce = j * bs, (j + 1) * bs
        w = W[:, cs:ce].clone()
        q, _ = hopt_snap(w, H_blk_p[j])
        Q[:, cs:ce] = q
        err = (w - q) / Hi.diagonal()[cs:ce].clone().unsqueeze(0)
        if ce < K:
            W[:, ce:] -= err @ Hi[cs:ce, ce:]

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# GPTQ-Ord + H-Opt + SPGL1 compensation
# ---------------------------------------------------------------------------

def gptq_ord_hopt_spgl1(W_orig, H, H_blocks, bs,
                        tau_frac=1.0, spgl1_iters=10, max_line_iters=10,
                        matvec_dtype=torch.float16, verbose=True):
    M, K = W_orig.shape
    nblk = K // bs
    dev = W_orig.device

    losses = torch.empty(nblk, device=dev)
    for j in range(nblk):
        w = W_orig[:, j*bs:(j+1)*bs]
        q, _ = hopt_snap(w, H_blocks[j])
        r = w - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    _, blk_perm = losses.sort(descending=True)
    col_perm = (blk_perm.unsqueeze(1) * bs + torch.arange(bs, device=dev).unsqueeze(0)).reshape(-1)

    W0 = W_orig[:, col_perm].contiguous()
    W = W0.clone()
    H_p = H[col_perm][:, col_perm].contiguous()
    H_blk_p = torch.stack([H_p[j*bs:(j+1)*bs, j*bs:(j+1)*bs] for j in range(nblk)])

    Q = torch.zeros_like(W)
    Delta = torch.zeros_like(W)

    t_snap = t_grad = t_spgl1 = 0.0
    total_li = 0

    for j in range(nblk):
        cs, ce = j * bs, (j + 1) * bs
        rem = K - ce

        t0 = time.time()
        w = W[:, cs:ce].clone()
        q, _ = hopt_snap(w, H_blk_p[j])
        Q[:, cs:ce] = q
        Delta[:, cs:ce] = q - W0[:, cs:ce]
        torch.cuda.synchronize(); t_snap += time.time() - t0

        if rem == 0:
            break

        t0 = time.time()
        H_red = H_p[ce:, ce:]
        ATb = -(Delta @ H_p[:, ce:])
        b_norm_sq = (Delta * (Delta @ H_p)).sum(-1).clamp(min=0)
        torch.cuda.synchronize(); t_grad += time.time() - t0

        tau = tau_frac * ATb.abs().sum(-1) / H_red.diagonal().mean().clamp(min=1e-12)

        t0 = time.time()
        delta, info = spgl1_lasso_reduced_batched(
            H_red, ATb, b_norm_sq, tau=tau,
            max_iter=spgl1_iters, max_line_iters=max_line_iters,
            verbose=False, matvec_dtype=matvec_dtype,
        )
        torch.cuda.synchronize(); t_spgl1 += time.time() - t0
        total_li += info["n_line_iters"]

        W[:, ce:] += delta
        Delta[:, ce:] += delta

        if verbose and (j < 3 or j % 10 == 0 or j == nblk - 1):
            atb_med = ATb.abs().sum(-1).median().item()
            d_med = delta.abs().sum(-1).median().item()
            del ATb, delta, b_norm_sq
            print(f"    blk {j:3d}/{nblk}  rem={rem:>5}  "
                  f"ATb_med={atb_med:.2e}  δ_med={d_med:.2e}  "
                  f"snap={t_snap:.1f}s  grad={t_grad:.1f}s  spgl1={t_spgl1:.1f}s",
                  flush=True)
        else:
            del ATb, delta, b_norm_sq

    if verbose:
        print(f"  snap={t_snap:.1f}s  grad={t_grad:.1f}s  "
              f"spgl1={t_spgl1:.1f}s  line_iters={total_li}", flush=True)

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt(t): return f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=128, choices=[32, 64, 128, 256])
    p.add_argument("--tau-frac", type=float, default=1.0)
    p.add_argument("--spgl1-iters", type=int, default=10)
    p.add_argument("--max-line-iters", type=int, default=10)
    p.add_argument("--matvec-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--standalone", action="store_true",
                   help="run Naive/SSE-Opt/H-Opt without GPTQ instead")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-spgl1", action="store_true")
    args = p.parse_args()

    bs = args.bs
    matvec_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

    print(f"MXINT8  bs={bs}  FP16 scales  total b/w={8 + 16/bs:.4f}", flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    if X_cpu.dtype == torch.float32:
        X_cpu = X_cpu.to(torch.bfloat16)
    M, K = W.shape
    assert K % bs == 0, f"K={K} not divisible by bs={bs}"
    nblk = K // bs
    print(f"W: {M}x{K}   X(cpu): {X_cpu.shape[0]}x{X_cpu.shape[1]}  nblk={nblk}", flush=True)

    print("Building H...", end=" ", flush=True)
    H = torch.zeros(K, K, device=DEVICE)
    for t0 in range(0, X_cpu.shape[0], 4096):
        Xc = X_cpu[t0:t0+4096].to(DEVICE, non_blocking=True).float()
        H.addmm_(Xc.T, Xc)
        del Xc
    H /= X_cpu.shape[0]
    H_blocks = torch.stack([H[j*bs:(j+1)*bs, j*bs:(j+1)*bs] for j in range(nblk)])
    print("done", flush=True)

    results = []

    if args.standalone:
        for label, snap_fn in [
            ("Naive", lambda w, Hb: naive_snap(w)),
            ("SSE-Opt", lambda w, Hb: sseopt_snap(w)),
            ("H-Opt", lambda w, Hb: hopt_snap(w, Hb)),
        ]:
            print(f"\n[standalone]  {label}", flush=True)
            t0 = time.time()
            Q = torch.zeros_like(W)
            for j in range(nblk):
                w = W[:, j*bs:(j+1)*bs]
                q, _ = snap_fn(w, H_blocks[j])
                Q[:, j*bs:(j+1)*bs] = q
            t = time.time() - t0
            m = compute_metrics_streamed(W, Q, X_cpu)
            del Q; torch.cuda.empty_cache()
            results.append((label, m, t))
            print(f"  W%={m['weight_error_pct']:.4f}  O%={m['output_error_pct']:.4f}  {_fmt(t)}", flush=True)

        print(f"\n### MXINT8 bs={bs} FP16 scales — standalone\n")
        print("| Config | W% | O% | Time |")
        print("|:--|:--:|:--:|--:|")
        for name, m, t in results:
            print(f"| {name} | {m['weight_error_pct']:.2f}% | {m['output_error_pct']:.2f}% | {_fmt(t)} |")
        return

    if not args.skip_baseline:
        print("\n[seq]  GPTQ-Seq + Naive", flush=True)
        t0 = time.time()
        Q = gptq_seq_naive(W, H, bs)
        t = time.time() - t0
        m = compute_metrics_streamed(W, Q, X_cpu)
        del Q; torch.cuda.empty_cache()
        results.append(("GPTQ-Seq+Naive", m, t))
        print(f"  W%={m['weight_error_pct']:.4f}  O%={m['output_error_pct']:.4f}  {_fmt(t)}", flush=True)

        print("\n[ord]  GPTQ-Ord + H-Opt", flush=True)
        t0 = time.time()
        Q = gptq_ord_hopt(W, H, H_blocks, bs)
        t = time.time() - t0
        m = compute_metrics_streamed(W, Q, X_cpu)
        del Q; torch.cuda.empty_cache()
        results.append(("GPTQ-Ord+H-Opt", m, t))
        print(f"  W%={m['weight_error_pct']:.4f}  O%={m['output_error_pct']:.4f}  {_fmt(t)}", flush=True)

    if not args.skip_spgl1:
        print(f"\n[spgl1]  GPTQ-Ord + H-Opt + SPGL1  (tau={args.tau_frac}  iters={args.spgl1_iters}  li={args.max_line_iters})", flush=True)
        t0 = time.time()
        Q = gptq_ord_hopt_spgl1(
            W, H, H_blocks, bs,
            tau_frac=args.tau_frac,
            spgl1_iters=args.spgl1_iters,
            max_line_iters=args.max_line_iters,
            matvec_dtype=matvec_map[args.matvec_dtype],
        )
        t = time.time() - t0
        m = compute_metrics_streamed(W, Q, X_cpu)
        del Q; torch.cuda.empty_cache()
        results.append(("GPTQ-Ord+H-Opt+SPGL1", m, t))
        print(f"  W%={m['weight_error_pct']:.4f}  O%={m['output_error_pct']:.4f}  {_fmt(t)}", flush=True)

    print(f"\n### MXINT8 bs={bs} FP16 scales\n")
    print("| Config | W% | O% | Time |")
    print("|:--|:--:|:--:|--:|")
    for name, m, t in results:
        print(f"| {name} | {m['weight_error_pct']:.2f}% | {m['output_error_pct']:.2f}% | {_fmt(t)} |")


if __name__ == "__main__":
    main()
