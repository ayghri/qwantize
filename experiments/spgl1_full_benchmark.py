#!/usr/bin/env python
"""Comprehensive benchmark with absolute reconstruction norms.

Records for every method:
  - W reconstruction:  ||W_q - W_0||_F  AND  ||W_q - W_0||_F / ||W_0||_F
  - Output reconstruction: ||X(W_q - W_0)^T||_F  AND ratio over ||X W_0^T||_F
  - Reference norms ||W_0||_F and ||X W_0^T||_F (one-time, per layer)
  - Wall time per method

Saves results to JSONL for later analysis. Prints summary table.

Methods covered:
  - Naive snap (no GPTQ)
  - H-Opt snap (no GPTQ)
  - GPTQ-Seq + H-Opt
  - GPTQ-Ord + H-Opt        (closed-form H_inv compensation)
  - OBS-ordered + SPGL1 LASSO compensation  (our new method, no H_inv)
"""

import json
import time

import torch

from qwantize.nvfp4.reference import (
    nvfp4_naive, _fp8_e4m3_snap, build_fp8_e4m3_scales,
    fp4_quantize, fp4_dequantize, compute_block_sse,
    Q_MAX, D_0,
)
from qwantize.spgl1 import spgl1_lasso_reduced_batched

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
OUT_PATH = "experiments/spgl1_full_benchmark_results.jsonl"
BS = 16
DAMP = 0.01


def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


# ---------------------------------------------------------------------------
# Absolute + relative metrics (replaces compute_metrics for raw numbers)
# ---------------------------------------------------------------------------

def compute_full_metrics(W, W_q, X, w_ref, xw_ref_sq):
    """Compute both absolute and relative reconstruction norms.

    Args:
        W:   (M, K) original full-precision weights, float32.
        W_q: (M, K) quantized weights, float32.
        X:   (T, K) calibration activations, bf16 or float32.
        w_ref:    ||W||_F (scalar, float)
        xw_ref_sq: ||X W^T||_F^2 (scalar, float). We pass squared form for FP
                  stability since it accumulates as sum of squares.

    Returns dict with keys:
        weight_l2, weight_l2_rel, output_l2, output_l2_rel,
        weight_error_pct, output_error_pct (compat with old reporting).
    """
    diff = W_q.float() - W.float()
    w_err = diff.norm().item()                            # ||W_q - W||_F
    w_err_rel = w_err / w_ref                              # relative

    # ||X diff^T||_F^2  =  trace(diff X^T X diff^T)  =  sum over rows
    # Compute via XTX matvec
    T = X.shape[0]
    K = X.shape[1]
    chunk = 8192
    # Reuse global H if precomputed — caller can also pass it; we redo it here
    XTX = torch.zeros(K, K, device=W.device, dtype=torch.float32)
    for t0 in range(0, T, chunk):
        Xc = X[t0:t0 + chunk].float()
        XTX.addmm_(Xc.T, Xc)
    xdiff_sq = (diff @ XTX * diff).sum().item()           # ||X diff^T||_F^2
    xdiff_l2 = max(xdiff_sq, 0.0) ** 0.5
    xdiff_l2_rel = xdiff_l2 / (xw_ref_sq ** 0.5)

    return {
        "weight_l2": w_err,
        "weight_l2_rel": w_err_rel,
        "weight_error_pct": w_err_rel * 100.0,
        "output_l2": xdiff_l2,
        "output_l2_rel": xdiff_l2_rel,
        "output_error_pct": xdiff_l2_rel * 100.0,
    }


# ---------------------------------------------------------------------------
# Block snap (H-Opt) reused across methods
# ---------------------------------------------------------------------------

def _qd(x, s):
    su = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x, su), su)


def _base_nvfp4(x):
    return _fp8_e4m3_snap((x.abs().amax(-1) / Q_MAX).clamp(min=1e-12))


def hopt_block_snap(x, H_blk, all_scales, bs):
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
# Methods
# ---------------------------------------------------------------------------

def method_naive(W, X, H_blocks, all_scales, **kw):
    M, K = W.shape
    _, _, dq = nvfp4_naive(W.view(M, K // BS, BS), return_dequant=True)
    return dq.view(M, K)


def method_hopt_only(W, X, H_blocks, all_scales, **kw):
    M, K = W.shape
    nblk = K // BS
    out = torch.empty_like(W)
    for j in range(nblk):
        cs = j * BS
        out[:, cs:cs + BS] = hopt_block_snap(
            W[:, cs:cs + BS], H_blocks[j], all_scales, BS,
        )
    return out


def _build_inverse_h(H, K, damp=DAMP):
    Hi = H.clone()
    dmu = damp * Hi.diagonal().mean()
    Hi.diagonal().add_(dmu)
    L = torch.linalg.cholesky(Hi)
    Hi = torch.cholesky_inverse(L).contiguous()
    del L
    return Hi


def _gptq_compensate(W_perm, Hi, j, bs, K, M):
    cs = j * bs
    ce = cs + bs
    rem = K - ce
    if rem == 0:
        return
    h_diag = torch.as_strided(
        Hi, size=(bs,), stride=(K + 1,), storage_offset=cs * K + cs,
    ).clone()
    err = (W_perm[:, cs:ce] - W_perm[:, cs:ce]) / 1.0  # placeholder
    # actually caller supplies err
    raise RuntimeError("unused helper; inline compensation below")


def method_gptq_seq(W, X, H_blocks, all_scales, H=None, **kw):
    """Plain GPTQ (no ordering) + H-Opt block snap."""
    M, K = W.shape
    nblk = K // BS
    Hi = _build_inverse_h(H, K)

    W = W.clone().contiguous()
    Q = torch.zeros_like(W)
    for j in range(nblk):
        cs = j * BS
        ce = cs + BS
        rem = K - ce
        w_blk = W[:, cs:ce].clone()
        q_blk = hopt_block_snap(w_blk, H_blocks[j], all_scales, BS)
        Q[:, cs:ce] = q_blk
        h_diag = torch.as_strided(
            Hi, size=(BS,), stride=(K + 1,), storage_offset=cs * K + cs,
        ).clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = torch.as_strided(
                Hi, size=(BS, rem), stride=(K, 1), storage_offset=cs * K + ce,
            )
            w_rem = torch.as_strided(
                W, size=(M, rem), stride=(K, 1), storage_offset=ce,
            )
            w_rem.sub_(err @ h_cross)
    return Q


def _block_saliencies(W, H_blocks, all_scales, bs):
    nblk = W.shape[1] // bs
    losses = torch.empty(nblk, device=W.device)
    for j in range(nblk):
        w_blk = W[:, j * bs:(j + 1) * bs]
        q = hopt_block_snap(w_blk, H_blocks[j], all_scales, bs)
        r = w_blk - q
        losses[j] = (r * (r @ H_blocks[j])).sum()
    return losses


def method_gptq_ord(W, X, H_blocks, all_scales, H=None, **kw):
    """GPTQ-Ord + H-Opt block snap (uses H_inv)."""
    M, K = W.shape
    nblk = K // BS
    dev = W.device

    losses = _block_saliencies(W, H_blocks, all_scales, BS)
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * BS
        + torch.arange(BS, device=dev).unsqueeze(0)
    ).reshape(-1)

    W_perm = W[:, col_perm].contiguous().clone()
    H_perm = H[col_perm][:, col_perm].contiguous()
    H_blocks_perm = torch.stack([
        H_perm[j * BS:(j + 1) * BS, j * BS:(j + 1) * BS] for j in range(nblk)
    ])
    Hi = _build_inverse_h(H_perm, K)

    Q = torch.zeros_like(W_perm)
    for j in range(nblk):
        cs = j * BS
        ce = cs + BS
        rem = K - ce
        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap(w_blk, H_blocks_perm[j], all_scales, BS)
        Q[:, cs:ce] = q_blk
        h_diag = torch.as_strided(
            Hi, size=(BS,), stride=(K + 1,), storage_offset=cs * K + cs,
        ).clone()
        err = (w_blk - q_blk) / h_diag.unsqueeze(0)
        if rem > 0:
            h_cross = torch.as_strided(
                Hi, size=(BS, rem), stride=(K, 1), storage_offset=cs * K + ce,
            )
            w_rem = torch.as_strided(
                W_perm, size=(M, rem), stride=(K, 1), storage_offset=ce,
            )
            w_rem.sub_(err @ h_cross)

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    return Q[:, inv_perm]


def method_obs_spgl1(W, X, H_blocks, all_scales, H=None,
                     tau_frac=1.0, spgl1_iters=10, **kw):
    """OBS-ordered + H-Opt snap + SPGL1 LASSO compensation. No H_inv."""
    M, K = W.shape
    nblk = K // BS
    dev = W.device

    losses = _block_saliencies(W, H_blocks, all_scales, BS)
    _, blk_perm = losses.sort(descending=True)
    col_perm = (
        blk_perm.unsqueeze(1) * BS
        + torch.arange(BS, device=dev).unsqueeze(0)
    ).reshape(-1)

    W_orig_perm = W[:, col_perm].contiguous()
    W_perm = W_orig_perm.clone()
    H_perm = H[col_perm][:, col_perm].contiguous()
    H_blocks_perm = torch.stack([
        H_perm[j * BS:(j + 1) * BS, j * BS:(j + 1) * BS] for j in range(nblk)
    ])

    Q = torch.zeros_like(W_perm)
    Delta_eff = torch.zeros_like(W_perm)

    for j in range(nblk):
        cs = j * BS
        ce = cs + BS
        rem_size = K - ce

        w_blk = W_perm[:, cs:ce].clone()
        q_blk = hopt_block_snap(w_blk, H_blocks_perm[j], all_scales, BS)
        Q[:, cs:ce] = q_blk
        Delta_eff[:, cs:ce] = q_blk - W_orig_perm[:, cs:ce]

        if rem_size == 0:
            break

        H_red = H_perm[ce:, ce:]
        ATb = -(Delta_eff @ H_perm[:, ce:])
        b_norm_sq = (Delta_eff * (Delta_eff @ H_perm)).sum(-1).clamp(min=0)

        diag_scale = H_red.diagonal().mean().clamp(min=1e-12)
        ref_l1 = ATb.abs().sum(-1) / diag_scale
        tau_vec = tau_frac * ref_l1

        delta, _ = spgl1_lasso_reduced_batched(
            H_red, ATb, b_norm_sq, tau=tau_vec,
            max_iter=spgl1_iters, verbose=False,
        )
        W_perm[:, ce:] += delta
        Delta_eff[:, ce:] += delta

    inv_perm = torch.empty_like(col_perm)
    inv_perm[col_perm] = torch.arange(K, device=dev)
    return Q[:, inv_perm]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    print("Comprehensive benchmark (absolute + relative norms)")
    print(f"  W_PATH={W_PATH}  X_PATH={X_PATH}", flush=True)

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True)
    if X.dtype == torch.float32:
        X = X.to(torch.bfloat16)
    M, K = W.shape
    T = X.shape[0]
    print(f"W: {M}x{K}  X: {T}x{K} ({X.dtype})", flush=True)

    print("Building Hessian (chunked)...", flush=True)
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    XTX = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    chunk = 8192
    for t0 in range(0, T, chunk):
        Xc = X[t0:t0 + chunk].float()
        XTX.addmm_(Xc.T, Xc)
    H = XTX / T

    nblk = K // BS
    H_blocks = torch.stack([
        H[j * BS:(j + 1) * BS, j * BS:(j + 1) * BS] for j in range(nblk)
    ])
    all_scales = build_fp8_e4m3_scales(device=DEVICE)

    # Reference norms (one-time, recorded so absolute values can be derived later)
    w_ref = W.norm().item()                              # ||W||_F
    xw_ref_sq = (W @ XTX * W).sum().item()                # ||X W^T||_F^2
    xw_ref = xw_ref_sq ** 0.5

    print(f"\nReference norms:")
    print(f"  ||W_0||_F      = {w_ref:.4e}")
    print(f"  ||X W_0^T||_F  = {xw_ref:.4e}    (T*M elements = {T*M:,})")
    print(f"  per-element output RMS = {xw_ref / (T * M) ** 0.5:.6e}",
          flush=True)

    methods = [
        ("Naive",                method_naive,    {}),
        ("H-Opt (no GPTQ)",      method_hopt_only,{}),
        ("GPTQ-Seq + H-Opt",     method_gptq_seq, {"H": H}),
        ("GPTQ-Ord + H-Opt",     method_gptq_ord, {"H": H}),
        ("OBS+SPGL1 (tau=0.5, 10 it)",
                                 method_obs_spgl1,
                                 {"H": H, "tau_frac": 0.5, "spgl1_iters": 10}),
        ("OBS+SPGL1 (tau=1.0, 10 it)",
                                 method_obs_spgl1,
                                 {"H": H, "tau_frac": 1.0, "spgl1_iters": 10}),
        ("OBS+SPGL1 (tau=2.0, 10 it)",
                                 method_obs_spgl1,
                                 {"H": H, "tau_frac": 2.0, "spgl1_iters": 10}),
    ]

    rows = []
    print(f"\n{'method':<33}  {'|W_q-W|_F':>12} {'rel':>8}  "
          f"{'|X(W_q-W)^T|_F':>16} {'rel':>8}  {'time':>8}")
    print("-" * 100)

    for name, fn, kwargs in methods:
        torch.cuda.synchronize()
        t0 = time.time()
        W_q = fn(W, X, H_blocks, all_scales, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        # Compute metrics using cached XTX (avoid rebuilding)
        diff = W_q.float() - W.float()
        w_l2 = diff.norm().item()
        x_diff_sq = (diff @ XTX * diff).sum().item()
        x_diff_l2 = max(x_diff_sq, 0.0) ** 0.5
        w_rel = w_l2 / w_ref
        o_rel = x_diff_l2 / xw_ref

        row = {
            "method": name,
            "kwargs": {k: v for k, v in kwargs.items() if k != "H"},
            "weight_l2": w_l2,
            "weight_l2_rel": w_rel,
            "weight_error_pct": w_rel * 100,
            "output_l2": x_diff_l2,
            "output_l2_rel": o_rel,
            "output_error_pct": o_rel * 100,
            "elapsed_s": elapsed,
            "w_ref": w_ref,
            "xw_ref": xw_ref,
            "M": M, "K": K, "T": T, "bs": BS,
        }
        rows.append(row)

        print(f"{name:<33}  {w_l2:>12.4e} {w_rel*100:>7.4f}%  "
              f"{x_diff_l2:>16.4e} {o_rel*100:>7.4f}%  {_fmt(elapsed):>8}",
              flush=True)

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
