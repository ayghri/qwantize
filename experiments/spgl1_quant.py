#!/usr/bin/env python
"""SPGL1 / SpaRSA quantization with moving FP4 target.

Implements the gradual L1 quantization scheme from
notes/L1_Quantization_via_SPGL1.md and tracked in
notes/progress_track_spgl1.md.

Per-iteration:
  1. Smooth gradient  g = (W - W_0) @ H
  2. Block-wise Barzilai-Borwein step alpha_g
  3. Gradient step    U = W - alpha_g * g
  4. Re-snap to FP4 grid: Q_{t+1} = quant_fn(U)   (moving target)
  5. Priority weights c_g per group (alpha controls how aggressively
     important blocks are pushed to quantize first)
  6. Block soft-thresholding around Q_{t+1} with threshold
     tau_g = lambda * alpha_g * c_g
  7. lambda *= lambda_grow

Final: hard FP4 snap of the converged iterate.

Usage:
    python experiments/spgl1_quant.py
"""

import argparse
import json
import time

import torch

from qwantize.nvfp4.reference import (
    fp4_quantize,
    fp4_dequantize,
    _fp8_e4m3_snap,
    build_fp8_e4m3_scales,
    compute_block_sse,
    Q_MAX,
    D_0,
)
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ===================================================================
# Snap operators: take a (M, ncb, bs) tensor, return (M, ncb, bs)
# dequantized FP4 values on a per-block FP8-E4M3 scale.
# ===================================================================

def _naive_snap(x_3d):
    """Per-block naive FP4 snap: s = FP8(amax/6)."""
    amax = x_3d.abs().amax(dim=-1)
    s = _fp8_e4m3_snap((amax / Q_MAX).clamp(min=1e-12))
    s_b = s.unsqueeze(-1)
    return fp4_dequantize(fp4_quantize(x_3d, s_b), s_b)


def _hoptimal_snap_factory(H_blocks, all_scales):
    """Build a per-block H-optimal FP4 snap that re-scans the FP8 scale grid.

    H_blocks: (ncb, bs, bs) block Hessians (col-wise).
    all_scales: pre-built sorted FP8 E4M3 scale candidates.
    """
    def snap(x_3d):
        M, ncb, bs = x_3d.shape
        x = x_3d.reshape(-1, bs)
        amax = x.abs().amax(-1)
        s0 = _fp8_e4m3_snap((amax / Q_MAX).clamp(min=1e-12))

        # Hessian-weighted baseline error E_H(s0)
        q0 = fp4_quantize(x, s0.unsqueeze(-1))
        dq0 = fp4_dequantize(q0, s0.unsqueeze(-1))
        r0 = x - dq0
        r0_3d = r0.reshape(M, ncb, bs)
        Hr0 = torch.einsum("jab,mjb->mja", H_blocks, r0_3d)
        E0_H = (r0_3d * Hr0).sum(-1).reshape(-1)

        # SSE-based bounding for pruning
        E0_sse = compute_block_sse(x, s0)
        noise = x.pow(2).sum(-1) <= E0_sse
        s_min = ((amax - E0_sse.sqrt()) / Q_MAX).clamp(min=0)
        sa, _ = x.abs().sort(-1)
        ks = (sa.pow(2).cumsum(-1) <= E0_sse.unsqueeze(-1)).sum(-1)
        noise |= ks >= bs
        s_max = sa.gather(-1, ks.clamp(max=bs - 1).unsqueeze(-1)).squeeze(-1) / D_0

        best_s = s0.clone()
        best_E = E0_H.clone()

        act = ~noise
        if act.any():
            xa = x[act]
            smn, smx = s_min[act], s_max[act]
            bE, bS = best_E[act].clone(), best_s[act].clone()
            act_idx = act.nonzero(as_tuple=True)[0]
            act_j = act_idx % ncb
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
                sf_t = torch.tensor(sf, device=dev)
                qq = fp4_quantize(xa, sf_t)
                dqs = fp4_dequantize(qq, sf_t)
                r = xa - dqs
                EH = torch.empty(xa.shape[0], device=dev)
                chunk = 4096
                for c0 in range(0, xa.shape[0], chunk):
                    c1 = min(c0 + chunk, xa.shape[0])
                    Hc = H_blocks[act_j[c0:c1]]
                    Hr = torch.bmm(Hc, r[c0:c1].unsqueeze(-1)).squeeze(-1)
                    EH[c0:c1] = (r[c0:c1] * Hr).sum(-1)
                imp = ev & (EH < bE)
                bE[imp], bS[imp] = EH[imp], sf
            best_s[act] = bS

        # Final dequant
        final_q = fp4_quantize(x, best_s.unsqueeze(-1))
        final_dq = fp4_dequantize(final_q, best_s.unsqueeze(-1))
        return final_dq.reshape(M, ncb, bs)

    return snap


# ===================================================================
# Core SpaRSA / Group L1 with moving FP4 target
# ===================================================================

def _estimate_lipschitz(H, n_iters=10):
    """Power iteration estimate of ||H||_op for step-size cap."""
    K = H.shape[0]
    v = torch.randn(K, device=H.device, dtype=H.dtype)
    v /= v.norm()
    for _ in range(n_iters):
        v = H @ v
        v /= v.norm().clamp(min=1e-30)
    L = (v @ (H @ v)).item()
    return max(L, 1e-12)


def sparsa_group_l1_lock(
    W0, H, snap_fn, H_blocks,
    block_size=16,
    alpha_priority=1.0,
    lam_init=1e-5,
    lam_grow=1.02,
    max_iter=500,
    eta_min=1e-8,
    eta_max=None,
    tol=1e-5,
    refresh_every=1,
    warmup_iters=0,
    verbose=False,
):
    """Moving-target SpaRSA with hard lock-at-snap-time.

    At each iteration:
      - Locked blocks: W is hard-pinned to its lock-point Q_locked
        (contributes to gradient via (Q_locked - W0) — propagates
        compensation to active blocks).
      - Active blocks: gradient descent + L1 prox with moving target
        Q_active = snap(U). When the prox step would fully snap a block
        (shrink → 0), its current Q_active becomes Q_locked, the block
        is added to the locked set, and W is pinned forever after.

    Args same as before; refresh_every controls how often Q_active is
    re-snapped for active blocks (1 = every iter).
    """
    """Gradual Group-L1 SpaRSA with moving FP4 target.

    Args:
        W0: (M, K) original full-precision weights.
        H: (K, K) full Hessian.
        snap_fn: callable (M, ncb, bs) -> (M, ncb, bs) FP4-dequantized target.
        H_blocks: (ncb, bs, bs) block Hessians for priority weights.
        block_size: FP4 block size (also the L1 group size for now).
        alpha_priority: $\\alpha$ in $c_g = ||H_g \\Delta w||_2 \\cdot s_g^\\alpha$.
            0 -> all blocks quantize together. 1 -> important blocks first.
        lam_init: initial L1 strength.
        lam_grow: geometric growth factor per iteration.
        max_iter: maximum outer iterations.
        eta_{min,max}: BB step clipping bounds.
        tol: stop when ||W_t - W_{t-1}||_2 / ||W_t|| < tol.

    Returns:
        W_dq: (M, K) final FP4-quantized weights.
        history: dict with diagnostic time series.
    """
    M, K = W0.shape
    assert K % block_size == 0
    bs = block_size
    ncb = K // bs

    W = W0.clone()
    W_prev = W0.clone()
    g_prev = torch.zeros_like(W)
    lam = lam_init

    # Lock state
    locked = torch.zeros(M, ncb, dtype=torch.bool, device=W.device)
    Q_locked = snap_fn(W.view(M, ncb, bs)).view(M, K)   # initial; updated at lock time

    def _compute_cg(Q_now):
        d3 = (W0 - Q_now).view(M, ncb, bs)
        Hd = torch.einsum("jab,mjb->mja", H_blocks, d3)
        q = (d3 * Hd).sum(-1).clamp(min=0)
        n = Hd.norm(dim=-1)
        c = n * q.pow(alpha_priority)
        return c / c.median().clamp(min=1e-30)

    history = {
        "iter": [],
        "lambda": [],
        "n_snapped": [],
        "loss": [],
        "rel_step": [],
    }

    for t in range(max_iter):
        # 1. Enforce locked blocks
        W_3d = W.view(M, ncb, bs)
        Q_locked_3d = Q_locked.view(M, ncb, bs)
        if locked.any():
            W_3d_new = torch.where(locked.unsqueeze(-1), Q_locked_3d, W_3d)
            W = W_3d_new.view(M, K)

        diff = W - W0
        g = diff @ H
        loss = 0.5 * (diff * g).sum().item()

        # 2. Block-wise BB step
        if t > 0:
            s = (W - W_prev).view(M, ncb, bs)
            y = (g - g_prev).view(M, ncb, bs)
            s_dot_s = (s * s).sum(-1)
            s_dot_y = (s * y).sum(-1)
            ok = s_dot_y > 1e-12
            alpha_bb = torch.where(
                ok,
                s_dot_s / s_dot_y.clamp(min=1e-12),
                torch.full_like(s_dot_s, 1.0),
            ).clamp(eta_min, eta_max)
        else:
            alpha_bb = torch.full((M, ncb), 1.0, device=W.device)

        # 3. Gradient step (no step for locked → kept at Q_locked)
        alpha_full = alpha_bb.unsqueeze(-1).expand(-1, -1, bs).reshape(M, K)
        U = W - alpha_full * g
        U_3d = U.view(M, ncb, bs)
        U_3d = torch.where(locked.unsqueeze(-1), Q_locked_3d, U_3d)
        U = U_3d.view(M, K)

        # 4. Moving target: snap U → Q_active. For locked blocks, keep Q_locked.
        if t % refresh_every == 0:
            Q_active = snap_fn(U.view(M, ncb, bs))                # (M, ncb, bs)
            Q_3d = torch.where(locked.unsqueeze(-1), Q_locked_3d, Q_active)
            Q = Q_3d.view(M, K)
            c_g = _compute_cg(Q)

        # 5. Proximal step (block soft-threshold around Q)
        shift_3d = (U - Q).view(M, ncb, bs)
        norm_shift = shift_3d.norm(dim=-1)
        tau = lam * alpha_bb * c_g
        shrink = (1.0 - tau / norm_shift.clamp(min=1e-30)).clamp(min=0.0)
        shrink = torch.where(locked, torch.zeros_like(shrink), shrink)
        W_next_3d = Q.view(M, ncb, bs) + shift_3d * shrink.unsqueeze(-1)
        W_next = W_next_3d.view(M, K)

        # 6. Lock newly-snapped blocks (snap point = current Q for that block)
        newly_locked = (~locked) & (shrink == 0)
        if newly_locked.any():
            Q_locked_3d_new = torch.where(newly_locked.unsqueeze(-1), Q.view(M, ncb, bs), Q_locked_3d)
            Q_locked = Q_locked_3d_new.view(M, K)
        locked = locked | newly_locked
        n_snapped = locked.sum().item()
        rel = (W_next - W).norm().item() / (W_next.norm().item() + 1e-30)

        history["iter"].append(t)
        history["lambda"].append(lam)
        history["n_snapped"].append(n_snapped)
        history["loss"].append(loss)
        history["rel_step"].append(rel)

        if verbose and (t % 20 == 0 or t == max_iter - 1):
            total = M * ncb
            print(f"  [iter {t:4d}] λ={lam:.4g}  loss={loss:.4g}  "
                  f"snapped={n_snapped}/{total} ({100*n_snapped/total:.1f}%)  "
                  f"rel_step={rel:.2e}", flush=True)

        W_prev = W
        g_prev = g
        W = W_next
        lam *= lam_grow

        if n_snapped == M * ncb:
            if verbose:
                print(f"  All blocks snapped at iter {t}.", flush=True)
            break
        if t > 5 and rel < tol:
            if verbose:
                print(f"  Converged (rel_step < {tol}) at iter {t}.", flush=True)
            break

    # Final: locked blocks → Q_locked; any remaining active → snap(W) once more
    final_3d = torch.where(
        locked.unsqueeze(-1),
        Q_locked.view(M, ncb, bs),
        snap_fn(W.view(M, ncb, bs)),
    )
    W_dq = final_3d.view(M, K)
    return W_dq, history


# ===================================================================
# Driver
# ===================================================================

def _fmt(t):
    return f"{t:.1f}s" if t >= 1 else f"{t * 1000:.0f}ms"


def run_experiment(name, snap_kind, block_size, alpha_priority,
                   lam_init, lam_grow, max_iter,
                   W, X, H, H_blocks, all_scales, verbose=True,
                   refresh_every=1, mode="lock"):
    if snap_kind == "naive":
        snap_fn = _naive_snap
    elif snap_kind == "hoptimal":
        snap_fn = _hoptimal_snap_factory(H_blocks, all_scales)
    else:
        raise ValueError(snap_kind)

    print(f"\n>>> {name}  (bs={block_size}, snap={snap_kind}, "
          f"α={alpha_priority}, λ0={lam_init}, ρ={lam_grow}, "
          f"refresh={refresh_every}, max_iter={max_iter})", flush=True)

    torch.cuda.synchronize()
    t0 = time.time()
    W_dq, hist = sparsa_group_l1_lock(
        W, H, snap_fn, H_blocks,
        block_size=block_size,
        alpha_priority=alpha_priority,
        lam_init=lam_init,
        lam_grow=lam_grow,
        max_iter=max_iter,
        refresh_every=refresh_every,
        verbose=verbose,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    m = compute_metrics(W, W_dq, X)
    print(f"  RESULT  W%={m['weight_error_pct']:.4f}  "
          f"O%={m['output_error_pct']:.4f}  time={_fmt(elapsed)}", flush=True)
    return {
        "name": name,
        "snap_kind": snap_kind,
        "block_size": block_size,
        "alpha_priority": alpha_priority,
        "lam_init": lam_init,
        "lam_grow": lam_grow,
        "max_iter": max_iter,
        "weight_error_pct": m["weight_error_pct"],
        "output_error_pct": m["output_error_pct"],
        "elapsed_s": elapsed,
        "n_iters_done": len(hist["iter"]),
        "final_snapped": hist["n_snapped"][-1] if hist["n_snapped"] else 0,
        "final_lambda": hist["lambda"][-1] if hist["lambda"] else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["E1"],
                        help="Which experiments to run: E1, E2, E3, ...")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--out", default="experiments/spgl1_results.jsonl")
    args = parser.parse_args()

    print("SPGL1 / SpaRSA Quantization Experiments")
    print(f"Device: {DEVICE}\n")

    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()
    M, K = W.shape
    print(f"W: {M}x{K}  X: {X.shape[0]}x{X.shape[1]}")

    print("Building Hessian...", end=" ", flush=True)
    H = (X.T @ X) / X.shape[0]
    print(f"done ({K}x{K})", flush=True)

    bs = args.bs
    ncb = K // bs
    H_blocks = torch.stack([
        H[j * bs:(j + 1) * bs, j * bs:(j + 1) * bs] for j in range(ncb)
    ])

    all_scales = build_fp8_e4m3_scales(device=DEVICE)

    configs = {
        # E1: Fixed Q (naive), α=1
        "E1": dict(name="E1: SpaRSA naive fixed-Q α=1",
                   snap_kind="naive", block_size=bs, alpha_priority=1.0,
                   lam_init=1e-5, lam_grow=1.02, max_iter=args.max_iter,
                   refresh_every=10**9),  # effectively fixed
        # E2: Fixed Q (H-optimal), α=1
        "E2": dict(name="E2: SpaRSA H-opt fixed-Q α=1",
                   snap_kind="hoptimal", block_size=bs, alpha_priority=1.0,
                   lam_init=1e-5, lam_grow=1.02, max_iter=args.max_iter,
                   refresh_every=10**9),  # effectively fixed
        # E3: Moving Q (naive), lock snapped blocks, α=1
        "E3": dict(name="E3: SpaRSA naive moving-Q α=1",
                   snap_kind="naive", block_size=bs, alpha_priority=1.0,
                   lam_init=1e-5, lam_grow=1.02, max_iter=args.max_iter,
                   refresh_every=1),
        # E4: Moving Q (H-optimal), lock snapped, α=1
        "E4": dict(name="E4: SpaRSA H-opt moving-Q α=1",
                   snap_kind="hoptimal", block_size=bs, alpha_priority=1.0,
                   lam_init=1e-5, lam_grow=1.02, max_iter=args.max_iter,
                   refresh_every=1),
        # Sanity: α=0 (no priority) with fixed Q naive
        "E1a": dict(name="E1a: fixed Q naive α=0",
                    snap_kind="naive", block_size=bs, alpha_priority=0.0,
                    lam_init=1e-5, lam_grow=1.02, max_iter=args.max_iter,
                    refresh_every=10**9),  # effectively fixed
    }

    results = []
    for key in args.experiments:
        if key not in configs:
            print(f"Skipping unknown experiment: {key}")
            continue
        r = run_experiment(W=W, X=X, H=H, H_blocks=H_blocks,
                           all_scales=all_scales, verbose=True,
                           **configs[key])
        r["key"] = key
        results.append(r)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
