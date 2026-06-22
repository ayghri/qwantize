#!/usr/bin/env python
"""Correctness + speed comparison: qwantize.spgl1 (GPU) vs spgl1 (CPU).

Step 1: tiny synthetic problem, B=1, compare batch-of-one GPU output to
        scipy spgl1 on the same (A, b, tau).
Step 2: full layer-0 batched problem (B=8 rows), GPU only, report timing.
Step 3: same problem solved row-by-row on CPU with the spgl1 package;
        report wall-time ratio.

Usage:
    python experiments/spgl1_gpu_test.py
"""

import argparse
import time

import numpy as np
import torch

import spgl1 as spgl1_pkg
from qwantize.spgl1 import (
    spgl1_lasso_batched, make_dense_op, project_l1_ball_batched,
    l1_norm_batched,
)
from qwantize.nvfp4.reference import nvfp4_optimal_hessian


DEVICE = torch.device("cuda:0")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def _fmt(t):
    return f"{t:.2f}s" if t >= 1 else f"{t * 1000:.1f}ms"


# ---------------------------------------------------------------------------
# Step 0: unit test the projection
# ---------------------------------------------------------------------------

def test_projection():
    print("\n=== Step 0: unit test L1 ball projection ===")
    torch.manual_seed(0)
    B, N = 5, 64
    x = torch.randn(B, N, dtype=torch.float64)
    for tau in [0.1, 1.0, 5.0, 100.0]:
        x_gpu = x.clone().cuda()
        proj_gpu = project_l1_ball_batched(x_gpu, tau).cpu()
        # CPU reference
        proj_cpu = torch.stack([
            torch.from_numpy(spgl1_pkg.oneprojector(x[i].numpy().copy(), 1.0, tau))
            for i in range(B)
        ])
        max_abs = (proj_gpu - proj_cpu).abs().max().item()
        gpu_l1 = proj_gpu.abs().sum(-1)
        cpu_l1 = proj_cpu.abs().sum(-1)
        print(f"  tau={tau:6.1f}  max|gpu-cpu|={max_abs:.3e}  "
              f"gpu ||.||_1 max={gpu_l1.max():.4f}  cpu={cpu_l1.max():.4f}")
        assert max_abs < 1e-9, "projection mismatch"
    print("  projection: OK")


# ---------------------------------------------------------------------------
# Step 1: tiny synthetic problem, GPU vs CPU
# ---------------------------------------------------------------------------

def test_tiny_synthetic():
    print("\n=== Step 1: tiny synthetic LASSO, B=1 ===")
    torch.manual_seed(42)
    np.random.seed(42)
    m, n = 256, 128

    A_np = np.random.randn(m, n).astype(np.float64) / np.sqrt(m)
    x_true_np = np.zeros(n)
    x_true_np[np.random.choice(n, 10, replace=False)] = np.random.randn(10)
    b_np = A_np @ x_true_np + 0.01 * np.random.randn(m)

    tau = 0.5 * float(np.linalg.norm(x_true_np, 1))

    # CPU
    print(f"  m={m} n={n}  tau={tau:.4f}  ||x_true||_1={np.linalg.norm(x_true_np, 1):.4f}")
    t0 = time.time()
    x_cpu, r_cpu, _, info_cpu = spgl1_pkg.spgl1(
        A_np, b_np, tau=tau, sigma=0.0, iter_lim=500, verbosity=0
    )
    t_cpu = time.time() - t0

    # GPU
    A_t = torch.from_numpy(A_np).to(DEVICE)
    b_t = torch.from_numpy(b_np).to(DEVICE).unsqueeze(0)        # (1, m)
    matvec, rmatvec = make_dense_op(A_t)
    torch.cuda.synchronize()
    t0 = time.time()
    x_gpu_t, r_gpu_t, info_gpu = spgl1_lasso_batched(
        matvec, rmatvec, b_t, tau=tau, n=n, max_iter=500, verbose=False,
    )
    torch.cuda.synchronize()
    t_gpu = time.time() - t0
    x_gpu = x_gpu_t.squeeze(0).cpu().numpy()
    r_gpu = r_gpu_t.squeeze(0).cpu().numpy()

    # Compare
    Ax_diff_cpu = float(np.linalg.norm(A_np @ x_cpu - b_np))
    Ax_diff_gpu = float(np.linalg.norm(A_np @ x_gpu - b_np))
    x_diff = float(np.linalg.norm(x_cpu - x_gpu) / max(1e-30, np.linalg.norm(x_cpu)))
    l1_cpu = float(np.linalg.norm(x_cpu, 1))
    l1_gpu = float(np.linalg.norm(x_gpu, 1))

    print(f"  CPU: ||Ax-b||={Ax_diff_cpu:.6f}  ||x||_1={l1_cpu:.4f}  "
          f"iters={info_cpu.get('niters', '?')}  time={_fmt(t_cpu)}")
    print(f"  GPU: ||Ax-b||={Ax_diff_gpu:.6f}  ||x||_1={l1_gpu:.4f}  "
          f"line iters={info_gpu['n_line_iters']}  time={_fmt(t_gpu)}")
    print(f"  rel ||x_cpu - x_gpu|| / ||x_cpu|| = {x_diff:.3e}")
    print(f"  CPU residual vs GPU residual: {Ax_diff_cpu:.6f} vs {Ax_diff_gpu:.6f}  "
          f"({100*(Ax_diff_gpu-Ax_diff_cpu)/Ax_diff_cpu:+.2f}%)")
    print(f"  CPU L1 vs GPU L1:               {l1_cpu:.4f} vs {l1_gpu:.4f}")
    return t_cpu, t_gpu


# ---------------------------------------------------------------------------
# Step 2: layer-0 problem on GPU, multi-row batch
# ---------------------------------------------------------------------------

def test_layer0_batched(n_rows=4, tau_frac=0.5):
    print(f"\n=== Step 2: layer-0 down_proj, B={n_rows} rows, GPU batch ===")
    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True).float()

    print(f"  W: {W.shape}  X: {X.shape}")
    bs = 16
    M, K = W.shape

    # Build H-optimal Q
    Q_blocked = nvfp4_optimal_hessian(
        W.view(M, K // bs, bs), return_dequant=True, X=X,
    )[2]
    Q = Q_blocked.view(M, K)

    # Pick rows
    rows = list(range(n_rows))
    W_sel = W[rows]                                          # (B, K)
    Q_sel = Q[rows]                                          # (B, K)
    diff = W_sel - Q_sel                                     # (B, K)

    # b = X @ (w_0 - q)  → for batch: b_i = X @ diff[i]   shape (B, T)
    b_t = diff @ X.T                                         # (B, T)
    print(f"  b: {b_t.shape}")

    # tau = tau_frac * ||w_0 - q||_1 per-row
    l1_init = diff.abs().sum(dim=-1)
    tau_vec = tau_frac * l1_init                              # (B,)

    snap_resid = b_t.norm(dim=-1)                             # (B,)
    xw0_norm = (W_sel @ X.T).norm(dim=-1)                     # (B,)

    print(f"  ||w0-q||_1: {l1_init.cpu().numpy()}")
    print(f"  tau:        {tau_vec.cpu().numpy()}")
    print(f"  ||X(w0-q)||_2 (snap residual): {snap_resid.cpu().numpy()}")
    print(f"  snap output err per row: "
          f"{(100 * snap_resid / xw0_norm).cpu().numpy()}")

    matvec, rmatvec = make_dense_op(X)

    torch.cuda.synchronize()
    t0 = time.time()
    d_gpu, r_gpu, info = spgl1_lasso_batched(
        matvec, rmatvec, b_t, tau=tau_vec, n=K,
        max_iter=200, verbose=True,
    )
    torch.cuda.synchronize()
    t_gpu = time.time() - t0

    print(f"\n  GPU total: {_fmt(t_gpu)}  line iters={info['n_line_iters']}  "
          f"matvecs={info['n_matvec']}  rmatvecs={info['n_rmatvec']}")
    print(f"  Exit: stat={info['exit_stat']} at iter {info['exit_iter']}")

    res_norm = r_gpu.norm(dim=-1)
    d_l1 = l1_norm_batched(d_gpu)
    d_l2 = d_gpu.norm(dim=-1)
    print(f"\n  Per-row results (GPU):")
    print(f"    row  res_norm  snap_resid  ratio  ||d||_1  (tau)  ||d||_2  out%")
    for i in range(n_rows):
        print(f"    {i:3d}  {res_norm[i]:.4f}   {snap_resid[i]:.4f}   "
              f"{(res_norm[i]/snap_resid[i]).item():.3f}  "
              f"{d_l1[i].item():.3f}  ({tau_vec[i].item():.3f})  "
              f"{d_l2[i].item():.3f}   "
              f"{100*res_norm[i].item()/xw0_norm[i].item():.4f}%")

    return t_gpu, b_t, X, tau_vec


# ---------------------------------------------------------------------------
# Step 3: same problem on CPU one-row-at-a-time
# ---------------------------------------------------------------------------

def test_layer0_cpu(n_rows, b_gpu, X_gpu, tau_vec, max_iter=200):
    print(f"\n=== Step 3: layer-0 same problem, CPU one row at a time ===")
    X_np = X_gpu.cpu().numpy().astype(np.float64)
    b_np = b_gpu.cpu().numpy().astype(np.float64)
    tau_np = tau_vec.cpu().numpy().astype(np.float64)

    times = []
    res_norms = []
    l1s = []
    for i in range(n_rows):
        print(f"  Row {i}: running spgl1 CPU (tau={tau_np[i]:.4f}, "
              f"||b||={np.linalg.norm(b_np[i]):.4f})", flush=True)
        t0 = time.time()
        x, r, _, info = spgl1_pkg.spgl1(
            X_np, b_np[i], tau=float(tau_np[i]), sigma=0.0,
            iter_lim=max_iter, verbosity=0,
        )
        t = time.time() - t0
        times.append(t)
        res_norms.append(float(np.linalg.norm(r)))
        l1s.append(float(np.linalg.norm(x, 1)))
        print(f"    done in {_fmt(t)}  res={res_norms[-1]:.4f}  "
              f"||x||_1={l1s[-1]:.4f}  iters={info.get('niters', '?')}")
    return sum(times), times, res_norms, l1s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip the slow CPU comparison")
    p.add_argument("--n-rows", type=int, default=4)
    p.add_argument("--tau-frac", type=float, default=0.5)
    p.add_argument("--max-iter", type=int, default=100)
    args = p.parse_args()

    test_projection()
    test_tiny_synthetic()

    t_gpu, b_t, X, tau_vec = test_layer0_batched(
        n_rows=args.n_rows, tau_frac=args.tau_frac,
    )

    if not args.skip_cpu:
        t_cpu, _, _, _ = test_layer0_cpu(
            args.n_rows, b_t, X, tau_vec, max_iter=args.max_iter,
        )
        print(f"\n=== Speed summary ===")
        print(f"  GPU batched ({args.n_rows} rows): {_fmt(t_gpu)}")
        print(f"  CPU one row at a time:    {_fmt(t_cpu)}  (avg {_fmt(t_cpu/args.n_rows)}/row)")
        print(f"  Speedup: {t_cpu / t_gpu:.1f}x")


if __name__ == "__main__":
    main()
