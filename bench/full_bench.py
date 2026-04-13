"""Full benchmark: all INT8 + NVFP4 + MXFP4 methods, torch vs Triton."""

import torch
import time
from qwantize.nvfp4.reference import nvfp4_naive, nvfp4_optimal, nvfp4_optimal_hessian
from qwantize.nvfp4.kernels import (
    nvfp4_naive_triton, nvfp4_optimal_triton, nvfp4_optimal_hessian_triton,
)
from qwantize.mxfp4.reference import mxfp4_naive, mxfp4_optimal, mxfp4_optimal_hessian
from qwantize.mxfp4.kernels import (
    mxfp4_naive_triton, mxfp4_optimal_triton,
)
from qwantize.int8.reference import int8_naive, int8_optimal, int8_optimal_hessian
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def bench(name, fn, W, M, K, X):
    print(f"  {name}...", flush=True)
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    result = fn()
    torch.cuda.synchronize()
    t = time.time() - t0
    W_dq = result[2].reshape(M, K)
    m = compute_metrics(W, W_dq, X)
    del result, W_dq
    torch.cuda.empty_cache()
    return (name, m, t)


def main():
    print(f"Device: {DEVICE}")
    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True)
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True)
    print(f"W: {W.shape} {W.dtype}")
    print(f"X: {X.shape} {X.dtype}")

    M, K = W.shape

    # INT8 block sizes
    for bs in [32, 64, 128, 256]:
        print(f"\n{'='*70}")
        print(f"INT8 Block size: {bs}")
        print(f"{'='*70}")

        W_b = W.reshape(M, K // bs, bs)
        methods = []

        methods.append(bench("Naive (torch)", lambda: int8_naive(W_b, return_dequant=True), W, M, K, X))
        methods.append(bench("SSE-Optimal (torch)", lambda: int8_optimal(W_b, return_dequant=True), W, M, K, X))
        methods.append(bench("H-Optimal (torch)", lambda: int8_optimal_hessian(W_b, return_dequant=True, X=X), W, M, K, X))

        print(f"\n{'Method':<28} {'Time':>10} {'Weight%':>10} {'Output%':>10}")
        print("-" * 60)
        for name, m, t in methods:
            if t >= 1.0:
                ts = f"{t:.2f} s"
            else:
                ts = f"{t*1000:.1f} ms"
            print(f"{name:<28} {ts:>10} {m['weight_error_pct']:>9.4f}% {m['output_error_pct']:>9.4f}%")

    # FP4 formats
    for fmt in ["NVFP4", "MXFP4"]:
        for bs in [16, 32]:
            print(f"\n{'='*70}")
            print(f"{fmt} Block size: {bs}")
            print(f"{'='*70}")

            W_b = W.reshape(M, K // bs, bs)
            methods = []

            if fmt == "NVFP4":
                methods.append(bench("Naive (torch)", lambda: nvfp4_naive(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("Naive (Triton)", lambda: nvfp4_naive_triton(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("SSE-Optimal (torch)", lambda: nvfp4_optimal(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("SSE-Optimal (Triton)", lambda: nvfp4_optimal_triton(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("H-Optimal (torch)", lambda: nvfp4_optimal_hessian(W_b, return_dequant=True, X=X), W, M, K, X))
                methods.append(bench("H-Optimal (Triton)", lambda: nvfp4_optimal_hessian_triton(W_b, return_dequant=True, X=X), W, M, K, X))
            else:
                methods.append(bench("Naive (torch)", lambda: mxfp4_naive(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("Naive (Triton)", lambda: mxfp4_naive_triton(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("SSE-Optimal (torch)", lambda: mxfp4_optimal(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("SSE-Optimal (Triton)", lambda: mxfp4_optimal_triton(W_b, return_dequant=True), W, M, K, X))
                methods.append(bench("H-Optimal (torch)", lambda: mxfp4_optimal_hessian(W_b, return_dequant=True, X=X), W, M, K, X))

            print(f"\n{'Method':<28} {'Time':>10} {'Weight%':>10} {'Output%':>10}")
            print("-" * 60)
            for name, m, t in methods:
                if t >= 1.0:
                    ts = f"{t:.2f} s"
                else:
                    ts = f"{t*1000:.1f} ms"
                print(f"{name:<28} {ts:>10} {m['weight_error_pct']:>9.4f}% {m['output_error_pct']:>9.4f}%")

            # Speedups
            print(f"\nSpeedups (Triton vs torch):")
            pairs = [("Naive", 0, 1), ("SSE-Optimal", 2, 3)]
            if fmt == "NVFP4":
                pairs.append(("H-Optimal", 4, 5))
            for label, ti, tj in pairs:
                t_torch = methods[ti][2]
                t_triton = methods[tj][2]
                print(f"  {label}: {t_torch/t_triton:.1f}x ({t_torch*1000:.1f} ms -> {t_triton*1000:.1f} ms)")


if __name__ == "__main__":
    main()
