"""Custom codebook optimization: learn optimal 4-bit quantization levels from data.

Approach (see docs/source/custom_codebook.md):
1. Sign reduction: 16 codes -> 8 unsigned + sign bit
2. Normalize each block by max (scale invariance)
3. 1D k-means on pooled normalized magnitudes -> 7 positive codebook values + zero
4. Quantize with learned codebook using FP8 E4M3 scales (same as NVFP4)
"""

import torch
import time

from qwantize.nvfp4.reference import (
    _fp8_e4m3_snap,
    build_fp8_e4m3_scales,
    fp4_quantize,
)
from qwantize.metrics import compute_metrics

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


# ---------------------------------------------------------------------------
# Codebook learning
# ---------------------------------------------------------------------------

def learn_codebook(W, block_size, n_positive=7, max_iter=100, X=None, q_max=6.0):
    """Learn optimal codebook via weighted 1D k-means on normalized magnitudes.

    Normalizes each block by its max, pools all normalized |x|/max values,
    and runs k-means to find n_positive cluster centers. When X is provided,
    uses Hessian-diagonal weights (activation energy per position) so that
    elements with higher output impact have more influence on the codebook.

    Args:
        W: Weight tensor of shape (M, K).
        block_size: Block size (16 or 32).
        n_positive: Number of positive codebook values (excluding zero).
        max_iter: Maximum Lloyd's iterations.
        X: Activation tensor of shape (T, K). If provided, uses Hessian-diagonal
           weights for weighted k-means.
        q_max: Maximum codebook value. Use 6.0 for FP4-like range, 1.0 for unit range.

    Returns:
        (codebook, boundaries) where:
        - codebook: shape (n_positive+1,) = {0, c1, ..., c7}, sorted ascending
        - boundaries: shape (n_positive,) = decision boundaries for bucketize
    """
    M, K = W.shape
    x = W.float().reshape(-1, block_size)  # (N, bs)
    N = x.shape[0]
    x_abs = x.abs()

    # Normalize each block by its max (scale invariance)
    amax = x_abs.amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (N, 1)
    y = x_abs / amax  # (N, bs), values in [0, 1]

    # Compute Hessian weights if X is provided
    # Full block Hessians H_j = X_j^T X_j, then use row-sum |H_j| @ 1
    # to capture cross-element correlations (not just diagonal)
    if X is not None:
        num_col_blocks = K // block_size
        H_blocks = torch.empty(num_col_blocks, block_size, block_size, device=W.device)
        batch_t = 8192
        for j in range(num_col_blocks):
            acc = torch.zeros(block_size, block_size, device=W.device)
            for t0 in range(0, X.shape[0], batch_t):
                Xj = X[t0 : t0 + batch_t, j * block_size : (j + 1) * block_size].float()
                acc.addmm_(Xj.T, Xj)
            H_blocks[j] = acc
        # Per-element weight = row sum of |H_j|: captures total influence of
        # element i on the quadratic form r^T H r (including cross-terms)
        h_rowsum = H_blocks.abs().sum(dim=-1)  # (num_col_blocks, block_size)
        # Broadcast to all M rows: (M, num_col_blocks, bs) -> (N, bs)
        weights = h_rowsum.unsqueeze(0).expand(M, -1, -1).reshape(N, block_size)
        # Scale by amax^2 to match actual error contribution (E_H ~ s^2 * ...)
        weights = weights * amax.pow(2)
        weights_flat = weights.reshape(-1).cpu().float()
        print(f"  Using full-Hessian row-sum weights from activations")
    else:
        weights_flat = None

    # Pool all values, move to CPU for k-means
    pool = y.reshape(-1).cpu().float()  # (N * bs,)
    P = pool.shape[0]
    print(f"  Codebook learning: {P:,} values from {N:,} blocks")

    pool_sorted, sort_idx = pool.sort()
    if weights_flat is not None:
        weights_sorted = weights_flat[sort_idx]

    # 1D k-means (Lloyd's algorithm) on [0, 1] normalized magnitudes
    # Initialize: n_positive centers for the positive values (excluding zero)
    nonzero_mask = pool_sorted > 0.01
    nonzero_sorted = pool_sorted[nonzero_mask]
    NZ = nonzero_sorted.shape[0]
    q_idx = torch.linspace(0, NZ - 1, n_positive + 2).long()[1:-1]
    centers = nonzero_sorted[q_idx].clone()

    for it in range(max_iter):
        bdry = (centers[:-1] + centers[1:]) / 2
        bdry_zero = centers[0] / 2

        pos_mask = pool > bdry_zero
        pool_pos = pool[pos_mask]

        assignments = torch.bucketize(pool_pos, bdry)

        if weights_flat is not None:
            w_pos = weights_flat[pos_mask]
            # Weighted k-means: center = sum(w * y) / sum(w)
            w_sums = torch.zeros(n_positive, dtype=pool.dtype)
            w_counts = torch.zeros(n_positive, dtype=pool.dtype)
            w_sums.scatter_add_(0, assignments, w_pos * pool_pos)
            w_counts.scatter_add_(0, assignments, w_pos)

            new_centers = centers.clone()
            valid = w_counts > 0
            new_centers[valid] = w_sums[valid] / w_counts[valid]
        else:
            sums = torch.zeros(n_positive, dtype=pool.dtype)
            counts_k = torch.zeros(n_positive, dtype=torch.long)
            sums.scatter_add_(0, assignments, pool_pos)
            counts_k.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=torch.long))

            new_centers = centers.clone()
            valid = counts_k > 0
            new_centers[valid] = sums[valid] / counts_k[valid].float()

        if (new_centers - centers).abs().max() < 1e-7:
            print(f"  k-means converged at iteration {it + 1}")
            break
        centers = new_centers
    else:
        print(f"  k-means reached max_iter={max_iter}")

    # Scale from [0,1]-normalized to target Q_MAX
    codebook_pos = centers.sort().values * q_max

    # Full codebook: {0, c1, ..., c7}
    codebook = torch.cat([torch.zeros(1), codebook_pos])

    # Decision boundaries: midpoints
    boundaries = torch.empty(n_positive)
    boundaries[0] = codebook_pos[0] / 2
    for k in range(1, n_positive):
        boundaries[k] = (codebook_pos[k - 1] + codebook_pos[k]) / 2

    return codebook.to(W.device), boundaries.to(W.device)


# ---------------------------------------------------------------------------
# Quantization with custom codebook
# ---------------------------------------------------------------------------

def custom_quantize(x, s, codebook, boundaries):
    """Quantize using a custom codebook.

    Args:
        x: Input tensor of shape (..., block_size).
        s: Per-block scale of shape (..., 1).
        codebook: Non-negative codebook values, shape (n_values,), sorted ascending.
        boundaries: Decision boundaries, shape (n_values-1,).

    Returns:
        Signed codebook values with same shape as x.
    """
    signs = x.sign()
    y = x.abs() / s
    bucket_idx = torch.bucketize(y, boundaries)
    q_mag = codebook[bucket_idx]
    return signs * q_mag


def custom_dequantize(quants, s):
    """Dequantize: quants * s."""
    return quants * s


def compute_custom_block_sse(x, s, codebook, boundaries):
    """Compute per-block SSE with custom codebook."""
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    quants = custom_quantize(x, s, codebook, boundaries)
    dq = custom_dequantize(quants, s)
    return (x - dq).pow(2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Naive and optimal quantization
# ---------------------------------------------------------------------------

def custom_naive(W, codebook, boundaries, dim=-1, return_dequant=False):
    """Naive quantization with custom codebook: s = FP8(max|x| / Q_MAX)."""
    dim = dim % W.ndim
    block_size = W.shape[dim]
    Q_MAX = codebook[-1].item()
    D_0 = boundaries[0].item()

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)

    amax = x.abs().amax(dim=-1)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    scales = _fp8_e4m3_snap(s_cont)

    quants = custom_quantize(x, scales.unsqueeze(-1), codebook, boundaries)
    result = (
        scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = custom_dequantize(quants, scales.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def custom_optimal(W, codebook, boundaries, dim=-1, return_dequant=False):
    """Optimal quantization with custom codebook via bounded FP8 E4M3 scale search."""
    dim = dim % W.ndim
    block_size = W.shape[dim]
    Q_MAX = codebook[-1].item()
    D_0 = boundaries[0].item()

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    all_scales = build_fp8_e4m3_scales(device=x.device)

    # Step 1: Baseline (naive) scale and error
    amax = x.abs().amax(dim=-1)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = _fp8_e4m3_snap(s_cont)

    E0 = compute_custom_block_sse(x, s0, codebook, boundaries)

    best_s = s0.clone()
    best_E = E0.clone()

    # Step 2: Edge case - noise blocks
    total_energy = x.pow(2).sum(dim=-1)
    noise_mask = total_energy <= E0

    # Step 3: Compute bounds
    sqrt_E0 = E0.sqrt()
    s_min = ((amax - sqrt_E0) / Q_MAX).clamp(min=0.0)

    sorted_abs, _ = x.abs().sort(dim=-1)
    cumsum_sq = sorted_abs.pow(2).cumsum(dim=-1)
    k_star = (cumsum_sq <= E0.unsqueeze(-1)).sum(dim=-1)
    noise_mask = noise_mask | (k_star >= block_size)

    k_star_idx = k_star.clamp(max=block_size - 1)
    y_k_plus_1 = sorted_abs.gather(dim=-1, index=k_star_idx.unsqueeze(-1)).squeeze(-1)
    s_max = y_k_plus_1 / D_0

    # Step 4: Bounded search
    active = ~noise_mask
    if active.any():
        x_active = x[active]
        s_min_a = s_min[active]
        s_max_a = s_max[active]
        best_E_a = best_E[active].clone()
        best_s_a = best_s[active].clone()

        for s_val in all_scales:
            s_f = s_val.item()

            in_range = (s_f >= s_min_a) & (s_f <= s_max_a)
            if not in_range.any():
                continue

            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)
            evaluate = in_range & (H_s < best_E_a)
            if not evaluate.any():
                continue

            s_broadcast = torch.full(
                (x_active.shape[0], 1), s_f, device=x_active.device, dtype=x_active.dtype
            )
            E_s = compute_custom_block_sse(x_active, s_broadcast.squeeze(-1), codebook, boundaries)

            improved = evaluate & (E_s < best_E_a)
            best_E_a[improved] = E_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = custom_quantize(x, best_s.unsqueeze(-1), codebook, boundaries)
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = custom_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def custom_optimal_hessian(W, codebook, boundaries, dim=-1, return_dequant=False, X=None, H_blocks=None):
    """Hessian-aware optimal quantization with custom codebook.

    Either X or H_blocks must be provided:
      - X: (T, K) activations, H computed as X_j^T @ X_j per block
      - H_blocks: (num_col_blocks, block_size, block_size) pre-computed block Hessians
    """
    assert X is not None or H_blocks is not None, "X or H_blocks required"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    Q_MAX = codebook[-1].item()
    D_0 = boundaries[0].item()

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    if H_blocks is not None:
        H = H_blocks.to(x.device)
        num_col_blocks = H.shape[0]
    else:
        K_dim = X.shape[1]
        num_col_blocks = K_dim // block_size
        H = torch.empty(num_col_blocks, block_size, block_size, device=x.device)
        batch_t = 8192
        for j in range(num_col_blocks):
            acc = torch.zeros(block_size, block_size, device=x.device)
            for t0 in range(0, X.shape[0], batch_t):
                Xj = X[t0 : t0 + batch_t, j * block_size : (j + 1) * block_size].float()
                acc.addmm_(Xj.T, Xj)
            H[j] = acc

    M_dim = N // num_col_blocks
    assert N == M_dim * num_col_blocks

    all_scales = build_fp8_e4m3_scales(device=x.device)

    # Baseline (naive) scale and SSE for bounding
    amax = x.abs().amax(dim=-1)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = _fp8_e4m3_snap(s_cont)

    E0_sse = compute_custom_block_sse(x, s0, codebook, boundaries)

    # Baseline Hessian error
    quants0 = custom_quantize(x, s0.unsqueeze(-1), codebook, boundaries)
    dq0 = custom_dequantize(quants0, s0.unsqueeze(-1))
    r0 = x - dq0
    r0_3d = r0.reshape(M_dim, num_col_blocks, block_size)
    Hr0 = torch.einsum("jab,mjb->mja", H, r0_3d)
    E0_H = (r0_3d * Hr0).sum(dim=-1).reshape(-1)

    best_s = s0.clone()
    best_E = E0_H.clone()

    # Noise blocks
    total_energy = x.pow(2).sum(dim=-1)
    noise_mask = total_energy <= E0_sse

    # SSE-based bounds for pruning
    sqrt_E0 = E0_sse.sqrt()
    s_min = ((amax - sqrt_E0) / Q_MAX).clamp(min=0.0)

    sorted_abs, _ = x.abs().sort(dim=-1)
    cumsum_sq = sorted_abs.pow(2).cumsum(dim=-1)
    k_star = (cumsum_sq <= E0_sse.unsqueeze(-1)).sum(dim=-1)
    noise_mask = noise_mask | (k_star >= block_size)

    k_star_idx = k_star.clamp(max=block_size - 1)
    y_k_plus_1 = sorted_abs.gather(dim=-1, index=k_star_idx.unsqueeze(-1)).squeeze(-1)
    s_max = y_k_plus_1 / D_0

    # Bounded search with Hessian error
    active = ~noise_mask
    if active.any():
        best_E_a = best_E[active].clone()
        best_s_a = best_s[active].clone()
        s_min_a = s_min[active]
        s_max_a = s_max[active]
        x_active = x[active]

        active_indices = active.nonzero(as_tuple=True)[0]
        active_j = active_indices % num_col_blocks

        for s_val in all_scales:
            s_f = s_val.item()

            in_range = (s_f >= s_min_a) & (s_f <= s_max_a)
            if not in_range.any():
                continue

            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)
            evaluate = in_range & (H_s < best_E_a * 10)
            if not evaluate.any():
                continue

            quants_s = custom_quantize(x_active, torch.tensor(s_f, device=x.device), codebook, boundaries)
            dq_s = custom_dequantize(quants_s, torch.tensor(s_f, device=x.device))
            r = x_active - dq_s

            H_active = H[active_j]
            Hr = torch.bmm(H_active, r.unsqueeze(-1)).squeeze(-1)
            E_H_s = (r * Hr).sum(dim=-1)

            improved = evaluate & (E_H_s < best_E_a)
            best_E_a[improved] = E_H_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = custom_quantize(x, best_s.unsqueeze(-1), codebook, boundaries)
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = custom_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(name, fn, W, M, K, X):
    """Benchmark a quantization method."""
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
    from qwantize.nvfp4.reference import nvfp4_naive, nvfp4_optimal, nvfp4_optimal_hessian
    from qwantize.mxfp4.reference import mxfp4_naive, mxfp4_optimal, mxfp4_optimal_hessian

    print(f"Device: {DEVICE}")
    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True)
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True)
    print(f"W: {W.shape} {W.dtype}")
    print(f"X: {X.shape} {X.dtype}")

    M, K = W.shape

    for bs in [16, 32]:
        print(f"\n{'='*70}")
        print(f"Block size: {bs}")
        print(f"{'='*70}")

        W_b = W.reshape(M, K // bs, bs)

        # Learn Hessian-weighted codebook (q_max=6)
        print(f"\nLearning Hessian codebook (q_max=6)...")
        t0 = time.time()
        cb_h6, bd_h6 = learn_codebook(W, bs, X=X, q_max=6.0)
        print(f"  Time: {time.time() - t0:.2f}s")
        print(f"  CB6: [{', '.join(f'{v:.4f}' for v in cb_h6.cpu().tolist())}]")

        # Learn Hessian-weighted codebook (q_max=1, 01codebook)
        print(f"\nLearning Hessian codebook (q_max=1, 01codebook)...")
        t0 = time.time()
        cb_h1, bd_h1 = learn_codebook(W, bs, X=X, q_max=1.0)
        print(f"  Time: {time.time() - t0:.2f}s")
        print(f"  CB1: [{', '.join(f'{v:.4f}' for v in cb_h1.cpu().tolist())}]")
        print(f"  (FP4 E2M1: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])")

        # INT4 codebook: 8 uniform levels in [0, 1] = {0, 1/7, 2/7, ..., 1}
        cb_int4 = torch.linspace(0, 1, 8, device=DEVICE)
        bd_int4 = (cb_int4[:-1] + cb_int4[1:]) / 2  # midpoints as boundaries
        bd_int4 = bd_int4[1:]  # drop first (boundary between 0 and 1/7 not needed for bucketize)
        # Actually need boundary between 0 and first positive value
        bd_int4 = torch.empty(7, device=DEVICE)
        bd_int4[0] = cb_int4[1] / 2  # 1/14
        for k in range(1, 7):
            bd_int4[k] = (cb_int4[k] + cb_int4[k + 1]) / 2
        print(f"  INT4: [{', '.join(f'{v:.4f}' for v in cb_int4.cpu().tolist())}]")

        methods = []

        # INT4 (uniform, q_max=1)
        methods.append(bench("INT4 Naive", lambda: custom_naive(W_b, cb_int4, bd_int4, return_dequant=True), W, M, K, X))
        methods.append(bench("INT4 Optimal", lambda: custom_optimal(W_b, cb_int4, bd_int4, return_dequant=True), W, M, K, X))
        methods.append(bench("INT4 H-Optimal", lambda: custom_optimal_hessian(W_b, cb_int4, bd_int4, return_dequant=True, X=X), W, M, K, X))

        # 01codebook (Hessian, q_max=1)
        methods.append(bench("01CB Naive", lambda: custom_naive(W_b, cb_h1, bd_h1, return_dequant=True), W, M, K, X))
        methods.append(bench("01CB Optimal", lambda: custom_optimal(W_b, cb_h1, bd_h1, return_dequant=True), W, M, K, X))
        methods.append(bench("01CB H-Optimal", lambda: custom_optimal_hessian(W_b, cb_h1, bd_h1, return_dequant=True, X=X), W, M, K, X))

        # Custom Hessian codebook (q_max=6)
        methods.append(bench("CB6 Naive", lambda: custom_naive(W_b, cb_h6, bd_h6, return_dequant=True), W, M, K, X))
        methods.append(bench("CB6 Optimal", lambda: custom_optimal(W_b, cb_h6, bd_h6, return_dequant=True), W, M, K, X))
        methods.append(bench("CB6 H-Optimal", lambda: custom_optimal_hessian(W_b, cb_h6, bd_h6, return_dequant=True, X=X), W, M, K, X))

        # NVFP4
        methods.append(bench("NVFP4 Naive", lambda: nvfp4_naive(W_b, return_dequant=True), W, M, K, X))
        methods.append(bench("NVFP4 Optimal", lambda: nvfp4_optimal(W_b, return_dequant=True), W, M, K, X))
        methods.append(bench("NVFP4 H-Optimal", lambda: nvfp4_optimal_hessian(W_b, return_dequant=True, X=X), W, M, K, X))

        # Print results
        print(f"\n{'Method':<24} {'Time':>10} {'Weight%':>10} {'Output%':>10}")
        print("-" * 60)
        for name, m, t in methods:
            if t >= 1.0:
                ts = f"{t:.2f} s"
            else:
                ts = f"{t*1000:.1f} ms"
            print(f"{name:<24} {ts:>10} {m['weight_error_pct']:>9.4f}% {m['output_error_pct']:>9.4f}%")


if __name__ == "__main__":
    main()
