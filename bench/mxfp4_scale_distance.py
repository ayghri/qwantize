"""Analyze scale distances and bounded search quality for MXFP4/UE8M0."""

import torch
from quantkit.mxfp4.reference import (
    build_ue8m0_scales,
    mxfp4_naive,
    mxfp4_optimal,
    compute_block_sse,
    fp4_quantize,
    fp4_dequantize,
    scales_to_ue8m0_exponent,
    Q_MAX,
    D_0,
)
from quantkit.metrics import compute_metrics

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def analyze_scale_distances(W, block_size):
    """Compute and analyze distances between naive and optimal UE8M0 scales."""
    M, K = W.shape
    W_blocked = W.reshape(M, K // block_size, block_size)

    naive_scales, _ = mxfp4_naive(W_blocked)
    optimal_scales, _ = mxfp4_optimal(W_blocked)

    naive_flat = naive_scales.reshape(-1)
    optimal_flat = optimal_scales.reshape(-1)

    naive_exp = scales_to_ue8m0_exponent(naive_flat)
    optimal_exp = scales_to_ue8m0_exponent(optimal_flat)

    exp_dist = (optimal_exp - naive_exp).to(torch.int32)

    N_total = exp_dist.shape[0]
    changed_mask = exp_dist != 0
    N_changed = changed_mask.sum().item()

    print(f"\n  Total blocks: {N_total}")
    print(f"  Changed:      {N_changed} ({N_changed/N_total*100:.1f}%)")
    print(f"  Unchanged:    {N_total - N_changed} ({(N_total-N_changed)/N_total*100:.1f}%)")

    print(f"\n  Exponent distance (optimal - naive) in UE8M0 steps (factors of 2):")
    print(f"    min:    {exp_dist.min().item()}")
    print(f"    max:    {exp_dist.max().item()}")
    print(f"    mean:   {exp_dist.float().mean().item():.3f}")
    print(f"    median: {exp_dist.float().median().item():.0f}")

    print(f"\n  Exponent distance distribution:")
    for d in range(exp_dist.min().item(), exp_dist.max().item() + 1):
        count = (exp_dist == d).sum().item()
        if count > 0:
            pct = count / N_total * 100
            bar = "#" * max(1, int(pct))
            print(f"    {d:+3d}: {count:>8d} ({pct:6.2f}%) {bar}")

    return exp_dist


def analyze_bounded_search(W, block_size, window):
    """Run bounded search with fixed window around naive scale and measure quality."""
    M, K = W.shape
    x = W.float().reshape(M * (K // block_size), block_size)

    all_scales = build_ue8m0_scales(device=W.device)
    num_scales = all_scales.shape[0]

    amax = x.abs().amax(dim=-1)
    safe_amax = amax.clamp(min=1e-30)
    log2_amax = safe_amax.log2()
    exponent = (log2_amax - 2 + 127).floor().clamp(min=1, max=254)
    s0 = torch.pow(2.0, exponent - 128.0)

    s0_idx = (exponent - 1).to(torch.long)

    E0 = compute_block_sse(x, s0)
    best_s = s0.clone()
    best_E = E0.clone()

    for delta in range(-window, window + 1):
        if delta == 0:
            continue
        cand_idx = (s0_idx + delta).clamp(0, num_scales - 1)
        s_cand = all_scales[cand_idx]

        E_s = compute_block_sse(x, s_cand)
        improved = E_s < best_E
        best_E[improved] = E_s[improved]
        best_s[improved] = s_cand[improved]

    quants = fp4_quantize(x, best_s.unsqueeze(-1))
    dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
    W_dq = dq.reshape(M, K)

    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    return w_err.item(), (w_err / w_norm * 100).item()


def main():
    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True)
    X = torch.load(X_PATH, map_location=DEVICE, weights_only=True)
    print(f"W: {W.shape} {W.dtype}")
    print(f"X: {X.shape} {X.dtype}")

    for block_size in [16, 32]:
        print(f"\n{'='*60}")
        print(f"Block size: {block_size}")
        print(f"{'='*60}")

        exp_dist = analyze_scale_distances(W, block_size)

        # Metrics for naive/optimal
        W_blocked = W.reshape(W.shape[0], W.shape[1] // block_size, block_size)
        _, _, W_naive = mxfp4_naive(W_blocked, return_dequant=True)
        _, _, W_opt = mxfp4_optimal(W_blocked, return_dequant=True)
        metrics_naive = compute_metrics(W, W_naive.reshape_as(W), X)
        metrics_opt = compute_metrics(W, W_opt.reshape_as(W), X)
        naive_pct = metrics_naive['weight_error_pct']
        opt_pct = metrics_opt['weight_error_pct']

        print(f"\n  Fixed-window search quality (||Q(W)-W||/||W||):")
        print(f"    Naive (window=0):    {naive_pct:.4f}%")

        for window in [1, 2, 3, 4, 5]:
            w_err, w_pct = analyze_bounded_search(W, block_size, window)
            gap = (w_pct - opt_pct) / (naive_pct - opt_pct) * 100 if naive_pct != opt_pct else 0
            print(f"    Window=+/-{window:<2d} ({2*window+1:>2d} cands): {w_pct:.4f}%  (gap to optimal: {gap:.1f}%)")

        print(f"    Full optimal (254):  {opt_pct:.4f}%")


if __name__ == "__main__":
    main()
