"""Analyze bit-level distance between naive and optimal FP8 E4M3 scales."""

import torch
from qwantize.nvfp4.reference import (
    build_fp8_e4m3_scales,
    nvfp4_naive,
    nvfp4_optimal,
    compute_block_sse,
    fp4_quantize,
    fp4_dequantize,
    Q_MAX,
    D_0,
)

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def scales_to_fp8_bytes(scales_f32):
    """Convert float32 scales to their FP8 E4M3 uint8 byte representation."""
    return scales_f32.to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_byte_to_index(fp8_bytes, scale_table_bytes):
    """Map FP8 bytes to their index in the sorted positive scale table."""
    return torch.searchsorted(scale_table_bytes.to(torch.int16), fp8_bytes.to(torch.int16))


def analyze_scale_distances(W, block_size):
    """Compute and analyze distances between naive and optimal scales."""
    M, K = W.shape
    W_blocked = W.reshape(M, K // block_size, block_size)

    naive_scales, _ = nvfp4_naive(W_blocked)
    optimal_scales, _ = nvfp4_optimal(W_blocked)

    naive_flat = naive_scales.reshape(-1)
    optimal_flat = optimal_scales.reshape(-1)

    naive_bytes = scales_to_fp8_bytes(naive_flat)
    optimal_bytes = scales_to_fp8_bytes(optimal_flat)

    scale_table = build_fp8_e4m3_scales(device=W.device)
    scale_table_bytes = scales_to_fp8_bytes(scale_table)

    naive_idx = fp8_byte_to_index(naive_bytes, scale_table_bytes)
    optimal_idx = fp8_byte_to_index(optimal_bytes, scale_table_bytes)

    idx_dist = (optimal_idx - naive_idx).to(torch.int32)

    xor_bytes = naive_bytes ^ optimal_bytes
    hamming = torch.zeros_like(xor_bytes, dtype=torch.int32)
    for bit in range(8):
        hamming += ((xor_bytes >> bit) & 1).to(torch.int32)

    byte_dist = (optimal_bytes.to(torch.int32) - naive_bytes.to(torch.int32))

    changed_mask = idx_dist != 0
    N_total = idx_dist.shape[0]
    N_changed = changed_mask.sum().item()

    print(f"\n  Total blocks: {N_total}")
    print(f"  Changed:      {N_changed} ({N_changed/N_total*100:.1f}%)")
    print(f"  Unchanged:    {N_total - N_changed} ({(N_total-N_changed)/N_total*100:.1f}%)")

    print(f"\n  Index distance (optimal - naive) in FP8 table steps:")
    print(f"    min:    {idx_dist.min().item()}")
    print(f"    max:    {idx_dist.max().item()}")
    print(f"    mean:   {idx_dist.float().mean().item():.3f}")
    print(f"    median: {idx_dist.float().median().item():.0f}")

    print(f"\n  Index distance distribution:")
    for d in range(idx_dist.min().item(), idx_dist.max().item() + 1):
        count = (idx_dist == d).sum().item()
        if count > 0:
            pct = count / N_total * 100
            bar = "#" * max(1, int(pct))
            print(f"    {d:+3d}: {count:>8d} ({pct:6.2f}%) {bar}")

    print(f"\n  Hamming distance (bit flips):")
    for h in range(9):
        count = (hamming == h).sum().item()
        if count > 0:
            pct = count / N_total * 100
            print(f"    {h} bits: {count:>8d} ({pct:6.2f}%)")

    print(f"\n  Raw byte distance (int8 view):")
    print(f"    min: {byte_dist.min().item()}")
    print(f"    max: {byte_dist.max().item()}")

    return idx_dist, hamming, byte_dist


def analyze_bounded_search(W, block_size, window):
    """Run bounded search with fixed window around naive scale and measure quality."""
    M, K = W.shape
    x = W.float().reshape(M * (K // block_size), block_size)

    scale_table = build_fp8_e4m3_scales(device=W.device)
    num_scales = scale_table.shape[0]

    amax = x.abs().amax(dim=-1)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = s_cont.to(torch.float8_e4m3fn).to(torch.float32)

    s0_bytes = s0.to(torch.float8_e4m3fn).view(torch.uint8)
    table_bytes = scale_table.to(torch.float8_e4m3fn).view(torch.uint8)
    s0_idx = torch.searchsorted(table_bytes.to(torch.int16), s0_bytes.to(torch.int16))

    E0 = compute_block_sse(x, s0)
    best_s = s0.clone()
    best_E = E0.clone()

    for delta in range(-window, window + 1):
        if delta == 0:
            continue
        cand_idx = (s0_idx + delta).clamp(0, num_scales - 1)
        s_cand = scale_table[cand_idx]

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

        idx_dist, hamming, byte_dist = analyze_scale_distances(W, block_size)

        print(f"\n  Fixed-window search quality (||Q(W)-W||/||W||):")

        W_blocked = W.reshape(W.shape[0], W.shape[1] // block_size, block_size)
        _, _, W_naive = nvfp4_naive(W_blocked, return_dequant=True)
        naive_err = (W_naive.reshape_as(W).float() - W.float()).norm().item()
        naive_pct = naive_err / W.float().norm().item() * 100

        _, _, W_opt = nvfp4_optimal(W_blocked, return_dequant=True)
        opt_err = (W_opt.reshape_as(W).float() - W.float()).norm().item()
        opt_pct = opt_err / W.float().norm().item() * 100

        print(f"    Naive (window=0):    {naive_pct:.4f}%")

        for window in [1, 2, 3, 4, 5, 6, 8, 10]:
            w_err, w_pct = analyze_bounded_search(W, block_size, window)
            gap = (w_pct - opt_pct) / (naive_pct - opt_pct) * 100 if naive_pct != opt_pct else 0
            print(f"    Window=+/-{window:<2d} ({2*window+1:>2d} cands): {w_pct:.4f}%  (gap to optimal: {gap:.1f}%)")

        print(f"    Full optimal (126):  {opt_pct:.4f}%")


if __name__ == "__main__":
    main()
