"""
MXFP4 quantization with UE8M0 scales: naive vs optimal.

UE8M0: 8-bit unsigned exponent, 0 mantissa bits -> pure powers of 2.
  scale = 2^(exponent - 127), exponent in {1, ..., 254} (0 and 255 reserved)

FP4 E2M1 codebook (standard): {0, 0.5, 1, 1.5, 2, 3, 4, 6} (and negatives)
  q_max = 6, decision boundary for zero d_0 = 0.25
"""

import torch

# Standard FP4 E2M1 codebook
FP4_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Decision boundaries (midpoints between consecutive codebook values)
FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
Q_MAX = 6.0   # max codebook value
D_0 = 0.25    # decision boundary for rounding to zero


def build_ue8m0_scales(device="cpu"):
    """Return sorted tensor of all 254 positive UE8M0 scale values.

    UE8M0 scale: ``2^(e - 127)`` for ``e in {1, ..., 254}``.
    (e=0 reserved for zero, e=255 reserved for NaN/Inf)

    Args:
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape ``(254,)`` with sorted UE8M0 power-of-2 scales as float32.
    """
    exponents = torch.arange(1, 255, device=device, dtype=torch.float32)
    return torch.pow(2.0, exponents - 127.0)  # sorted ascending, 254 values


def fp4_quantize(x, s):
    """Quantize to FP4 E2M1 codebook values given a per-block scale.

    Maps each element to the nearest value in ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``
    (with sign preserved).

    Args:
        x: Input tensor of shape ``(..., block_size)``.
        s: Per-block scale of shape ``(..., 1)``, broadcastable to *x*.

    Returns:
        Signed codebook values with the same shape as *x*.
    """
    boundaries = FP4_BOUNDARIES.to(x.device)
    codebook = FP4_CODEBOOK.to(x.device)
    signs = x.sign()
    y = x.abs() / s  # normalized magnitude
    bucket_idx = torch.bucketize(y, boundaries)
    q_mag = codebook[bucket_idx]
    return signs * q_mag


def fp4_dequantize(quants, s):
    """Dequantize FP4 codebook values back to float: ``dequant = quants * s``.

    Args:
        quants: Signed codebook values of shape ``(..., block_size)``.
        s: Per-block scale of shape ``(..., 1)``, broadcastable to *quants*.

    Returns:
        Dequantized tensor with the same shape as *quants*.
    """
    return quants * s


def compute_block_sse(x, s):
    """Compute per-block sum of squared quantization error.

    Args:
        x: Block values of shape ``(num_blocks, block_size)``.
        s: Per-block scales of shape ``(num_blocks,)`` or ``(num_blocks, 1)``.

    Returns:
        Tensor of shape ``(num_blocks,)`` with the SSE for each block.
    """
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    quants = fp4_quantize(x, s)
    dq = fp4_dequantize(quants, s)
    return (x - dq).pow(2).sum(dim=-1)


def mxfp4_naive(W, dim=-1, return_dequant=False):
    """Naive MXFP4 quantization: ``s = 2^(floor(log2(amax)) - 2)`` per block.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block UE8M0 (power-of-2) scales. Shape is *W.shape*
          with dimension *dim* removed.
        - **quants**: Signed FP4 codebook values. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)

    amax = x.abs().amax(dim=-1)  # (N,)
    safe_amax = amax.clamp(min=1e-30)
    log2_amax = safe_amax.log2()
    exponent = (log2_amax - 2 + 127).floor()
    exponent = exponent.clamp(min=1, max=254)
    scales = torch.pow(2.0, exponent - 127.0)  # (N,)

    quants = fp4_quantize(x, scales.unsqueeze(-1))
    result = (
        scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, scales.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def mxfp4_optimal(W, dim=-1, return_dequant=False):
    """Optimal MXFP4 quantization via bounded search over UE8M0 scales.

    Same algorithm as :func:`~qwantize.nvfp4.reference.nvfp4_optimal`,
    adapted for UE8M0 power-of-2 scales. Since consecutive UE8M0 scales
    differ by a factor of 2, the optimal is always within 1 step of naive.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block optimal UE8M0 scales. Shape is *W.shape*
          with dimension *dim* removed.
        - **quants**: Signed FP4 codebook values. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    all_scales = build_ue8m0_scales(device=x.device)  # 254 power-of-2 scales

    # Step 1: Baseline (naive) scale and error
    amax = x.abs().amax(dim=-1)
    safe_amax = amax.clamp(min=1e-30)
    log2_amax = safe_amax.log2()
    exponent = (log2_amax - 2 + 127).floor().clamp(min=1, max=254)
    s0 = torch.pow(2.0, exponent - 127.0)

    E0 = compute_block_sse(x, s0)

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

            # Fast-fail: clipping error
            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)

            evaluate = in_range & (H_s < best_E_a)
            if not evaluate.any():
                continue

            # Full SSE
            s_broadcast = torch.full(
                (x_active.shape[0], 1), s_f, device=x_active.device, dtype=x_active.dtype
            )
            E_s = compute_block_sse(x_active, s_broadcast.squeeze(-1))

            improved = evaluate & (E_s < best_E_a)
            best_E_a[improved] = E_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    quants = fp4_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def mxfp4_optimal_hessian(W, dim=-1, return_dequant=False, X=None):
    """Hessian-aware optimal MXFP4 scale search.

    Like :func:`mxfp4_optimal`, searches over UE8M0 scale candidates using
    SSE bounds for pruning, but selects the scale minimizing the Hessian-weighted
    error ``(x - sq)^T H (x - sq)`` instead of raw SSE.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.
        X: Activation tensor of shape ``(T, K)``. Required.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.
    """
    assert X is not None, "X (activations) required for Hessian-aware scale search"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    # --- Block Hessians from X ---
    K_dim = X.shape[1]
    num_col_blocks = K_dim // block_size
    M_dim = N // num_col_blocks
    assert N == M_dim * num_col_blocks

    H = torch.empty(num_col_blocks, block_size, block_size, device=x.device)
    batch_t = 8192
    for j in range(num_col_blocks):
        acc = torch.zeros(block_size, block_size, device=x.device)
        for t0 in range(0, X.shape[0], batch_t):
            Xj = X[t0 : t0 + batch_t, j * block_size : (j + 1) * block_size].float()
            acc.addmm_(Xj.T, Xj)
        H[j] = acc

    all_scales = build_ue8m0_scales(device=x.device)

    # Step 1: Baseline (naive) scale and SSE error (for bounding)
    amax = x.abs().amax(dim=-1)
    safe_amax = amax.clamp(min=1e-30)
    log2_amax = safe_amax.log2()
    exponent = (log2_amax - 2 + 127).floor().clamp(min=1, max=254)
    s0 = torch.pow(2.0, exponent - 127.0)

    E0_sse = compute_block_sse(x, s0)

    # Hessian baseline error
    quants0 = fp4_quantize(x, s0.unsqueeze(-1))
    dq0 = fp4_dequantize(quants0, s0.unsqueeze(-1))
    r0 = x - dq0
    r0_3d = r0.reshape(M_dim, num_col_blocks, block_size)
    Hr0 = torch.einsum("jab,mjb->mja", H, r0_3d)
    E0_H = (r0_3d * Hr0).sum(dim=-1).reshape(-1)

    best_s = s0.clone()
    best_E = E0_H.clone()

    # Step 2: Edge case - noise blocks
    total_energy = x.pow(2).sum(dim=-1)
    noise_mask = total_energy <= E0_sse

    # Step 3: SSE-based bounds
    sqrt_E0 = E0_sse.sqrt()
    s_min = ((amax - sqrt_E0) / Q_MAX).clamp(min=0.0)

    sorted_abs, _ = x.abs().sort(dim=-1)
    cumsum_sq = sorted_abs.pow(2).cumsum(dim=-1)
    k_star = (cumsum_sq <= E0_sse.unsqueeze(-1)).sum(dim=-1)
    noise_mask = noise_mask | (k_star >= block_size)

    k_star_idx = k_star.clamp(max=block_size - 1)
    y_k_plus_1 = sorted_abs.gather(dim=-1, index=k_star_idx.unsqueeze(-1)).squeeze(-1)
    s_max = y_k_plus_1 / D_0

    # Step 4: Bounded search, evaluate Hessian error
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

            # Fast-fail: SSE clipping error
            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)
            evaluate = in_range & (H_s < E0_sse[active])
            if not evaluate.any():
                continue

            # Hessian-weighted error
            quants_s = fp4_quantize(x_active, torch.tensor(s_f, device=x.device))
            dq_s = fp4_dequantize(quants_s, torch.tensor(s_f, device=x.device))
            r = x_active - dq_s

            H_active = H[active_j]
            Hr = torch.bmm(H_active, r.unsqueeze(-1)).squeeze(-1)
            E_H_s = (r * Hr).sum(dim=-1)

            improved = evaluate & (E_H_s < best_E_a)
            best_E_a[improved] = E_H_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    quants = fp4_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def mxfp4_dequantize(scales, quants, dim=-1):
    """Dequantize MXFP4: ``dequant = quants * scales``.

    Args:
        scales: Per-block scales. Shape is the original *W.shape* with
            dimension *dim* removed.
        quants: Signed FP4 codebook values. Same shape as the original *W*.
        dim: Block dimension in *quants* (default: -1).

    Returns:
        Dequantized tensor with the same shape as *quants*.
    """
    return quants * scales.unsqueeze(dim)


def scales_to_ue8m0_exponent(scales_f32):
    """Convert float32 scales back to UE8M0 exponent (integer).

    Args:
        scales_f32: Float32 tensor of UE8M0 scale values.

    Returns:
        Int32 tensor of UE8M0 exponents (``log2(scale) + 127``).
    """
    # scale = 2^(e - 127) -> e = log2(scale) + 127
    safe = scales_f32.clamp(min=1e-30)
    return (safe.log2() + 127.0).round().to(torch.int32)
