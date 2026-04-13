import torch

Q_MAX = 127  # max quantized magnitude (symmetric INT8)
D_0 = 0.5  # dead-zone boundary: |x|/s < 0.5 rounds to zero
VALID_BLOCK_SIZES = (32, 64, 128, 256)


@torch.compiler.disable
def _fp8_e4m3_snap(x):
    """Snap float32 values to nearest FP8 E4M3 representable value.

    Excluded from torch.compile because inductor emits float8e4nv which
    is unsupported on pre-Ada GPUs.
    """
    return x.to(torch.float8_e4m3fn).to(torch.float32)


@torch.compiler.disable
def build_fp8_e4m3_scales(device="cpu"):
    """Return sorted tensor of all 126 positive FP8 E4M3 representable values.

    Args:
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape ``(126,)`` with sorted positive FP8 E4M3 values as float32.
    """
    all_bytes = torch.arange(256, dtype=torch.uint8, device=device)
    fp8_vals = all_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    pos = fp8_vals[(fp8_vals > 0) & (~fp8_vals.isnan())]
    return pos.unique().sort().values


def int8_quantize(x, s):
    """Symmetric INT8 quantization: ``round(clamp(x / s, -127, 127))``.

    Args:
        x: Input tensor of shape ``(..., block_size)``.
        s: Per-block effective scale of shape ``(..., 1)``, broadcastable to *x*.

    Returns:
        Integer-valued tensor in ``[-127, 127]``, same shape as *x*.
    """
    return torch.clamp(torch.round(x / s), -Q_MAX, Q_MAX)


def int8_dequantize_block(quants, s):
    """Dequantize INT8 block values: ``dequant = quants * s``.

    Args:
        quants: Integer-valued tensor of shape ``(..., block_size)``.
        s: Per-block effective scale of shape ``(..., 1)``, broadcastable.

    Returns:
        Dequantized tensor with the same shape as *quants*.
    """
    return quants * s


def compute_block_sse(x, s):
    """Compute per-block sum of squared INT8 quantization error.

    Args:
        x: Block values of shape ``(num_blocks, block_size)``.
        s: Per-block effective scales of shape ``(num_blocks,)`` or ``(num_blocks, 1)``.

    Returns:
        Tensor of shape ``(num_blocks,)`` with the SSE for each block.
    """
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    quants = int8_quantize(x, s)
    dq = int8_dequantize_block(quants, s)
    return (x - dq).pow(2).sum(dim=-1)


def int8_naive(W, dim=-1, return_dequant=False):
    """Naive INT8 quantization: ``s = FP8_E4M3(max|x_i|) / 127`` per block.

    The per-block amax is snapped to FP8 E4M3 and divided by 127 to get
    the effective scale. This keeps the stored amax within FP8 range.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be in ``{32, 64, 128, 256}``.
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block effective scales (FP8 amax / 127). Shape is
          *W.shape* with dimension *dim* removed.
        - **quants**: Integer values in ``[-127, 127]``. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in VALID_BLOCK_SIZES, (
        f"block_size={block_size}, expected one of {VALID_BLOCK_SIZES}"
    )

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)

    amax = x.abs().amax(dim=-1)
    amax_fp8 = _fp8_e4m3_snap(amax.clamp(min=1e-12))
    scales = amax_fp8 / Q_MAX  # effective scale

    quants = int8_quantize(x, scales.unsqueeze(-1))
    result = (
        scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = int8_dequantize_block(quants, scales.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def int8_optimal(W, dim=-1, return_dequant=False):
    """SSE-Optimal INT8 quantization via bounded search over FP8 E4M3 scales.

    The effective scale grid is ``{a / 127 : a in FP8_E4M3_positive}``,
    giving 126 discrete candidates. Uses clipping and dead-zone bounds
    to prune the search, with a fast-fail clipping check per candidate.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be in ``{32, 64, 128, 256}``.
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block optimal effective scales. Shape is *W.shape*
          with dimension *dim* removed.
        - **quants**: Integer values in ``[-127, 127]``. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in VALID_BLOCK_SIZES

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    # Effective scale grid: FP8 amax values / 127
    all_fp8 = build_fp8_e4m3_scales(device=x.device)
    all_scales = all_fp8 / Q_MAX

    # Step 1: Baseline (naive) scale and error
    amax = x.abs().amax(dim=-1)  # (N,)
    amax_fp8 = _fp8_e4m3_snap(amax.clamp(min=1e-12))
    s0 = amax_fp8 / Q_MAX  # (N,)

    E0 = compute_block_sse(x, s0)  # (N,)

    best_s = s0.clone()
    best_E = E0.clone()

    # Step 2: Edge case - noise blocks (sum(x^2) <= E0)
    total_energy = x.pow(2).sum(dim=-1)  # (N,)
    noise_mask = total_energy <= E0

    # Step 3: Compute bounds
    sqrt_E0 = E0.sqrt()
    s_min = ((amax - sqrt_E0) / Q_MAX).clamp(min=0.0)  # (N,)

    # Upper bound: sort |x| ascending, cumsum of squares, find k*
    sorted_abs, _ = x.abs().sort(dim=-1)  # (N, block_size)
    cumsum_sq = sorted_abs.pow(2).cumsum(dim=-1)  # (N, block_size)
    k_star = (cumsum_sq <= E0.unsqueeze(-1)).sum(dim=-1)  # (N,)

    noise_mask = noise_mask | (k_star >= block_size)

    k_star_idx = k_star.clamp(max=block_size - 1)
    y_k_plus_1 = sorted_abs.gather(dim=-1, index=k_star_idx.unsqueeze(-1)).squeeze(-1)
    s_max = y_k_plus_1 / D_0  # (N,)

    # Step 4: Bounded search over candidate scales
    active = ~noise_mask
    if active.any():
        x_active = x[active]
        s_min_a = s_min[active]
        s_max_a = s_max[active]
        best_E_a = best_E[active].clone()
        best_s_a = best_s[active].clone()

        for s_val in all_scales:
            s_f = s_val.item()

            # Per-block range check
            in_range = (s_f >= s_min_a) & (s_f <= s_max_a)
            if not in_range.any():
                continue

            # Fast-fail: clipping error H(s) = sum(max(|x_i| - 127*s, 0)^2)
            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)

            evaluate = in_range & (H_s < best_E_a)
            if not evaluate.any():
                continue

            # Full SSE computation
            s_broadcast = torch.full(
                (x_active.shape[0], 1), s_f, device=x_active.device, dtype=x_active.dtype
            )
            E_s = compute_block_sse(x_active, s_broadcast.squeeze(-1))

            improved = evaluate & (E_s < best_E_a)
            best_E_a[improved] = E_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = int8_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = int8_dequantize_block(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def _compute_block_hessian_error(x, s, H_blocks, M_dim, num_col_blocks, block_size):
    """Compute per-block Hessian-weighted INT8 quantization error.

    Args:
        x: Block values of shape ``(N, block_size)`` where ``N = M_dim * num_col_blocks``.
        s: Per-block effective scales of shape ``(N,)``.
        H_blocks: Block Hessians of shape ``(num_col_blocks, block_size, block_size)``.
        M_dim: Number of rows (M dimension).
        num_col_blocks: Number of column blocks.
        block_size: Block size.

    Returns:
        Tensor of shape ``(N,)`` with per-block Hessian-weighted error.
    """
    quants = int8_quantize(x, s.unsqueeze(-1))
    dq = int8_dequantize_block(quants, s.unsqueeze(-1))
    r = x - dq  # (N, bs)
    r_3d = r.reshape(M_dim, num_col_blocks, block_size)
    Hr = torch.einsum("jab,mjb->mja", H_blocks, r_3d)  # (M, ncb, bs)
    E_H = (r_3d * Hr).sum(dim=-1)  # (M, ncb)
    return E_H.reshape(-1)  # (N,)


def int8_optimal_hessian(W, dim=-1, return_dequant=False, X=None, H_blocks=None):
    """Hessian-aware optimal INT8 scale search over FP8 E4M3 candidates.

    Like :func:`int8_optimal`, searches over FP8 E4M3 scale candidates using
    SSE bounds for pruning, but selects the scale minimizing the Hessian-weighted
    error ``(x - sq)^T H (x - sq)`` instead of raw SSE.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be in ``{32, 64, 128, 256}``.
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.
        X: Activation tensor of shape ``(T, K)``. H computed as ``X_j^T @ X_j``.
        H_blocks: Pre-computed block Hessians of shape
            ``(num_col_blocks, bs, bs)``. If provided, *X* is ignored.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block optimal effective scales. Shape is *W.shape*
          with dimension *dim* removed.
        - **quants**: Integer values in ``[-127, 127]``. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    assert X is not None or H_blocks is not None, "X or H_blocks required"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in VALID_BLOCK_SIZES

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)
    N = x.shape[0]

    # --- Block Hessians ---
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

    # Effective scale grid: FP8 amax values / 127
    all_fp8 = build_fp8_e4m3_scales(device=x.device)
    all_scales = all_fp8 / Q_MAX

    # Step 1: Baseline (naive) scale and SSE error (for bounding)
    amax = x.abs().amax(dim=-1)  # (N,)
    amax_fp8 = _fp8_e4m3_snap(amax.clamp(min=1e-12))
    s0 = amax_fp8 / Q_MAX  # (N,)

    E0_sse = compute_block_sse(x, s0)  # (N,) -- SSE for bounding
    E0_H = _compute_block_hessian_error(x, s0, H, M_dim, num_col_blocks, block_size)

    best_s = s0.clone()
    best_E = E0_H.clone()

    # Step 2: Edge case - noise blocks (sum(x^2) <= E0_sse)
    total_energy = x.pow(2).sum(dim=-1)
    noise_mask = total_energy <= E0_sse

    # Step 3: Compute SSE-based bounds (still valid for pruning)
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

        # Map active indices to (m, j) for Hessian lookup
        active_indices = active.nonzero(as_tuple=True)[0]  # (N_active,)
        active_j = active_indices % num_col_blocks  # column block index

        for s_val in all_scales:
            s_f = s_val.item()

            # Per-block range check (SSE bounds)
            in_range = (s_f >= s_min_a) & (s_f <= s_max_a)
            if not in_range.any():
                continue

            # Fast-fail: SSE clipping error
            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)
            evaluate = in_range & (H_s < best_E_a * 10)  # looser fast-fail for H-error
            if not evaluate.any():
                continue

            # Compute Hessian-weighted error for active blocks (chunked to avoid OOM)
            quants_s = int8_quantize(x_active, torch.tensor(s_f, device=x.device))
            dq_s = int8_dequantize_block(quants_s, torch.tensor(s_f, device=x.device))
            r = x_active - dq_s  # (N_active, bs)

            # Hr = H[j] @ r for each active block, chunked
            E_H_s = torch.empty(x_active.shape[0], device=x.device)
            chunk = 4096
            for c0 in range(0, x_active.shape[0], chunk):
                c1 = min(c0 + chunk, x_active.shape[0])
                H_chunk = H[active_j[c0:c1]]  # (chunk, bs, bs)
                Hr = torch.bmm(H_chunk, r[c0:c1].unsqueeze(-1)).squeeze(-1)
                E_H_s[c0:c1] = (r[c0:c1] * Hr).sum(dim=-1)

            improved = evaluate & (E_H_s < best_E_a)
            best_E_a[improved] = E_H_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = int8_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = int8_dequantize_block(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def int8_dequantize(scales, quants, dim=-1):
    """Dequantize INT8: ``dequant = quants * scales``.

    Args:
        scales: Per-block effective scales (FP8 amax / 127). Shape is the
            original *W.shape* with dimension *dim* removed.
        quants: Integer values in ``[-127, 127]``. Same shape as the
            original *W*.
        dim: Block dimension in *quants* (default: -1).

    Returns:
        Dequantized tensor with the same shape as *quants*.
    """
    return quants * scales.unsqueeze(dim)
