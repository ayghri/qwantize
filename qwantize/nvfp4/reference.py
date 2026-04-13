import torch

@torch.compiler.disable
def _fp8_e4m3_snap(x):
    """Snap float32 values to nearest FP8 E4M3 representable value.

    Excluded from torch.compile because inductor emits float8e4nv which
    is unsupported on pre-Ada GPUs.
    """
    return x.to(torch.float8_e4m3fn).to(torch.float32)


# FP4 E2M1 codebook (actual values)
FP4_CODEBOOK = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Decision boundaries: midpoints between consecutive codebook values
FP4_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
Q_MAX = 6.0
D_0 = 0.25  # decision boundary for rounding to zero


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


def nvfp4_naive(W, dim=-1, return_dequant=False):
    """Naive NVFP4 quantization: ``s = FP8_E4M3(max|x_i| / 6)`` per block.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block FP8 E4M3 scales. Shape is *W.shape* with
          dimension *dim* removed.
        - **quants**: Signed FP4 codebook values. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)

    amax = x.abs().amax(dim=-1)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    scales = _fp8_e4m3_snap(s_cont)

    quants = fp4_quantize(x, scales.unsqueeze(-1))
    result = (
        scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, scales.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def nvfp4_optimal(W, dim=-1, return_dequant=False):
    """Optimal NVFP4 quantization via bounded search over FP8 E4M3 scales.

    Uses clipping and dead-zone bounds to reduce the search from 126
    FP8 candidates to ~4-8, with a fast-fail clipping check per candidate.
    See :doc:`../optimal_scale_search` for the algorithm.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block optimal FP8 E4M3 scales. Shape is *W.shape*
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
    N = x.shape[0]  # total number of blocks

    all_scales = build_fp8_e4m3_scales(device=x.device)

    # Step 1: Baseline (naive) scale and error
    amax = x.abs().amax(dim=-1)  # (N,)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = _fp8_e4m3_snap(s_cont)  # (N,)

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

    # Blocks where all elements are "affordable" to quantize to zero -> noise
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

            # Fast-fail: clipping error H(s) = sum(max(|x_i| - 6*s, 0)^2)
            H_s = (x_active.abs() - Q_MAX * s_f).clamp(min=0).pow(2).sum(dim=-1)

            # Only evaluate blocks in range and passing fast-fail
            evaluate = in_range & (H_s < best_E_a)
            if not evaluate.any():
                continue

            # Full SSE computation
            s_broadcast = torch.full(
                (x_active.shape[0], 1), s_f, device=x_active.device, dtype=x_active.dtype
            )
            E_s = compute_block_sse(x_active, s_broadcast.squeeze(-1))

            # Update best where improved
            improved = evaluate & (E_s < best_E_a)
            best_E_a[improved] = E_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = fp4_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def _compute_block_hessian_error(x, s, H_blocks, M_dim, num_col_blocks, block_size):
    """Compute per-block Hessian-weighted quantization error.

    Args:
        x: Block values of shape ``(N, block_size)`` where ``N = M_dim * num_col_blocks``.
        s: Per-block scales of shape ``(N,)``.
        H_blocks: Block Hessians of shape ``(num_col_blocks, block_size, block_size)``.
        M_dim: Number of rows (M dimension).
        num_col_blocks: Number of column blocks.
        block_size: Block size.

    Returns:
        Tensor of shape ``(N,)`` with per-block Hessian-weighted error.
    """
    quants = fp4_quantize(x, s.unsqueeze(-1))
    dq = fp4_dequantize(quants, s.unsqueeze(-1))
    r = x - dq  # (N, bs)
    # Reshape to (M, ncb, bs) for einsum with H (ncb, bs, bs)
    r_3d = r.reshape(M_dim, num_col_blocks, block_size)
    Hr = torch.einsum("jab,mjb->mja", H_blocks, r_3d)  # (M, ncb, bs)
    E_H = (r_3d * Hr).sum(dim=-1)  # (M, ncb)
    return E_H.reshape(-1)  # (N,)


def nvfp4_optimal_hessian(W, dim=-1, return_dequant=False, X=None, H_blocks=None):
    """Hessian-aware optimal NVFP4 scale search.

    Like :func:`nvfp4_optimal`, searches over FP8 E4M3 scale candidates using
    SSE bounds for pruning, but selects the scale minimizing the Hessian-weighted
    error ``(x - sq)^T H (x - sq)`` instead of raw SSE. This directly minimizes
    each block's contribution to the output error ``||W_q X - WX||_F^2``.

    See :doc:`../hessian_scale_search` for the math.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.
        X: Activation tensor of shape ``(T, K)``. H computed as X_j^T @ X_j.
        H_blocks: Pre-computed block Hessians of shape ``(num_col_blocks, bs, bs)``.
            If provided, X is ignored.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block optimal FP8 E4M3 scales. Shape is *W.shape*
          with dimension *dim* removed.
        - **quants**: Signed FP4 codebook values. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    assert X is not None or H_blocks is not None, "X or H_blocks required"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

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

    all_scales = build_fp8_e4m3_scales(device=x.device)

    # Step 1: Baseline (naive) scale and SSE error (for bounding)
    amax = x.abs().amax(dim=-1)  # (N,)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = _fp8_e4m3_snap(s_cont)  # (N,)

    E0_sse = compute_block_sse(x, s0)  # (N,) — SSE for bounding
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
        # We need 3D indexing for Hessian error, so work with all blocks
        # but mask inactive ones
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

            # Compute Hessian-weighted error for active blocks
            quants_s = fp4_quantize(x_active, torch.tensor(s_f, device=x.device))
            dq_s = fp4_dequantize(quants_s, torch.tensor(s_f, device=x.device))
            r = x_active - dq_s  # (N_active, bs)

            # Hr = H[j] @ r for each active block
            H_active = H[active_j]  # (N_active, bs, bs)
            Hr = torch.bmm(H_active, r.unsqueeze(-1)).squeeze(-1)  # (N_active, bs)
            E_H_s = (r * Hr).sum(dim=-1)  # (N_active,)

            improved = evaluate & (E_H_s < best_E_a)
            best_E_a[improved] = E_H_s[improved]
            best_s_a[improved] = s_f

        best_E[active] = best_E_a
        best_s[active] = best_s_a

    # Final quantization with optimal scales
    quants = fp4_quantize(x, best_s.unsqueeze(-1))
    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def nvfp4_admm(W, dim=-1, return_dequant=False, X=None, n_outer=3, n_inner=10):
    """ADMM-based NVFP4 quantization using input Hessian.

    Starts from the Hessian-aware optimal scale found by
    :func:`nvfp4_optimal_hessian`, then uses ADMM to refine the FP4
    quantization values by minimizing the Hessian-weighted error
    ``(x - sq)^T H (x - sq)`` per block, where ``H = X_block^T @ X_block``
    captures input activation correlations.
    This further reduces output error beyond :func:`nvfp4_optimal` at the
    cost of slightly higher weight error.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.
        X: Activation tensor of shape ``(T, K)``. Required.
        n_outer: Outer loop iterations (scale updates). Default 3.
        n_inner: Inner ADMM iterations per outer step. Default 10.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``.

        - **scales**: Per-block FP8 E4M3 scales. Shape is *W.shape* with
          dimension *dim* removed.
        - **quants**: Signed FP4 codebook values. Same shape as *W*.
        - **dequant**: ``quants * scales`` broadcast. Same shape as *W*.
    """
    assert X is not None, "X (activations) required for ADMM quantization"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)  # (N, bs)
    N = x.shape[0]

    # --- Block Hessians from X ---
    K_dim = X.shape[1]
    num_col_blocks = K_dim // block_size
    M_dim = N // num_col_blocks
    assert N == M_dim * num_col_blocks, (
        f"Weight blocks ({N}) must equal M * K//bs ({M_dim} * {num_col_blocks})"
    )

    # Compute H_j = X_j^T @ X_j in batches to avoid OOM on large X
    H = torch.empty(num_col_blocks, block_size, block_size, device=x.device)
    batch_t = 8192
    for j in range(num_col_blocks):
        acc = torch.zeros(block_size, block_size, device=x.device)
        for t0 in range(0, X.shape[0], batch_t):
            Xj = X[t0 : t0 + batch_t, j * block_size : (j + 1) * block_size].float()
            acc.addmm_(Xj.T, Xj)
        H[j] = acc

    # --- ADMM constants ---
    # Normalize H by max eigenvalue so ADMM is well-conditioned
    eigvals = torch.linalg.eigvalsh(H)
    max_eig = eigvals[:, -1].clamp(min=1e-6)  # (ncb,)
    H_norm = H / max_eig.reshape(-1, 1, 1)

    rho = torch.ones(num_col_blocks, device=x.device, dtype=x.dtype)

    eye = torch.eye(block_size, device=x.device, dtype=x.dtype).unsqueeze(0)
    M_inv = torch.linalg.inv(H_norm + rho.reshape(-1, 1, 1) * eye)  # (ncb, bs, bs)

    # --- 3D layout: (M, ncb, bs) ---
    x_3d = x.reshape(M_dim, num_col_blocks, block_size)

    # --- Initialize scale via naive: s = FP8(max|x| / 6) ---
    amax = x.abs().amax(dim=-1)  # (N,)
    s_cont = (amax / Q_MAX).clamp(min=1e-12)
    s0 = _fp8_e4m3_snap(s_cont)
    s = s0.reshape(M_dim, num_col_blocks)

    # --- Outer loop: alternate scale and quants ---
    ones = torch.ones((*x_3d.shape[:-1], 1), device=x.device, dtype=x.dtype)
    q_star = None

    for outer in range(n_outer):
        w = x_3d / s.unsqueeze(-1)  # (M, ncb, bs)

        # z = FP4 projection of w (quantize with s=1)
        z = fp4_quantize(w, ones)
        u = torch.zeros_like(w)

        # Precompute H_norm @ w (use normalized H in ADMM)
        Hw = torch.einsum("jab,mjb->mja", H_norm, w)

        # Inner ADMM loop
        for k in range(n_inner):
            rhs = Hw + rho.reshape(1, -1, 1) * (z - u)
            q = torch.einsum("jab,mjb->mja", M_inv, rhs)
            z = fp4_quantize(q + u, ones)
            u = u + q - z

        q_star = z  # discrete FP4 codebook values

        # --- Scale update: s = FP8(x^T H q* / (q*^T H q*)) ---
        Hq = torch.einsum("jab,mjb->mja", H, q_star)
        numer = (x_3d * Hq).sum(dim=-1)  # (M, ncb)
        denom = (q_star * Hq).sum(dim=-1)  # (M, ncb)

        valid = (numer > 0) & (denom > 1e-12)
        s_cont = (numer / denom.clamp(min=1e-12)).clamp(min=1e-12)
        s_snapped = _fp8_e4m3_snap(s_cont)
        s_new = torch.where(valid, s_snapped, s)

        if (s_new == s).all():
            break
        s = s_new

    # --- Final output ---
    best_s = s.reshape(N)
    quants = q_star.reshape(N, block_size)

    result = (
        best_s.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dq = fp4_dequantize(quants, best_s.unsqueeze(-1))
        result = result + (dq.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def nvfp4_dequantize(scales, quants, dim=-1):
    """Dequantize NVFP4: ``dequant = quants * scales``.

    Args:
        scales: Per-block scales. Shape is the original *W.shape* with
            dimension *dim* removed.
        quants: Signed FP4 codebook values. Same shape as the original *W*.
        dim: Block dimension in *quants* (default: -1).

    Returns:
        Dequantized tensor with the same shape as *quants*.
    """
    return quants * scales.unsqueeze(dim)
