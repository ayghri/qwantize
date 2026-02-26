import torch
import triton
import triton.language as tl

from .reference import build_fp8_e4m3_scales
from ..fp4 import fp4_sse_block, fp4_hessian_block, fp4_dequant_block
from ..fp8 import fp8_e4m3_snap_asm

Q_MAX = 6.0
D_0 = 0.25


# ---------------------------------------------------------------------------
# Naive kernel
# ---------------------------------------------------------------------------


@triton.jit
def nvfp4_naive_kernel(
    x_ptr,
    out_ptr,
    out_scale_ptr,
    total_blocks,
    block_stride,
    element_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_blocks:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * block_stride + offs * element_stride).to(tl.float32)
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=0)

    # Snap amax/6 to FP8 E4M3 using inline ASM bitwise ops
    s_cont = amax / 6.0
    s_cont = tl.maximum(s_cont, 1e-12)
    # fp8_e4m3_snap_asm expects a tensor, broadcast scalar to size-1
    s_cont_vec = s_cont + tl.zeros([1], dtype=tl.float32)
    s0_vec = fp8_e4m3_snap_asm(s_cont_vec)
    s0 = tl.sum(s0_vec, axis=0)  # extract scalar

    # Dequantize with naive scale
    dq = fp4_dequant_block(x, s0, BLOCK_SIZE)

    tl.store(out_ptr + pid * BLOCK_SIZE + offs, dq)
    tl.store(out_scale_ptr + pid, s0)


# ---------------------------------------------------------------------------
# Optimal kernel
# ---------------------------------------------------------------------------


@triton.jit
def nvfp4_optimal_kernel(
    x_ptr,
    scale_table_ptr,
    out_ptr,
    out_scale_ptr,
    total_blocks,
    block_stride,
    element_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_SCALES: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_blocks:
        return

    # Step 1: Load block
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * block_stride + offs * element_stride).to(tl.float32)
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=0)

    # Step 2: Baseline scale via inline ASM FP8 snap
    s_cont = amax / 6.0
    s_cont = tl.maximum(s_cont, 1e-12)
    s_cont_vec = s_cont + tl.zeros([1], dtype=tl.float32)
    s0_vec = fp8_e4m3_snap_asm(s_cont_vec)
    s0 = tl.sum(s0_vec, axis=0)
    # Ensure s0 > 0
    s0 = tl.maximum(s0, 1e-12)

    # Step 3: Compute baseline error E0
    E0 = fp4_sse_block(x, x_abs, s0, BLOCK_SIZE)

    best_s = s0
    best_E = E0

    # Step 4: Edge case -- noise block
    total_energy = tl.sum(x * x, axis=0)
    is_noise = total_energy <= E0

    if is_noise == 0:
        # Step 5: Lower bound
        sqrt_E0 = tl.sqrt(E0)
        s_min = tl.maximum(0.0, (amax - sqrt_E0) / 6.0)

        # Step 6: Upper bound -- sort, cumsum, find k*
        sorted_abs = tl.sort(x_abs)
        sorted_sq = sorted_abs * sorted_abs
        cumsum_sq = tl.cumsum(sorted_sq, axis=0)

        # k_star = number of elements whose cumsum of squares <= E0
        k_mask = (cumsum_sq <= E0).to(tl.int32)
        k_star = tl.sum(k_mask, axis=0)

        # Extract sorted_abs[min(k_star, BLOCK_SIZE-1)]
        k_idx = tl.minimum(k_star, BLOCK_SIZE - 1)
        # Use masked reduction to pick element at index k_idx
        y_k = tl.sum(tl.where(offs == k_idx, sorted_abs, 0.0), axis=0)
        s_max = y_k / 0.25

        # Step 7: Bounded search over FP8 E4M3 scale candidates
        for i in range(NUM_SCALES):
            s_cand = tl.load(scale_table_ptr + i)

            in_range = (s_cand >= s_min) & (s_cand <= s_max)
            if in_range:
                # Fast-fail: clipping error H(s)
                clip_excess = tl.maximum(x_abs - 6.0 * s_cand, 0.0)
                H_s = tl.sum(clip_excess * clip_excess, axis=0)

                if H_s < best_E:
                    # Full SSE via inline ASM FP4 dequant
                    E_s = fp4_sse_block(x, x_abs, s_cand, BLOCK_SIZE)
                    if E_s < best_E:
                        best_E = E_s
                        best_s = s_cand

    # Step 8: Final dequantization with optimal scale
    dq = fp4_dequant_block(x, best_s, BLOCK_SIZE)

    tl.store(out_ptr + pid * BLOCK_SIZE + offs, dq)
    tl.store(out_scale_ptr + pid, best_s)


# ---------------------------------------------------------------------------
# Hessian-aware optimal kernel
# ---------------------------------------------------------------------------


@triton.jit
def nvfp4_optimal_hessian_kernel(
    x_ptr,
    scale_table_ptr,
    H_ptr,
    out_ptr,
    out_scale_ptr,
    total_blocks,
    num_col_blocks,
    block_stride,
    element_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_SCALES: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_blocks:
        return

    # Column-block index for Hessian lookup
    j = pid % num_col_blocks

    # Step 1: Load block
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * block_stride + offs * element_stride).to(tl.float32)
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=0)

    # Load Hessian block H[j]: (BLOCK_SIZE, BLOCK_SIZE)
    row_offs = tl.arange(0, BLOCK_SIZE)
    col_offs = tl.arange(0, BLOCK_SIZE)
    H_base = j * BLOCK_SIZE * BLOCK_SIZE
    H_block = tl.load(
        H_ptr + H_base + row_offs[:, None] * BLOCK_SIZE + col_offs[None, :]
    )

    # Step 2: Baseline scale via inline ASM FP8 snap
    s_cont = amax / 6.0
    s_cont = tl.maximum(s_cont, 1e-12)
    s_cont_vec = s_cont + tl.zeros([1], dtype=tl.float32)
    s0_vec = fp8_e4m3_snap_asm(s_cont_vec)
    s0 = tl.sum(s0_vec, axis=0)
    s0 = tl.maximum(s0, 1e-12)

    # Step 3: Baseline errors -- SSE for bounding, Hessian for selection
    E0_sse = fp4_sse_block(x, x_abs, s0, BLOCK_SIZE)
    E0_H = fp4_hessian_block(x, x_abs, s0, H_block, BLOCK_SIZE)

    best_s = s0
    best_E = E0_H

    # Step 4: Edge case -- noise block
    total_energy = tl.sum(x * x, axis=0)
    is_noise = total_energy <= E0_sse

    if is_noise == 0:
        # Step 5: Lower bound (SSE-based, still valid for pruning)
        sqrt_E0 = tl.sqrt(E0_sse)
        s_min = tl.maximum(0.0, (amax - sqrt_E0) / 6.0)

        # Step 6: Upper bound -- sort, cumsum, find k*
        sorted_abs = tl.sort(x_abs)
        sorted_sq = sorted_abs * sorted_abs
        cumsum_sq = tl.cumsum(sorted_sq, axis=0)

        k_mask = (cumsum_sq <= E0_sse).to(tl.int32)
        k_star = tl.sum(k_mask, axis=0)

        k_idx = tl.minimum(k_star, BLOCK_SIZE - 1)
        y_k = tl.sum(tl.where(offs == k_idx, sorted_abs, 0.0), axis=0)
        s_max = y_k / 0.25

        # Step 7: Bounded search -- evaluate Hessian error
        for i in range(NUM_SCALES):
            s_cand = tl.load(scale_table_ptr + i)

            in_range = (s_cand >= s_min) & (s_cand <= s_max)
            if in_range:
                # Fast-fail: SSE clipping error (cheap necessary condition)
                clip_excess = tl.maximum(x_abs - 6.0 * s_cand, 0.0)
                H_s = tl.sum(clip_excess * clip_excess, axis=0)

                # Only skip if clipping error alone exceeds baseline SSE
                if H_s < E0_sse:
                    # Full Hessian-weighted error
                    E_H = fp4_hessian_block(x, x_abs, s_cand, H_block, BLOCK_SIZE)
                    if E_H < best_E:
                        best_E = E_H
                        best_s = s_cand

    # Step 8: Final dequantization with optimal scale
    dq = fp4_dequant_block(x, best_s, BLOCK_SIZE)

    tl.store(out_ptr + pid * BLOCK_SIZE + offs, dq)
    tl.store(out_scale_ptr + pid, best_s)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def nvfp4_naive_triton(W, dim=-1, return_dequant=False):
    """Naive NVFP4 quantization using Triton kernel with inline PTX ASM.

    GPU-accelerated version of :func:`~quantkit.nvfp4.reference.nvfp4_naive`.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.nvfp4.reference.nvfp4_naive` for shape details.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float()
    batch_shape = x.shape[:dim] + x.shape[dim + 1:]
    num_blocks = 1
    for s in batch_shape:
        num_blocks *= s

    x_contig = x.movedim(dim, -1).reshape(-1, block_size)
    block_stride = x_contig.stride(0)
    element_stride = x_contig.stride(1)

    out = torch.empty(num_blocks * block_size, device=W.device, dtype=torch.float32)
    out_scales = torch.empty(num_blocks, device=W.device, dtype=torch.float32)

    grid = (num_blocks,)
    nvfp4_naive_kernel[grid](
        x_contig, out, out_scales, num_blocks,
        block_stride, element_stride, BLOCK_SIZE=block_size,
    )

    quants = out.reshape(-1, block_size) / out_scales.unsqueeze(-1)
    result = (
        out_scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        result = result + (out.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def nvfp4_optimal_triton(W, dim=-1, return_dequant=False):
    """Optimal NVFP4 quantization using Triton kernel with inline PTX ASM.

    GPU-accelerated version of :func:`~quantkit.nvfp4.reference.nvfp4_optimal`.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.nvfp4.reference.nvfp4_optimal` for shape details.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float()
    batch_shape = x.shape[:dim] + x.shape[dim + 1:]
    num_blocks = 1
    for s in batch_shape:
        num_blocks *= s

    x_contig = x.movedim(dim, -1).reshape(-1, block_size)
    block_stride = x_contig.stride(0)
    element_stride = x_contig.stride(1)

    scale_table = build_fp8_e4m3_scales(device=W.device)
    num_scales = scale_table.shape[0]

    out = torch.empty(num_blocks * block_size, device=W.device, dtype=torch.float32)
    out_scales = torch.empty(num_blocks, device=W.device, dtype=torch.float32)

    grid = (num_blocks,)
    nvfp4_optimal_kernel[grid](
        x_contig,
        scale_table,
        out,
        out_scales,
        num_blocks,
        block_stride,
        element_stride,
        BLOCK_SIZE=block_size,
        NUM_SCALES=num_scales,
    )

    quants = out.reshape(-1, block_size) / out_scales.unsqueeze(-1)
    result = (
        out_scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        result = result + (out.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def nvfp4_optimal_hessian_triton(W, dim=-1, return_dequant=False, X=None):
    """Hessian-aware optimal NVFP4 quantization using Triton kernel.

    GPU-accelerated version of
    :func:`~quantkit.nvfp4.reference.nvfp4_optimal_hessian`.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.
        X: Activation tensor of shape ``(T, K)``. Required.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.nvfp4.reference.nvfp4_optimal_hessian` for shape details.
    """
    assert X is not None, "X (activations) required for Hessian-aware scale search"
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    x = W.float()
    batch_shape = x.shape[:dim] + x.shape[dim + 1:]
    num_blocks = 1
    for s in batch_shape:
        num_blocks *= s

    x_contig = x.movedim(dim, -1).reshape(-1, block_size)
    block_stride = x_contig.stride(0)
    element_stride = x_contig.stride(1)

    # Compute block Hessians: H[j] = X_j^T @ X_j
    K_dim = X.shape[1]
    num_col_blocks = K_dim // block_size
    H = torch.empty(num_col_blocks, block_size, block_size, device=W.device, dtype=torch.float32)
    batch_t = 8192
    for j in range(num_col_blocks):
        acc = torch.zeros(block_size, block_size, device=W.device, dtype=torch.float32)
        for t0 in range(0, X.shape[0], batch_t):
            Xj = X[t0 : t0 + batch_t, j * block_size : (j + 1) * block_size].float()
            acc.addmm_(Xj.T, Xj)
        H[j] = acc
    H = H.contiguous()

    scale_table = build_fp8_e4m3_scales(device=W.device)
    num_scales = scale_table.shape[0]

    out = torch.empty(num_blocks * block_size, device=W.device, dtype=torch.float32)
    out_scales = torch.empty(num_blocks, device=W.device, dtype=torch.float32)

    grid = (num_blocks,)
    nvfp4_optimal_hessian_kernel[grid](
        x_contig,
        scale_table,
        H,
        out,
        out_scales,
        num_blocks,
        num_col_blocks,
        block_stride,
        element_stride,
        BLOCK_SIZE=block_size,
        NUM_SCALES=num_scales,
    )

    quants = out.reshape(-1, block_size) / out_scales.unsqueeze(-1)
    result = (
        out_scales.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        result = result + (out.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result
