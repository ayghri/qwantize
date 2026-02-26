"""
MXFP4 quantization with UE8M0 scales -- Triton kernel with inline PTX ASM.

UE8M0 scales are pure powers of 2. The optimal scale is always either the
naive scale s0 or 2*s0 (one step up). So the "search" is just: compute
SSE for both, pick the winner.

Uses standard FP4 E2M1 codebook {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
"""

import torch
import triton
import triton.language as tl

from ..fp4 import fp4_sse_block, fp4_dequant_block

Q_MAX = 6.0   # max codebook value
D_0 = 0.25    # decision boundary for zero


# ---------------------------------------------------------------------------
# Inline ASM helpers
# ---------------------------------------------------------------------------


@triton.jit
def ue8m0_snap_asm(val):
    """
    Snap a positive float32 to UE8M0 scale: 2^(floor(log2(val)) - 2).

    This computes the naive MXFP4 scale from amax using only bitwise ops.
    The scale is 2^(e-127) where e = floor(log2(amax))-1+127.

    PTX approach: extract the float32 exponent via bfe, subtract 2, reconstruct.
    Result = 2^(biased_exp - 127 - 2) = 2^(biased_exp - 129).
    Equivalently, set mantissa to 0 and exponent to (biased_exp - 2).
    """
    result = tl.inline_asm_elementwise(
        asm="""
    {
    .reg .b32 bits, exp32, new_exp, res;
    .reg .pred p_uf;

    mov.b32 bits, $1;

    // Extract biased exponent (bits 23..30)
    bfe.u32 exp32, bits, 23, 8;

    // new_exp = exp32 - 2 (we want 2^(exp32-127-2) = float with exp = exp32-2)
    add.s32 new_exp, exp32, -2;

    // Underflow check: if new_exp < 1, clamp to 0 (smallest positive is 2^(1-127))
    setp.lt.s32 p_uf, new_exp, 1;

    // Reconstruct: new_exp << 23, mantissa = 0
    shl.b32 res, new_exp, 23;

    // Underflow -> return smallest UE8M0 scale: 2^(1-127) = 2^(-126)
    // which is 0x00800000 as float32 (biased_exp=1, mant=0)
    @p_uf mov.b32 res, 0x00800000;

    mov.b32 $0, res;
    }
    """,
        constraints="=r,r",
        args=[val],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return result


# ---------------------------------------------------------------------------
# Naive kernel
# ---------------------------------------------------------------------------


@triton.jit
def mxfp4_naive_kernel(
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

    # UE8M0 snap: s = 2^(floor(log2(amax)) - 2) via inline ASM
    amax_safe = tl.maximum(amax, 1e-30)
    amax_vec = amax_safe + tl.zeros([1], dtype=tl.float32)
    s0_vec = ue8m0_snap_asm(amax_vec)
    s0 = tl.sum(s0_vec, axis=0)

    dq = fp4_dequant_block(x, s0, BLOCK_SIZE)

    tl.store(out_ptr + pid * BLOCK_SIZE + offs, dq)
    tl.store(out_scale_ptr + pid, s0)


# ---------------------------------------------------------------------------
# Optimal kernel -- just compare s0 vs 2*s0
# ---------------------------------------------------------------------------


@triton.jit
def mxfp4_optimal_kernel(
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

    # Naive scale via inline ASM
    amax_safe = tl.maximum(amax, 1e-30)
    amax_vec = amax_safe + tl.zeros([1], dtype=tl.float32)
    s0_vec = ue8m0_snap_asm(amax_vec)
    s0 = tl.sum(s0_vec, axis=0)
    s0 = tl.maximum(s0, 1e-30)

    # Compute SSE for s0
    E0 = fp4_sse_block(x, x_abs, s0, BLOCK_SIZE)

    # Candidate: s1 = 2 * s0 (one UE8M0 step up)
    s1 = s0 * 2.0
    E1 = fp4_sse_block(x, x_abs, s1, BLOCK_SIZE)

    # Pick the better scale
    best_s = s0
    if E1 < E0:
        best_s = s1

    dq = fp4_dequant_block(x, best_s, BLOCK_SIZE)

    tl.store(out_ptr + pid * BLOCK_SIZE + offs, dq)
    tl.store(out_scale_ptr + pid, best_s)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def mxfp4_naive_triton(W, dim=-1, return_dequant=False):
    """Naive MXFP4 quantization using Triton kernel with inline PTX ASM.

    GPU-accelerated version of :func:`~quantkit.mxfp4.reference.mxfp4_naive`.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.mxfp4.reference.mxfp4_naive` for shape details.
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
    mxfp4_naive_kernel[grid](
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


def mxfp4_optimal_triton(W, dim=-1, return_dequant=False):
    """Optimal MXFP4 quantization using Triton kernel with inline PTX ASM.

    GPU-accelerated version of :func:`~quantkit.mxfp4.reference.mxfp4_optimal`.
    Compares naive scale ``s0`` with ``2*s0`` (one UE8M0 step up) and picks
    whichever has lower SSE.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.mxfp4.reference.mxfp4_optimal` for shape details.
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
    mxfp4_optimal_kernel[grid](
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


# ---------------------------------------------------------------------------
# Torch reference implementations (for benchmarking)
# ---------------------------------------------------------------------------


def mxfp4_naive_torch(W, dim=-1, return_dequant=False):
    """Naive MXFP4 quantization using pure PyTorch operations.

    Functionally identical to :func:`~quantkit.mxfp4.reference.mxfp4_naive`
    but uses vectorized ``argmin`` over the full signed codebook instead of
    ``bucketize``.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.mxfp4.reference.mxfp4_naive` for shape details.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=W.device, dtype=torch.float32,
    )

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)  # (N, block_size)

    # UE8M0 scale
    scale_exponent = x.abs().amax(dim=-1).clamp(min=1e-30).log2().add(-2 + 127).floor()
    scale_exponent = scale_exponent.clamp(min=1, max=254)
    scale = torch.pow(2.0, scale_exponent - 127.0)  # (N,)

    # Quantize: find closest codebook value
    possible = (scale.unsqueeze(-1) * codebook.view(1, 16)).unsqueeze(-2)  # (N, 1, 16)
    deltas = x.unsqueeze(-1) - possible  # (N, bs, 16)
    quants = codebook[deltas.abs().argmin(dim=-1)]  # (N, bs)

    result = (
        scale.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dequants = scale.unsqueeze(-1) * quants
        result = result + (dequants.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result


def mxfp4_optimal_torch(W, dim=-1, return_dequant=False):
    """Optimal MXFP4 quantization using pure PyTorch operations.

    Tries naive scale ``s0`` and ``2*s0``, picks whichever has lower SSE.

    Args:
        W: Input tensor. ``W.shape[dim]`` must be 16 or 32 (the block size).
        dim: Dimension along which to quantize (default: -1).
        return_dequant: If ``True``, also return the dequantized tensor.

    Returns:
        ``(scales, quants)`` by default, or ``(scales, quants, dequant)``
        if *return_dequant* is ``True``. See
        :func:`~quantkit.mxfp4.reference.mxfp4_optimal` for shape details.
    """
    dim = dim % W.ndim
    block_size = W.shape[dim]
    assert block_size in (16, 32)

    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=W.device, dtype=torch.float32,
    )

    x = W.float().movedim(dim, -1)
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, block_size)  # (N, block_size)

    # s0: naive UE8M0 scale
    scale_exponent = x.abs().amax(dim=-1).clamp(min=1e-30).log2().add(-2 + 127).floor()
    scale_exponent = scale_exponent.clamp(min=1, max=254)
    s0 = torch.pow(2.0, scale_exponent - 127.0)  # (N,)

    # s1: one step up
    s1_exponent = (scale_exponent + 1).clamp(max=254)
    s1 = torch.pow(2.0, s1_exponent - 127.0)

    def quant_dequant_sse(x, scale):
        possible = (scale.unsqueeze(-1) * codebook.view(1, 16)).unsqueeze(-2)  # (N, 1, 16)
        deltas = x.unsqueeze(-1) - possible  # (N, bs, 16)
        q = codebook[deltas.abs().argmin(dim=-1)]
        dequants = scale.unsqueeze(-1) * q
        sse = (x - dequants).pow(2).sum(dim=-1)
        return dequants, q, sse

    dq0, q0, sse0 = quant_dequant_sse(x, s0)
    dq1, q1, sse1 = quant_dequant_sse(x, s1)

    # Pick best per block
    use_s1 = (sse1 < sse0).unsqueeze(-1)
    quants = torch.where(use_s1, q1, q0)
    best_scale = torch.where(sse1 < sse0, s1, s0)

    result = (
        best_scale.reshape(*batch_shape),
        quants.reshape(*batch_shape, block_size).movedim(-1, dim),
    )
    if return_dequant:
        dequants = torch.where(use_s1, dq1, dq0)
        result = result + (dequants.reshape(*batch_shape, block_size).movedim(-1, dim),)
    return result
