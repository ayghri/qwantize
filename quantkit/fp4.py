"""FP4 E2M1 shared utilities: Triton ASM quantize-dequantize and nibble unpacking.

Standard FP4 E2M1 codebook: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
Decision boundaries: {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}

Used by both NVFP4 and MXFP4 kernels.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inline ASM: FP4 quantize-dequantize
# ---------------------------------------------------------------------------


@triton.jit
def fp4_dequant_asm(x, s):
    """
    FP4 E2M1 quantize-dequantize via inline PTX ASM.

    Maps |x|/s to the nearest FP4 codebook value {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    using 7 boundary comparisons (setp) and a selp chain, then reconstructs
    dq = sign(x) * q * s.

    Inputs: x (float32), s (float32)
    Output: dq (float32 dequantized value)
    """
    dq = tl.inline_asm_elementwise(
        asm="""
    {
    .reg .f32 xi, si, ax, yy, qq, dv;
    .reg .pred p0, p1, p2, p3, p4, p5, p6, pn;

    mov.f32 xi, $2;
    mov.f32 si, $1;

    // yy = |x| / s
    abs.f32 ax, xi;
    div.full.f32 yy, ax, si;

    // 7 boundary comparisons (boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    setp.le.f32 p0, yy, 0f3E800000;
    setp.le.f32 p1, yy, 0f3F400000;
    setp.le.f32 p2, yy, 0f3FA00000;
    setp.le.f32 p3, yy, 0f3FE00000;
    setp.le.f32 p4, yy, 0f40200000;
    setp.le.f32 p5, yy, 0f40600000;
    setp.le.f32 p6, yy, 0f40A00000;

    // selp chain: codebook {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    mov.f32 qq, 0f40C00000;              // default: 6.0
    selp.f32 qq, 0f40800000, qq, p6;     // y <= 5.0  -> 4.0
    selp.f32 qq, 0f40400000, qq, p5;     // y <= 3.5  -> 3.0
    selp.f32 qq, 0f40000000, qq, p4;     // y <= 2.5  -> 2.0
    selp.f32 qq, 0f3FC00000, qq, p3;     // y <= 1.75 -> 1.5
    selp.f32 qq, 0f3F800000, qq, p2;     // y <= 1.25 -> 1.0
    selp.f32 qq, 0f3F000000, qq, p1;     // y <= 0.75 -> 0.5
    selp.f32 qq, 0f00000000, qq, p0;     // y <= 0.25 -> 0.0

    // dv = sign(x) * qq * s
    mul.f32 dv, qq, si;
    setp.lt.f32 pn, xi, 0f00000000;
    @pn neg.f32 dv, dv;

    mov.f32 $0, dv;
    }
    """,
        constraints="=r,r,r",
        args=[s, x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return dq


# ---------------------------------------------------------------------------
# Triton JIT helpers
# ---------------------------------------------------------------------------


@triton.jit
def fp4_sse_block(x, x_abs, s, BLOCK_SIZE: tl.constexpr):
    """Compute SSE of FP4 quantization for a block with scale s."""
    s_vec = s + tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dq = fp4_dequant_asm(x, s_vec)
    err = x - dq
    return tl.sum(err * err, axis=0)


@triton.jit
def fp4_hessian_block(x, x_abs, s, H_block, BLOCK_SIZE: tl.constexpr):
    """Compute Hessian-weighted FP4 quantization error: r^T H r."""
    s_vec = s + tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dq = fp4_dequant_asm(x, s_vec)
    r = x - dq  # (BLOCK_SIZE,)
    # Mat-vec: Hr = H @ r, where H is (BLOCK_SIZE, BLOCK_SIZE), r is (BLOCK_SIZE,)
    Hr = tl.sum(H_block * r[None, :], axis=1)  # (BLOCK_SIZE,)
    return tl.sum(r * Hr, axis=0)  # scalar: r^T H r


@triton.jit
def fp4_dequant_block(x, s, BLOCK_SIZE: tl.constexpr):
    """Dequantize a block with scale s, return dequantized values."""
    s_vec = s + tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    return fp4_dequant_asm(x, s_vec)


# ---------------------------------------------------------------------------
# FP4 nibble unpacking
# ---------------------------------------------------------------------------


@triton.jit
def fp4_decode_kernel(
    input_ptr,  # Pointer to uint8 data
    output_ptr,  # Pointer to int8 data
    n_elements,  # Number of input bytes
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n_int32 = n_elements // 4

    ofs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = ofs < n_int32

    in_ptr_32 = input_ptr.to(tl.pointer_type(tl.int32))
    out_ptr_32 = output_ptr.to(tl.pointer_type(tl.int32))

    packed_input = tl.load(in_ptr_32 + ofs, mask=mask)

    out0, out1 = tl.inline_asm_elementwise(
        asm="""
    .reg .b32 r_in, r_lut;
    .reg .b32 r_idx, r_sign, r_shift, r_mag, r_neg, r_val, r_tmp;
    .reg .b32 r_out0, r_out1;
    .reg .pred p_neg;

    mov.b32 r_in, $2;             // Load Input
    mov.b32 r_lut, 0xC8643210;    // LUT: 12, 8, 6, 4, 3, 2, 1, 0
    mov.b32 r_out0, 0;            // Init Output 0
    mov.b32 r_out1, 0;            // Init Output 1

    // Decode Bytes 0 and 1 -> Output 0
    // Nibble 0 (Byte 0 Low) -> Out0 Bits 0-7
    bfe.u32 r_idx,  r_in, 0, 3;
    bfe.u32 r_sign, r_in, 3, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    or.b32  r_out0, r_out0, r_val;

    // Nibble 1 (Byte 0 High) -> Out0 Bits 8-15
    bfe.u32 r_idx,  r_in, 4, 3;
    bfe.u32 r_sign, r_in, 7, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 8;
    or.b32  r_out0, r_out0, r_tmp;

    // Nibble 2 (Byte 1 Low) -> Out0 Bits 16-23
    bfe.u32 r_idx,  r_in, 8, 3;
    bfe.u32 r_sign, r_in, 11, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 16;
    or.b32  r_out0, r_out0, r_tmp;

    // Nibble 3 (Byte 1 High) -> Out0 Bits 24-31
    bfe.u32 r_idx,  r_in, 12, 3;
    bfe.u32 r_sign, r_in, 15, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 24;
    or.b32  r_out0, r_out0, r_tmp;

    // Decode Bytes 2 and 3 -> Output 1
    // Nibble 4 (Byte 2 Low) -> Out1 Bits 0-7
    bfe.u32 r_idx,  r_in, 16, 3;
    bfe.u32 r_sign, r_in, 19, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    or.b32  r_out1, r_out1, r_val;

    // Nibble 5 (Byte 2 High) -> Out1 Bits 8-15
    bfe.u32 r_idx,  r_in, 20, 3;
    bfe.u32 r_sign, r_in, 23, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 8;
    or.b32  r_out1, r_out1, r_tmp;

    // Nibble 6 (Byte 3 Low) -> Out1 Bits 16-23
    bfe.u32 r_idx,  r_in, 24, 3;
    bfe.u32 r_sign, r_in, 27, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 16;
    or.b32  r_out1, r_out1, r_tmp;

    // Nibble 7 (Byte 3 High) -> Out1 Bits 24-31
    bfe.u32 r_idx,  r_in, 28, 3;
    bfe.u32 r_sign, r_in, 31, 1;
    shl.b32 r_shift, r_idx, 2;
    bfe.u32 r_mag, r_lut, r_shift, 4;
    neg.s32 r_neg, r_mag;
    setp.ne.u32 p_neg, r_sign, 0;
    selp.s32 r_val, r_neg, r_mag, p_neg;
    and.b32 r_val, r_val, 0xFF;
    shl.b32 r_tmp, r_val, 24;
    or.b32  r_out1, r_out1, r_tmp;

    mov.b32 $0, r_out0;
    mov.b32 $1, r_out1;
    """,
        constraints="=r,=r,r",
        args=[packed_input],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )

    tl.store(out_ptr_32 + (ofs * 2), out0, mask=mask)
    tl.store(out_ptr_32 + (ofs * 2 + 1), out1, mask=mask)


def fp4_unpack(input_data: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 nibbles from uint8 tensor to int8 tensor.

    Each input byte contains two FP4 values (low/high nibble).
    Output has 2x the number of elements, with decoded int8 values
    from the doubled codebook {0, 1, 2, 3, 4, 6, 8, 12} with sign.

    Args:
        input_data: (N,) uint8 tensor of packed FP4 nibbles (on CUDA).

    Returns:
        (2*N,) int8 tensor of decoded values.
    """
    assert input_data.dtype == torch.uint8
    assert input_data.is_cuda
    n_bytes = input_data.numel()
    assert n_bytes % 4 == 0, "Input size must be a multiple of 4 bytes"

    output_data = torch.empty(n_bytes * 2, dtype=torch.int8, device=input_data.device)

    BLOCK_SIZE = 256
    n_int32 = n_bytes // 4
    grid = ((n_int32 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fp4_decode_kernel[grid](input_data, output_data, n_bytes, BLOCK_SIZE=BLOCK_SIZE)

    return output_data
