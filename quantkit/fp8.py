"""FP8 E4M3 Triton ASM helper: snap float32 to nearest FP8 E4M3 value."""

import triton
import triton.language as tl


@triton.jit
def fp8_e4m3_snap_asm(val):
    """
    Snap a positive float32 to the nearest FP8 E4M3 representable value.

    FP32: sign(1) | exponent(8, bias=127) | mantissa(23)
    FP8 E4M3: sign(1) | exponent(4, bias=7) | mantissa(3)

    Normal path: round mantissa to 3 bits in-place by adding rounding bias
    (bit 19) to the float32 bit pattern, then masking off bits 19..0.
    Mantissa overflow carry propagates into the exponent via integer addition.
    Round-to-nearest-even: suppress rounding on ties when mant3 LSB is 0.

    Subnormal path (FP8 biased exp < 1, i.e. FP32 biased exp <= 120):
    val * 512 -> round to nearest int -> clamp [0, 8] -> * (1/512).
    """
    result = tl.inline_asm_elementwise(
        asm="""
    {
    .reg .b32 bits, tmp, res;
    .reg .f32 fv;
    .reg .pred p_sub, p_tie, p_even, p_no_round, p_of;

    mov.b32 bits, $1;

    // Subnormal check: FP32 biased exponent <= 120 means FP8 exp < 1
    bfe.u32 tmp, bits, 23, 8;
    setp.le.u32 p_sub, tmp, 120;

    // --- Normal path: round mantissa to 3 bits in-place ---
    // Tie detection: bits[19:0] == 0x80000 (round bit set, no sticky bits)
    and.b32 tmp, bits, 0xFFFFF;
    setp.eq.u32 p_tie, tmp, 0x80000;
    // Round-to-even: suppress rounding on tie if mant3 LSB (bit 20) is 0
    bfe.u32 tmp, bits, 20, 1;
    setp.eq.u32 p_even, tmp, 0;
    and.pred p_no_round, p_tie, p_even;

    // Add rounding bias at bit 19; carry propagates through mantissa -> exponent
    add.u32 res, bits, 0x80000;
    @p_no_round mov.u32 res, bits;
    and.b32 res, res, 0xFFF00000;     // keep sign + exponent + top 3 mantissa bits

    // Clamp to max FP8 E4M3 = 448.0 (0x43E00000)
    setp.gt.u32 p_of, res, 0x43E00000;
    @p_of mov.b32 res, 0x43E00000;

    // --- Subnormal path: val * 512, round to int, clamp to 8, / 512 ---
    // FP8 E4M3 subnormals: k * 2^(-9) for k=0..7; k=8 = smallest normal.
    // Input is always positive, so no lower clamp needed.
    @p_sub mul.f32 fv, $1, 0f44000000;     // * 512.0
    @p_sub cvt.rni.f32.f32 fv, fv;         // round to nearest even int
    @p_sub min.f32 fv, fv, 0f41000000;     // clamp to 8.0
    @p_sub mul.f32 fv, fv, 0f3B000000;     // * (1/512)
    @p_sub mov.b32 res, fv;

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
