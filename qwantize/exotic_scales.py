"""Exotic FP8 scale formats for block-scaled quantization.

Extends the NVFP4 (FP8 E4M3) scale grid with two unsigned 8-bit alternatives:

- **UE4M4**: 4 exp + 4 mantissa, bias 7, no sign bit. Trades sign for one extra
  mantissa bit -> denser scale grid near 1, same dynamic range as E4M3 when
  E4M3 magnitudes are reused.
- **UE5M3**: 5 exp + 3 mantissa, bias 15, no sign bit. Wider dynamic range
  than E4M3 with the same mantissa precision.

All codes are treated as finite (no NaN/Inf reserved).
"""

import torch


def _build_unsigned_fp8(exp_bits, mant_bits, device="cpu"):
    """Enumerate every positive value of an unsigned (E,M) 8-bit float.

    Encoding follows IEEE-754 conventions:
      bias = 2^(E-1) - 1
      e == 0 (subnormal):  v = (m / 2^M) * 2^(1 - bias)
      e > 0  (normal):     v = (1 + m / 2^M) * 2^(e - bias)

    All 2^(E+M) codes are treated as finite (no NaN/Inf reserved).

    Args:
        exp_bits: Number of exponent bits.
        mant_bits: Number of mantissa bits. exp_bits + mant_bits must equal 8.
        device: Torch device for the output tensor.

    Returns:
        Tensor of shape (n,) with sorted unique positive values as float32.
    """
    assert exp_bits + mant_bits == 8
    bias = 2 ** (exp_bits - 1) - 1
    n_exp = 2 ** exp_bits
    n_mant = 2 ** mant_bits

    e = torch.arange(n_exp, device=device, dtype=torch.float64).unsqueeze(-1)  # (n_exp, 1)
    m = torch.arange(n_mant, device=device, dtype=torch.float64).unsqueeze(0)  # (1, n_mant)

    sub = (m / n_mant) * (2.0 ** (1 - bias))                  # e == 0 row
    nrm = (1.0 + m / n_mant) * torch.pow(2.0, e - bias)       # e >= 1 rows
    vals = torch.where(e == 0, sub, nrm).reshape(-1)
    pos = vals[vals > 0].unique().sort().values
    return pos.float()


def build_ue4m4_scales(device="cpu"):
    """Return sorted positive UE4M4 values (4-exp, 4-mantissa, unsigned)."""
    return _build_unsigned_fp8(4, 4, device=device)


def build_ue5m3_scales(device="cpu"):
    """Return sorted positive UE5M3 values (5-exp, 3-mantissa, unsigned)."""
    return _build_unsigned_fp8(5, 3, device=device)


def snap_to_table(x, sorted_table):
    """Snap each element of *x* to the nearest value in *sorted_table*.

    Args:
        x: Tensor of positive values.
        sorted_table: 1-D tensor of sorted positive scale values on the same device.

    Returns:
        Tensor with the same shape and dtype as *x* with each entry replaced by
        the closest value in *sorted_table*.
    """
    table = sorted_table.to(x.device).to(x.dtype)
    boundaries = (table[:-1] + table[1:]) * 0.5
    idx = torch.bucketize(x.contiguous(), boundaries)
    return table[idx]
