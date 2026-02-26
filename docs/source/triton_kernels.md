# Triton Kernels

This page describes the Triton GPU kernels that implement both naive and optimal quantization for NVFP4 and MXFP4.
```{note}
These are provided as proof of concept at this point and are far from being optimal.
```

## Architecture

- **Grid**: `(total_blocks,)` where `total_blocks = product of all non-block dimensions`
- **One program per block**: each program loads `BLOCK_SIZE` elements using stride-based access, runs the quantization algorithm, writes back dequantized values + scale
- **Stride-based access**: kernels receive `block_stride` and `element_stride` parameters, enabling quantization along any tensor dimension without requiring contiguous memory layout

## NVFP4 Kernels

### Inline PTX Assembly

Two inline PTX ASM helpers avoid round-trips through FP8 hardware:

#### `fp8_e4m3_snap_asm` -- FP32 to Nearest FP8 E4M3

Snaps a positive float32 to its nearest FP8 E4M3 representable value using only PTX bitwise operations.

**FP32 layout**: sign(1) | exponent(8, bias=127) | mantissa(23)
**FP8 E4M3 layout**: sign(1) | exponent(4, bias=7) | mantissa(3)
**Bias difference**: 127 - 7 = 120

- **Normal path** (exp8 >= 1): Extract FP32 exponent, subtract bias offset of 120, round mantissa to 3 bits with round-to-nearest-even, handle mantissa overflow carry and exponent overflow clamping to 448.0.
- **Subnormal path** (exp8 < 1): FP8 E4M3 subnormal values are $k \cdot 2^{-9}$ for $k = 0, 1, \ldots, 7$. Multiply by 512, round to nearest integer, clamp to [0, 7], multiply back by $2^{-9}$.

#### `fp4_dequant_asm` -- FP4 Quantize-Dequantize

Maps $|x|/s$ to the nearest FP4 codebook value using 7 boundary comparisons (`setp.le.f32`) and a selection chain (`selp.f32`), then reconstructs $\text{dq} = \text{sign}(x) \cdot q \cdot s$.

The `setp`/`selp` pattern avoids branching entirely -- all 7 comparisons execute unconditionally and the `selp` chain narrows to the correct codebook entry.

**Register scoping**: each inline ASM invocation is wrapped in PTX `{ }` braces, creating local register scopes. This prevents name collisions when the function is inlined multiple times in the optimal kernel's search loop.

### Optimal Kernel Structure

```
1. Load block using stride-based offsets
2. Compute naive scale s0 via fp8_e4m3_snap_asm(amax / 6.0)
3. Compute baseline SSE E0 using fp4_dequant_asm
4. Noise check: skip search if sum(x^2) <= E0
5. Lower bound: s_min = max(0, (amax - sqrt(E0)) / 6)
6. Upper bound: sort |x|, cumsum of squares, find k*
7. Search over 126 FP8 scale candidates (constexpr unrolled loop)
   - Fast-fail with clipping error H(s)
   - Full SSE only if H(s) < best_E
8. Final dequant with best scale
```

The `NUM_SCALES` loop is a `tl.constexpr` so Triton unrolls it at compile time. The bounds and fast-fail skip ~95% of iterations.

## MXFP4 Kernels

### Inline PTX Assembly

#### `ue8m0_snap_asm` -- FP32 to Nearest UE8M0

Extracts the FP32 exponent and converts to a UE8M0 power-of-2 scale. Simpler than FP8 E4M3 since UE8M0 has no mantissa bits.

MXFP4 uses the same shared `fp4_dequant_asm` as NVFP4 (standard FP4 codebook $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$), with UE8M0 power-of-2 scales computed by `ue8m0_snap_asm`.

### Torch Reference Implementations

MXFP4 also provides pure-PyTorch implementations (`mxfp4_naive_torch`, `mxfp4_optimal_torch`) that use vectorized tensor operations instead of Triton kernels. These serve as an additional correctness reference and are useful on hardware without Triton support.

## Zero-Divergence Design

Based on the [Scale Distance Analysis](scale_distance.md), both NVFP4 and MXFP4 optimal kernels can use a fixed-window search:

- **NVFP4**: $\pm 5$ FP8 table steps (11 candidates) -- captures 100% of optimal improvement
- **MXFP4**: $\pm 1$ UE8M0 exponent step (3 candidates) -- captures 100% of optimal improvement

This eliminates the `tl.sort` and `tl.cumsum` operations and replaces the full-table loop with a fixed-count loop, achieving zero warp divergence.

## Performance

> Reproduce with: `python bench/full_bench.py`

```{note}
Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B (W: 2560x9728, bfloat16), with activations collected from WikiText-2 (max_seq_len=512, num_samples=2048, X: 244449x9728, bfloat16).
```

- **Weight error**: $\lVert Q(W) - W \rVert_F / \lVert W \rVert_F$
- **Output error**: $\lVert X W_q^T - X W^T \rVert_F / \lVert X W^T \rVert_F$

### NVFP4 (FP8 E4M3 scales)

| Implementation | Block Size | Weight Error | Output Error | Time | Speedup |
|:--|:--:|:--:|:--:|--:|--:|
| Naive (torch) | 16 | 10.05% | 6.89% | 2.8 ms | |
| Naive (Triton) | 16 | 10.05% | 6.89% | 1.9 ms | 1.5x |
| SSE-Optimal (torch) | 16 | 8.74% | 6.04% | 234 ms | |
| SSE-Optimal (Triton) | 16 | 8.74% | 6.04% | 33 ms | **7.0x** |
| H-Optimal (torch) | 16 | 9.35% | **5.31%** | 866 ms | |
| H-Optimal (Triton) | 16 | 9.35% | **5.31%** | 470 ms | 1.8x |
| Naive (torch) | 32 | 10.42% | 7.15% | 2.9 ms | |
| Naive (Triton) | 32 | 10.42% | 7.15% | 1.2 ms | 2.4x |
| SSE-Optimal (torch) | 32 | 9.57% | 6.61% | 179 ms | |
| SSE-Optimal (Triton) | 32 | 9.57% | 6.61% | 18 ms | **10.2x** |
| H-Optimal (torch) | 32 | 10.12% | **5.95%** | 676 ms | |
| H-Optimal (Triton) | 32 | 10.12% | **5.95%** | 236 ms | 2.9x |

### MXFP4 (UE8M0 power-of-2 scales)

| Implementation | Block Size | Weight Error | Output Error | Time | Speedup |
|:--|:--:|:--:|:--:|--:|--:|
| Naive (torch) | 16 | 11.77% | 8.48% | 3.0 ms | |
| Naive (Triton) | 16 | 11.77% | 8.48% | 1.8 ms | 1.7x |
| SSE-Optimal (torch) | 16 | 11.02% | 7.67% | 86 ms | |
| SSE-Optimal (Triton) | 16 | 11.02% | 7.67% | 2.6 ms | **33.6x** |
| H-Optimal (torch) | 16 | 11.10% | **7.62%** | 545 ms | |
| Naive (torch) | 32 | 11.75% | 8.37% | 3.0 ms | |
| Naive (Triton) | 32 | 11.75% | 8.37% | 1.2 ms | 2.6x |
| SSE-Optimal (torch) | 32 | 11.32% | 7.91% | 74 ms | |
| SSE-Optimal (Triton) | 32 | 11.32% | 7.91% | 1.6 ms | **45.7x** |
| H-Optimal (torch) | 32 | 11.42% | **7.80%** | 361 ms | |

See [Hessian-Aware Scale Search](hessian_scale_search.md) for the mathematical framework
and analysis of why NVFP4 benefits much more from Hessian-awareness than MXFP4.

### Correctness notes

**NVFP4 Triton vs Python reference**: Scale computation matches exactly (0 disagreements). The max element-level abs diff (~5e-2) comes from FP4 decision-boundary tie-breaking: when $|x|/s$ lands exactly on a codebook boundary (e.g. 0.75, 1.75, 3.5), the PTX ``div.full.f32`` and PyTorch ``/`` produce results that round to different FP4 values. This affects ~0.01% of elements and does not affect the error metrics.

**MXFP4 Triton vs Python reference**: Naive kernel matches exactly (0.00 max abs diff). For the optimal kernel, in rare tie-breaking cases (1 in ~800k blocks), ``tl.sum`` tree reduction and PyTorch sequential ``.sum()`` accumulate float32 rounding differently, causing one to pick ``s0`` and the other ``2*s0`` when their SSEs are identical. This produces a max abs diff of one scale step but does not affect the error metrics.
