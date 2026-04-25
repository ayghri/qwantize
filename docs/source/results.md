# Results

> Reproduce with: `python bench/full_bench.py`

```{note}
Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B (W: 2560x9728, bfloat16), with activations collected from WikiText-2 (max_seq_len=512, num_samples=2048, X: 244449x9728, bfloat16).
```

- **Weight error**: $\lVert Q(W) - W \rVert_F / \lVert W \rVert_F$
- **Output error**: $\lVert X W_q^T - X W^T \rVert_F / \lVert X W^T \rVert_F$

## INT8 (FP8 E4M3 scales)

Symmetric INT8 quantization ([-127, 127]) with per-block amax stored in FP8 E4M3.
The effective scale is ``amax_fp8 / 127``, keeping the stored value within FP8 range
while the division by 127 is performed in float32.

| Implementation | Block Size | Weight Error | Output Error | Time |
|:--|:--:|:--:|:--:|--:|
| Naive (torch) | 32 | 1.01% | 0.79% | 1.7 ms |
| SSE-Optimal (torch) | 32 | 0.57% | 0.40% | 236 ms |
| H-Optimal (torch) | 32 | 0.60% | **0.37%** | 1.2 s |
| Naive (torch) | 64 | 0.93% | 0.72% | 1.7 ms |
| SSE-Optimal (torch) | 64 | 0.64% | 0.45% | 204 ms |
| H-Optimal (torch) | 64 | 0.66% | **0.42%** | 1.5 s |
| Naive (torch) | 128 | 0.88% | 0.68% | 1.6 ms |
| SSE-Optimal (torch) | 128 | 0.71% | 0.49% | 173 ms |
| H-Optimal (torch) | 128 | 0.73% | **0.48%** | 2.8 s |
| Naive (torch) | 256 | 0.87% | 0.66% | 1.6 ms |
| SSE-Optimal (torch) | 256 | 0.77% | 0.54% | 165 ms |
| H-Optimal (torch) | 256 | 0.79% | **0.52%** | 4.9 s |

**SSE-Optimal vs Naive** (output error reduction):
- Block size 32: **-49.9%** (0.79% $\to$ 0.40%)
- Block size 64: **-38.1%** (0.72% $\to$ 0.45%)
- Block size 128: **-27.6%** (0.68% $\to$ 0.49%)
- Block size 256: **-18.6%** (0.66% $\to$ 0.54%)

**H-Optimal vs SSE-Optimal** (further output error reduction):
- Block size 32: **+7.0%** further reduction (0.40% $\to$ 0.37%)
- Block size 64: **+4.8%** further reduction (0.45% $\to$ 0.42%)
- Block size 128: **+3.3%** further reduction (0.49% $\to$ 0.48%)
- Block size 256: **+2.4%** further reduction (0.54% $\to$ 0.52%)

**H-Optimal vs Naive** (total output error reduction):
- Block size 32: **-53.4%** (0.79% $\to$ 0.37%)
- Block size 64: **-41.1%** (0.72% $\to$ 0.42%)
- Block size 128: **-29.9%** (0.68% $\to$ 0.48%)
- Block size 256: **-20.6%** (0.66% $\to$ 0.52%)

The massive naive-to-optimal improvement (up to 50%) is driven by the FP8 E4M3
scale grid: with only 126 discrete scale values, the naive ``amax`` snap often
lands on a scale that is significantly suboptimal, and the bounded search finds
a much better candidate. This is analogous to NVFP4's scale search, but the effect
is even stronger because INT8's 127 quantization levels amplify scale misalignment
(a scale error of $\delta$ causes $127\delta$ in the worst case, vs $6\delta$ for FP4).

H-Optimal provides a further 2--7% reduction over SSE-Optimal by prioritizing
output-sensitive weights.

## NVFP4 (FP8 E4M3 scales)

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

**H-Optimal vs SSE-Optimal** (output error reduction):
- Block size 16: **+12.0%** further reduction (6.04% $\to$ 5.31%)
- Block size 32: **+10.0%** further reduction (6.61% $\to$ 5.95%)

**H-Optimal vs Naive** (total output error reduction):
- Block size 16: **-22.9%** (6.89% $\to$ 5.31%)
- Block size 32: **-16.7%** (7.15% $\to$ 5.95%)

Weight error increases slightly (by 0.6--0.5pp) because H-Optimal optimizes for output
error rather than weight error. This is the correct trade-off: a model's quality depends
on output error, not weight error.

## MXFP4 (UE8M0 power-of-2 scales)

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

**H-Optimal vs SSE-Optimal** (output error reduction):
- Block size 16: **+0.7%** further reduction (7.67% $\to$ 7.62%)
- Block size 32: **+1.4%** further reduction (7.91% $\to$ 7.80%)

The improvement is much smaller for MXFP4 because UE8M0 scales are powers of 2 --
consecutive scales differ by a factor of 2, leaving only 1--2 candidates near the optimum.
With so few choices, the Hessian criterion rarely selects a different scale than SSE.

## Why NVFP4 benefits much more from Hessian-awareness

FP8 E4M3 has 126 finely-spaced positive scale values with non-uniform spacing.
The SSE-optimal and H-optimal scales can differ by several FP8 steps, because the
Hessian re-weights the importance of each element. With UE8M0's coarse power-of-2
grid, this re-weighting almost always lands on the same scale.

## Correctness notes

**NVFP4 Triton vs Python reference**: Scale computation matches exactly (0 disagreements). The max element-level abs diff (~5e-2) comes from FP4 decision-boundary tie-breaking: when $|x|/s$ lands exactly on a codebook boundary (e.g. 0.75, 1.75, 3.5), the PTX ``div.full.f32`` and PyTorch ``/`` produce results that round to different FP4 values. This affects ~0.01% of elements and does not affect the error metrics.

**MXFP4 Triton vs Python reference**: Naive kernel matches exactly (0.00 max abs diff). For the optimal kernel, in rare tie-breaking cases (1 in ~800k blocks), ``tl.sum`` tree reduction and PyTorch sequential ``.sum()`` accumulate float32 rounding differently, causing one to pick ``s0`` and the other ``2*s0`` when their SSEs are identical. This produces a max abs diff of one scale step but does not affect the error metrics.

## GPTQ Quantization

> Reproduce with: `/misc/envs/quant/bin/python experiments/quant_gptq_strided.py`

GPTQ (Frantar et al., 2022) applies Optimal Brain Surgeon error compensation
to sequential column-block quantization. After quantizing each block of columns,
the quantization error is propagated to remaining columns using the inverse Hessian,
minimizing the total output error.

Our implementation uses `torch.as_strided` for zero-copy sub-matrix views during
error propagation. The GPTQ block size equals the quantization block size, so each
column block is quantized and its error immediately compensated across all remaining
columns:

```python
# After quantizing columns [cs:ce], propagate error via as_strided views:
h_cross = torch.as_strided(H_inv, (bs, rem), (K, 1), offset=cs*K + ce)
w_rem   = torch.as_strided(W,     (M, rem),  (K, 1), offset=ce)
w_rem.sub_(err @ h_cross)  # in-place, zero-copy
```

Three modes are compared: **baseline** (no GPTQ), **sequential** GPTQ (natural
column order), and **ordered** GPTQ (column blocks sorted by descending
Hessian-weighted quantization loss, so the highest-error blocks are quantized
first and their error is compensated across the most remaining columns).

### Block Size 16

| Format | Approach | GPTQ | Weight Error | Output Error | Time |
|:--|:--|:--:|:--:|:--:|--:|
| NVFP4 | Naive | — | 10.05% | 6.89% | 95ms |
| NVFP4 | GPTQ+Naive | Seq | 12.58% | 5.53% | 402ms |
| NVFP4 | GPTQ-Ord+Naive | Ord | 13.18% | 5.18% | 490ms |
| NVFP4 | Optimal | — | 8.74% | 6.04% | 7.4s |
| NVFP4 | GPTQ+Optimal | Seq | 10.94% | 4.82% | 7.7s |
| NVFP4 | GPTQ-Ord+Optimal | Ord | 11.45% | 4.52% | 15.1s |
| NVFP4 | H-Optimal | — | 9.37% | 5.34% | 7.7s |
| NVFP4 | GPTQ+H-Optimal | Seq | 11.14% | 4.37% | 8.0s |
| NVFP4 | GPTQ-Ord+H-Optimal | Ord | 11.53% | **4.21%** | 15.9s |
| MXFP4 | Naive | — | 11.77% | 8.48% | 102ms |
| MXFP4 | GPTQ+Naive | Seq | 14.61% | 6.67% | 400ms |
| MXFP4 | GPTQ-Ord+Naive | Ord | 15.27% | 6.20% | 517ms |
| MXFP4 | Optimal | — | 11.02% | 7.67% | 6.7s |
| MXFP4 | GPTQ+Optimal | Seq | 13.79% | 6.13% | 7.0s |
| MXFP4 | GPTQ-Ord+Optimal | Ord | 14.43% | 5.72% | 13.8s |
| MXFP4 | H-Optimal | — | 11.10% | 7.62% | 6.9s |
| MXFP4 | GPTQ+H-Optimal | Seq | 13.82% | 6.10% | 7.1s |
| MXFP4 | GPTQ-Ord+H-Optimal | Ord | 14.45% | **5.71%** | 14.0s |
| NVINT4 | Naive | — | 9.46% | 6.55% | 65ms |
| NVINT4 | GPTQ+Naive | Seq | 11.84% | 5.23% | 376ms |
| NVINT4 | GPTQ-Ord+Naive | Ord | 12.37% | 4.89% | 414ms |
| NVINT4 | Optimal | — | 9.20% | 6.40% | 5.6s |
| NVINT4 | GPTQ+Optimal | Seq | 11.54% | 5.12% | 5.9s |
| NVINT4 | GPTQ-Ord+Optimal | Ord | 12.06% | 4.76% | 11.5s |
| NVINT4 | H-Optimal | — | 9.60% | 6.04% | 5.9s |
| NVINT4 | GPTQ+H-Optimal | Seq | 11.73% | 4.88% | 6.1s |
| NVINT4 | GPTQ-Ord+H-Optimal | Ord | 12.20% | **4.65%** | 12.0s |

### Block Size 32

| Format | Approach | GPTQ | Weight Error | Output Error | Time |
|:--|:--|:--:|:--:|:--:|--:|
| NVFP4 | Naive | — | 10.42% | 7.15% | 37ms |
| NVFP4 | GPTQ+Naive | Seq | 13.04% | 5.74% | 272ms |
| NVFP4 | GPTQ-Ord+Naive | Ord | 13.53% | 5.43% | 320ms |
| NVFP4 | Optimal | — | 9.57% | 6.61% | 3.6s |
| NVFP4 | GPTQ+Optimal | Seq | 11.98% | 5.29% | 3.8s |
| NVFP4 | GPTQ-Ord+Optimal | Ord | 12.42% | 5.01% | 7.3s |
| NVFP4 | H-Optimal | — | 10.16% | 6.02% | 3.7s |
| NVFP4 | GPTQ+H-Optimal | Seq | 12.21% | 4.91% | 4.0s |
| NVFP4 | GPTQ-Ord+H-Optimal | Ord | 12.57% | **4.75%** | 7.7s |
| MXFP4 | Naive | — | 11.75% | 8.37% | 47ms |
| MXFP4 | GPTQ+Naive | Seq | 14.62% | 6.62% | 273ms |
| MXFP4 | GPTQ-Ord+Naive | Ord | 15.14% | 6.24% | 335ms |
| MXFP4 | Optimal | — | 11.32% | 7.91% | 3.4s |
| MXFP4 | GPTQ+Optimal | Seq | 14.16% | 6.32% | 3.5s |
| MXFP4 | GPTQ-Ord+Optimal | Ord | 14.66% | 5.95% | 6.8s |
| MXFP4 | H-Optimal | — | 11.42% | 7.80% | 3.4s |
| MXFP4 | GPTQ+H-Optimal | Seq | 14.19% | 6.25% | 3.6s |
| MXFP4 | GPTQ-Ord+H-Optimal | Ord | 14.68% | **5.92%** | 7.0s |
| NVINT4 | Naive | — | 10.36% | 7.18% | 24ms |
| NVINT4 | GPTQ+Naive | Seq | 13.00% | 5.72% | 248ms |
| NVINT4 | GPTQ-Ord+Naive | Ord | 13.45% | 5.42% | 282ms |
| NVINT4 | Optimal | — | 10.13% | 7.10% | 2.8s |
| NVINT4 | GPTQ+Optimal | Seq | 12.71% | 5.65% | 3.0s |
| NVINT4 | GPTQ-Ord+Optimal | Ord | 13.14% | 5.33% | 5.8s |
| NVINT4 | H-Optimal | — | 10.59% | 6.92% | 2.9s |
| NVINT4 | GPTQ+H-Optimal | Seq | 13.12% | 5.57% | 3.1s |
| NVINT4 | GPTQ-Ord+H-Optimal | Ord | 13.54% | **5.34%** | 6.0s |

### Ordered vs Sequential GPTQ

Additional output error reduction from reordering (pp over sequential):

| Format | Approach | BS=16 | BS=32 |
|:--|:--|:--:|:--:|
| NVFP4 | Naive | **-0.34pp** | -0.30pp |
| NVFP4 | Optimal | **-0.31pp** | -0.28pp |
| NVFP4 | H-Optimal | -0.16pp | -0.15pp |
| MXFP4 | Naive | **-0.47pp** | **-0.38pp** |
| MXFP4 | Optimal | **-0.41pp** | **-0.37pp** |
| MXFP4 | H-Optimal | **-0.39pp** | -0.33pp |
| NVINT4 | Naive | **-0.34pp** | -0.31pp |
| NVINT4 | Optimal | **-0.36pp** | -0.32pp |
| NVINT4 | H-Optimal | -0.24pp | -0.23pp |

Ordered GPTQ (quantizing highest-loss blocks first) consistently outperforms
sequential GPTQ by 0.15--0.47pp. The gain is largest for MXFP4 (coarser scales
create bigger per-block errors to redistribute) and for naive/optimal approaches
(H-Optimal already concentrates error where it matters least, leaving less room
for reordering to help). Weight error increases slightly more (~0.4--0.6pp over
sequential) as a natural consequence of the stronger output-error optimization.

## Exotic Scales

> Reproduce with:
> - `/misc/envs/quant/bin/python experiments/quant_exotic_scales.py` (no GPTQ)
> - `/misc/envs/quant/bin/python experiments/quant_gptq_exotic_scales.py` (with GPTQ-Seq, GPTQ-Ord)

NVFP4 stores per-block scales in **FP8 E4M3** (signed, 1+4+3 bits, 126 positive
values). Scales are always non-negative, so the sign bit is wasted. We try
two unsigned 8-bit alternatives that re-purpose the sign bit:

- **UE4M4** -- 4-exp, 4-mantissa, bias 7. Trades the sign for one extra
  mantissa bit. Same dynamic range as E4M3 (max $\approx$ 496 vs 448), but
  **2x denser** scale grid (255 distinct positive values).
- **UE5M3** -- 5-exp, 3-mantissa, bias 15. Same mantissa precision as E4M3
  but **much wider** dynamic range (max $\approx$ 122880). Also 255 positive
  values.

All codes are treated as finite (no NaN/Inf reserved). The FP4 codebook
$\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ is unchanged; only the per-block scale
representation differs. Each table below crosses
{Naive, SSE-Optimal, H-Optimal} with {no-GPTQ, GPTQ-Seq, GPTQ-Ord} for
each scale grid.

### Block Size 16

| Scale | Approach | GPTQ | Weight Error | Output Error | Time |
|:--|:--|:--:|:--:|:--:|--:|
| E4M3 | Naive | — | 10.05% | 6.89% | 87ms |
| E4M3 | GPTQ+Naive | Seq | 12.58% | 5.53% | 402ms |
| E4M3 | GPTQ-Ord+Naive | Ord | 13.18% | 5.18% | 492ms |
| E4M3 | Optimal | — | 8.74% | 6.04% | 7.4s |
| E4M3 | GPTQ+Optimal | Seq | 10.94% | 4.82% | 7.5s |
| E4M3 | GPTQ-Ord+Optimal | Ord | 11.45% | 4.52% | 14.8s |
| E4M3 | H-Optimal | — | 9.37% | 5.34% | 7.6s |
| E4M3 | GPTQ+H-Optimal | Seq | 11.14% | 4.37% | 7.9s |
| E4M3 | GPTQ-Ord+H-Optimal | Ord | 11.53% | 4.21% | 15.5s |
| UE4M4 | Naive | — | 9.54% | 6.55% | 84ms |
| UE4M4 | GPTQ+Naive | Seq | 11.97% | 5.25% | 394ms |
| UE4M4 | GPTQ-Ord+Naive | Ord | 12.54% | 4.93% | 505ms |
| UE4M4 | Optimal | — | 8.19% | 5.66% | 14.1s |
| UE4M4 | GPTQ+Optimal | Seq | 10.26% | 4.52% | 14.5s |
| UE4M4 | GPTQ-Ord+Optimal | Ord | 10.75% | 4.23% | 28.3s |
| UE4M4 | H-Optimal | — | 8.95% | 4.97% | 14.7s |
| UE4M4 | GPTQ+H-Optimal | Seq | 10.58% | 4.08% | 15.1s |
| UE4M4 | GPTQ-Ord+H-Optimal | Ord | 10.94% | **3.94%** | 29.9s |
| UE5M3 | Naive | — | 9.47% | 6.51% | 85ms |
| UE5M3 | GPTQ+Naive | Seq | 11.89% | 5.22% | 393ms |
| UE5M3 | GPTQ-Ord+Naive | Ord | 12.46% | 4.89% | 504ms |
| UE5M3 | Optimal | — | 8.13% | 5.63% | 12.5s |
| UE5M3 | GPTQ+Optimal | Seq | 10.19% | 4.49% | 12.7s |
| UE5M3 | GPTQ-Ord+Optimal | Ord | 10.67% | 4.21% | 25.1s |
| UE5M3 | H-Optimal | — | 8.92% | 4.99% | 12.8s |
| UE5M3 | GPTQ+H-Optimal | Seq | 10.56% | 4.09% | 13.2s |
| UE5M3 | GPTQ-Ord+H-Optimal | Ord | 10.92% | **3.95%** | 25.9s |

### Block Size 32

| Scale | Approach | GPTQ | Weight Error | Output Error | Time |
|:--|:--|:--:|:--:|:--:|--:|
| E4M3 | Naive | — | 10.42% | 7.15% | 37ms |
| E4M3 | GPTQ+Naive | Seq | 13.04% | 5.74% | 271ms |
| E4M3 | GPTQ-Ord+Naive | Ord | 13.53% | 5.43% | 318ms |
| E4M3 | Optimal | — | 9.57% | 6.61% | 3.5s |
| E4M3 | GPTQ+Optimal | Seq | 11.98% | 5.29% | 3.7s |
| E4M3 | GPTQ-Ord+Optimal | Ord | 12.42% | 5.01% | 7.2s |
| E4M3 | H-Optimal | — | 10.16% | 6.02% | 3.7s |
| E4M3 | GPTQ+H-Optimal | Seq | 12.21% | 4.91% | 3.9s |
| E4M3 | GPTQ-Ord+H-Optimal | Ord | 12.57% | 4.75% | 7.5s |
| UE4M4 | Naive | — | 10.18% | 6.99% | 42ms |
| UE4M4 | GPTQ+Naive | Seq | 12.76% | 5.61% | 271ms |
| UE4M4 | GPTQ-Ord+Naive | Ord | 13.24% | 5.31% | 326ms |
| UE4M4 | Optimal | — | 9.16% | 6.32% | 6.8s |
| UE4M4 | GPTQ+Optimal | Seq | 11.47% | 5.06% | 7.0s |
| UE4M4 | GPTQ-Ord+Optimal | Ord | 11.88% | 4.79% | 13.8s |
| UE4M4 | H-Optimal | — | 9.90% | 5.73% | 7.1s |
| UE4M4 | GPTQ+H-Optimal | Seq | 11.85% | 4.70% | 7.3s |
| UE4M4 | GPTQ-Ord+H-Optimal | Ord | 12.19% | **4.56%** | 14.4s |
| UE5M3 | Naive | — | 10.16% | 6.98% | 42ms |
| UE5M3 | GPTQ+Naive | Seq | 12.74% | 5.61% | 271ms |
| UE5M3 | GPTQ-Ord+Naive | Ord | 13.22% | 5.31% | 327ms |
| UE5M3 | Optimal | — | 9.14% | 6.31% | 5.9s |
| UE5M3 | GPTQ+Optimal | Seq | 11.44% | 5.06% | 6.2s |
| UE5M3 | GPTQ-Ord+Optimal | Ord | 11.86% | 4.78% | 12.1s |
| UE5M3 | H-Optimal | — | 9.89% | 5.75% | 6.1s |
| UE5M3 | GPTQ+H-Optimal | Seq | 11.86% | 4.71% | 6.3s |
| UE5M3 | GPTQ-Ord+H-Optimal | Ord | 12.20% | **4.58%** | 12.3s |

### Best output error per scale

| Scale | BS=16 best | BS=32 best |
|:--|:--:|:--:|
| E4M3 | 4.21% | 4.75% |
| UE4M4 | **3.94%** | **4.56%** |
| UE5M3 | 3.95% | 4.58% |

(All bests are achieved by GPTQ-Ord+H-Optimal.)

### Output error reduction vs E4M3 (same approach + mode)

| Approach + mode | BS=16: UE4M4 | BS=16: UE5M3 | BS=32: UE4M4 | BS=32: UE5M3 |
|:--|:--:|:--:|:--:|:--:|
| Naive (no GPTQ) | -0.34pp | -0.38pp | -0.16pp | -0.17pp |
| Optimal (no GPTQ) | -0.38pp | -0.41pp | -0.29pp | -0.30pp |
| H-Optimal (no GPTQ) | -0.37pp | -0.35pp | -0.29pp | -0.27pp |
| GPTQ+Naive | -0.28pp | -0.31pp | -0.13pp | -0.13pp |
| GPTQ+Optimal | -0.30pp | -0.33pp | -0.23pp | -0.23pp |
| GPTQ+H-Optimal | -0.29pp | -0.28pp | -0.21pp | -0.20pp |
| GPTQ-Ord+Naive | -0.25pp | -0.29pp | -0.12pp | -0.12pp |
| GPTQ-Ord+Optimal | -0.29pp | -0.31pp | -0.22pp | -0.23pp |
| GPTQ-Ord+H-Optimal | **-0.27pp** | **-0.26pp** | **-0.19pp** | **-0.17pp** |

Both unsigned formats beat E4M3 across every approach × mode × block size.
The relative gain shrinks somewhat once GPTQ is layered on (GPTQ already
compensates for some of the per-block scale-snapping loss), but the absolute
output error keeps falling -- the **best result of every scale grid is
GPTQ-Ord+H-Optimal**, and UE4M4/UE5M3 still beat E4M3 there by 0.17--0.27pp.

UE4M4 and UE5M3 perform almost identically (within 0.01--0.03pp) across the
full grid, even though UE5M3 has $\sim$250x more dynamic range. Weight
magnitudes in this layer fall well within E4M3's range, so extra range is
wasted -- what matters is **grid density near the optimal scale**, and both
formats double the density relative to E4M3.

Caveat: standard FP8 E4M3 hardware support exists on Hopper/Ada; UE4M4 and
UE5M3 do not have hardware encoders, so naive-mode quantization is slower
in production (the snap requires a table lookup rather than a hardware
cast). Optimal/H-Optimal modes are unaffected since they iterate over the
scale table either way; the ~2x slowdown there is purely from doubling the
candidate count (255 vs 126).
