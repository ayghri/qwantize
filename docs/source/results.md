# Results

> Reproduce with: `python bench/full_bench.py`

```{note}
Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B (W: 2560x9728, bfloat16), with activations collected from WikiText-2 (max_seq_len=512, num_samples=2048, X: 244449x9728, bfloat16).
```

- **Weight error**: $\lVert Q(W) - W \rVert_F / \lVert W \rVert_F$
- **Output error**: $\lVert X W_q^T - X W^T \rVert_F / \lVert X W^T \rVert_F$

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
