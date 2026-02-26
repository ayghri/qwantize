# Scale Distance Analysis

This page presents empirical analysis of the distance between naive and optimal scales for both NVFP4 and MXFP4 quantization formats. The results justify using a fixed-window search as a practical alternative to the full bounded search.

> Reproduce with: `python bench/nvfp4_scale_distance.py` and `python bench/mxfp4_scale_distance.py`

```{note}
Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B (W: 2560x9728, bfloat16).
```

## Motivation

The full optimal scale search (Section 5 of [Optimal Scale Search](optimal_scale_search.md)) uses data-dependent bounds that vary per block. On GPU, this causes **warp divergence**: threads within a warp may iterate over different numbers of scale candidates, leaving some threads idle.

If optimal scales always fall within a fixed window of table steps from the naive scale, we can replace the variable-length search with a constant-count loop -- achieving **zero divergence** with no loss in quality.

## NVFP4: FP8 E4M3 Scale Distances

NVFP4 uses FP8 E4M3 as the scale format, giving 126 positive representable values. We convert both naive and optimal scales to their FP8 byte representation, map them to indices in the sorted 126-entry FP8 table, and compute the signed index distance $\delta = \text{idx}\_{\text{optimal}} - \text{idx}\_{\text{naive}}$.

Key property exploited: for positive FP8 E4M3 values, byte ordering equals value ordering, so `searchsorted` on uint8 bytes works directly.

### Distance Distribution

| Metric | Block Size 16 | Block Size 32 |
|:--|:--:|:--:|
| Total blocks | 1,553,920 | 776,960 |
| Blocks changed | 58.7% | 52.8% |
| Index distance range | [-2, +11] | [-2, +9] |
| Mean distance | +1.2 | +0.9 |
| Median distance | +1 | +1 |
| Direction | Mostly upward | Mostly upward |

The optimal scale tends to be **larger** than naive. This makes sense: the naive scale $\text{FP8}(\max/6)$ rounds the continuous scale to FP8, often rounding down. A slightly larger scale can reduce rounding error for the bulk of elements at the cost of marginal clipping on the maximum.

### Fixed-Window Search Quality

| Window | Candidates | BS16 Gap to Optimal | BS32 Gap to Optimal |
|:--:|:--:|:--:|:--:|
| $\pm 1$ | 3 | ~30% remaining gap | ~35% |
| $\pm 3$ | 7 | ~2-4% | ~2-4% |
| $\pm 5$ | 11 | ~0% | ~0% |
| Full (126) | 126 | 0% (reference) | 0% |

**A window of $\pm 5$ FP8 table steps (11 candidates) captures 100% of the optimal improvement.** This is a fixed, data-independent loop count suitable for a zero-divergence GPU kernel.

### Quantization Quality

| Method | Block Size | $\|Q(W)-W\|/\|W\|$ | $\|W_q X - WX\|/\|WX\|$ | Weight Error Reduction | Output Error Reduction |
|:--|:--:|:--:|:--:|:--:|:--:|
| Naive | 16 | 10.05% | -- | -- | -- |
| Optimal | 16 | 8.74% | -- | 13.07% | 12.03% |
| Naive | 32 | 10.42% | -- | -- | -- |
| Optimal | 32 | 9.57% | -- | 8.15% | 7.53% |

## MXFP4: UE8M0 Scale Distances

MXFP4 uses UE8M0 (power-of-2) scales: $s = 2^{e-128}$ for exponent $e \in \{1, \ldots, 254\}$, giving 254 scale values. Since successive scales differ by exactly a factor of 2, the "distance" between naive and optimal is measured in exponent steps.

### Distance Distribution

| Metric | Block Size 16 | Block Size 32 |
|:--|:--:|:--:|
| Total blocks | 1,553,920 | 776,960 |
| Blocks changed | 15.8% | 16.4% |
| Exponent distance range | [0, +1] | [0, +1] |
| Mean distance | +0.158 | +0.164 |

The optimal scale for MXFP4 is **always within 1 step** of the naive scale. This is expected: with power-of-2 scales, consecutive scales differ by a factor of 2, so the naive $s_0 = 2^{\lfloor \log_2(\max|x_i|) - 2 + 127 \rfloor - 128}$ is already very close to optimal. The only scenario where the optimal differs is when most elements in the block are much smaller than the maximum -- making a one-step-larger scale (which doubles the scale) beneficial.

### Fixed-Window Search Quality

| Window | Candidates | BS16 Gap to Optimal | BS32 Gap to Optimal |
|:--:|:--:|:--:|:--:|
| $\pm 1$ | 3 | 0% | 0% |
| Full (254) | 254 | 0% (reference) | 0% |

**A window of $\pm 1$ UE8M0 exponent step (3 candidates) is sufficient to capture the full optimal improvement.** This trivially allows a zero-divergence GPU kernel.

### Quantization Quality

| Method | Block Size | $\|Q(W)-W\|/\|W\|$ |
|:--|:--:|:--:|
| Naive | 16 | 10.34% |
| Optimal | 16 | 10.13% |
| Naive | 32 | 10.76% |
| Optimal | 32 | 10.58% |

The improvement for MXFP4 is smaller than NVFP4 (as expected, since UE8M0 scales are already near-optimal due to the factor-of-2 spacing).

## Summary

| Format | Scale Type | Scale Values | Max Distance | Required Window |
|:--|:--|:--:|:--:|:--:|
| NVFP4 | FP8 E4M3 | 126 | $\pm 5$ steps | $\pm 5$ (11 candidates) |
| MXFP4 | UE8M0 | 254 | +1 step | $\pm 1$ (3 candidates) |

These results enable **fixed-window kernels** that achieve optimal quantization quality with zero warp divergence.
