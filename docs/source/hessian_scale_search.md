# Hessian-Aware Optimal Scale Search

## 1. Motivation

The [standard optimal scale search](optimal_scale_search.md) minimizes the per-block
Sum of Squared Errors (SSE):

$$E(s) = \sum_{i=1}^K \left(x_i - s \cdot Q\!\left(\frac{x_i}{s}\right)\right)^2 = \|x - s\,Q(x/s)\|^2$$

This treats each weight element independently. In reality, weight errors interact through
the input activations. Consider the output error for column-block $j$ of the weight matrix:

$$\|X_j(x - s\,q)\|^2 = (x - s\,q)^T\underbrace{(X_j^T X_j)}_{H_j}(x - s\,q)$$

where $X_j \in \mathbb{R}^{T \times K}$ is the submatrix of activations corresponding to
block $j$, and $H_j = X_j^T X_j \in \mathbb{R}^{K \times K}$ is the **block Hessian**.

Minimizing $E_H(s) = (x - s\,q)^T H\,(x - s\,q)$ directly minimizes each block's
contribution to the total output error $\|W_q X^T - W X^T\|_F^2$, rather than the weight
error $\|W_q - W\|_F^2$.

## 2. Block Hessian Structure

For a weight matrix $W \in \mathbb{R}^{M \times K}$ and activations $X \in \mathbb{R}^{T \times K}$:

- Partition the $K$ columns into blocks of size $b$ (16 or 32), giving $J = K/b$ column-blocks.
- The block Hessian for column-block $j$ is:

$$H_j = X_j^T X_j \in \mathbb{R}^{b \times b}, \quad X_j = X[:, jb:(j+1)b]$$

- There are only $J$ distinct Hessians, shared across all $M$ rows of $W$.
- Storage: $J \cdot b^2$ floats. For $K = 9728$, $b = 32$: $304 \cdot 1024 = 311\text{K}$ floats (1.2 MB).

**Precomputation** is done in batches to avoid materializing the full $X$ in float32:

$$H_j = \sum_{t=0}^{\lceil T/B \rceil - 1} X_j^{(t)\,T} X_j^{(t)}$$

where $X_j^{(t)}$ is a batch of $B$ rows (default $B = 8192$).

## 3. Hessian-Weighted Error

For a block $x \in \mathbb{R}^K$ with scale $s$ and quantization $q(s) = Q(x/s)$:

$$E_H(s) = r(s)^T\, H\, r(s), \quad r(s) = x - s \cdot q(s)$$

This is a quadratic form in the residual $r$, weighted by the Hessian $H$. Elements with
larger diagonal entries in $H$ (i.e., activations with higher energy) contribute more to the error.

## 4. Adapting the Bounded Search

The [SSE-based bounds](optimal_scale_search.md) carry over as **necessary conditions** for pruning:

- **Lower bound**: $s_{\min} = \max(0,\; (x_{\max} - \sqrt{E_0}) / q_{\max})$
- **Upper bound**: $s_{\max} = y_{k^*+1} / d_0$ (dead-zone bound)
- **Fast-fail**: Clipping error $\sum_i \max(|x_i| - q_{\max} s, 0)^2 < E_0$

These are computed from the baseline SSE $E_0$ and used to prune the candidate set.
For each surviving candidate $s$, we evaluate the full Hessian error $E_H(s)$ and select
the scale minimizing it.

**Algorithm:**

1. Compute baseline scale $s_0$ and its errors $E_0^{\text{SSE}}$, $E_0^H$.
2. Compute SSE-based bounds $[s_{\min}, s_{\max}]$.
3. For each representable scale $s \in [s_{\min}, s_{\max}]$:
   - If clipping error $> E_0^{\text{SSE}}$: skip (fast-fail).
   - Compute $r = x - s \cdot Q(x/s)$.
   - Compute $E_H(s) = r^T H\, r$.
   - If $E_H(s) < E_H^{\text{best}}$: update best.
4. Return best scale.

## 5. Efficient Computation

**Per-candidate evaluation** requires a matrix-vector product $Hr$ of cost $O(K^2)$ per block,
versus $O(K)$ for SSE. For $K = 32$, this is a 32$\times$ increase per candidate.

**Vectorization**: Since $H$ depends only on the column-block index $j$ (not the row $m$),
the Triton kernel loads $H_j$ once into registers and reuses it across all candidates:

```
H_block = load(H_ptr + j * K * K)           # (K, K) in registers
for each candidate s:
    r = x - dequant(x, s)                   # (K,)
    Hr = sum(H_block * r[None, :], axis=1)  # mat-vec: O(K^2)
    E_H = sum(r * Hr)                       # dot: O(K)
```

The bounds typically reduce the candidate set to 4--8 scales (same as SSE search), so the
total per-block cost is roughly $8 \cdot K^2$ FLOPs.

## 6. Closed-Form Optimal Continuous Scale

For fixed quantization levels $q$, the Hessian-weighted error is quadratic in $s$:

$$E_H(s) = x^T H x - 2s\,(x^T H q) + s^2\,(q^T H q)$$

The minimum is at:

$$s^* = \frac{x^T H\, q}{q^T H\, q}$$

This closed form is used in the ADMM algorithm for the scale update step.
The Hessian-aware scale search instead evaluates all representable scales in $[s_{\min}, s_{\max}]$,
which is more robust since $q(s) = Q(x/s)$ changes with $s$.

## 7. Results

> Reproduce with: `python bench/full_bench.py`

```{note}
Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B (W: 2560$\times$9728, bfloat16), with activations collected from WikiText-2 (max_seq_len=512, num_samples=2048, X: 244449$\times$9728, bfloat16).
```

### NVFP4 (FP8 E4M3 scales)

| Method | Block Size | Weight Error | Output Error | torch | Triton | Speedup |
|:--|:--:|:--:|:--:|--:|--:|--:|
| Naive | 16 | 10.05% | 6.89% | 2.8 ms | 1.9 ms | 1.5x |
| SSE-Optimal | 16 | 8.74% | 6.04% | 234 ms | 33 ms | **7.0x** |
| H-Optimal | 16 | 9.35% | **5.31%** | 866 ms | 470 ms | 1.8x |
| Naive | 32 | 10.42% | 7.15% | 2.9 ms | 1.2 ms | 2.4x |
| SSE-Optimal | 32 | 9.57% | 6.61% | 179 ms | 18 ms | **10.2x** |
| H-Optimal | 32 | 10.12% | **5.95%** | 676 ms | 236 ms | 2.9x |

**H-Optimal vs SSE-Optimal** (output error reduction):
- Block size 16: **+12.0%** further reduction (6.04% $\to$ 5.31%)
- Block size 32: **+10.0%** further reduction (6.61% $\to$ 5.95%)

**H-Optimal vs Naive** (total output error reduction):
- Block size 16: **-22.9%** (6.89% $\to$ 5.31%)
- Block size 32: **-16.7%** (7.15% $\to$ 5.95%)

Weight error increases slightly (by 0.6--0.5pp) because H-Optimal optimizes for output
error rather than weight error. This is the correct trade-off: a model's quality depends
on output error, not weight error.

### MXFP4 (UE8M0 power-of-2 scales)

| Method | Block Size | Weight Error | Output Error | torch | Triton | Speedup |
|:--|:--:|:--:|:--:|--:|--:|--:|
| Naive | 16 | 11.77% | 8.48% | 3.0 ms | 1.8 ms | 1.7x |
| SSE-Optimal | 16 | 11.02% | 7.67% | 86 ms | 2.6 ms | **33.6x** |
| H-Optimal | 16 | 11.10% | **7.62%** | 545 ms | -- | |
| Naive | 32 | 11.75% | 8.37% | 3.0 ms | 1.2 ms | 2.6x |
| SSE-Optimal | 32 | 11.32% | 7.91% | 74 ms | 1.6 ms | **45.7x** |
| H-Optimal | 32 | 11.42% | **7.80%** | 361 ms | -- | |

**H-Optimal vs SSE-Optimal** (output error reduction):
- Block size 16: **+0.7%** further reduction (7.67% $\to$ 7.62%)
- Block size 32: **+1.4%** further reduction (7.91% $\to$ 7.80%)

The improvement is much smaller for MXFP4 because UE8M0 scales are powers of 2 --
consecutive scales differ by a factor of 2, leaving only 1--2 candidates near the optimum.
With so few choices, the Hessian criterion rarely selects a different scale than SSE.

### Why NVFP4 benefits much more

FP8 E4M3 has 126 finely-spaced positive scale values with non-uniform spacing.
The SSE-optimal and H-optimal scales can differ by several FP8 steps, because the
Hessian re-weights the importance of each element. With UE8M0's coarse power-of-2
grid, this re-weighting almost always lands on the same scale.
