# Custom Codebook Optimization

This page derives an algorithm for learning an optimal 4-bit codebook from data, replacing the fixed FP4 E2M1 values with a custom set of 16 quantization levels that minimize total quantization error.

## 1. Problem Statement

Let $W \in \mathbb{R}^{M \times K}$ be a weight matrix, partitioned along the $K$ dimension into blocks of size $b$.
For row $m$ and column-block $j$, denote the block as $x \in \mathbb{R}^b$ where $x_i = W_{m, jb+i}$.

We seek a **codebook** $\mathcal{C} = \{c_0, c_1, \ldots, c_{15}\}$ of 16 values (4 bits), per-block scales $s_j \in \mathcal{S}$, and assignments $k_{j,i} \in \{0, \ldots, 15\}$ that minimize the total Sum of Squared Errors:

$$E(\mathcal{C}, \{s_j\}, \{k_{j,i}\}) = \sum_{j=1}^{MK/b} \sum_{i=1}^{b} \left(x_{j,i} - s_j \cdot c_{k_{j,i}}\right)^2$$

This is a joint optimization over:
- The 16 codebook values $\mathcal{C}$
- The per-block scales $\{s_j\}$
- The per-element code assignments $\{k_{j,i}\}$

The fixed FP4 E2M1 codebook $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ is one feasible solution.
We aim to find a codebook that achieves lower error for a given weight matrix.

## 2. Sign Reduction

Weights are distributed symmetrically around zero, so we use a **sign bit** and restrict
the codebook to non-negative values:

$$\mathcal{C} = \mathcal{C}^+ \cup (-\mathcal{C}^+), \quad \mathcal{C}^+ = \{0, c_1, c_2, \ldots, c_7\}, \quad 0 < c_1 < c_2 < \cdots < c_7$$

This gives 16 codes: $\{+0, -0, \pm c_1, \ldots, \pm c_7\}$.

The quantization of $x_i$ becomes:
$$\hat{x}_i = \text{sign}(x_i) \cdot s_j \cdot Q^+(|x_i|/s_j)$$

where $Q^+(\cdot)$ maps to the nearest value in $\mathcal{C}^+$. The total error decomposes as:

$$E = \sum_{j,i} \left(|x_{j,i}| - s_j \cdot Q^+\!\left(\frac{|x_{j,i}|}{s_j}\right)\right)^2$$

**We only need to find 7 positive codebook values** (plus zero). The sign is handled separately.

## 3. Scale Normalization

Each block has a per-block scale $s_j$, so the quantization operates on the **normalized
magnitudes** $y_{j,i} = |x_{j,i}|/s_j$. Substituting into the error:

$$E = \sum_j s_j^2 \sum_{i=1}^{b} \left(y_{j,i} - Q^+(y_{j,i})\right)^2$$

The codebook $\mathcal{C}^+$ determines the quantizer $Q^+$, and $y_{j,i} \in [0, c_7]$
(values above $c_7$ clip). The factor $s_j^2$ weights blocks by their magnitude, but for
a fixed codebook, minimizing the inner sum for each block independently is equivalent to
minimizing the total.

### Scale invariance

The key observation: for the naive scale $s_j = \text{snap}(\max_i|x_{j,i}| / c_7)$, the
normalized values $y_{j,i} = |x_{j,i}|/s_j$ all lie in $[0, c_7]$ with the block maximum
near $c_7$. **The distribution of normalized values is approximately independent of the
block's absolute scale.** This means we can pool normalized values across all blocks to
learn a single codebook.

## 4. The Optimization Problem

Define the normalized magnitudes for all blocks:

$$y_{j,i} = \frac{|x_{j,i}|}{\max_k|x_{j,k}|}, \quad y_{j,i} \in [0, 1]$$

where we normalize by the block maximum (rather than the snapped scale) to obtain a
clean $[0,1]$ distribution independent of the scale format.

Pool all normalized magnitudes from all blocks:

$$\mathcal{Y} = \{y_{j,i} : i = 1, \ldots, b, \; j = 1, \ldots, MK/b\}$$

We seek 7 positive centers $\tilde{c}_1 < \cdots < \tilde{c}_7$ in $[0, 1]$ that minimize:

$$\min_{\tilde{c}_1, \ldots, \tilde{c}_7} \sum_{y \in \mathcal{Y},\, y > d_0} \min_k \left(y - \tilde{c}_k\right)^2$$

where $d_0 = \tilde{c}_1 / 2$ is the decision boundary between zero and the smallest positive
value. Values $y \le d_0$ are assigned to $c_0 = 0$ and excluded from the positive-cluster
optimization.

This is **1D k-means** (Lloyd's algorithm) with $k = 7$ on the non-zero portion of $\mathcal{Y}$.

### Algorithm

1. **Normalize**: For each block, compute $y_{j,i} = |x_{j,i}| / \max_k|x_{j,k}|$.
2. **Pool**: Collect all $y_{j,i}$ values into a single set $\mathcal{Y}$.
3. **Initialize**: Set initial centers $\tilde{c}_1, \ldots, \tilde{c}_7$ at evenly-spaced
   quantiles of the non-zero values in $\mathcal{Y}$.
4. **Iterate** (Lloyd's algorithm):
   - Compute decision boundary $d_0 = \tilde{c}_1 / 2$ and inter-center boundaries
     $d_k = (\tilde{c}_k + \tilde{c}_{k+1})/2$.
   - **Assign**: each $y > d_0$ to its nearest center.
   - **Update**: $\tilde{c}_k \leftarrow \text{mean}(S_k)$ where $S_k$ is the set of values
     assigned to center $k$.
   - Repeat until convergence.
5. **Rescale**: Multiply by $c_7^{\text{target}}$ (e.g., 6.0) to obtain the final codebook
   in the standard scale range: $c_k = \tilde{c}_k \cdot c_7^{\text{target}}$.

### Why this works

The fixed FP4 E2M1 codebook $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ allocates half its positive
values to the range $[0, 1.5]$ (linear spacing) and the other half to $[2, 6]$ (exponential
spacing). This is a compromise designed for general-purpose hardware.

The learned codebook adapts to the **actual distribution of normalized weight magnitudes**.
If the distribution is approximately uniform in $[0, 1]$ (as is typical for Gaussian-like
weights after normalization), the optimal codebook will be more evenly spaced — placing
more resolution where there is more probability mass.

### Complexity

- **Normalization**: $O(MK)$ — one pass over the weight matrix.
- **Sorting for initialization**: $O(N \log N)$ where $N = MK$.
- **Each k-means iteration**: $O(N)$ (binary search in 7 sorted centers is $O(1)$).
- **Total iterations**: Typically $< 100$ for 1D k-means.

For a $2560 \times 9728$ weight matrix with block size 32, $N \approx 25\text{M}$ values.
The entire optimization runs in seconds on CPU.

## 5. Per-Block Scale Selection

Given the custom codebook $\mathcal{C}^+$, per-block scale selection follows the same framework
as the [Optimal Scale Search](optimal_scale_search.md):

- **Naive**: $s = \text{snap}(\max_i|x_i| / c_7)$
- **SSE-Optimal**: Bounded search minimizing $E(s) = \sum_i (x_i - s \cdot Q(x_i/s))^2$,
  using clipping and dead-zone bounds with $q_{\max} = c_7$ and $d_0 = c_1/2$
- **Hessian-Optimal**: Same bounded search minimizing $r^T H r$ with the custom codebook

The bounds from [Optimal Scale Search](optimal_scale_search.md) apply directly — only the
codebook values and boundaries change.

## 6. Discussion

**vs. FP4 E2M1.** The fixed FP4 E2M1 codebook $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ is designed
for hardware decode efficiency (the bit pattern maps directly to a floating-point value).
A learned codebook requires a lookup table for decode, but can achieve lower quantization error
by adapting to the actual weight distribution.

**Weight-dependent.** The optimal codebook depends on the weight distribution. Different layers
(or layer types) may benefit from different codebooks. In practice, one could learn a single
codebook per model, per layer type, or per individual layer — trading off storage (16 values
per codebook) against quantization quality.

**Relationship to scalar k-means.** This derivation shows that optimal block-scaled 4-bit
quantization with a learned codebook reduces to 1D k-means on the pooled normalized magnitudes.
Scale normalization provides invariance to per-block magnitude, reducing the problem from a
joint codebook-and-scale optimization to a single scalar quantization problem.
