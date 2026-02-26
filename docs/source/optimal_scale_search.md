# Optimal Scale Search

This page describes the mathematical foundation for finding per-block optimal scales in block-scaled FP4 quantization.

## 1. Problem Formulation

Let $x \in \mathbb{R}^K$ be a block of weights or activations.
Let $\mathcal{Q}$ be the set of representable values in FP4 E2M1. The representable positive values are:

$$\mathcal{Q}^+ = \{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$$

- Maximum representable value: $q_{\max} = 6$
- Smallest non-zero magnitude: $q_{\min} = 0.5$
- Decision boundary for zero: $d_0 = 0.25$ (any value $|y| < 0.25$ rounds to 0)

Let $\mathcal{S}$ be the set of representable positive scale values (FP8 E4M3 for NVFP4, UE8M0 for MXFP4).
The quantized value of $x_i$ given a scale $s$ is $s \cdot Q(x_i/s)$, where $Q(\cdot)$ is the nearest-neighbor mapping to $\mathcal{Q}$.

Our objective is to find the optimal scale $s^*$ that minimizes the Sum of Squared Errors (SSE):

$$E(s) = \sum_{i=1}^K \left(x_i - s \cdot Q\left(\frac{x_i}{s}\right)\right)^2$$

$$s^* = \arg\min_{s \in \mathcal{S}} E(s)$$

## 2. Baseline Error Anchor

Standard practice uses the baseline continuous scale $s_{\text{base}} = \max_i |x_i| / q_{\max}$.
Let $s_0 \in \mathcal{S}$ be the closest representable scale to $s_{\text{base}}$.
We compute its error: $E_0 = E(s_0)$.

Since $s^*$ is the absolute optimum, it must strictly satisfy:

$$E(s^*) \le E_0$$

This inequality $E(s) \le E_0$ is the anchor we use to prove that scales too large or too small cannot be optimal.

**Edge case**: If $\sum x_i^2 \le E_0$, quantizing everything to zero is no worse than the baseline. The block is effectively noise; return $s_0$ immediately.

## 3. Lower Bound (Clipping Dominance)

If a scale $s$ is too small, large values in $x$ will clip to $s \cdot q_{\max}$, generating large error.

**Proof.** For any element $x_i$, the maximum representable magnitude is $s \cdot q_{\max}$. If $|x_i| > s \cdot q_{\max}$, it clips. The error for that single element is strictly $(|x_i| - s \cdot q_{\max})^2$.
Since the squared error of all other elements is $\ge 0$:

$$E(s) \ge \sum_{i=1}^K \max(|x_i| - s \cdot q_{\max}, 0)^2 \ge \max_i (|x_i| - s \cdot q_{\max})^2$$

For $s$ to be a valid candidate, we must have $E(s) \le E_0$. Let $x_{\max} = \max_i |x_i|$. Assuming $s < x_{\max}/q_{\max}$ (clipping occurs):

$$(x_{\max} - s \cdot q_{\max})^2 \le E_0$$
$$x_{\max} - s \cdot q_{\max} \le \sqrt{E_0}$$
$$s \ge \frac{x_{\max} - \sqrt{E_0}}{q_{\max}} = s_{\min}$$

**Strict algorithmic lower bound.** A stronger per-candidate fast-fail uses the full clipping sum:

$$H(s) = \sum_{i=1}^K \max(|x_i| - s \cdot q_{\max}, 0)^2$$

$H(s)$ is monotonically decreasing with $s$. Any scale $s$ where $H(s) > E_{\text{best}}$ can be immediately discarded without computing the full quantization SSE.

## 4. Upper Bound (Dead-Zone Dominance)

If a scale $s$ is too large, the values $x_i/s$ become so small that they fall below the decision boundary $d_0 = 0.25$, causing elements to snap to $0$.

**Proof.** If $|x_i| / s < d_0$, then $Q(x_i/s) = 0$. The error contributed by this element is exactly $x_i^2$.
Let $A(s) = \{i \mid |x_i| < s \cdot d_0\}$ be the set of indices quantized to zero.

$$E(s) = \sum_{i \in A(s)} x_i^2 + \sum_{i \notin A(s)} (x_i - s \cdot Q(x_i/s))^2$$

Because the second term is non-negative:

$$E(s) \ge \sum_{i \in A(s)} x_i^2$$

To ensure $E(s) \le E_0$, the sum of the squares of the elements quantized to zero cannot exceed $E_0$.
Sort the absolute values in ascending order: $y_1 \le y_2 \le \dots \le y_K$.
Compute the cumulative sum of squares: $C_k = \sum_{j=1}^k y_j^2$.

Find the maximum index $k^*$ such that $C_{k^*} \le E_0$.
This tells us we can afford to quantize at most $k^*$ elements to zero. The $(k^*+1)$-th smallest element must NOT quantize to zero:

$$|y_{k^*+1}| / s \ge d_0$$
$$s \le \frac{y_{k^*+1}}{d_0} = s_{\max}$$

## 5. The Optimized Search Algorithm

Instead of brute-forcing all representable scales, we execute the following sequence:

**Step 1: Setup and Baseline**
1. Extract the block $x \in \mathbb{R}^K$. Compute $x_{\max} = \max|x_i|$.
2. Set analytical baseline: $s_{\text{cont}} = x_{\max} / q_{\max}$.
3. Snap $s_{\text{cont}}$ to the nearest representable scale to get $s_0$.
4. Quantize $x$ using $s_0$ and calculate the baseline error $E_0$.
5. *Edge Case Check:* If $\sum x_i^2 \le E_0$, the block is effectively noise. Return $s_0$ immediately.

**Step 2: Calculate Bounds**
1. **Upper Bound:** Sort $|x|$ ascending as $y_1, \dots, y_K$. Find highest $k^*$ where $\sum_{j=1}^{k^*} y_j^2 \le E_0$. Compute $s_{\max} = y_{k^*+1} / d_0$.
2. **Lower Bound:** Compute analytical floor: $s_{\min} = \max(0, (x_{\max} - \sqrt{E_0}) / q_{\max})$.

**Step 3: Bounded Search**
1. Filter the scale table to only those in the range $[s_{\min}, s_{\max}]$.
2. Initialize `best_s = s_0` and `min_E = E_0`.
3. For each $s$ in the filtered set:
   - *Fast-Fail Check:* Calculate clipping error $H(s) = \sum \max(|x_i| - q_{\max} \cdot s, 0)^2$. If $H(s) > \text{min\_E}$, skip.
   - Otherwise, compute full quantization error $E(s)$.
   - If $E(s) < \text{min\_E}$, update `min_E = E(s)` and `best_s = s`.
4. Return `best_s`.

### Efficiency

- **Sorting $K$ elements** (where $K$ is typically 16 or 32) takes negligible time compared to memory bandwidth and repeated quantization math.
- **Search space reduction:** The bounds usually restrict candidates to a window of **4 to 8 scales** around $s_0$. The fast-fail check further skips full evaluations, bringing the heavy $Q(\cdot)$ operations down to an absolute minimum.
