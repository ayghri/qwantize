# SPGL1 LASSO Compensation

## 1. Motivation

[GPTQ](https://arxiv.org/abs/2210.17323) snaps weights to the quantization
grid one block at a time, then propagates the snap error to the
not-yet-quantized columns via a closed-form inverse-Hessian step:

$$\Delta W_{\text{rem}} \;=\; -\, \frac{q_g - w_g}{[H^{-1}]_{gg}} \; [H^{-1}]_{g,\text{rem}}$$

That step is the **unconstrained** least-squares minimizer over the
remaining columns. It works well, but it has three drawbacks:

1. It requires forming and storing $H^{-1}$ (Cholesky + damping, $K^2$
   memory).
2. The compensation can drift the remaining columns *far* from their
   original values, making them harder to quantize cleanly when their turn
   comes.
3. There is no way to interpolate between "no compensation" and "full
   compensation" -- the closed form is a single point in the trade-off.

**SPGL1 compensation** replaces this step with an L1-constrained LASSO
solve. Instead of letting the remaining columns absorb the snap error
unconstrained, we cap the L1 magnitude of the compensation, keeping the
remaining columns close to where they started.

## 2. Problem Setup

After block $j$ is snapped (in descending-saliency order), let

- $W^{(j-1)}$ = the current iterate of the weight matrix
- $W_0$ = the original (unquantized) weight matrix
- $\Delta_{\text{eff}} = W^{(j-1)} - W_0$ -- the **effective shift** so far
- $\text{rem}$ = the set of column indices not yet snapped

We want a compensation $\delta \in \mathbb{R}^{M \times |\text{rem}|}$
applied to columns $\text{rem}$ that minimizes the output error:

$$\min_{\delta} \; \big\| X\,(\Delta_{\text{eff}} + \delta\,\mathbf{1}_{\text{rem}})^\top \big\|_F^2 \quad \text{s.t.} \quad \|\delta_m\|_1 \le \tau_m \;\;\forall\, m$$

per row $m$. The L1 constraint is what makes this different from GPTQ:
it bounds how much *total* drift any single row absorbs.

## 3. Reduced (Gram) Form

Materializing the $M \times T$ residual $X\,(\Delta_{\text{eff}} + \delta)^\top$
is wasteful (and OOM-prone for long calibration sequences). Expand:

$$\big\| X(\Delta + \delta)^\top \big\|^2_F = \delta\,H_{\text{rem}}\delta^\top \;-\; 2\,\delta\,(\Delta_{\text{eff}} H)_{\text{rem}}^\top \;+\; \text{const}(\Delta_{\text{eff}}, H)$$

where $H = X^\top X$ is the full $K \times K$ Hessian. Defining

$$H_{\text{red}} \;=\; H_{\text{rem},\,\text{rem}}, \qquad A^\top b \;=\; -\,\Delta_{\text{eff}}\, H_{:,\,\text{rem}}, \qquad b_{\text{norm-sq}} \;=\; \Delta_{\text{eff}}\, H\, \Delta_{\text{eff}}^\top$$

the LASSO becomes a pure Gram-form problem:

$$\min_{\delta} \;\; \tfrac{1}{2}\, \delta\, H_{\text{red}}\, \delta^\top \;-\; \langle A^\top b,\, \delta \rangle \;+\; \tfrac{1}{2}\, b_{\text{norm-sq}} \quad \text{s.t.} \quad \|\delta_m\|_1 \le \tau_m$$

This is solved by **SPGL1** (Spectral Projected Gradient for L1) in its
batched GPU form (`qwantize/spgl1.py`):

- Gradient: $\nabla_\delta = \delta\, H_{\text{red}} - A^\top b$
- Step: $\delta \leftarrow P_{\|\cdot\|_1 \le \tau}\big(\delta - \alpha\,\nabla\big)$
- Step size $\alpha$: Barzilai-Borwein with non-monotone backtracking
- Projection: $P_{\|\cdot\|_1 \le \tau}$ via sort + soft-threshold (Duchi
  et al. 2008)

The Hessian matvec $\delta\,H_{\text{red}}$ runs in fp16 on tensor cores
for ~1.4-1.6× speedup with no measurable accuracy regression.

**No matrix inverse appears anywhere.** No Cholesky, no damping, no
$H^{-1}$ storage. The pipeline is robust to ill-conditioned or
rank-deficient $H$.

## 4. The Full Pipeline

```text
1. Compute block saliencies: loss_j = r_j^T H_j r_j  (cheap, block-diag)
2. Sort blocks in descending loss order  →  col_perm
3. Permute W, H to col_perm
4. For each block j in order:
     a. Snap block j to grid via per-block H-Opt scale search.
     b. Update Δ_eff[:, block_j].
     c. Form H_red, ATb, b_norm_sq for remaining columns.
     d. Solve LASSO: δ = SPGL1(H_red, ATb, b_norm_sq, τ).
     e. Apply: W[:, rem] += δ;  Δ_eff[:, rem] += δ.
5. Inverse-permute W back to original column order.
```

### Choice of $\tau$

The L1 budget is set per-row from the gradient magnitude at $\delta = 0$:

$$\tau_m \;=\; \texttt{tau\_frac} \cdot \frac{\| (A^\top b)_m \|_1}{\overline{\mathrm{diag}}(H_{\text{red}})}$$

`tau_frac` controls how aggressive the compensation is. Empirically the
result is **robust across two decades** of `tau_frac` (0.1 -- 5.0); the
SPGL1 line search adapts the actual step size regardless of where the L1
ball radius starts. `tau_frac = 1.0` is a good default.

### Why block-diag saliency, not strict OBS

The standard OBS saliency uses $r^\top [H^{-1}]_{gg}^{-1} r$. We use the
cheaper block-Hessian form $r^\top H_g r$ instead. Empirically the two
orderings differ by **+0.004pp** output error on layer_0 -- noise. The
cheap form keeps the pipeline inverse-free.

## 5. What It Replaces in GPTQ

| GPTQ step | SPGL1 compensation |
|:--|:--|
| Build $H^{-1}$ via Cholesky + damping | Use $H$ directly |
| Closed-form $-\,\text{err} / [H^{-1}]_{gg} \cdot [H^{-1}]_{g,\text{rem}}$ | Iterative LASSO solve |
| Unconstrained -- columns can drift freely | $\|\delta_m\|_1 \le \tau_m$ trust region |
| Single fixed answer | Parametric family in `tau_frac` |
| Requires PSD + invertible $H$ | Works on rank-deficient $H$ |

What stays from the GPTQ family:

- **OBS-style block saliency** for ordering, in the cheap block-diag form.
- **Per-block H-Opt snap** -- the SSE / Hessian-weighted scale search over
  the FP8 grid is unchanged.

This is why the framing is *"OBS-ordered SPGL1 compensation"*, not
*"GPTQ + SPGL1"*: SPGL1 replaces GPTQ's defining move rather than
augmenting it.

## 6. Cost

For layer_0 (`W = 2560 x 9728`, `X = 244449 x 9728`):

| Configuration | GPTQ-Ord + H-Opt | + SPGL1 (10 iters, fp16 matvec) |
|:--|:--:|:--:|
| Block size 16, FP8 E4M3 | ~16s | ~200s |
| Block size 64, FP8 E4M3 | ~150-200s | ~370s |
| Block size 128, FP16 | ~0.6s | ~30s |

Dominant cost is the per-iter $\delta\, H_{\text{red}}$ matvec and the L1
ball projection. Both have known optimization paths (radix sort for the
projection, sparse incremental updates for the matvec). See the
[research log](https://github.com/ayghri/qwantize/blob/main/notes/progress_track_spgl1.md)
for benchmarks and deferred optimizations.

## 7. Results

| Config | b/w | GPTQ-Ord+Opt O% | +SPGL1 O% | ΔO from SPGL1 |
|:--|:--:|:--:|:--:|:--:|
| FP4, bs=16, E4M3 | 4.500 | 4.21 (H-Opt) | **3.64** | -0.57 |
| FP4, bs=32, FP16 | 4.500 | 4.91 (SSE-Opt) | **4.22** | -0.69 |
| FP4, bs=64, E4M3 | 4.125 | 5.22 (H-Opt) | **4.65** | -0.57 |
| FP4, bs=128, FP16 | 4.125 | 5.47 (H-Opt) | **4.77** | -0.70 |
| INT4, bs=64, E4M3 | 4.125 | 5.99 (H-Opt) | **5.22** | -0.77 |
| INT4, bs=128, FP16 | 4.125 | 6.17 (H-Opt) | **5.35** | -0.82 |

SPGL1 contributes a consistent **0.55--0.82 pp** output-error reduction
across codebooks (FP4/INT4), block sizes (16/32/64/128), and scale formats
(FP8/FP16). At every operating point tested, the SPGL1 variant is the best
result. For bs=32 FP16, SSE-Opt is used for the per-block scale (the FP16
grid is too fine to enumerate; SSE and H-weighted formulas converge to
nearly identical results at this resolution).

See the [Results page](results.md) for the full table and the
[research log](https://github.com/ayghri/qwantize/blob/main/notes/progress_track_spgl1.md)
for the iteration-by-iteration progression that led to this method.
