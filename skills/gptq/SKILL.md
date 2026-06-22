# SparseGPT & GPTQ — Pruning and Quantization Skill

Self-contained OBS (Optimal Brain Surgeon) implementations for one-shot weight pruning and post-training quantization of linear layers.

## Setup

```bash
pip install torch transformers datasets
```

## Module overview

| File | What it does |
|------|-------------|
| `obs.py` | Shared utilities: `find_layers`, `compute_hessian`, `prepare_hessian`, `HessianAccumulator`, `get_transformer_layers` |
| `sparsegpt.py` | SparseGPT pruning: core algorithm + full LLM pipeline + calibration data loader |
| `gptq.py` | GPTQ quantization: core algorithm + example quantizer factories |

## SparseGPT

### Core algorithm — single layer

```python
from gptq import sparsegpt, compute_hessian

# W: [out_features, in_features] weight matrix
# X: [N, in_features] calibration activations for this layer
H = compute_hessian(X)

# 2:4 structured sparsity
W_pruned = sparsegpt(W, H, prune_n=2, prune_m=4)

# 50% unstructured sparsity
W_pruned = sparsegpt(W, H, sparsity=0.5, prune_n=0)

# Custom block size and damping
W_pruned = sparsegpt(W, H, prune_n=2, prune_m=4, blocksize=128, percdamp=0.01)
```

### Full LLM pipeline

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptq import prune_model, get_calibration_data

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Get calibration data (128 sequences of length 2048 from C4)
tokens = get_calibration_data(tokenizer, nsamples=128, seqlen=2048)

# Prune with 2:4 structured sparsity
model = prune_model(model, tokens, prune_n=2, prune_m=4, device="cuda:0")
```

`prune_model` handles the full SparseGPT pipeline:
1. Captures inputs to the first transformer layer via calibration forward passes
2. For each decoder layer: attaches Hessian accumulators on all linear sublayers, runs forward to accumulate H, prunes, then re-runs forward with pruned weights to propagate corrected outputs
3. Prints per-sublayer RMSE and density

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sparsity` | 0.5 | Unstructured sparsity ratio (ignored when prune_n > 0) |
| `prune_n` | 2 | Nonzeros to keep per group (0 = unstructured) |
| `prune_m` | 4 | Group size for N:M structured sparsity |
| `blocksize` | 128 | OBS block size (columns processed together) |
| `percdamp` | 0.01 | Diagonal damping as fraction of mean Hessian diagonal |

## GPTQ

### Core algorithm — quantization-agnostic

The `gptq()` function performs OBS error compensation while delegating the actual quantization to a user-supplied function. This makes it agnostic to quantization scheme (uniform, group, non-uniform, mixed precision, etc.).

```python
from gptq import gptq, compute_hessian

H = compute_hessian(X)

# Simple interface: just quantize a column
def my_quantizer(w_col):
    scale = w_col.abs().max() / 7
    return (w_col / scale).round().clamp(-8, 7) * scale

Q, losses = gptq(W, H, my_quantizer)
```

### quantize_fn interface

Two signatures are supported:

**Simple** — one argument, receives and returns a single column:
```python
def quantize_fn(w_col: Tensor) -> Tensor:
    # w_col: [out_features]
    # return: [out_features] quantized
```

**Full context** — four arguments, for context-aware quantization:
```python
def quantize_fn(W, H, done_mask, target_mask) -> Tensor:
    # W:           [out, in]  current weight matrix (processed cols have quantized values)
    # H:           [in, in]   Hessian
    # done_mask:   [in]       bool, True for already-quantized columns
    # target_mask: [in]       bool, True for the column to quantize now
    # return:      [out, target_mask.sum()] quantized values
```

The full interface gives the quantizer access to:
- The whole weight matrix for outlier detection or scale computation
- The Hessian for sensitivity-aware decisions (e.g. higher precision for sensitive columns)
- Which columns are already done vs. pending

### Built-in quantizer factories

```python
from gptq import gptq, make_uniform_quantizer, make_group_quantizer

# INT4 symmetric uniform
Q, losses = gptq(W, H, make_uniform_quantizer(bits=4))

# INT4 with per-channel scales
Q, losses = gptq(W, H, make_uniform_quantizer(bits=4, per_channel=True))

# INT4 with group-128 quantization
Q, losses = gptq(W, H, make_group_quantizer(bits=4, group_size=128))

# INT8 symmetric
Q, losses = gptq(W, H, make_uniform_quantizer(bits=8))
```

### Custom quantizer example — mixed precision

```python
def mixed_precision_quantizer(W, H, done, target):
    """Use INT8 for sensitive columns, INT4 for the rest."""
    col_idx = target.nonzero().item()
    sensitivity = H[col_idx, col_idx]

    w = W[:, target].float()
    if sensitivity > threshold:
        bits, qmax = 8, 127
    else:
        bits, qmax = 4, 7

    scale = w.abs().amax().clamp(min=1e-10) / qmax
    qmin = -qmax
    return (w / scale).round().clamp(qmin, qmax) * scale

Q, losses = gptq(W, H, mixed_precision_quantizer)
```

### Passing raw activations

`gptq()` and `sparsegpt()` accept either a Hessian (square matrix) or raw activations (non-square). Raw activations are converted to H automatically:

```python
# These are equivalent:
Q, losses = gptq(W, X, quantize_fn)               # X: [N, in_features]
Q, losses = gptq(W, compute_hessian(X), quantize_fn)  # H: [in, in]
```

## Shared utilities

```python
from gptq import find_layers, HessianAccumulator, get_transformer_layers

# Find all linear layers in a module
sublayers = find_layers(transformer_layer)
# → {"self_attn.q_proj": Linear, "self_attn.k_proj": Linear, ...}

# Accumulate Hessian via hooks (no need to store activations)
acc = HessianAccumulator(linear_layer)
# ... run forward passes through the model ...
acc.remove()
H = acc.H  # [in_features, in_features]

# Get decoder layers from any common HF architecture
layers = get_transformer_layers(model)  # works for Llama, Qwen, GPT-2, Pythia, ...
```

## Algorithm summary

Both SparseGPT and GPTQ use the same OBS framework:

1. Compute Hessian H = X^T X / N from calibration activations
2. Dead column handling + diagonal damping + Cholesky inverse
3. Process columns left-to-right in blocks of `blocksize`:
   - **SparseGPT**: zero out weights with lowest saliency (w^2 / H_inv[j,j]^2)
   - **GPTQ**: call `quantize_fn` to quantize the column
   - Propagate error: `W[:, j+1:] -= (w - q) / H_inv[j,j] * H_inv[j, j+1:]`
4. Inter-block error propagation after each block

The only difference is step 3's column processing — pruning vs. quantization.
