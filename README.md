# QuantKit

Optimal quantization methods for block-scaled formats.

## Formats

- **NVFP4** — FP4 E2M1 with FP8 E4M3 scales (block sizes 16, 32)
- **MXFP4** — FP4 E2M1 with UE8M0 (power-of-2) scales (block sizes 16, 32)

## Methods

Each format supports multiple scale selection strategies:

| Method | Description |
|:--|:--|
| **Naive** | Standard heuristic: `s = snap(amax / Q_MAX)` |
| **SSE-Optimal** | Bounded search minimizing sum of squared quantization error |
| **Hessian-Optimal** | Bounded search minimizing Hessian-weighted error `r^T H r` using activations |

All methods have both pure-PyTorch (reference) and Triton (GPU-accelerated) implementations.

## Install

```bash
pip install -e .
```

Requires PyTorch and Triton (for GPU kernels).

## Usage

```python
from quantkit import nvfp4_naive, nvfp4_optimal, nvfp4_dequantize, compute_metrics

# W has shape (..., block_size) where block_size is 16 or 32
W_blocked = W.reshape(M, K // 32, 32)

# Quantize: returns (scales, quants)
scales, quants = nvfp4_optimal(W_blocked, dim=-1)

# Dequantize
W_dq = nvfp4_dequantize(scales, quants, dim=-1)

# Or get dequantized output directly
scales, quants, W_dq = nvfp4_optimal(W_blocked, dim=-1, return_dequant=True)

# Compute metrics: ||Q(W)-W||/||W|| and ||XW_q^T - XW^T||/||XW^T||
metrics = compute_metrics(W, W_dq.reshape(M, K), X)
```

Triton-accelerated versions:

```python
from quantkit import nvfp4_optimal_triton, nvfp4_optimal_hessian_triton

scales, quants, W_dq = nvfp4_optimal_triton(W_blocked, dim=-1, return_dequant=True)

# Hessian-aware (requires activations X)
scales, quants, W_dq = nvfp4_optimal_hessian_triton(W_blocked, dim=-1, return_dequant=True, X=X)
```

## Benchmarks

Benchmarked on the `down_proj` weight of the first decoder layer from Qwen3-4B, with activations from WikiText-2 (max_seq_len=512, num_samples=2048).

```bash
python bench/full_bench.py
```

### NVFP4 (block size 16)

| Method | Weight Error | Output Error | Triton Speedup |
|:--|:--:|:--:|:--:|
| Naive | 10.05% | 6.89% | 1.7x |
| SSE-Optimal | 8.74% | 6.04% | 7.0x |
| H-Optimal | 9.35% | 5.31% | 1.8x |

### MXFP4 (block size 16)

| Method | Weight Error | Output Error | Triton Speedup |
|:--|:--:|:--:|:--:|
| Naive | 11.77% | 8.48% | 1.7x |
| SSE-Optimal | 11.02% | 7.67% | 33x |
| H-Optimal | 11.10% | 7.62% | — |

## Documentation

Full documentation: [quantkit.readthedocs.io](https://quantkit.readthedocs.io/)

Build locally:

```bash
pip install -r docs/requirements.txt
cd docs && make html
```

## Contact

- **Author**: Ayoub Ghriss, [research@ayghri.me](mailto:research@ayghri.me)
- **GitHub**: [github.com/ayghri/quantkit](https://github.com/ayghri/quantkit)
