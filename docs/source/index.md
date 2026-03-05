# Qwantize

Optimal quantization methods for block-scaled formats.

## Installation

```bash
pip install qwantize
```

Requires PyTorch (>=2.0) and Triton (>=3.0).

## Repository

- **GitHub**: [github.com/ayghri/qwantize](https://github.com/ayghri/qwantize)

## Formats

- **NVFP4** -- FP4 E2M1 with FP8 E4M3 scales (block sizes 16, 32)
- **MXFP4** -- FP4 E2M1 with UE8M0 (power-of-2) scales (block sizes 16, 32)

## Quick Start

```python
from qwantize import nvfp4_naive, nvfp4_optimal, nvfp4_dequantize, compute_metrics

# W has shape (..., block_size) where block_size is 16 or 32
# dim specifies which dimension is the block dimension (default: -1)
W_blocked = W.reshape(M, K // 32, 32)

# Quantize: returns (scales, quants)
scales, quants = nvfp4_optimal(W_blocked, dim=-1)

# Dequantize separately
W_dq = nvfp4_dequantize(scales, quants, dim=-1)

# Or get dequantized output directly
scales, quants, W_dq = nvfp4_optimal(W_blocked, dim=-1, return_dequant=True)

metrics = compute_metrics(W, W_dq.reshape(M, K), X)
```

```{toctree}
:maxdepth: 2
:caption: Methods

optimal_scale_search
hessian_scale_search
scale_distance
triton_kernels
```

```{toctree}
:maxdepth: 2
:caption: Results

results
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/nvfp4
api/mxfp4
api/metrics
```

