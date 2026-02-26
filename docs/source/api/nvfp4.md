# NVFP4

## Reference Implementation

```{eval-rst}
.. autofunction:: quantkit.nvfp4.reference.nvfp4_naive

.. autofunction:: quantkit.nvfp4.reference.nvfp4_optimal

.. autofunction:: quantkit.nvfp4.reference.nvfp4_optimal_hessian

.. autofunction:: quantkit.nvfp4.reference.nvfp4_dequantize

.. autofunction:: quantkit.nvfp4.reference.build_fp8_e4m3_scales

.. autofunction:: quantkit.nvfp4.reference.fp4_quantize

.. autofunction:: quantkit.nvfp4.reference.fp4_dequantize

.. autofunction:: quantkit.nvfp4.reference.compute_block_sse
```

## Triton Kernels

```{eval-rst}
.. autofunction:: quantkit.nvfp4.kernels.nvfp4_naive_triton

.. autofunction:: quantkit.nvfp4.kernels.nvfp4_optimal_triton

.. autofunction:: quantkit.nvfp4.kernels.nvfp4_optimal_hessian_triton
```

## Constants

- `quantkit.nvfp4.Q_MAX = 6.0` -- Maximum FP4 E2M1 codebook value
- `quantkit.nvfp4.D_0 = 0.25` -- Decision boundary for rounding to zero
- `quantkit.nvfp4.FP4_CODEBOOK = [0, 0.5, 1, 1.5, 2, 3, 4, 6]` -- FP4 E2M1 codebook
- `quantkit.nvfp4.FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]` -- Decision boundaries
