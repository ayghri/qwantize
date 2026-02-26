# MXFP4

## Reference Implementation

```{eval-rst}
.. autofunction:: quantkit.mxfp4.reference.mxfp4_naive

.. autofunction:: quantkit.mxfp4.reference.mxfp4_optimal

.. autofunction:: quantkit.mxfp4.reference.mxfp4_optimal_hessian

.. autofunction:: quantkit.mxfp4.reference.mxfp4_dequantize

.. autofunction:: quantkit.mxfp4.reference.build_ue8m0_scales

.. autofunction:: quantkit.mxfp4.reference.fp4_quantize

.. autofunction:: quantkit.mxfp4.reference.fp4_dequantize

.. autofunction:: quantkit.mxfp4.reference.compute_block_sse
```

## Triton Kernels

```{eval-rst}
.. autofunction:: quantkit.mxfp4.kernels.mxfp4_naive_triton

.. autofunction:: quantkit.mxfp4.kernels.mxfp4_optimal_triton

.. autofunction:: quantkit.mxfp4.kernels.mxfp4_naive_torch

.. autofunction:: quantkit.mxfp4.kernels.mxfp4_optimal_torch
```

## Constants

- `quantkit.mxfp4.Q_MAX = 6.0` -- Maximum FP4 codebook value
- `quantkit.mxfp4.D_0 = 0.25` -- Decision boundary for rounding to zero
- `quantkit.mxfp4.FP4_CODEBOOK = [0, 0.5, 1, 1.5, 2, 3, 4, 6]` -- Standard FP4 E2M1 codebook
- `quantkit.mxfp4.FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]` -- Decision boundaries
