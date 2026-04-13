# INT8

## Reference Implementation

```{eval-rst}
.. autofunction:: qwantize.int8.reference.int8_naive

.. autofunction:: qwantize.int8.reference.int8_optimal

.. autofunction:: qwantize.int8.reference.int8_optimal_hessian

.. autofunction:: qwantize.int8.reference.int8_dequantize

.. autofunction:: qwantize.int8.reference.build_fp16_scales

.. autofunction:: qwantize.int8.reference.int8_quantize

.. autofunction:: qwantize.int8.reference.int8_dequantize_block

.. autofunction:: qwantize.int8.reference.compute_block_sse
```

## Constants

- `qwantize.int8.Q_MAX = 127` -- Maximum symmetric INT8 magnitude
- `qwantize.int8.D_0 = 0.5` -- Dead-zone boundary for rounding to zero
- `qwantize.int8.VALID_BLOCK_SIZES = (32, 64, 128, 256)` -- Supported block sizes
