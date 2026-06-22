"""GPTQ: post-training quantization with OBS error compensation.

The quantization itself is **user-provided** — the OBS loop handles
error propagation while a caller-supplied ``quantize_fn`` decides how
each column is quantized.

Provides:
  - gptq()                   — core algorithm
  - make_uniform_quantizer() — example: symmetric uniform INT-N
  - make_group_quantizer()   — example: group-wise INT-N

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers", ICLR 2023.
"""

import inspect

import torch

from .obs import compute_hessian, prepare_hessian


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def gptq(W, H, quantize_fn, blocksize=128, percdamp=0.01):
    """Quantize a weight matrix using GPTQ (OBS error compensation).

    Processes columns left-to-right in blocks.  For each column the
    user-supplied ``quantize_fn`` produces a quantized value; the
    framework then propagates the quantization error to remaining
    columns using Hessian information.

    Two ``quantize_fn`` signatures are accepted:

    **Simple** — quantize a single column in isolation::

        def quantize_fn(w_col: Tensor) -> Tensor:
            '''w_col: [out_features].  Return quantized [out_features].'''

    **Full context** — access the entire weight matrix, Hessian, and
    column masks for context-aware decisions (e.g. outlier handling,
    sensitivity-adaptive bit widths)::

        def quantize_fn(W, H, done_mask, target_mask) -> Tensor:
            '''
            W:           [out, in] current weight matrix.
            H:           [in, in]  Hessian (X^T X / N).
            done_mask:   [in]      bool, True for already-quantized columns.
            target_mask: [in]      bool, True for columns to quantize now.
            Returns:     [out, target_mask.sum()] quantized target columns.
            '''

    The simple form is auto-wrapped into the full form internally.

    Args:
        W: [out_features, in_features] weight matrix.
        H: [in_features, in_features] Hessian (X^T X / N).
           If H has shape [N, in_features] (i.e. raw activations),
           it is converted to X^T X / N automatically.
        quantize_fn: Quantization function (see above).
        blocksize: OBS block size.
        percdamp: Diagonal damping as fraction of mean diagonal.

    Returns:
        Q:      [out, in] quantized weight matrix (float32).
        losses: [out, in] per-element quantization loss
                (``(w - q)^2 / Hinv[j,j]^2``).
    """
    W = W.clone().float()
    rows, cols = W.shape
    device = W.device

    # Accept raw activations X as H
    if H.dim() == 2 and H.shape[0] != H.shape[1]:
        H = compute_hessian(H)
    H_orig = H.clone()

    Hinv, dead = prepare_hessian(H, percdamp)
    W[:, dead] = 0

    # Auto-detect simple vs full quantize_fn
    n_params = len(inspect.signature(quantize_fn).parameters)
    if n_params == 1:
        _inner = quantize_fn

        def quantize_fn(W, H, done, target):  # noqa: F811
            return _inner(W[:, target])

    losses = torch.zeros_like(W)
    done = torch.zeros(cols, dtype=torch.bool, device=device)
    target = torch.zeros(cols, dtype=torch.bool, device=device)

    for b_start in range(0, cols, blocksize):
        b_end = min(b_start + blocksize, cols)
        bs = b_end - b_start

        W_blk = W[:, b_start:b_end].clone()
        Err = torch.zeros(rows, bs, device=device)
        Hinv_blk = Hinv[b_start:b_end, b_start:b_end]

        for j in range(bs):
            j_g = b_start + j

            # Sync error-corrected value into W so quantize_fn sees it
            W[:, j_g] = W_blk[:, j]

            target.zero_()
            target[j_g] = True

            q = quantize_fn(W, H_orig, done, target)
            q = q.reshape(-1)

            w = W_blk[:, j]
            d = Hinv_blk[j, j]

            err = (w - q) / d
            Err[:, j] = err
            losses[:, j_g] = (w - q) ** 2 / (d**2 + 1e-30)

            W_blk[:, j] = q
            W[:, j_g] = q

            if j + 1 < bs:
                W_blk[:, j + 1 :] -= err.unsqueeze(1) * Hinv_blk[j : j + 1, j + 1 :]

            done[j_g] = True

        W[:, b_start:b_end] = W_blk
        if b_end < cols:
            W[:, b_end:] -= Err @ Hinv[b_start:b_end, b_end:]

    return W, losses


# ---------------------------------------------------------------------------
# Example quantizers
# ---------------------------------------------------------------------------


def make_uniform_quantizer(bits=4, symmetric=True, per_channel=False):
    """Create a uniform quantization function for use with :func:`gptq`.

    Quantizes to ``bits``-bit integers then de-quantizes back to float.

    Args:
        bits: Bit width (e.g. 4 for INT4).
        symmetric: Symmetric range ``[-2^(b-1)+1, 2^(b-1)-1]``.
        per_channel: Compute scale per output channel (row) rather than
            per column.

    Returns:
        A ``quantize_fn(W, H, done, target) -> Q`` compatible with
        :func:`gptq`.
    """
    if symmetric:
        qmin = -(2 ** (bits - 1)) + 1
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2**bits - 1

    def fn(W, H, done, target):
        w = W[:, target].float()
        if per_channel:
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
        else:
            scale = w.abs().amax().clamp(min=1e-10) / qmax
        if symmetric:
            q = (w / scale).round().clamp(qmin, qmax) * scale
        else:
            zero = (-w.amin(dim=0, keepdim=True) / scale).round().clamp(qmin, qmax)
            q = ((w / scale + zero).round().clamp(qmin, qmax) - zero) * scale
        return q

    fn.__doc__ = (
        f"Uniform {bits}-bit {'symmetric' if symmetric else 'asymmetric'} quantizer"
    )
    return fn


def make_group_quantizer(bits=4, group_size=128, symmetric=True):
    """Create a group-wise quantization function for use with :func:`gptq`.

    Each group of ``group_size`` elements along the output dimension
    shares one scale factor.

    Args:
        bits: Bit width.
        group_size: Number of elements per quantization group.
        symmetric: Symmetric range.

    Returns:
        A ``quantize_fn(W, H, done, target) -> Q`` compatible with
        :func:`gptq`.
    """
    if symmetric:
        qmin = -(2 ** (bits - 1)) + 1
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2**bits - 1

    def fn(W, H, done, target):
        w = W[:, target].float()  # [out, n_targets]
        out = w.shape[0]
        pad = (group_size - out % group_size) % group_size
        if pad:
            w = torch.nn.functional.pad(w, (0, 0, 0, pad))
        # [n_groups, group_size, n_targets]
        groups = w.reshape(-1, group_size, w.shape[1])
        scale = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / qmax
        q = (groups / scale).round().clamp(qmin, qmax) * scale
        return q.reshape(-1, w.shape[1])[:out]

    fn.__doc__ = (
        f"Group-{group_size} {bits}-bit {'sym' if symmetric else 'asym'} quantizer"
    )
    return fn
