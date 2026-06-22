"""Optimal Brain Surgeon utilities.

Shared Hessian computation, preparation, and layer discovery
for SparseGPT and GPTQ.
"""

import torch
import torch.nn as nn


def find_layers(module, layer_types=(nn.Linear,), prefix=""):
    """Recursively find layers of given types in a module.

    Args:
        module: Root nn.Module.
        layer_types: Tuple of layer types to find.
        prefix: Name prefix (used in recursion).

    Returns:
        Dict mapping dotted name to layer module.
    """
    if isinstance(module, tuple(layer_types)):
        return {prefix: module}
    result = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        result.update(find_layers(child, layer_types, full))
    return result


def get_transformer_layers(model):
    """Get the sequential decoder layers from a HuggingFace model.

    Tries common patterns: model.model.layers (Llama/Qwen/Mistral),
    model.transformer.h (GPT-2/Neo), model.gpt_neox.layers (Pythia).

    Returns:
        nn.ModuleList of transformer decoder layers.

    Raises:
        ValueError: if no known pattern matches.
    """
    candidates = [
        ("model", "layers"),  # Llama, Qwen, Mistral, Gemma
        ("transformer", "h"),  # GPT-2, GPT-Neo
        ("gpt_neox", "layers"),  # Pythia, GPT-NeoX
        ("transformer", "blocks"),  # MPT, Falcon
    ]
    for backbone, attr in candidates:
        obj = getattr(model, backbone, None)
        if obj is not None:
            layers = getattr(obj, attr, None)
            if layers is not None:
                return layers
    raise ValueError(
        "Cannot auto-detect transformer layers. "
        "Pass them explicitly via the `layers` argument."
    )


def compute_hessian(X):
    """Compute Hessian H = X^T X / N.

    Args:
        X: [N, K] flattened input activations (float32 recommended).

    Returns:
        H: [K, K] Hessian matrix in float32.
    """
    X = X.float()
    return X.t().mm(X).div_(X.shape[0])


def prepare_hessian(H, percdamp=0.01):
    """Dead-column handling, diagonal damping, Cholesky factorization.

    Args:
        H: [K, K] Hessian. Not modified (cloned internally).
        percdamp: Damping as fraction of mean diagonal.

    Returns:
        Hinv: [K, K] upper Cholesky factor of H^{-1}.
        dead: [K] bool mask of dead (zero-variance) columns.
    """
    H = H.clone().float()
    dead = H.diagonal() == 0
    H[dead, dead] = 1
    H.diagonal().add_(percdamp * H.diagonal().mean())

    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        # Extra damping on near-singular H
        H.diagonal().add_(1e-5 * H.diagonal().mean())
        L = torch.linalg.cholesky(H)

    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    return Hinv, dead


class HessianAccumulator:
    """Accumulates H = X^T X / N via a forward pre-hook on a Linear layer.

    Usage::

        acc = HessianAccumulator(linear_layer)
        # ... run forward passes ...
        acc.remove()
        H = acc.H  # [in_features, in_features]
    """

    def __init__(self, layer):
        self.H = None
        self.nsamples = 0
        self._hook = layer.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        x = args[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]

        if self.H is None:
            self.H = torch.zeros(x.shape[1], x.shape[1], device=x.device)

        # Running weighted average: H = (n_old * H + x^T x) / (n_old + n)
        self.H.mul_(self.nsamples / (self.nsamples + n))
        self.H.addmm_(x.t(), x, alpha=1.0 / (self.nsamples + n))
        self.nsamples += n

    def remove(self):
        """Remove the forward hook."""
        self._hook.remove()

    def free(self):
        """Release Hessian memory."""
        self.H = None
        torch.cuda.empty_cache()
