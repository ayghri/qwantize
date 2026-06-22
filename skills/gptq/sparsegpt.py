"""SparseGPT: one-shot pruning via greedy Optimal Brain Surgeon.

Provides:
  - sparsegpt()     — core algorithm (weight matrix + Hessian in, pruned matrix out)
  - prune_model()   — full pipeline (model + calibration tokens in, pruned model out)
  - get_calibration_data() — load C4/WikiText calibration tokens

Reference: Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot", ICML 2023.
"""

import torch
import torch.nn as nn

from .obs import (
    find_layers,
    get_transformer_layers,
    prepare_hessian,
    HessianAccumulator,
)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def sparsegpt(W, H, sparsity=0.5, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
    """Prune a weight matrix using the SparseGPT (greedy OBS) algorithm.

    Processes columns left-to-right in blocks. For each column, removes
    weights with lowest saliency and propagates the error to remaining
    columns via the Hessian inverse.

    Supports both unstructured sparsity (via ``sparsity``) and N:M
    structured sparsity (via ``prune_n`` / ``prune_m``).

    Args:
        W: [out_features, in_features] weight matrix.
        H: [in_features, in_features] Hessian (X^T X / N).
        sparsity: Target sparsity for unstructured pruning (ignored when
            prune_n > 0).
        prune_n: Nonzeros to **keep** per group of ``prune_m`` (e.g. 2 for
            2:4).  Set to 0 for unstructured.
        prune_m: Group size for N:M structured sparsity (e.g. 4 for 2:4).
        blocksize: OBS block size (columns processed together).
        percdamp: Diagonal damping as fraction of mean Hessian diagonal.

    Returns:
        W_pruned: [out_features, in_features] pruned weight matrix (float32).
    """
    W = W.clone().float()
    rows, cols = W.shape
    device = W.device

    Hinv, dead = prepare_hessian(H, percdamp)
    W[:, dead] = 0

    # Align blocksize to prune_m so N:M groups don't cross block boundaries
    if prune_n > 0 and blocksize % prune_m != 0:
        blocksize = ((blocksize // prune_m) + 1) * prune_m

    for b_start in range(0, cols, blocksize):
        b_end = min(b_start + blocksize, cols)
        bs = b_end - b_start

        W_blk = W[:, b_start:b_end].clone()
        Err = torch.zeros(rows, bs, device=device)
        Hinv_blk = Hinv[b_start:b_end, b_start:b_end]

        # Masks: True = **prune** (zero out)
        if prune_n > 0:
            mask = torch.zeros(rows, bs, dtype=torch.bool, device=device)
        else:
            # Unstructured: per-block saliency threshold
            sal = W_blk**2 / (Hinv_blk.diagonal().unsqueeze(0) ** 2 + 1e-30)
            thresh = sal.flatten().sort()[0][int(sal.numel() * sparsity)]
            mask = sal <= thresh

        for j in range(bs):
            w = W_blk[:, j]
            d = Hinv_blk[j, j]

            # N:M structured: determine mask at the start of each group
            if prune_n > 0 and (b_start + j) % prune_m == 0:
                m_end = min(j + prune_m, bs)
                m = m_end - j
                group = W_blk[:, j:m_end]
                diags = Hinv_blk.diagonal()[j:m_end]
                sal = group**2 / (diags.unsqueeze(0) ** 2 + 1e-30)
                # Mark the (M - N) smallest as pruned
                n_prune = m - min(prune_n, m)
                if n_prune > 0:
                    mask[:, j:m_end].scatter_(
                        1, sal.topk(n_prune, dim=1, largest=False).indices, True
                    )

            q = w.clone()
            q[mask[:, j]] = 0

            err = (w - q) / d
            Err[:, j] = err
            W_blk[:, j] = q
            if j + 1 < bs:
                W_blk[:, j + 1 :] -= err.unsqueeze(1) * Hinv_blk[j : j + 1, j + 1 :]

        W[:, b_start:b_end] = W_blk
        if b_end < cols:
            W[:, b_end:] -= Err @ Hinv[b_start:b_end, b_end:]

    return W


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class _Catcher(nn.Module):
    """Wraps a transformer layer to capture its inputs during forward."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.inputs = []
        self.kwargs = {}

    def forward(self, *args, **kwargs):
        self.inputs.append(args[0].detach())
        if not self.kwargs:
            self.kwargs = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }
        raise ValueError


@torch.no_grad()
def prune_model(
    model,
    calibration_tokens,
    sparsity=0.5,
    prune_n=2,
    prune_m=4,
    blocksize=128,
    percdamp=0.01,
    device="cuda:0",
    verbose=True,
):
    """Prune all linear layers in a transformer LM using SparseGPT.

    Full pipeline: capture layer inputs from calibration data, accumulate
    Hessians per sublayer, prune, and propagate corrected outputs to the
    next layer.

    Args:
        model: HuggingFace ``AutoModelForCausalLM`` (moved to *device*
            internally).
        calibration_tokens: [nsamples, seqlen] int tensor of token IDs.
        sparsity: Target sparsity for unstructured (ignored when prune_n>0).
        prune_n: N in N:M structured sparsity (0 = unstructured).
        prune_m: M in N:M structured sparsity.
        blocksize: OBS block size.
        percdamp: Diagonal damping fraction.
        device: Compute device.
        verbose: Print per-layer progress.

    Returns:
        The pruned model (modified in place).
    """
    model.eval()
    model = model.to(device)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_transformer_layers(model)
    nsamples = calibration_tokens.shape[0]

    # --- Capture inputs to the first decoder layer ---
    layers[0] = _Catcher(layers[0])
    for i in range(nsamples):
        try:
            model(calibration_tokens[i : i + 1].to(device))
        except ValueError:
            pass
    inps = layers[0].inputs  # list of [1, seqlen, hidden]
    layer_kwargs = layers[0].kwargs
    layers[0] = layers[0].layer  # restore original layer

    # Only keep kwargs the layer actually needs
    safe_keys = {"attention_mask", "position_ids", "cache_position"}
    layer_kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in layer_kwargs.items()
        if k in safe_keys
    }

    outs = [torch.empty_like(inp) for inp in inps]

    # --- Layer-by-layer pruning ---
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        sublayers = find_layers(layer)

        # Attach Hessian accumulators
        accumulators = {}
        for name, sub in sublayers.items():
            accumulators[name] = HessianAccumulator(sub)

        # Forward pass 1: accumulate Hessians
        for i in range(nsamples):
            outs[i] = layer(inps[i].to(device), **layer_kwargs)[0]

        for acc in accumulators.values():
            acc.remove()

        # Prune each sublayer
        for name, sub in sublayers.items():
            W = sub.weight.data.float()
            H = accumulators[name].H

            W_pruned = sparsegpt(
                W,
                H,
                sparsity=sparsity,
                prune_n=prune_n,
                prune_m=prune_m,
                blocksize=blocksize,
                percdamp=percdamp,
            )

            if verbose:
                rmse = (W - W_pruned).norm() / (W.norm() + 1e-12)
                nnz = (W_pruned != 0).float().mean()
                print(
                    f"  [{layer_idx}] {name:<25s}  " f"RMSE={rmse:.4f}  nnz={nnz:.2%}"
                )

            sub.weight.data = W_pruned.to(sub.weight.dtype)
            accumulators[name].free()

        # Forward pass 2: propagate corrected outputs
        for i in range(nsamples):
            outs[i] = layer(inps[i].to(device), **layer_kwargs)[0]

        inps, outs = outs, inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return model


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------


def get_calibration_data(tokenizer, nsamples=128, seqlen=2048, dataset="c4", seed=42):
    """Load calibration tokens from C4 or WikiText.

    Requires the ``datasets`` package.

    Args:
        tokenizer: HuggingFace tokenizer.
        nsamples: Number of calibration sequences.
        seqlen: Sequence length.
        dataset: ``"c4"`` or ``"wikitext"``.
        seed: Random seed for shuffling.

    Returns:
        tokens: [nsamples, seqlen] int64 tensor.
    """
    from datasets import load_dataset

    if dataset == "c4":
        data = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )
    elif dataset == "wikitext":
        data = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    import random

    rng = random.Random(seed)

    # Collect long-enough texts
    texts = []
    for example in data:
        text = example.get("text", "")
        if len(text) >= seqlen // 2:
            texts.append(text)
        if len(texts) >= nsamples * 4:
            break

    rng.shuffle(texts)

    samples = []
    for text in texts:
        tok = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seqlen + 64
        )
        if tok.input_ids.shape[1] >= seqlen:
            samples.append(tok.input_ids[:, :seqlen])
        if len(samples) >= nsamples:
            break

    if len(samples) < nsamples:
        raise RuntimeError(
            f"Only found {len(samples)} sequences of length {seqlen}, "
            f"need {nsamples}. Try a shorter seqlen or different dataset."
        )

    return torch.cat(samples, dim=0)
