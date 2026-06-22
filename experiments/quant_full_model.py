"""Full model INT4 H-Cal quantization, one decoder layer at a time.

Usage:
    python quant_full_model.py

For each of the 36 decoder layers:
  1. Collect activations from C4-EN + C4-ZH
  2. Build per-linear block Hessians
  3. Quantize all linears with INT4 H-Cal
  4. Every 3 layers, report PPL + KL

After all decoder layers, quantize lm_head with H-Cal and report final metrics.
"""

import gc
import os
import sys
import time

import torch
import torch.nn as nn

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"
torch.backends.cuda.enable_flash_sdp(True)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from custom_codebook import custom_optimal_hessian
from eval_ppl import evaluate

MODEL_NAME = "Qwen/Qwen3-4B"
TOP_P = 0.9
MAX_TOKENS = 91154
BLOCK_SIZE = 16
CHUNK_ROWS = 4096
CAL_TOKENS = 4096
REPORT_EVERY = 3

CAL_DATASETS = [
    ("allenai/c4", "en", "validation"),
    ("allenai/c4", "zh", "validation"),
]

EVAL_DATASETS = [
    ("wikitext", "wikitext", None, "test"),
    ("c4zh", "allenai/c4", "zh", "validation"),
]


def load_model(device):
    print(f"  Loading {MODEL_NAME} on {device}...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f"done ({time.time() - t0:.1f}s)", flush=True)
    return model


def get_cal_texts():
    """Load calibration texts once."""
    all_texts = []
    for ds_name, ds_config, ds_split in CAL_DATASETS:
        if ds_name == "wikitext":
            ds = load_dataset("EleutherAI/wikitext_document_level",
                              "wikitext-2-raw-v1", split=ds_split)
            texts = [doc["page"] for doc in ds if doc["page"].strip()]
        else:
            ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=True)
            texts = []
            for doc in ds:
                texts.append(doc["text"])
                if len(texts) >= 200:
                    break
        all_texts.append((ds_config or ds_name, texts))
    return all_texts


def collect_layer_activations(model, tokenizer, layer_idx, device, cal_texts):
    """Collect input activations for each linear in a decoder layer."""
    layer = model.model.layers[layer_idx]
    activations = {}
    hooks = []

    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            activations[name] = []

            def make_hook(n):
                def hook_fn(mod, inp, out):
                    activations[n].append(
                        inp[0].detach().reshape(-1, inp[0].shape[-1])
                    )
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

    for label, texts in cal_texts:
        total_tokens = 0
        for text in texts:
            if total_tokens >= CAL_TOKENS:
                break
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, return_tensors="pt",
                                       truncation=True, max_length=2048).to(device)
            if tokens.shape[1] == 0:
                continue
            with torch.no_grad():
                model(tokens)
            total_tokens += tokens.shape[1]

    for h in hooks:
        h.remove()

    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)

    return activations


def collect_lmhead_activations(model, tokenizer, device, cal_texts):
    """Collect input activations for lm_head."""
    activations = []

    def hook_fn(mod, inp, out):
        activations.append(inp[0].detach().reshape(-1, inp[0].shape[-1]))

    handle = model.lm_head.register_forward_hook(hook_fn)

    for label, texts in cal_texts:
        total_tokens = 0
        for text in texts:
            if total_tokens >= CAL_TOKENS:
                break
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, return_tensors="pt",
                                       truncation=True, max_length=2048).to(device)
            if tokens.shape[1] == 0:
                continue
            with torch.no_grad():
                model(tokens)
            total_tokens += tokens.shape[1]

    handle.remove()
    return torch.cat(activations, dim=0)


def build_hblocks(X, block_size, device):
    """Build block Hessians from activations."""
    in_feat = X.shape[1]
    n = X.shape[0]
    if in_feat % block_size != 0:
        return None
    num_blocks = in_feat // block_size
    H_blocks = torch.zeros(num_blocks, block_size, block_size, device=device)
    X = X.float().to(device)
    for j in range(num_blocks):
        Xj = X[:, j * block_size : (j + 1) * block_size]
        H_blocks[j] = (Xj.T @ Xj) / n
    return H_blocks


def quantize_linear_hcal(module, cb, bd, H_blocks, block_size):
    """Quantize a linear layer with H-Cal."""
    W = module.weight.data.float()
    out_feat, in_feat = W.shape

    if in_feat % block_size != 0 or H_blocks is None:
        return None

    W_blocked = W.reshape(out_feat, in_feat // block_size, block_size)
    t0 = time.time()
    dq_parts = []
    for r0 in range(0, out_feat, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_feat)
        res = custom_optimal_hessian(W_blocked[r0:r1], cb, bd,
                                      return_dequant=True, H_blocks=H_blocks)
        dq_parts.append(res[2])
        del res
    W_dq = torch.cat(dq_parts, dim=0)
    t_q = time.time() - t0
    module.weight.data = W_dq.reshape(out_feat, in_feat).to(module.weight.dtype)
    return t_q


def run_eval(model, tokenizer, model_ref):
    """Evaluate on all datasets, return dict of results."""
    results = {}
    for ds_label, ds_name, ds_config, ds_split in EVAL_DATASETS:
        metrics = evaluate(model, tokenizer, model_ref=model_ref, top_p=TOP_P,
                           max_length=2048, max_tokens=MAX_TOKENS,
                           dataset_name=ds_name, dataset_config=ds_config,
                           split=ds_split)
        results[ds_label] = metrics
    return results


def print_results(all_results):
    print(f"\n{'=' * 90}")
    print(f"{'Checkpoint':<25} {'byte_ppl':>10} {'bpb':>8} {'KL_mean':>10} {'dataset':>10}")
    print(f"{'-' * 90}")
    for label, ds_results in all_results.items():
        for ds_label, m in ds_results.items():
            bppl = f"{m['byte_perplexity']:.4f}"
            bpb = f"{m['bits_per_byte']:.4f}"
            kl = f"{m.get('kl_mean', 0):.6f}" if 'kl_mean' in m else "-"
            print(f"{label:<25} {bppl:>10} {bpb:>8} {kl:>10} {ds_label:>10}")
    print(f"{'=' * 90}")
    sys.stdout.flush()


def main():
    DEVICE_REF = "cuda:0"
    DEVICE_QUANT = "cuda:1"

    num_layers = 36  # Qwen3-4B

    print(f"Full model INT4 H-Cal quantization: {MODEL_NAME}")
    print(f"  BS={BLOCK_SIZE}  top_p={TOP_P}  report every {REPORT_EVERY} layers")
    print(f"  ref={DEVICE_REF}  quant={DEVICE_QUANT}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # INT4 codebook
    cb = torch.linspace(0, 1, 8, device=DEVICE_QUANT)
    bd = torch.empty(7, device=DEVICE_QUANT)
    bd[0] = cb[1] / 2
    for k in range(1, 7):
        bd[k] = (cb[k] + cb[k + 1]) / 2

    # Load calibration texts once
    print("[Loading calibration texts]", flush=True)
    cal_texts = get_cal_texts()
    for label, texts in cal_texts:
        print(f"  {label}: {len(texts)} docs", flush=True)

    # Load models
    print("\n[Reference model]", flush=True)
    model_ref = load_model(DEVICE_REF)

    print("\n[Quant model]", flush=True)
    model = load_model(DEVICE_QUANT)

    all_results = {}
    total_quant_time = 0

    # Quantize decoder layers
    for li in range(num_layers):
        print(f"\n--- Layer {li}/{num_layers} ---", flush=True)

        # Collect activations
        act_dict = collect_layer_activations(model, tokenizer, li, DEVICE_QUANT,
                                              cal_texts)

        # Build H_blocks and quantize each linear
        layer = model.model.layers[li]
        layer_t = 0
        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            X = act_dict.get(name)
            if X is None:
                continue
            H_blocks = build_hblocks(X, BLOCK_SIZE, DEVICE_QUANT)
            t = quantize_linear_hcal(module, cb, bd, H_blocks, BLOCK_SIZE)
            if t is not None:
                layer_t += t
                print(f"  {name}: {t:.1f}s", flush=True)

        total_quant_time += layer_t
        del act_dict
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  Layer {li} done ({layer_t:.1f}s, total={total_quant_time:.1f}s)", flush=True)

        # Report every N layers
        if (li + 1) % REPORT_EVERY == 0:
            label = f"L0-{li}"
            print(f"\n[Eval after {li+1} layers]", flush=True)
            all_results[label] = run_eval(model, tokenizer, model_ref)
            for ds_label, m in all_results[label].items():
                kl = f"{m.get('kl_mean', 0):.6f}" if 'kl_mean' in m else "-"
                print(f"  {ds_label}: byte_ppl={m['byte_perplexity']:.4f}  KL={kl}", flush=True)
            print_results(all_results)

    # Final eval after all decoder layers (if not already done)
    if num_layers % REPORT_EVERY != 0:
        label = f"L0-{num_layers-1}"
        print(f"\n[Eval after all {num_layers} layers]", flush=True)
        all_results[label] = run_eval(model, tokenizer, model_ref)
        print_results(all_results)

    # Quantize lm_head
    print(f"\n--- lm_head ---", flush=True)
    X_lmhead = collect_lmhead_activations(model, tokenizer, DEVICE_QUANT, cal_texts)
    print(f"  lm_head activations: {X_lmhead.shape}", flush=True)
    H_blocks_lmhead = build_hblocks(X_lmhead, BLOCK_SIZE, DEVICE_QUANT)
    del X_lmhead
    print(f"  H_blocks: {H_blocks_lmhead.shape}", flush=True)

    W = model.lm_head.weight.data.float()
    out_feat, in_feat = W.shape
    W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)
    t0 = time.time()
    dq_parts = []
    for r0 in range(0, out_feat, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_feat)
        res = custom_optimal_hessian(W_blocked[r0:r1], cb, bd,
                                      return_dequant=True, H_blocks=H_blocks_lmhead)
        dq_parts.append(res[2])
        del res
    W_dq = torch.cat(dq_parts, dim=0)
    t_lmhead = time.time() - t0
    model.lm_head.weight.data = W_dq.reshape(out_feat, in_feat).to(model.lm_head.weight.dtype)
    total_quant_time += t_lmhead
    print(f"  lm_head quantized in {t_lmhead:.1f}s", flush=True)
    del H_blocks_lmhead, W_dq
    gc.collect()
    torch.cuda.empty_cache()

    # Final eval
    label = "All + lm_head"
    print(f"\n[Final eval]", flush=True)
    all_results[label] = run_eval(model, tokenizer, model_ref)
    print(f"\nTotal quantization time: {total_quant_time:.1f}s", flush=True)
    print_results(all_results)


if __name__ == "__main__":
    main()
