"""Compare MLP Hessian from specific layers for INT4 H-Optimal lm_head.

Usage:
    python hessian_mlp_compare.py --device cuda:0 --layers 0 8
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"
torch.backends.cuda.enable_flash_sdp(True)

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from custom_codebook import custom_optimal_hessian
from eval_ppl import evaluate

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
CHUNK_ROWS = 4096
TOP_P = 0.9
MAX_DOCS = 20
MLP_PROJS = ["gate_proj", "up_proj"]


def build_hessian_from_layer(model, layer_idx, device, proj_names, block_size=BLOCK_SIZE):
    hidden_dim = model.lm_head.weight.shape[1]
    num_col_blocks = hidden_dim // block_size
    H_blocks = torch.zeros(num_col_blocks, block_size, block_size, device=device)
    n = 0
    prefix = f"model.layers.{layer_idx}."
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        if not any(p in name for p in proj_names):
            continue
        if module.weight.shape[1] != hidden_dim:
            continue
        W = module.weight.data.float()
        for j in range(num_col_blocks):
            Wj = W[:, j * block_size : (j + 1) * block_size]
            H_blocks[j].addmm_(Wj.T, Wj)
        n += 1
    if n > 0:
        H_blocks /= n
    return H_blocks


def quantize_chunked(W_blocked, H_blocks, quant_fn, chunk_rows=CHUNK_ROWS):
    out_feat = W_blocked.shape[0]
    dq_parts = []
    for r0 in range(0, out_feat, chunk_rows):
        r1 = min(r0 + chunk_rows, out_feat)
        res = quant_fn(W_blocked[r0:r1], H_blocks)
        dq_parts.append(res[2])
        del res
        torch.cuda.empty_cache()
    return torch.cat(dq_parts, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 8])
    args = parser.parse_args()
    device = args.device
    layers = args.layers

    print(f"INT4 H-Optimal lm_head: MLP Hessian comparison")
    print(f"Layers: {layers}  device={device}  BS={BLOCK_SIZE}  top_p={TOP_P}  max_docs={MAX_DOCS}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cb = torch.linspace(0, 1, 8, device=device)
    bd = torch.empty(7, device=device)
    bd[0] = cb[1] / 2
    for k in range(1, 7):
        bd[k] = (cb[k] + cb[k + 1]) / 2
    quant_fn = lambda W, H_blocks: custom_optimal_hessian(W, cb, bd, return_dequant=True, H_blocks=H_blocks)

    print(f"[Loading model on {device}]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    out_feat, in_feat = model.lm_head.weight.shape
    orig_lm_head_weight = model.lm_head.weight.data.clone()
    print(f"  done (lm_head {out_feat}x{in_feat})\n", flush=True)

    results = {}
    for li in layers:
        label = f"L{li}_mlp"
        print(f"[{label}] Building Hessian...", end=" ", flush=True)
        H_blocks = build_hessian_from_layer(model, li, device, MLP_PROJS)
        print("done", flush=True)

        W = orig_lm_head_weight.float()
        W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)

        print(f"  Quantizing...", end=" ", flush=True)
        t0 = time.time()
        W_dq = quantize_chunked(W_blocked, H_blocks, quant_fn)
        t_q = time.time() - t0
        model.lm_head.weight.data = W_dq.reshape(out_feat, in_feat).to(orig_lm_head_weight.dtype)
        print(f"done ({t_q:.1f}s)", flush=True)

        print(f"  Evaluating...", flush=True)
        metrics = evaluate(model, tokenizer, ref_lm_head_weight=orig_lm_head_weight,
                           top_p=TOP_P, max_length=2048, max_docs=MAX_DOCS)
        metrics["quant_time_s"] = t_q
        results[label] = metrics

        print(f"  => kl={metrics['kl_mean']:.6f}  word_ppl={metrics['word_perplexity']:.4f}\n", flush=True)

        del W, W_blocked, W_dq, H_blocks
        torch.cuda.empty_cache()

    model.lm_head.weight.data = orig_lm_head_weight

    print(f"\n{'=' * 60}")
    print(f"{'Label':<16} {'KL_mean':>12} {'KL_median':>12} {'word_ppl':>10}")
    print(f"{'-' * 60}")
    for label, m in results.items():
        print(f"{label:<16} {m['kl_mean']:>12.6f} {m['kl_median']:>12.6f} {m['word_perplexity']:>10.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
