"""H-Optimal lm_head: attn (q/k/v) vs MLP (gate/up) Hessian per layer.

Usage:
    python hessian_attn_vs_mlp.py INT4 --device cuda:0

For each decoder layer, builds H_blocks from attn projections (q/k/v_proj)
and MLP projections (gate/up_proj) separately, quantizes lm_head with each,
and reports KL divergence to compare which group provides a better Hessian.

Single GPU, weight-swap approach. Results saved incrementally.
"""

import argparse
import gc
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
from qwantize.nvfp4.reference import nvfp4_optimal_hessian
from qwantize.mxfp4.reference import mxfp4_optimal_hessian
from custom_codebook import custom_optimal_hessian
from eval_ppl import evaluate

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
CHUNK_ROWS = 4096
TOP_P = 0.9
MAX_DOCS = 20

PROJ_GROUPS = {
    "attn": ["q_proj", "k_proj", "v_proj"],
    "mlp": ["gate_proj", "up_proj"],
}


def get_quant_fn(method, device):
    if method == "NVFP4":
        return lambda W, H_blocks: nvfp4_optimal_hessian(W, return_dequant=True, H_blocks=H_blocks)
    elif method == "INT4":
        cb = torch.linspace(0, 1, 8, device=device)
        bd = torch.empty(7, device=device)
        bd[0] = cb[1] / 2
        for k in range(1, 7):
            bd[k] = (cb[k] + cb[k + 1]) / 2
        return lambda W, H_blocks: custom_optimal_hessian(W, cb, bd, return_dequant=True, H_blocks=H_blocks)
    elif method == "MXFP4":
        return lambda W, H_blocks: mxfp4_optimal_hessian(W, return_dequant=True, H_blocks=H_blocks)
    else:
        raise ValueError(f"Unknown method: {method}. Use NVFP4, INT4, or MXFP4.")


def results_file(method):
    return os.path.join(os.path.dirname(__file__), f"hessian_attn_vs_mlp_{method}_results.txt")


def load_results(path):
    results = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    results[parts[0]] = json.loads(parts[1])
    return results


def save_result(path, label, metrics):
    with open(path, "a") as f:
        f.write(f"{label}\t{json.dumps(metrics)}\n")


def init_results_file(path, method, device):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            f.write(f"# {method} H-Optimal lm_head: attn vs MLP Hessian\n")
            f.write(f"# model: {MODEL_NAME}  block_size: {BLOCK_SIZE}  top_p: {TOP_P}  max_docs: {MAX_DOCS}\n")
            f.write(f"# device: {device}\n")
            f.write(f"# label\tmetrics_json\n")


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


def print_summary(done):
    print(f"\n{'=' * 100}")
    print(f"{'Label':<16} {'word_ppl':>10} {'byte_ppl':>10} {'bpb':>8} {'KL_mean':>10} {'KL_median':>10} {'time':>8}")
    print(f"{'-' * 100}")
    # Sort by layer number, then group
    def sort_key(label):
        parts = label.split("_")
        layer = int(parts[0][1:])
        group = parts[1] if len(parts) > 1 else ""
        return (layer, group)
    for label in sorted(done.keys(), key=sort_key):
        m = done[label]
        kl = f"{m['kl_mean']:.6f}" if 'kl_mean' in m else "-"
        kl_med = f"{m['kl_median']:.6f}" if 'kl_median' in m else "-"
        t = f"{m.get('elapsed_s', 0):.0f}s"
        print(f"{label:<16} {m['word_perplexity']:>10.4f} {m['byte_perplexity']:>10.4f} "
              f"{m['bits_per_byte']:>8.4f} {kl:>10} {kl_med:>10} {t:>8}")
    print(f"{'=' * 100}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="H-Optimal lm_head: attn vs MLP Hessian")
    parser.add_argument("method", choices=["NVFP4", "INT4", "MXFP4"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    method = args.method
    device = args.device

    rfile = results_file(method)
    init_results_file(rfile, method, device)

    print(f"{method} H-Optimal lm_head: attn vs MLP Hessian comparison")
    print(f"Model: {MODEL_NAME}  device={device}  BS={BLOCK_SIZE}  top_p={TOP_P}  max_docs={MAX_DOCS}")
    print(f"Results: {rfile}\n", flush=True)

    done = load_results(rfile)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quant_fn = get_quant_fn(method, device)

    # Load model once
    print(f"[Loading model on {device}]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    out_feat, in_feat = model.lm_head.weight.shape
    orig_lm_head_weight = model.lm_head.weight.data.clone()
    print(f"  done ({num_layers} layers, lm_head {out_feat}x{in_feat})", flush=True)

    if done:
        print_summary(done)

    for li in range(num_layers):
        for group_name, proj_names in PROJ_GROUPS.items():
            label = f"L{li}_{group_name}"
            if label in done:
                m = done[label]
                print(f"  [{label}] kl={m.get('kl_mean', 0):.6f}  (cached)", flush=True)
                continue

            print(f"\n[{label}] ({', '.join(proj_names)})", flush=True)

            print(f"  Building Hessian from layer {li} {group_name}...", end=" ", flush=True)
            H_blocks = build_hessian_from_layer(model, li, device, proj_names)
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
            metrics["proj_group"] = group_name
            metrics["proj_names"] = proj_names

            save_result(rfile, label, metrics)
            done[label] = metrics

            print(f"  => kl={metrics['kl_mean']:.6f}  word_ppl={metrics['word_perplexity']:.4f}", flush=True)

            del W, W_blocked, W_dq, H_blocks
            torch.cuda.empty_cache()

    # Restore original weights
    model.lm_head.weight.data = orig_lm_head_weight

    # Final summary
    print_summary(done)

    # Per-layer comparison
    print(f"\n{'=' * 60}")
    print(f"{'Layer':<8} {'attn KL':>12} {'mlp KL':>12} {'winner':>8}")
    print(f"{'-' * 60}")
    for li in range(num_layers):
        attn = done.get(f"L{li}_attn", {})
        mlp = done.get(f"L{li}_mlp", {})
        attn_kl = attn.get("kl_mean", float("inf"))
        mlp_kl = mlp.get("kl_mean", float("inf"))
        winner = "attn" if attn_kl < mlp_kl else "mlp"
        print(f"L{li:<7} {attn_kl:>12.6f} {mlp_kl:>12.6f} {winner:>8}")
    print(f"{'=' * 60}")

    best_label = min(done.keys(), key=lambda k: done[k].get("kl_mean", float("inf")))
    best = done[best_label]
    print(f"\nBest overall: {best_label}  kl={best['kl_mean']:.6f}  "
          f"word_ppl={best['word_perplexity']:.4f}")


if __name__ == "__main__":
    main()
