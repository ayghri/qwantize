"""lm_head quantization benchmark: PPL + KL divergence.

Reference model on cuda:0, quantized model on cuda:1.
Uses eval_ppl.evaluate() for all metrics in one pass.

Results saved to experiments/quant_bench_results.txt (skips completed runs).
"""

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
from custom_codebook import custom_optimal, custom_optimal_hessian
from qwantize.nvfp4.reference import nvfp4_optimal, nvfp4_optimal_hessian
from qwantize.nvint4.reference import nvint4_optimal, nvint4_optimal_hessian
from eval_ppl import evaluate

MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE_REF = "cuda:0"
DEVICE_QUANT = "cuda:1"
BLOCK_SIZE = 16
CHUNK_ROWS = 16384
TOP_P = 0.9
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "quant_bench_results.txt")


def load_results():
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    results[parts[0]] = json.loads(parts[1])
    return results


def save_result(label, metrics):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{label}\t{json.dumps(metrics)}\n")


def print_results(results):
    print(f"\n{'=' * 90}")
    print(f"{'Method':<30} {'word_ppl':>10} {'byte_ppl':>10} {'bpb':>8} {'KL_mean':>10} {'time':>8}")
    print(f"{'-' * 90}")
    for label, m in results.items():
        kl = f"{m.get('kl_mean', 0):.6f}" if 'kl_mean' in m else "-"
        t = f"{m.get('elapsed_s', 0):.0f}s" if 'elapsed_s' in m else "-"
        print(f"{label:<30} {m['word_perplexity']:>10.4f} {m['byte_perplexity']:>10.4f} "
              f"{m['bits_per_byte']:>8.4f} {kl:>10} {t:>8}")
    print(f"{'=' * 90}")
    sys.stdout.flush()


def load_model(device):
    print(f"  Loading {MODEL_NAME} on {device}...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print(f"done ({time.time() - t0:.1f}s)", flush=True)
    return model


def build_hessian_from_layer(model, layer_idx, block_size=BLOCK_SIZE):
    hidden_dim = model.lm_head.weight.shape[1]
    num_col_blocks = hidden_dim // block_size
    H_blocks = torch.zeros(num_col_blocks, block_size, block_size, device=DEVICE_QUANT)
    n = 0
    prefix = f"model.layers.{layer_idx}."
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        if not any(p in name for p in PROJ_NAMES):
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


def quantize_chunked(W_blocked, quant_fn, chunk_rows=CHUNK_ROWS):
    out_feat = W_blocked.shape[0]
    n_chunks = (out_feat + chunk_rows - 1) // chunk_rows
    dq_parts = []
    for ci, r0 in enumerate(range(0, out_feat, chunk_rows)):
        r1 = min(r0 + chunk_rows, out_feat)
        res = quant_fn(W_blocked[r0:r1], return_dequant=True)
        dq_parts.append(res[2])
        del res
        torch.cuda.empty_cache()
    return torch.cat(dq_parts, dim=0)


def quantize_lm_head(model, quant_fn, method_name):
    """Quantize lm_head in-place. Returns quantization time."""
    W = model.lm_head.weight.data.float()
    out_feat, in_feat = W.shape
    W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)

    print(f"  Quantizing lm_head ({out_feat}x{in_feat})...", flush=True)
    t0 = time.time()
    chunk_sz = 4096 if "H-Optimal" in method_name else CHUNK_ROWS
    if out_feat <= chunk_sz:
        result = quant_fn(W_blocked, return_dequant=True)
        W_dq = result[2]
    else:
        W_dq = quantize_chunked(W_blocked, quant_fn, chunk_rows=chunk_sz)
    t_q = time.time() - t0

    model.lm_head.weight.data = W_dq.reshape(out_feat, in_feat).to(model.lm_head.weight.dtype)
    print(f"  Quantized in {t_q:.1f}s", flush=True)
    return t_q


def main():
    print(f"Quantization Benchmark: {MODEL_NAME}")
    print(f"  ref={DEVICE_REF}  quant={DEVICE_QUANT}  BS={BLOCK_SIZE}  top_p={TOP_P}")
    print(f"  results: {RESULTS_FILE}\n", flush=True)

    done = load_results()
    if done:
        print_results(done)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Reference model stays on cuda:0 for the entire run
    print("[Reference model]", flush=True)
    model_ref = load_model(DEVICE_REF)

    # Baseline (known from previous runs)
    if "Baseline" not in done:
        baseline = {
            "word_perplexity": 18.4559,
            "byte_perplexity": 1.7249,
            "bits_per_byte": 0.7865,
        }
        save_result("Baseline", baseline)
        done["Baseline"] = baseline
        print_results(done)

    # Build H_blocks from best layers (NVFP4: L23, INT4: L0)
    print("\n[Building Hessians]", flush=True)
    model_tmp = load_model(DEVICE_QUANT)
    H_blocks_L23 = build_hessian_from_layer(model_tmp, 23)
    print(f"  H_blocks L23: {H_blocks_L23.shape}", flush=True)
    H_blocks_L0 = build_hessian_from_layer(model_tmp, 0)
    print(f"  H_blocks L0: {H_blocks_L0.shape}", flush=True)
    del model_tmp; gc.collect(); torch.cuda.empty_cache()

    # INT4 codebook
    cb_int4 = torch.linspace(0, 1, 8, device=DEVICE_QUANT)
    bd_int4 = torch.empty(7, device=DEVICE_QUANT)
    bd_int4[0] = cb_int4[1] / 2
    for i in range(1, 7):
        bd_int4[i] = (cb_int4[i] + cb_int4[i + 1]) / 2

    # --- Methods ---
    methods = [
        ("NVFP4 Optimal",
         lambda W, return_dequant=True: nvfp4_optimal(W, return_dequant=return_dequant)),
        ("NVFP4 H-Optimal L23",
         lambda W, return_dequant=True: nvfp4_optimal_hessian(W, return_dequant=return_dequant, H_blocks=H_blocks_L23)),
        ("INT4 Optimal",
         lambda W, return_dequant=True: custom_optimal(W, cb_int4, bd_int4, return_dequant=return_dequant)),
        ("INT4 H-Optimal L0",
         lambda W, return_dequant=True: custom_optimal_hessian(W, cb_int4, bd_int4, return_dequant=return_dequant, H_blocks=H_blocks_L0)),
        ("NVINT4 Optimal",
         lambda W, return_dequant=True: nvint4_optimal(W, return_dequant=return_dequant)),
        ("NVINT4 H-Optimal L0",
         lambda W, return_dequant=True: nvint4_optimal_hessian(W, return_dequant=return_dequant, H_blocks=H_blocks_L0)),
    ]

    for mi, (method_name, quant_fn) in enumerate(methods):
        label = f"{method_name} lm_head"
        if label in done:
            m = done[label]
            print(f"\n  [{mi+1}/{len(methods)}] {label}: word_ppl={m['word_perplexity']:.4f} (cached)", flush=True)
            continue

        print(f"\n{'=' * 60}")
        print(f"[{mi+1}/{len(methods)}] {label}")
        print(f"{'=' * 60}", flush=True)

        model = load_model(DEVICE_QUANT)
        t_q = quantize_lm_head(model, quant_fn, method_name)

        print(f"  Evaluating (PPL + KL)...", flush=True)
        results = evaluate(model, tokenizer, model_ref=model_ref, top_p=TOP_P, max_length=2048)
        results["quant_time_s"] = t_q

        save_result(label, results)
        done[label] = results
        print_results(done)

        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n\nDONE")
    print_results(done)


if __name__ == "__main__":
    main()
