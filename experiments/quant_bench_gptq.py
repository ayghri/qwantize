"""lm_head quantization benchmark with GPTQ on C4 Chinese + wikitext.

Compares: NVFP4 Optimal, INT4 Optimal, GPTQ+NVFP4, GPTQ+INT4
Reference model on cuda:0, quantized model on cuda:1.

Results saved to experiments/quant_bench_gptq_results.txt (skips completed runs).
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
from custom_codebook import custom_optimal
from qwantize.nvfp4.reference import nvfp4_optimal
from eval_ppl import evaluate
from gptq_lmhead import (collect_lmhead_activations, gptq_quantize_lmhead,
                          make_nvfp4_block_fn, make_int4_block_fn,
                          make_int4_hessian_block_fn)

MODEL_NAME = "Qwen/Qwen3-4B"
DEVICE_REF = "cuda:0"
DEVICE_QUANT = "cuda:1"
BLOCK_SIZE = 16
CHUNK_ROWS = 16384
TOP_P = 0.9
MAX_TOKENS = 91154
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "quant_bench_gptq_results.txt")


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
    print(f"\n{'=' * 100}")
    print(f"{'Method':<35} {'byte_ppl':>10} {'bpb':>8} {'KL_mean':>10} {'time':>8} {'dataset':>10}")
    print(f"{'-' * 100}")
    for label, m in results.items():
        kl = f"{m.get('kl_mean', 0):.6f}" if 'kl_mean' in m else "-"
        t = f"{m.get('elapsed_s', 0):.0f}s" if 'elapsed_s' in m else "-"
        bppl = f"{m['byte_perplexity']:.4f}" if 'byte_perplexity' in m else "-"
        bpb = f"{m['bits_per_byte']:.4f}" if 'bits_per_byte' in m else "-"
        ds = m.get('dataset', '-')
        print(f"{label:<35} {bppl:>10} {bpb:>8} {kl:>10} {t:>8} {ds:>10}")
    print(f"{'=' * 100}")
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


def quantize_lm_head_simple(model, quant_fn):
    """Quantize lm_head with block-wise optimal scale search (no GPTQ)."""
    W = model.lm_head.weight.data.float()
    out_feat, in_feat = W.shape
    W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)

    t0 = time.time()
    chunk_sz = CHUNK_ROWS
    if out_feat <= chunk_sz:
        result = quant_fn(W_blocked, return_dequant=True)
        W_dq = result[2]
    else:
        dq_parts = []
        for r0 in range(0, out_feat, chunk_sz):
            r1 = min(r0 + chunk_sz, out_feat)
            res = quant_fn(W_blocked[r0:r1], return_dequant=True)
            dq_parts.append(res[2])
            del res
            torch.cuda.empty_cache()
        W_dq = torch.cat(dq_parts, dim=0)
    t_q = time.time() - t0

    model.lm_head.weight.data = W_dq.reshape(out_feat, in_feat).to(model.lm_head.weight.dtype)
    return t_q


def build_hessian_from_layer(model, layer_idx, device, block_size=BLOCK_SIZE):
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


def run_eval(model, tokenizer, model_ref, dataset_name, dataset_config, split):
    return evaluate(model, tokenizer, model_ref=model_ref, top_p=TOP_P,
                    max_length=2048, max_tokens=MAX_TOKENS,
                    dataset_name=dataset_name, dataset_config=dataset_config,
                    split=split)


def main():
    print(f"Quantization Benchmark (with GPTQ): {MODEL_NAME}")
    print(f"  ref={DEVICE_REF}  quant={DEVICE_QUANT}  BS={BLOCK_SIZE}  top_p={TOP_P}")
    print(f"  results: {RESULTS_FILE}\n", flush=True)

    done = load_results()
    if done:
        print_results(done)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # INT4 codebook
    cb_int4 = torch.linspace(0, 1, 8, device=DEVICE_QUANT)
    bd_int4 = torch.empty(7, device=DEVICE_QUANT)
    bd_int4[0] = cb_int4[1] / 2
    for i in range(1, 7):
        bd_int4[i] = (cb_int4[i] + cb_int4[i + 1]) / 2

    # Reference model
    print("[Reference model]", flush=True)
    model_ref = load_model(DEVICE_REF)

    # Datasets to evaluate on
    eval_datasets = [
        ("wikitext", "wikitext", None, "test"),
        ("c4zh", "allenai/c4", "zh", "validation"),
    ]

    # Methods: (label_prefix, setup_fn)
    # setup_fn(model) -> t_q
    methods = []

    # 1. NVFP4 Optimal (no GPTQ)
    def setup_nvfp4(model):
        print(f"  Quantizing lm_head NVFP4 Optimal...", flush=True)
        return quantize_lm_head_simple(
            model, lambda W, return_dequant=True: nvfp4_optimal(W, return_dequant=return_dequant))
    methods.append(("NVFP4 Optimal", setup_nvfp4))

    # 2. INT4 Optimal (no GPTQ)
    def setup_int4(model):
        print(f"  Quantizing lm_head INT4 Optimal...", flush=True)
        return quantize_lm_head_simple(
            model, lambda W, return_dequant=True: custom_optimal(W, cb_int4, bd_int4, return_dequant=return_dequant))
    methods.append(("INT4 Optimal", setup_int4))

    # 3. GPTQ + NVFP4
    def setup_gptq_nvfp4(model):
        print(f"  Collecting calibration data (wikitext)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="wikitext",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_nvfp4_block_fn(BLOCK_SIZE, DEVICE_QUANT)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+NVFP4", setup_gptq_nvfp4))

    # 4. GPTQ + INT4
    def setup_gptq_int4(model):
        print(f"  Collecting calibration data (wikitext)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="wikitext",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_int4_block_fn(BLOCK_SIZE, DEVICE_QUANT)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+INT4", setup_gptq_int4))

    # 5. GPTQ + NVFP4 (calibrated on C4 Chinese)
    def setup_gptq_nvfp4_c4zh(model):
        print(f"  Collecting calibration data (C4 Chinese)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="allenai/c4",
                                        dataset_config="zh",
                                        split="validation",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_nvfp4_block_fn(BLOCK_SIZE, DEVICE_QUANT)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+NVFP4 (c4zh cal)", setup_gptq_nvfp4_c4zh))

    # 6. GPTQ + INT4 (calibrated on C4 Chinese)
    def setup_gptq_int4_c4zh(model):
        print(f"  Collecting calibration data (C4 Chinese)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="allenai/c4",
                                        dataset_config="zh",
                                        split="validation",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_int4_block_fn(BLOCK_SIZE, DEVICE_QUANT)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+INT4 (c4zh cal)", setup_gptq_int4_c4zh))

    # Build H_blocks L0 for H-Optimal variants
    print("\n[Building H_blocks L0]", flush=True)
    model_tmp = load_model(DEVICE_QUANT)
    H_blocks_L0 = build_hessian_from_layer(model_tmp, 0, DEVICE_QUANT)
    print(f"  H_blocks L0: {H_blocks_L0.shape}", flush=True)
    del model_tmp; gc.collect(); torch.cuda.empty_cache()

    # 7. GPTQ + INT4 H-Optimal L0 (wiki calibration)
    def setup_gptq_int4h_wiki(model):
        print(f"  Collecting calibration data (wikitext)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="wikitext",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_int4_hessian_block_fn(BLOCK_SIZE, DEVICE_QUANT, H_blocks_L0)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+INT4-H L0", setup_gptq_int4h_wiki))

    # 8. GPTQ + INT4 H-Optimal L0 (c4zh calibration)
    def setup_gptq_int4h_c4zh(model):
        print(f"  Collecting calibration data (C4 Chinese)...", flush=True)
        X = collect_lmhead_activations(model, tokenizer, dataset_name="allenai/c4",
                                        dataset_config="zh", split="validation",
                                        max_tokens=8192, device=DEVICE_QUANT)
        block_fn = make_int4_hessian_block_fn(BLOCK_SIZE, DEVICE_QUANT, H_blocks_L0)
        return gptq_quantize_lmhead(model, X, block_fn, block_size=BLOCK_SIZE)
    methods.append(("GPTQ+INT4-H L0 (c4zh cal)", setup_gptq_int4h_c4zh))

    for mi, (method_name, setup_fn) in enumerate(methods):
        for ds_label, ds_name, ds_config, ds_split in eval_datasets:
            label = f"{method_name} [{ds_label}]"
            if label in done:
                m = done[label]
                print(f"\n  [{mi+1}/{len(methods)}] {label}: kl={m.get('kl_mean', 'N/A')} (cached)", flush=True)
                continue

            print(f"\n{'=' * 60}")
            print(f"[{mi+1}/{len(methods)}] {label}")
            print(f"{'=' * 60}", flush=True)

            model = load_model(DEVICE_QUANT)
            t_q = setup_fn(model)

            print(f"  Evaluating on {ds_label}...", flush=True)
            results = run_eval(model, tokenizer, model_ref, ds_name, ds_config, ds_split)
            results["quant_time_s"] = t_q
            results["dataset"] = ds_label

            save_result(label, results)
            done[label] = results
            print_results(done)

            del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\n\nDONE")
    print_results(done)


if __name__ == "__main__":
    main()
