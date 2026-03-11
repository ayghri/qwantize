"""lm_head quantization evaluation: Qwen3-4B wikitext perplexity.

For H-Optimal: the lm_head input is the hidden state. We approximate it
by sampling rows from the transformer's projection weight matrices
(q/k/v/o_proj, gate/up/down_proj), since those rows represent the
directions the model actually uses to read the hidden state.

Results saved to experiments/quant_eval_results.txt (skips completed runs).
"""

import os
import sys
import time

import torch
import torch.nn as nn

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

sys.path.insert(0, os.path.dirname(__file__))
from custom_codebook import (
    learn_codebook,
    custom_naive,
    custom_optimal,
    custom_optimal_hessian,
)
from qwantize.nvfp4.reference import nvfp4_naive, nvfp4_optimal, nvfp4_optimal_hessian

DEVICE = "cuda:1"
MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
CHUNK_ROWS = 16384
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "quant_eval_results.txt")


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
                    results[parts[0]] = {"ppl": float(parts[1]),
                                         "quant_time": parts[2] if len(parts) > 2 else "-"}
    return results


def save_result(label, ppl, quant_time="-"):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{label}\t{ppl:.4f}\t{quant_time}\n")


def print_results(results):
    print(f"\n{'=' * 60}")
    print(f"{'Method':<30} {'PPL':>10} {'Time':>10}")
    print(f"{'-' * 60}")
    for label, info in results.items():
        print(f"{label:<30} {info['ppl']:>10.4f} {info['quant_time']:>10}")
    print(f"{'=' * 60}")
    sys.stdout.flush()


def make_int4_codebook(device):
    cb = torch.linspace(0, 1, 8, device=device)
    bd = torch.empty(7, device=device)
    bd[0] = cb[1] / 2
    for k in range(1, 7):
        bd[k] = (cb[k] + cb[k + 1]) / 2
    return cb, bd


def quantize_chunked(W_blocked, quant_fn, chunk_rows=CHUNK_ROWS):
    out_feat = W_blocked.shape[0]
    n_chunks = (out_feat + chunk_rows - 1) // chunk_rows
    dq_parts = []
    for ci, r0 in enumerate(range(0, out_feat, chunk_rows)):
        r1 = min(r0 + chunk_rows, out_feat)
        print(f"      chunk {ci+1}/{n_chunks} [{r0}:{r1}]", flush=True)
        res = quant_fn(W_blocked[r0:r1], return_dequant=True)
        dq_parts.append(res[2])
        del res
        torch.cuda.empty_cache()
    return torch.cat(dq_parts, dim=0)


def evaluate_wikitext(model, tokenizer):
    hf_model = HFLM(model, tokenizer=tokenizer, batch_size=1, max_length=2048)
    task_manager = TaskManager()
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        with torch.no_grad():
            results = simple_evaluate(
                model=hf_model, tasks=["wikitext"], num_fewshot=0,
                task_manager=task_manager, log_samples=False, verbosity="ERROR",
            )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()
    wt = results["results"].get("wikitext", {})
    return wt.get("word_perplexity,none", float("nan"))


def load_model():
    print(f"  Loading {MODEL_NAME}...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    print(f"done ({time.time() - t0:.1f}s)", flush=True)
    return model


def build_lm_head_hessian(model, block_size=BLOCK_SIZE):
    """Build block Hessian for lm_head from projection weight matrices.

    The lm_head input is the hidden state. The projection weights
    (q/k/v_proj, gate/up_proj) have rows in hidden_dim — they represent
    directions the model reads from the hidden state. We accumulate
    H_j = (1/N) * sum_layers(W_j^T @ W_j) per block as the Hessian.

    Returns H_blocks: (num_col_blocks, block_size, block_size) on DEVICE
    """
    hidden_dim = model.lm_head.weight.shape[1]
    num_col_blocks = hidden_dim // block_size
    print(f"  Building lm_head block Hessian (hidden={hidden_dim}, {num_col_blocks} blocks)...", flush=True)

    H_blocks = torch.zeros(num_col_blocks, block_size, block_size, device=DEVICE)
    n_layers = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(p in name for p in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            continue
        # Only use first decoder layer
        if not name.startswith("model.layers.0."):
            continue
        if module.weight.shape[1] != hidden_dim:
            continue

        W = module.weight.data.float()  # (out_feat, hidden_dim) on GPU
        for j in range(num_col_blocks):
            Wj = W[:, j * block_size : (j + 1) * block_size]  # (out_feat, bs)
            H_blocks[j].addmm_(Wj.T, Wj)
        n_layers += 1
        print(f"    + {name} ({W.shape[0]}x{W.shape[1]})", flush=True)

    if n_layers > 0:
        H_blocks /= n_layers
    print(f"  H_blocks: {H_blocks.shape} from {n_layers} layers", flush=True)
    return H_blocks


def main():
    print(f"Model: {MODEL_NAME}  Device: {DEVICE}  BS: {BLOCK_SIZE}")
    print(f"Results: {RESULTS_FILE}", flush=True)

    done = load_results()
    if done:
        print_results(done)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cb_int4, bd_int4 = make_int4_codebook(torch.device(DEVICE))

    # Baseline
    if "Baseline" not in done:
        print(f"\n[Baseline]", flush=True)
        model = load_model()
        print("  Evaluating...", flush=True)
        t0 = time.time()
        ppl = evaluate_wikitext(model, tokenizer)
        print(f"  PPL = {ppl:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        save_result("Baseline", ppl)
        done["Baseline"] = {"ppl": ppl, "quant_time": "-"}
        del model; torch.cuda.empty_cache()
    ppl_base = done["Baseline"]["ppl"]
    print(f"\nBaseline: {ppl_base:.4f}", flush=True)

    # Build block Hessian for lm_head H-Optimal
    model = load_model()
    H_blocks = build_lm_head_hessian(model)
    del model; torch.cuda.empty_cache()

    # --- Methods ---
    def nvfp4_optimal_fn(W, return_dequant=True):
        return nvfp4_optimal(W, return_dequant=return_dequant)

    def nvfp4_hoptimal_fn(W, return_dequant=True):
        return nvfp4_optimal_hessian(W, return_dequant=return_dequant, H_blocks=H_blocks)

    def int4_optimal_fn(W, return_dequant=True):
        return custom_optimal(W, cb_int4, bd_int4, return_dequant=return_dequant)

    def int4_hoptimal_fn(W, return_dequant=True):
        return custom_optimal_hessian(W, cb_int4, bd_int4, return_dequant=return_dequant, H_blocks=H_blocks)

    methods = [
        ("NVFP4 Optimal", nvfp4_optimal_fn),
        ("NVFP4 H-Optimal", nvfp4_hoptimal_fn),
        ("INT4 Optimal", int4_optimal_fn),
        ("INT4 H-Optimal", int4_hoptimal_fn),
    ]

    # --- Quantize lm_head only ---
    for mi, (method_name, quant_fn) in enumerate(methods):
        label = f"{method_name} lm_head"
        if label in done:
            ppl = done[label]["ppl"]
            print(f"\n  [{mi+1}/{len(methods)}] {label}: PPL={ppl:.4f} (cached)", flush=True)
            continue

        print(f"\n{'=' * 60}")
        print(f"[{mi+1}/{len(methods)}] {label}")
        print(f"{'=' * 60}", flush=True)
        model = load_model()

        lm_head = model.lm_head
        W = lm_head.weight.data.float()
        out_feat, in_feat = W.shape
        W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)

        print(f"  Quantizing lm_head ({out_feat}x{in_feat})...", flush=True)
        t0 = time.time()
        # H-Optimal needs smaller chunks (block Hessian gather is memory-heavy)
        chunk_sz = 4096 if "H-Optimal" in method_name else CHUNK_ROWS
        if out_feat <= chunk_sz:
            result = quant_fn(W_blocked, return_dequant=True)
            W_dq = result[2]
        else:
            W_dq = quantize_chunked(W_blocked, quant_fn, chunk_rows=chunk_sz)
        t_q = time.time() - t0

        lm_head.weight.data = W_dq.reshape(out_feat, in_feat).to(lm_head.weight.dtype)
        print(f"  Quantized in {t_q:.1f}s", flush=True)

        print(f"  Evaluating...", flush=True)
        t0 = time.time()
        ppl = evaluate_wikitext(model, tokenizer)
        t_eval = time.time() - t0
        delta = ppl - ppl_base
        print(f"  PPL={ppl:.4f}  (delta={delta:+.4f})  ({t_eval:.0f}s)", flush=True)

        save_result(label, ppl, f"{t_q:.1f}s")
        done[label] = {"ppl": ppl, "quant_time": f"{t_q:.1f}s"}
        print_results(done)

        del model; torch.cuda.empty_cache()

    print(f"\n\nDONE")
    print_results(done)


if __name__ == "__main__":
    main()
