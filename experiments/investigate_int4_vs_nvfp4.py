"""Investigate why NVFP4 H-Cal outperforms INT4 H-Cal at full model scale.

Collects per-layer per-linear diagnostics:
  Phase 1: Diagnostic pass (FP16 activations, both methods side-by-side)
  Phase 2: Propagation passes (error accumulation through quantized layers)

Usage:
    python investigate_int4_vs_nvfp4.py
"""

import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"
torch.backends.cuda.enable_flash_sdp(True)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from custom_codebook import custom_optimal_hessian, custom_quantize, custom_dequantize
from qwantize.nvfp4.reference import (
    nvfp4_optimal_hessian,
    fp4_quantize,
    fp4_dequantize,
    FP4_CODEBOOK,
)

MODEL_NAME = "Qwen/Qwen3-4B"
BLOCK_SIZE = 16
CHUNK_ROWS = 4096
CAL_TOKENS = 4096
DEVICE_QUANT = "cuda:0"
DEVICE_REF = "cuda:1"

CAL_DATASETS = [
    ("allenai/c4", "en", "validation"),
    ("allenai/c4", "zh", "validation"),
]

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "investigate_int4_vs_nvfp4.jsonl")


def append_result(entry):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    all_texts = []
    for ds_name, ds_config, ds_split in CAL_DATASETS:
        ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=True,
                          trust_remote_code=True)
        texts = []
        for doc in ds:
            texts.append(doc["text"])
            if len(texts) >= 200:
                break
        all_texts.append((ds_config or ds_name, texts))
    return all_texts


def collect_layer_activations(model, tokenizer, layer_idx, device, cal_texts):
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


def build_hblocks(X, block_size, device):
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


def entropy(hist):
    """Shannon entropy of a histogram."""
    p = torch.tensor(hist, dtype=torch.float32)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return -(p * p.log()).sum().item() / math.log(2)  # bits


def compute_diagnostics(W_blocked, dq_int4, dq_nvfp4, scales_int4, scales_nvfp4,
                        quants_int4, quants_nvfp4, X, H_blocks, block_size,
                        cb_int4):
    """Compute all per-linear diagnostic metrics."""
    out_feat = W_blocked.shape[0]
    in_feat = W_blocked.shape[1] * W_blocked.shape[2]
    W_flat = W_blocked.reshape(out_feat, in_feat).float()
    dq_i_flat = dq_int4.reshape(out_feat, in_feat).float()
    dq_n_flat = dq_nvfp4.reshape(out_feat, in_feat).float()

    rec = {}

    # --- Weight MSE (normalized) ---
    W_norm_sq = W_flat.pow(2).sum().item()
    rec["int4_weight_mse_norm"] = (dq_i_flat - W_flat).pow(2).sum().item() / max(W_norm_sq, 1e-12)
    rec["nvfp4_weight_mse_norm"] = (dq_n_flat - W_flat).pow(2).sum().item() / max(W_norm_sq, 1e-12)

    # --- Output MSE (normalized) ---
    # X: (T, in_feat), W: (out_feat, in_feat)
    # output = X @ W^T -> (T, out_feat)
    X_f = X.float()
    WX = X_f @ W_flat.T  # (T, out_feat)
    WX_norm_sq = WX.pow(2).sum().item()
    WqX_int4 = X_f @ dq_i_flat.T
    WqX_nvfp4 = X_f @ dq_n_flat.T
    rec["int4_output_mse_norm"] = (WqX_int4 - WX).pow(2).sum().item() / max(WX_norm_sq, 1e-12)
    rec["nvfp4_output_mse_norm"] = (WqX_nvfp4 - WX).pow(2).sum().item() / max(WX_norm_sq, 1e-12)
    del WX, WqX_int4, WqX_nvfp4

    # --- Hessian-weighted error ---
    # r^T H r summed over all blocks
    num_col_blocks = H_blocks.shape[0]
    M_dim = out_feat

    r_int4 = (W_blocked - dq_int4).float().reshape(M_dim, num_col_blocks, block_size)
    Hr_int4 = torch.einsum("jab,mjb->mja", H_blocks, r_int4)
    rec["int4_hessian_error"] = (r_int4 * Hr_int4).sum().item()

    r_nvfp4 = (W_blocked - dq_nvfp4).float().reshape(M_dim, num_col_blocks, block_size)
    Hr_nvfp4 = torch.einsum("jab,mjb->mja", H_blocks, r_nvfp4)
    rec["nvfp4_hessian_error"] = (r_nvfp4 * Hr_nvfp4).sum().item()
    del r_int4, Hr_int4, r_nvfp4, Hr_nvfp4

    # --- Scale analysis ---
    s_i = scales_int4.reshape(-1).float()
    s_n = scales_nvfp4.reshape(-1).float()
    rec["int4_scale_mean"] = s_i.mean().item()
    rec["int4_scale_std"] = s_i.std().item()
    rec["nvfp4_scale_mean"] = s_n.mean().item()
    rec["nvfp4_scale_std"] = s_n.std().item()
    rec["scale_diff_frac"] = (s_i != s_n).float().mean().item()

    # Scale ratio where both > 0
    both_pos = (s_i > 0) & (s_n > 0)
    if both_pos.any():
        rec["scale_ratio_mean"] = (s_n[both_pos] / s_i[both_pos]).mean().item()
    else:
        rec["scale_ratio_mean"] = float("nan")

    # --- Dead-zone analysis ---
    q_i_flat = quants_int4.reshape(-1).float()
    q_n_flat = quants_nvfp4.reshape(-1).float()
    rec["int4_zero_frac"] = (q_i_flat == 0).float().mean().item()
    rec["nvfp4_zero_frac"] = (q_n_flat == 0).float().mean().item()

    # --- Codebook utilization ---
    # INT4 levels: {0, 1/7, ..., 1} -> absolute values
    int4_levels = cb_int4.to(q_i_flat.device)
    q_i_abs = q_i_flat.abs()
    hist_int4 = []
    for lv in int4_levels:
        hist_int4.append((q_i_abs - lv).abs().lt(1e-5).sum().item())
    rec["int4_level_hist"] = hist_int4
    rec["int4_entropy"] = entropy(hist_int4)

    # NVFP4 levels: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    nvfp4_levels = FP4_CODEBOOK.to(q_n_flat.device)
    q_n_abs = q_n_flat.abs()
    hist_nvfp4 = []
    for lv in nvfp4_levels:
        hist_nvfp4.append((q_n_abs - lv).abs().lt(1e-5).sum().item())
    rec["nvfp4_level_hist"] = hist_nvfp4
    rec["nvfp4_entropy"] = entropy(hist_nvfp4)

    # --- Disagreement: same sign, different magnitude bucket ---
    # Normalize both to [0,7] bucket indices for comparison
    q_i_bucket = torch.bucketize(q_i_abs, (int4_levels[:-1] + int4_levels[1:]) / 2)
    nvfp4_bd = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=q_n_abs.device)
    q_n_bucket = torch.bucketize(q_n_abs, nvfp4_bd)
    # Can't directly compare bucket indices since codebooks differ,
    # but we can check if both map to zero vs non-zero
    rec["both_zero_frac"] = ((q_i_flat == 0) & (q_n_flat == 0)).float().mean().item()
    rec["int4_zero_nvfp4_nonzero"] = ((q_i_flat == 0) & (q_n_flat != 0)).float().mean().item()
    rec["int4_nonzero_nvfp4_zero"] = ((q_i_flat != 0) & (q_n_flat == 0)).float().mean().item()

    return rec


def dual_quantize_chunked(W_blocked, H_blocks, cb, bd, device, chunk_rows=CHUNK_ROWS):
    """Quantize with both INT4 and NVFP4 H-Cal, return all intermediates."""
    out_feat = W_blocked.shape[0]

    all_s_i, all_q_i, all_dq_i = [], [], []
    all_s_n, all_q_n, all_dq_n = [], [], []

    for r0 in range(0, out_feat, chunk_rows):
        r1 = min(r0 + chunk_rows, out_feat)
        W_chunk = W_blocked[r0:r1]

        # INT4 H-Cal
        s_i, q_i, dq_i = custom_optimal_hessian(
            W_chunk, cb, bd, return_dequant=True, H_blocks=H_blocks)
        all_s_i.append(s_i)
        all_q_i.append(q_i)
        all_dq_i.append(dq_i)

        # NVFP4 H-Cal
        s_n, q_n, dq_n = nvfp4_optimal_hessian(
            W_chunk, return_dequant=True, H_blocks=H_blocks)
        all_s_n.append(s_n)
        all_q_n.append(q_n)
        all_dq_n.append(dq_n)

    return (
        (torch.cat(all_s_i), torch.cat(all_q_i), torch.cat(all_dq_i)),
        (torch.cat(all_s_n), torch.cat(all_q_n), torch.cat(all_dq_n)),
    )


def quantize_linear_single(module, quant_fn, H_blocks, cb, bd, block_size):
    """Quantize a single linear layer in-place with one method."""
    W = module.weight.data.float()
    out_feat, in_feat = W.shape
    if in_feat % block_size != 0:
        return
    W_blocked = W.reshape(out_feat, in_feat // block_size, block_size)
    dq_parts = []
    for r0 in range(0, out_feat, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, out_feat)
        res = quant_fn(W_blocked[r0:r1], H_blocks, cb, bd)
        dq_parts.append(res[2])
        del res
    W_dq = torch.cat(dq_parts, dim=0)
    module.weight.data = W_dq.reshape(out_feat, in_feat).to(module.weight.dtype)


def _quant_int4_hcal(W_blocked, H_blocks, cb, bd):
    return custom_optimal_hessian(W_blocked, cb, bd,
                                   return_dequant=True, H_blocks=H_blocks)

def _quant_nvfp4_hcal(W_blocked, H_blocks, cb, bd):
    return nvfp4_optimal_hessian(W_blocked, return_dequant=True, H_blocks=H_blocks)


# ---------------------------------------------------------------------------
# Phase 1: Diagnostic pass
# ---------------------------------------------------------------------------

def diagnostic_pass(model, tokenizer, cal_texts, device, cb, bd):
    """For each layer, collect FP16 activations, quantize both ways, collect diagnostics."""
    num_layers = len(model.model.layers)
    records = []

    print(f"\n{'='*80}")
    print(f"PHASE 1: DIAGNOSTIC PASS (FP16 activations, no weight modification)")
    print(f"{'='*80}\n", flush=True)

    for li in range(num_layers):
        t0 = time.time()
        act_dict = collect_layer_activations(model, tokenizer, li, device, cal_texts)
        layer = model.model.layers[li]

        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            X = act_dict.get(name)
            if X is None:
                continue
            H_blocks = build_hblocks(X, BLOCK_SIZE, device)
            if H_blocks is None:
                continue

            W = module.weight.data.float()
            out_feat, in_feat = W.shape
            if in_feat % BLOCK_SIZE != 0:
                continue
            W_blocked = W.reshape(out_feat, in_feat // BLOCK_SIZE, BLOCK_SIZE)

            (s_i, q_i, dq_i), (s_n, q_n, dq_n) = dual_quantize_chunked(
                W_blocked, H_blocks, cb, bd, device)

            rec = compute_diagnostics(
                W_blocked, dq_i, dq_n, s_i, s_n, q_i, q_n,
                X, H_blocks, BLOCK_SIZE, cb)
            rec["phase"] = "diagnostic"
            rec["layer"] = li
            rec["linear"] = name
            rec["shape"] = [out_feat, in_feat]
            records.append(rec)
            append_result(rec)

            del s_i, q_i, dq_i, s_n, q_n, dq_n, H_blocks, W_blocked

        del act_dict
        gc.collect()
        torch.cuda.empty_cache()

        t_layer = time.time() - t0
        # Quick per-layer summary
        layer_recs = [r for r in records if r["layer"] == li]
        if layer_recs:
            avg_w_i = sum(r["int4_weight_mse_norm"] for r in layer_recs) / len(layer_recs)
            avg_w_n = sum(r["nvfp4_weight_mse_norm"] for r in layer_recs) / len(layer_recs)
            avg_o_i = sum(r["int4_output_mse_norm"] for r in layer_recs) / len(layer_recs)
            avg_o_n = sum(r["nvfp4_output_mse_norm"] for r in layer_recs) / len(layer_recs)
            avg_h_i = sum(r["int4_hessian_error"] for r in layer_recs) / len(layer_recs)
            avg_h_n = sum(r["nvfp4_hessian_error"] for r in layer_recs) / len(layer_recs)
            w_win = "INT4" if avg_w_i < avg_w_n else "NVFP4"
            o_win = "INT4" if avg_o_i < avg_o_n else "NVFP4"
            h_win = "INT4" if avg_h_i < avg_h_n else "NVFP4"
            print(f"  L{li:02d} ({t_layer:.1f}s): W_err INT4={avg_w_i:.6f} NV={avg_w_n:.6f} [{w_win}]  "
                  f"O_err INT4={avg_o_i:.6f} NV={avg_o_n:.6f} [{o_win}]  "
                  f"H_err INT4={avg_h_i:.4f} NV={avg_h_n:.4f} [{h_win}]", flush=True)

    return records


# ---------------------------------------------------------------------------
# Phase 2: Propagation passes
# ---------------------------------------------------------------------------

def compare_hidden_states(model_q, model_ref, tokenizer, cal_texts, device_q, device_ref, layer_idx):
    """Run calibration through both models, compare hidden states after layer_idx."""
    hidden_q = []
    hidden_ref = []

    def make_hook(storage):
        def hook_fn(mod, inp, out):
            # out is a tuple for transformer layers; first element is hidden state
            h = out[0] if isinstance(out, tuple) else out
            storage.append(h.detach().float().cpu())
        return hook_fn

    hook_q = model_q.model.layers[layer_idx].register_forward_hook(make_hook(hidden_q))
    hook_ref = model_ref.model.layers[layer_idx].register_forward_hook(make_hook(hidden_ref))

    total_tokens = 0
    for label, texts in cal_texts:
        for text in texts:
            if total_tokens >= 2048:
                break
            if not text.strip():
                continue
            tokens_q = tokenizer.encode(text, return_tensors="pt",
                                         truncation=True, max_length=512)
            if tokens_q.shape[1] == 0:
                continue
            tokens_ref = tokens_q.clone()
            with torch.no_grad():
                model_q(tokens_q.to(device_q))
                model_ref(tokens_ref.to(device_ref))
            total_tokens += tokens_q.shape[1]

    hook_q.remove()
    hook_ref.remove()

    if not hidden_q or not hidden_ref:
        return float("nan"), float("nan")

    h_q = torch.cat([h.reshape(-1, h.shape[-1]) for h in hidden_q], dim=0)
    h_ref = torch.cat([h.reshape(-1, h.shape[-1]) for h in hidden_ref], dim=0)

    cos_sim = F.cosine_similarity(h_q, h_ref, dim=-1).mean().item()
    l2_div = (h_q - h_ref).pow(2).sum().item() / max(h_ref.pow(2).sum().item(), 1e-12)

    return cos_sim, l2_div


def propagation_pass(model, model_ref, tokenizer, cal_texts, device_q, device_ref,
                     method_name, quant_fn, cb, bd):
    """Quantize layer by layer, measure hidden-state divergence vs FP16 reference."""
    num_layers = len(model.model.layers)
    records = []

    print(f"\n{'='*80}")
    print(f"PHASE 2: PROPAGATION PASS ({method_name})")
    print(f"{'='*80}\n", flush=True)

    total_quant_time = 0
    for li in range(num_layers):
        t0 = time.time()
        act_dict = collect_layer_activations(model, tokenizer, li, device_q, cal_texts)
        layer = model.model.layers[li]

        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            X = act_dict.get(name)
            if X is None:
                continue
            H_blocks = build_hblocks(X, BLOCK_SIZE, device_q)
            if H_blocks is None:
                continue
            quantize_linear_single(module, quant_fn, H_blocks, cb, bd, BLOCK_SIZE)

        del act_dict
        gc.collect()
        torch.cuda.empty_cache()
        total_quant_time += time.time() - t0

        # Compare hidden states every 6 layers and at the end
        if (li + 1) % 6 == 0 or li == num_layers - 1:
            cos_sim, l2_div = compare_hidden_states(
                model, model_ref, tokenizer, cal_texts, device_q, device_ref, li)
            rec = {
                "phase": "propagation",
                "method": method_name,
                "layer": li,
                "cos_sim": cos_sim,
                "l2_div": l2_div,
                "quant_time": total_quant_time,
            }
            records.append(rec)
            append_result(rec)
            print(f"  After L{li:02d}: cos_sim={cos_sim:.6f}  l2_div={l2_div:.6f}  "
                  f"(quant: {total_quant_time:.1f}s)", flush=True)

    return records


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def print_diagnostic_summary(records):
    diag = [r for r in records if r["phase"] == "diagnostic"]
    if not diag:
        return

    print(f"\n{'='*100}")
    print(f"DIAGNOSTIC SUMMARY (per-layer averages)")
    print(f"{'='*100}")
    print(f"{'Layer':>5} {'Linear':<12} {'INT4 W%':>10} {'NV W%':>10} {'INT4 O%':>10} "
          f"{'NV O%':>10} {'INT4 H':>12} {'NV H':>12} {'I0%':>6} {'N0%':>6} {'Win(H)':>7}")
    print(f"{'-'*100}")

    for r in diag:
        h_win = "INT4" if r["int4_hessian_error"] < r["nvfp4_hessian_error"] else "NVFP4"
        print(f"{r['layer']:>5} {r['linear']:<12} "
              f"{r['int4_weight_mse_norm']*100:>10.4f} {r['nvfp4_weight_mse_norm']*100:>10.4f} "
              f"{r['int4_output_mse_norm']*100:>10.4f} {r['nvfp4_output_mse_norm']*100:>10.4f} "
              f"{r['int4_hessian_error']:>12.4f} {r['nvfp4_hessian_error']:>12.4f} "
              f"{r['int4_zero_frac']*100:>5.1f}% {r['nvfp4_zero_frac']*100:>5.1f}% "
              f"{h_win:>7}")

    # Aggregate by linear type
    print(f"\n{'='*100}")
    print(f"AGGREGATE BY LINEAR TYPE")
    print(f"{'='*100}")

    from collections import defaultdict
    by_type = defaultdict(list)
    for r in diag:
        by_type[r["linear"]].append(r)

    print(f"{'Type':<12} {'INT4 W%':>10} {'NV W%':>10} {'INT4 O%':>10} {'NV O%':>10} "
          f"{'INT4 H':>12} {'NV H':>12} {'I0%':>6} {'N0%':>6} {'IEnt':>6} {'NEnt':>6}")
    print(f"{'-'*100}")

    for ltype in sorted(by_type.keys()):
        recs = by_type[ltype]
        n = len(recs)
        print(f"{ltype:<12} "
              f"{sum(r['int4_weight_mse_norm'] for r in recs)/n*100:>10.4f} "
              f"{sum(r['nvfp4_weight_mse_norm'] for r in recs)/n*100:>10.4f} "
              f"{sum(r['int4_output_mse_norm'] for r in recs)/n*100:>10.4f} "
              f"{sum(r['nvfp4_output_mse_norm'] for r in recs)/n*100:>10.4f} "
              f"{sum(r['int4_hessian_error'] for r in recs)/n:>12.4f} "
              f"{sum(r['nvfp4_hessian_error'] for r in recs)/n:>12.4f} "
              f"{sum(r['int4_zero_frac'] for r in recs)/n*100:>5.1f}% "
              f"{sum(r['nvfp4_zero_frac'] for r in recs)/n*100:>5.1f}% "
              f"{sum(r['int4_entropy'] for r in recs)/n:>6.2f} "
              f"{sum(r['nvfp4_entropy'] for r in recs)/n:>6.2f}")

    # Overall winner counts
    int4_wins_w = sum(1 for r in diag if r["int4_weight_mse_norm"] < r["nvfp4_weight_mse_norm"])
    int4_wins_o = sum(1 for r in diag if r["int4_output_mse_norm"] < r["nvfp4_output_mse_norm"])
    int4_wins_h = sum(1 for r in diag if r["int4_hessian_error"] < r["nvfp4_hessian_error"])
    total = len(diag)

    print(f"\n  Winner counts ({total} linears total):")
    print(f"    Weight MSE:    INT4 {int4_wins_w}, NVFP4 {total - int4_wins_w}")
    print(f"    Output MSE:    INT4 {int4_wins_o}, NVFP4 {total - int4_wins_o}")
    print(f"    Hessian Error: INT4 {int4_wins_h}, NVFP4 {total - int4_wins_h}")

    # Scale ratio analysis
    ratios = [r["scale_ratio_mean"] for r in diag if not math.isnan(r.get("scale_ratio_mean", float("nan")))]
    if ratios:
        print(f"\n  Scale ratio (NVFP4/INT4): mean={sum(ratios)/len(ratios):.4f}")

    # Dead-zone disagreement
    i0_n1 = sum(r["int4_zero_nvfp4_nonzero"] for r in diag) / len(diag)
    i1_n0 = sum(r["int4_nonzero_nvfp4_zero"] for r in diag) / len(diag)
    print(f"  Dead-zone disagreement: INT4→0,NV→nonzero: {i0_n1*100:.2f}%  "
          f"INT4→nonzero,NV→0: {i1_n0*100:.2f}%")

    sys.stdout.flush()


def print_propagation_summary(prop_records):
    if not prop_records:
        return

    print(f"\n{'='*80}")
    print(f"PROPAGATION COMPARISON (hidden-state divergence from FP16)")
    print(f"{'='*80}")
    print(f"{'Layer':>5} {'INT4 cos':>12} {'NV cos':>12} {'INT4 L2':>12} {'NV L2':>12} {'Winner(L2)':>10}")
    print(f"{'-'*80}")

    int4_recs = {r["layer"]: r for r in prop_records if r["method"] == "int4_hcal"}
    nvfp4_recs = {r["layer"]: r for r in prop_records if r["method"] == "nvfp4_hcal"}

    for li in sorted(set(int4_recs.keys()) | set(nvfp4_recs.keys())):
        ri = int4_recs.get(li, {})
        rn = nvfp4_recs.get(li, {})
        cos_i = ri.get("cos_sim", float("nan"))
        cos_n = rn.get("cos_sim", float("nan"))
        l2_i = ri.get("l2_div", float("nan"))
        l2_n = rn.get("l2_div", float("nan"))
        win = "INT4" if l2_i < l2_n else "NVFP4"
        print(f"{li:>5} {cos_i:>12.6f} {cos_n:>12.6f} {l2_i:>12.6f} {l2_n:>12.6f} {win:>10}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Investigation: INT4 H-Cal vs NVFP4 H-Cal")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Block size: {BLOCK_SIZE}")
    print(f"  Quant device: {DEVICE_QUANT}, Ref device: {DEVICE_REF}")
    print(f"  Results: {RESULTS_FILE}\n", flush=True)

    # Clear previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # INT4 codebook
    cb = torch.linspace(0, 1, 8, device=DEVICE_QUANT)
    bd = torch.empty(7, device=DEVICE_QUANT)
    bd[0] = cb[1] / 2
    for k in range(1, 7):
        bd[k] = (cb[k] + cb[k + 1]) / 2

    # Calibration texts
    print("[Loading calibration texts]", flush=True)
    cal_texts = get_cal_texts()
    for label, texts in cal_texts:
        print(f"  {label}: {len(texts)} docs", flush=True)

    # Reference model (stays FP16 throughout)
    print("\n[Reference model]", flush=True)
    model_ref = load_model(DEVICE_REF)

    # --- Phase 1: Diagnostic pass ---
    print("\n[Working model for diagnostic pass]", flush=True)
    model = load_model(DEVICE_QUANT)
    diag_records = diagnostic_pass(model, tokenizer, cal_texts, DEVICE_QUANT, cb, bd)
    print_diagnostic_summary(diag_records)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2a: INT4 propagation ---
    print("\n[Loading model for INT4 propagation]", flush=True)
    model = load_model(DEVICE_QUANT)
    int4_prop = propagation_pass(
        model, model_ref, tokenizer, cal_texts, DEVICE_QUANT, DEVICE_REF,
        "int4_hcal", _quant_int4_hcal, cb, bd)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2b: NVFP4 propagation ---
    print("\n[Loading model for NVFP4 propagation]", flush=True)
    model = load_model(DEVICE_QUANT)
    nvfp4_prop = propagation_pass(
        model, model_ref, tokenizer, cal_texts, DEVICE_QUANT, DEVICE_REF,
        "nvfp4_hcal", _quant_nvfp4_hcal, cb, bd)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Final summary ---
    print_propagation_summary(int4_prop + nvfp4_prop)

    print(f"\n=== INVESTIGATION COMPLETE ===")
    print(f"Full results in: {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
