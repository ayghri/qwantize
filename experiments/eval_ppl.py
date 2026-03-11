"""Standalone perplexity & top-k KL divergence evaluator.

Replicates lm_eval's wikitext metrics exactly (word_perplexity, byte_perplexity,
bits_per_byte), and adds top-k KL divergence between a reference and quantized model.

Usage:
    from eval_ppl import evaluate

    # All metrics in one pass (PPL + KL if model_ref provided)
    results = evaluate(model, tokenizer, model_ref=model_ref, k=100)

    # PPL only
    results = evaluate(model, tokenizer)
"""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Wikitext detokenizer (exact copy from lm_eval)
# ---------------------------------------------------------------------------
def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


# ---------------------------------------------------------------------------
# Rolling window generation (matches lm_eval logic)
# ---------------------------------------------------------------------------
def get_rolling_windows(token_ids, max_length, prefix_token_id):
    """Generate (input_ids, target_ids) windows covering the full sequence.

    Matches lm_eval's get_rolling_token_windows + make_disjoint_window.
    Each target token is scored exactly once.
    """
    n = len(token_ids)
    if n == 0:
        return

    # First window
    first_len = min(max_length, n)
    ctx = [prefix_token_id] + token_ids[: first_len - 1]
    tgt = token_ids[:first_len]
    disjoint_ctx_len = len(ctx) - (len(tgt) - 1)
    yield ctx[:disjoint_ctx_len], tgt

    predicted = first_len
    pred_len = max_length  # context_len=1

    while predicted < n:
        window_pred_len = min(n - predicted, pred_len)
        window_end = predicted + window_pred_len
        ctx = token_ids[window_end - max_length - 1: window_end - 1]
        tgt = token_ids[window_end - window_pred_len: window_end]
        disjoint_ctx_len = len(ctx) - (len(tgt) - 1)
        yield ctx[:disjoint_ctx_len], tgt
        predicted += window_pred_len


def _forward_window(model, input_ids, target_ids, device):
    """Forward pass for a single window. Returns logits for continuation positions."""
    full_ids = input_ids + target_ids
    inp = torch.tensor(full_ids[:-1] if len(full_ids) > 1 else full_ids,
                       dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp).logits  # (1, seq_len, vocab)
    cont_len = len(target_ids)
    return logits[0, -cont_len:, :]  # (cont_len, vocab)


def _loglik_from_logits(logits, target_ids, device):
    """Compute sum of log-probs for target tokens from logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = torch.tensor(target_ids, dtype=torch.long, device=device)
    token_lp = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return token_lp.sum().item()


def _topp_kl_chunk(logits_ref, logits_quant, p_threshold):
    """Top-p KL for a chunk of positions. (chunk_len, vocab) inputs. Runs in fp32."""
    logits_ref = logits_ref.float()
    logits_quant = logits_quant.float()
    sorted_ref, idx_ref = logits_ref.sort(dim=-1, descending=True)
    sorted_quant, idx_quant = logits_quant.sort(dim=-1, descending=True)

    probs_ref = F.softmax(sorted_ref, dim=-1)
    mask_ref = (probs_ref.cumsum(dim=-1) - probs_ref) < p_threshold

    probs_quant = F.softmax(sorted_quant, dim=-1)
    mask_quant = (probs_quant.cumsum(dim=-1) - probs_quant) < p_threshold

    # Scatter masks back to vocab positions, union them
    vmask_ref = torch.zeros_like(logits_ref, dtype=torch.bool)
    vmask_ref.scatter_(-1, idx_ref, mask_ref)
    vmask_quant = torch.zeros_like(logits_quant, dtype=torch.bool)
    vmask_quant.scatter_(-1, idx_quant, mask_quant)
    union_mask = vmask_ref | vmask_quant

    neg_inf = torch.tensor(float('-inf'), device=logits_ref.device, dtype=logits_ref.dtype)
    log_p = F.log_softmax(torch.where(union_mask, logits_ref, neg_inf), dim=-1)
    log_q = F.log_softmax(torch.where(union_mask, logits_quant, neg_inf), dim=-1)
    p = log_p.exp()
    # Zero out non-union positions to avoid 0 * -inf = nan
    kl_terms = torch.where(union_mask, p * (log_p - log_q), torch.zeros_like(p))
    return kl_terms.sum(dim=-1)


def _topp_kl_from_logits(logits_ref, logits_quant, p_threshold, chunk_size=64):
    """Compute per-position top-p KL divergence, chunked along seq_len.

    Returns tensor of shape (seq_len,).
    """
    seq_len = logits_ref.shape[0]
    kls = []
    for i in range(0, seq_len, chunk_size):
        kls.append(_topp_kl_chunk(
            logits_ref[i:i+chunk_size],
            logits_quant[i:i+chunk_size],
            p_threshold,
        ))
    return torch.cat(kls)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _load_dataset(dataset_name, split):
    if dataset_name == "wikitext":
        ds = load_dataset("EleutherAI/wikitext_document_level",
                          "wikitext-2-raw-v1", split=split)
        return ds, "page", True
    else:
        ds = load_dataset(dataset_name, split=split)
        text_field = "text" if "text" in ds.column_names else ds.column_names[0]
        return ds, text_field, False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(model, tokenizer, model_ref=None, ref_lm_head_weight=None,
             top_p=0.9, max_length=2048, max_docs=None,
             dataset_name="wikitext", split="test", verbose=True):
    """Evaluate perplexity and optionally top-p KL divergence.

    Computes all lm_eval wikitext metrics:
      - word_perplexity: exp(-sum(ll) / sum(words))
      - byte_perplexity: exp(-sum(ll) / sum(bytes))
      - bits_per_byte:   -sum(ll) / sum(bytes) / log(2)

    For KL divergence, provide either:
      - model_ref: a separate reference model (can be on different GPU)
      - ref_lm_head_weight: original lm_head weight tensor; the model's
        lm_head is swapped to this for the ref forward, then swapped back.
        This avoids loading two models on the same GPU.

    Args:
        model: Model to evaluate (quantized or baseline)
        tokenizer: Corresponding tokenizer
        model_ref: Optional reference model for KL divergence
        ref_lm_head_weight: Optional original lm_head weight for KL (same GPU)
        top_p: Cumulative probability threshold for KL (default 0.9)
        max_length: Max sequence length for rolling windows
        max_docs: Max number of documents to evaluate (None = all)
        dataset_name: "wikitext" or HF dataset name
        split: Dataset split
        verbose: Print progress

    Returns:
        dict with word_perplexity, byte_perplexity, bits_per_byte,
        and optionally kl_mean, kl_std, kl_median
    """
    compute_kl = model_ref is not None or ref_lm_head_weight is not None
    device = next(model.parameters()).device
    device_ref = next(model_ref.parameters()).device if model_ref else None

    ds, text_field, detokenize = _load_dataset(dataset_name, split)

    prefix_token_id = tokenizer.bos_token_id
    if prefix_token_id is None:
        prefix_token_id = tokenizer.eos_token_id

    # Only use threads when models are on different devices
    use_threads = model_ref is not None and device != device_ref
    executor = ThreadPoolExecutor(max_workers=2) if use_threads else None

    t_start = time.time()
    total_loglik = 0.0
    total_words = 0
    total_bytes = 0
    total_tokens = 0
    all_kls = [] if compute_kl else None

    n_docs = max_docs if max_docs else len(ds)
    for di, doc in enumerate(ds):
        if di >= n_docs:
            break
        raw_text = doc[text_field]

        # Word/byte counts from original text (matches lm_eval)
        words = len(re.split(r"\s+", raw_text))
        nbytes = len(raw_text.encode("utf-8"))

        # Detokenize for wikitext
        text = wikitext_detokenizer(doc) if detokenize else raw_text
        if not text.strip():
            continue

        token_ids = tokenizer.encode(text)
        if len(token_ids) == 0:
            continue

        # Rolling windows
        doc_loglik = 0.0
        doc_tokens = 0
        for ctx, tgt in get_rolling_windows(token_ids, max_length, prefix_token_id):
            # Build input tensor once
            full_ids = ctx + tgt
            inp_ids = full_ids[:-1] if len(full_ids) > 1 else full_ids
            cont_len = len(tgt)

            if compute_kl:
                inp = torch.tensor(inp_ids, dtype=torch.long, device=device).unsqueeze(0)

                if model_ref is not None and use_threads:
                    inp_ref = torch.tensor(inp_ids, dtype=torch.long, device=device_ref).unsqueeze(0)
                    def _fwd_quant():
                        with torch.no_grad():
                            return model(inp).logits
                    def _fwd_ref():
                        with torch.no_grad():
                            return model_ref(inp_ref).logits
                    fut_q = executor.submit(_fwd_quant)
                    fut_r = executor.submit(_fwd_ref)
                    logits_quant_full = fut_q.result()
                    logits_ref_full = fut_r.result()
                elif model_ref is not None:
                    inp_ref = torch.tensor(inp_ids, dtype=torch.long, device=device_ref).unsqueeze(0)
                    with torch.no_grad():
                        logits_quant_full = model(inp).logits
                        logits_ref_full = model_ref(inp_ref).logits
                else:
                    # Weight-swap mode: run quant, swap to ref, run ref, swap back
                    quant_w = model.lm_head.weight.data
                    with torch.no_grad():
                        logits_quant_full = model(inp).logits
                        model.lm_head.weight.data = ref_lm_head_weight
                        logits_ref_full = model(inp).logits
                        model.lm_head.weight.data = quant_w

                logits = logits_quant_full[0, -cont_len:, :]
                logits_ref = logits_ref_full[0, -cont_len:, :]

                doc_loglik += _loglik_from_logits(logits, tgt, device)
                kl = _topp_kl_from_logits(
                    logits_ref.to(device),
                    logits,
                    top_p,
                )
                all_kls.append(kl.cpu())
            else:
                logits = _forward_window(model, ctx, tgt, device)
                doc_loglik += _loglik_from_logits(logits, tgt, device)

            doc_tokens += cont_len

        total_loglik += doc_loglik
        total_words += words
        total_bytes += nbytes
        total_tokens += doc_tokens

        if verbose and (di + 1) % 10 == 0:
            w_ppl = math.exp(-total_loglik / total_words)
            b_ppl = math.exp(-total_loglik / total_bytes)
            bpb = -total_loglik / total_bytes / math.log(2)
            msg = (f"  doc {di+1}/{n_docs}  tokens={total_tokens}  "
                   f"word_ppl={w_ppl:.4f}  byte_ppl={b_ppl:.4f}  bpb={bpb:.4f}")
            if all_kls:
                running_kl = torch.cat(all_kls)
                msg += f"  topp_kl={running_kl.mean():.6f}"
            print(msg, flush=True)

    # Final metrics
    elapsed = time.time() - t_start
    result = {
        "word_perplexity": math.exp(-total_loglik / total_words),
        "byte_perplexity": math.exp(-total_loglik / total_bytes),
        "bits_per_byte": -total_loglik / total_bytes / math.log(2),
        "total_loglik": total_loglik,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "num_docs": n_docs,
        "elapsed_s": elapsed,
    }

    if all_kls:
        kls = torch.cat(all_kls)
        result.update({
            "kl_mean": kls.mean().item(),
            "kl_std": kls.std().item(),
            "kl_median": kls.median().item(),
            "kl_positions": kls.shape[0],
            "top_p": top_p,
        })

    if verbose:
        print(f"\n  word_perplexity  = {result['word_perplexity']:.4f}")
        print(f"  byte_perplexity  = {result['byte_perplexity']:.4f}")
        print(f"  bits_per_byte    = {result['bits_per_byte']:.4f}")
        print(f"  elapsed          = {elapsed:.1f}s")
        if all_kls:
            print(f"  top-{top_p} KL mean = {result['kl_mean']:.6f} "
                  f"+/- {result['kl_std']:.6f}  "
                  f"(median={result['kl_median']:.6f})")


    return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    os.environ["HF_HOME"] = "/buckets/datasets/huggingface"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_flash_sdp(True)

    MODEL_NAME = "Qwen/Qwen3-4B"
    DEVICE = "cuda:0"

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    print("\nEvaluating (should match lm_eval word_ppl ~18.4559)...", flush=True)
    results = evaluate(model, tokenizer, max_length=2048)
    print(f"\nResults: {results}")
