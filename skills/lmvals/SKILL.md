# LLM Evaluation Guide

## Setup

Install required packages:
```bash
pip install lm-eval transformers torch accelerate datasets
```

The evaluation script uses the `lm_eval` framework (EleutherAI). See `evals.txt` for all 14000+ supported tasks.

## As a library

```python
from lmvals.eval import evaluate_model, evaluate_single, evaluate_and_save
from lmvals.eval import COMMON_TASKS, ALL_TASKS, load_checkpoints

# Evaluate a model instance on common benchmarks
results = evaluate_model(model, tokenizer, tasks="common")

# Single task
metrics = evaluate_single(model, tokenizer, "wikitext")
# → {"wikitext/word_perplexity,none": 12.21, "wikitext/byte_perplexity,none": 1.59, ...}

# Evaluate and save incrementally to JSON
results = evaluate_and_save(model, tokenizer, tasks="all", output="results.json")

# With pruned checkpoints
load_checkpoints(model, "path/to/checkpoints", device)
results = evaluate_model(model, tokenizer)

# Suppress logs (call once)
from lmvals.eval import suppress_loggers
suppress_loggers()
```

## As a script

### Dense model baseline
```bash
python lmvals/eval.py --model Qwen/Qwen3-8B --tasks common
```

### Common benchmarks (default)
Tasks: wikitext, arc_easy, arc_challenge, piqa, winogrande, boolq, lambada_openai

### All benchmarks (adds mmlu, hellaswag — slower)
```bash
python lmvals/eval.py --model Qwen/Qwen3-8B --tasks all
```

### Specific tasks
```bash
python lmvals/eval.py --model Qwen/Qwen3-8B --tasks wikitext,mmlu
```

### Pruned model with layer checkpoints
```bash
python lmvals/eval.py --model Qwen/Qwen3-8B \
    --ckpt-dir autoresearch/results/admm_Qwen3-8B_20260406/checkpoints
```

Checkpoints are `layer_NNN.pt` files produced by `autoresearch/bench_prune_llm.py`.

### Options
- `--device cuda:1` — use specific GPU
- `--verbose` — show lm_eval progress bars (suppressed by default)
- `--output path.json` — custom output path (default: `lmvals/<model>_<label>_<timestamp>.json`)
- `--batch-size 8` — batch size for evaluation (default 4)

## Output format

JSON with all numeric metrics from lm_eval, keyed as `task/metric_name`:
```json
{
  "model": "Qwen/Qwen3-8B",
  "ckpt_dir": null,
  "tasks": ["wikitext", "arc_easy", ...],
  "wikitext/word_perplexity,none": 12.21,
  "wikitext/byte_perplexity,none": 1.59,
  "arc_easy/acc,none": 0.833,
  "arc_easy/acc_norm,none": 0.807,
  ...
  "eval_time_s": 3461.8,
  "status": "done"
}
```

Results are saved incrementally after each task completes.

## Benchmark reference

### Fast benchmarks (~1-5 min each on 8B)
- `wikitext` — WikiText-2 perplexity (word_ppl, byte_ppl, bits_per_byte)
- `arc_easy` — ARC Easy (acc, acc_norm)
- `arc_challenge` — ARC Challenge (acc, acc_norm)
- `piqa` — Physical Intuition QA (acc, acc_norm)
- `winogrande` — Winogrande commonsense (acc)
- `boolq` — Boolean Questions (acc)
- `lambada_openai` — LAMBADA next-word prediction (acc, perplexity)

### Slow benchmarks (~15-60 min on 8B)
- `mmlu` — Massive Multitask Language Understanding, 57 subjects (acc)
- `hellaswag` — Sentence completion (acc, acc_norm)

### Metric conventions
- `acc` — accuracy
- `acc_norm` — length-normalized accuracy (used for multiple-choice with varying option lengths)
- `word_perplexity` — per-word perplexity (lower is better)
- For reporting: use `acc_norm` when available, otherwise `acc`

## Approximate timing (RTX 3090, 8B model)
- wikitext: ~5 min
- arc_easy/challenge: ~2 min each
- piqa/winogrande/boolq: ~1-3 min each
- lambada_openai: ~3 min
- mmlu: ~17 min
- hellaswag: ~10 min
- All common (7 tasks): ~20 min
- All (9 tasks): ~45 min

## Notes
- Token embeddings and lm_head are never pruned in our experiments
- Only transformer layer projections (q/k/v/o_proj, gate/up/down_proj) are pruned
- All evaluations are 0-shot unless otherwise noted
- MMLU is typically run 5-shot in papers but we use 0-shot for consistency
