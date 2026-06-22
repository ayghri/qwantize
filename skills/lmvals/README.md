# lmvals — LLM Evaluation Skill

A self-contained LLM evaluation module built on [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness). Works as a Python library, CLI script, or Claude Code agent skill.

## Adding as an Agent Skill

To give a Claude Code agent access to this evaluation tool, add `lmvals/SKILL.md` to the agent's skill files. This teaches the agent how to run benchmarks, interpret results, and use the library API.

### In Claude Code CLI

Reference the skill file in your project's `CLAUDE.md`:

```markdown
For LLM evaluation, follow the guide in lmvals/SKILL.md
```

### In a custom agent (Claude Agent SDK)

Include `SKILL.md` in the agent's system prompt or tool instructions:

```python
skill = open("lmvals/SKILL.md").read()
system_prompt = f"""
You are a research assistant.

{skill}
"""
```

### As an MCP resource

Serve `SKILL.md` as a resource from an MCP server so any connected agent can discover it.

## Quick Start

### Install dependencies

```bash
pip install lm-eval transformers torch accelerate datasets
```

### As a library

```python
from lmvals.eval import evaluate_model, evaluate_single, evaluate_and_save
from lmvals.eval import COMMON_TASKS, ALL_TASKS, load_checkpoints

# Evaluate on common benchmarks (wikitext, arc, piqa, winogrande, boolq, lambada)
results = evaluate_model(model, tokenizer, tasks="common")

# Single task
metrics = evaluate_single(model, tokenizer, "wikitext")

# Evaluate and save incrementally
results = evaluate_and_save(model, tokenizer, tasks="all", output="results.json")

# Load pruned checkpoints first
load_checkpoints(model, "path/to/checkpoints", device)
results = evaluate_model(model, tokenizer)
```

### As a script

```bash
# Dense model
python lmvals/eval.py --model Qwen/Qwen3-8B --tasks common

# Pruned model with layer checkpoints
python lmvals/eval.py --model Qwen/Qwen3-8B \
    --ckpt-dir path/to/checkpoints

# All benchmarks (adds mmlu, hellaswag)
python lmvals/eval.py --model Qwen/Qwen3-8B --tasks all

# Options
python lmvals/eval.py --model Qwen/Qwen3-8B \
    --device cuda:1 --batch-size 8 --verbose --output results.json
```

## Files

| File | Purpose |
|------|---------|
| `eval.py` | Evaluation module — importable functions and CLI |
| `SKILL.md` | Agent skill file — teaches agents how to use this tool |
| `evals.txt` | All 14000+ supported lm_eval task names |

## Supported Benchmarks

**Fast** (~1-5 min each on 8B): wikitext, arc_easy, arc_challenge, piqa, winogrande, boolq, lambada_openai

**Slow** (~15-60 min on 8B): mmlu, hellaswag

Use `tasks="common"` for the 7 fast benchmarks, `tasks="all"` for all 9.
