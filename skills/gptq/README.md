# gptq — SparseGPT & GPTQ Skill

Self-contained one-shot pruning (SparseGPT) and post-training quantization (GPTQ) for linear layers, built on PyTorch. Works as a Python library or as a Claude Code agent skill.

## Adding as an Agent Skill

To give a Claude Code agent access to this pruning/quantization tool, add `gptq/SKILL.md` to the agent's skill files.

### In Claude Code CLI

Reference the skill file in your project's `CLAUDE.md`:

```markdown
For weight pruning and quantization, follow the guide in gptq/SKILL.md
```

### In a custom agent (Claude Agent SDK)

Include `SKILL.md` in the agent's system prompt or tool instructions:

```python
skill = open("gptq/SKILL.md").read()
system_prompt = f"""
You are a research assistant for neural network compression.

{skill}
"""
```

### As an MCP resource

Serve `SKILL.md` as a resource from an MCP server so any connected agent can discover it.

## Quick Start

### Install dependencies

```bash
pip install torch transformers datasets
```

### SparseGPT — prune a full model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptq import prune_model, get_calibration_data

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

tokens = get_calibration_data(tokenizer, nsamples=128, seqlen=2048)
model = prune_model(model, tokens, prune_n=2, prune_m=4, device="cuda:0")
```

### SparseGPT — prune a single layer

```python
from gptq import sparsegpt, compute_hessian

H = compute_hessian(X)  # X: [N, in_features] activations
W_pruned = sparsegpt(layer.weight.data, H, prune_n=2, prune_m=4)
layer.weight.data = W_pruned.to(layer.weight.dtype)
```

### GPTQ — quantize with a custom scheme

```python
from gptq import gptq, make_uniform_quantizer

# Built-in INT4
Q, losses = gptq(W, H, make_uniform_quantizer(bits=4))

# Or bring your own quantizer
def my_quantizer(w_col):
    scale = w_col.abs().max() / 7
    return (w_col / scale).round().clamp(-8, 7) * scale

Q, losses = gptq(W, H, my_quantizer)
```

The `quantize_fn` can also accept `(W, H, done_mask, target_mask)` for full context (see `SKILL.md` for details).

## Files

| File | Purpose |
|------|---------|
| `obs.py` | Shared OBS utilities: Hessian, Cholesky, layer discovery, hooks |
| `sparsegpt.py` | SparseGPT: core algorithm + full LLM pipeline + calibration data |
| `gptq.py` | GPTQ: core algorithm + example quantizer factories |
| `SKILL.md` | Agent skill file with full API reference |
