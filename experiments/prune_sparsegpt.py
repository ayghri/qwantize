from lm_eval.evaluator import simple_evaluate
from lm_eval.models import huggingface
from lm_eval.tasks import TaskManager
from torch.nn.attention import sdpa_kernel, SDPBackend
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import os
import pandas as pd
import sys
import torch
from astra.data.llm import get_c4
from astra.hooks import ModuleInputCatcher, ModuleOutputCatcher
from astra.misc import transfer_to_device
from astra.sparsegpt import SparseGPT

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"
print(os.environ["HF_HOME"])

# Store original stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
devnull = open(os.devnull, "w")

# Baseline: {'alias': 'wikitext', 'word_perplexity,none': 26.13244747530861, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 1.840855757730666, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 0.8803765871662008, 'bits_per_byte_stderr,none': 'N/A'}
# 0.6B 'alias': 'wikitext', 'word_perplexity,none': 21950376.716961656, 'word_perplexity_stderr,none': 'N/A', 'byte_perplexity,none': 23.598605914052282, 'byte_perplexity_stderr,none': 'N/A', 'bits_per_byte,none': 4.560629729885957, 'bits_per_byte_stderr,none': 'N/A'}}, 'group_subtasks'
model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-1.7B"
# model_name = "Qwen/Qwen3-4B"

teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0",
)
print(teacher)

tokenizer = AutoTokenizer.from_pretrained(model_name)

seq_length = 2048
num_samples = 1024

c4_data = get_c4(
    num_samples=num_samples, seq_len=seq_length, tokenizer=tokenizer, seed=42
)

# CSV output file
csv_output_path = "sparsegpt_benchmark_results.csv"
benchmark_results = []


def save_and_print_results(results_list, csv_path):
    """Save results to CSV and print using pandas."""
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 60)
    print("Benchmark Results:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")
    return df


print("=" * 5, "Model sizes")
print(teacher.model.embed_tokens.weight.numel() / 1000**2)
print(
    sum(
        p.numel()
        for n, p in teacher.model.named_parameters()
        if "self_attn" in n
    )
    / 1000**2
)
print(
    sum(p.numel() for n, p in teacher.model.named_parameters() if "mlp" in n)
    / 1000**2
)
print(teacher.lm_head.weight.numel() / 1000**2)

input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
output_catcher = ModuleOutputCatcher(device=torch.device("cpu"))

hf_model = huggingface.HFLM(teacher, tokenizer=tokenizer)
task_manager = TaskManager()


sys.stdout = devnull
sys.stderr = devnull
with torch.no_grad():
    results = simple_evaluate(
        model=hf_model,
        tasks=["wikitext"],
        num_fewshot=0,
        task_manager=task_manager,
        log_samples=False,
        batch_size=2,
        verbosity="ERROR",
        # delete_requests_cache=True,
    )

sys.stdout = original_stdout
sys.stderr = original_stderr

if results is not None:
    wikitext_results = results["results"].get("wikitext", {})
    benchmark_results.append(
        {
            "layer_idx": -1,
            "stage": "baseline",
            "word_perplexity": wikitext_results.get("word_perplexity,none", None),
            "byte_perplexity": wikitext_results.get("byte_perplexity,none", None),
            "bits_per_byte": wikitext_results.get("bits_per_byte,none", None),
        }
    )
    print(f"Baseline: {wikitext_results}")
    save_and_print_results(benchmark_results, csv_output_path)


tokenized_inputs = [
    {"input_ids": d[0], "attention_mask": torch.ones_like(d[0])}
    for d in c4_data
]

layer_idx = 0
target_layer = teacher.model.layers[layer_idx]
target_layer.device = list(target_layer.parameters())[0].device
layer_name = f"decoder_{layer_idx}"

output_catcher.attach(target_layer, layer_name, raise_error=True)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    with torch.no_grad():
        for layer_ins in tqdm(tokenized_inputs):
            try:
                _ = teacher(
                    **transfer_to_device(layer_ins, teacher.device),
                    labels=None,
                    use_cache=False,
                )
            except Exception:
                pass

layer_targets = output_catcher.outputs[layer_name]
output_catcher.detach(layer_name)

# layer_targets not used in SparseGPT, free it
del layer_targets
torch.cuda.empty_cache()

input_catcher.attach(target_layer, layer_name, raise_error=True)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    with torch.no_grad():
        for layer_ins in tqdm(tokenized_inputs):
            try:
                _ = teacher(
                    **transfer_to_device(layer_ins, teacher.device),
                    labels=None,
                    use_cache=False,
                )
            except Exception:
                pass
layer_inputs = input_catcher.inputs[layer_name]
input_catcher.detach(layer_name)


for layer_idx in range(len(teacher.model.layers)):
    print(f"Pruning linear layer of decoder layer{layer_idx}")
    layer_name = f"decoder_{layer_idx}"
    target_layer = teacher.model.layers[layer_idx]
    layers_to_prune = {}

    for layer_name, layer in target_layer.self_attn.named_children():
        if "_proj" in layer_name:
            print(layer_name, layer)
            layers_to_prune[layer_name] = layer

    for layer_name, layer in target_layer.mlp.named_children():
        if "_proj" in layer_name:
            # print(layer_name, layer)
            layers_to_prune[layer_name] = layer

    for layer_name, layer in layers_to_prune.items():
        input_catcher.attach(layer, layer_name)

    with torch.no_grad():
        for layer_ins in tqdm(layer_inputs):
            # print(layer_ins.keys())
            layer_ins = transfer_to_device(layer_ins, target_layer.device)  # type: ignore[assignment]
            # print(layer_ins)

            # model_inputs = transfer_to_device(layer_inputs[idx], network.device)
            # target = transfer_to_device(layer_targets[idx],network.device)
            # num_batches += 1
            # pred = network(model_inputs["args"][0], **model_inputs["kwargs"])
            _ = target_layer(
                *layer_ins["args"], **layer_ins["kwargs"]  # type: ignore[index]
            )

    for layer_name, layer in layers_to_prune.items():
        gpt = SparseGPT(layer)

        for ins in input_catcher.inputs[layer_name]:
            # [:10]:
            gpt.add_batch(ins["args"][0][0].to(layer.weight.device), None)

        print("Pruning layer:", layer_name)
        gpt.fasterprune(0.5, prune_n=2, prune_m=4)
        gpt.free()

        torch.cuda.empty_cache()

    for layer_name in layers_to_prune:
        input_catcher.detach(layer_name)
    assert len(input_catcher.inputs) == 0

    torch.cuda.empty_cache()

    # Evaluate every 2 layers
    if (layer_idx + 1) % 2 == 0:
        print(f"\nEvaluating after pruning layer {layer_idx}...")
        sys.stdout = devnull
        sys.stderr = devnull
        with torch.no_grad():
            results = simple_evaluate(
                model=hf_model,
                tasks=["wikitext"],
                num_fewshot=0,
                task_manager=task_manager,
                log_samples=False,
                batch_size=2,
                verbosity="ERROR",
            )
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        if results is not None:
            wikitext_results = results["results"].get("wikitext", {})
            benchmark_results.append(
                {
                    "layer_idx": layer_idx,
                    "stage": f"after_layer_{layer_idx}",
                    "word_perplexity": wikitext_results.get("word_perplexity,none", None),
                    "byte_perplexity": wikitext_results.get("byte_perplexity,none", None),
                    "bits_per_byte": wikitext_results.get("bits_per_byte,none", None),
                }
            )
            print(f"Layer {layer_idx} results: {wikitext_results}")
            save_and_print_results(benchmark_results, csv_output_path)

    print("=" * 20, "\n" * 2)
    if layer_idx < len(teacher.model.layers) - 1:
        target_layer = teacher.model.layers[layer_idx + 1]
        target_layer.device = list(target_layer.parameters())[0].device

        output_catcher.attach(target_layer, layer_name, raise_error=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.no_grad():
                for layer_ins in tqdm(layer_inputs):
                    try:
                        layer_ins_dev = transfer_to_device(layer_ins, target_layer.device)
                        _ = target_layer(
                            *layer_ins_dev["args"], **layer_ins_dev["kwargs"]  # type: ignore[index]
                        )
                    except Exception:
                        pass

        layer_targets = output_catcher.outputs[layer_name]
        output_catcher.detach(layer_name)

        # layer_targets not used in SparseGPT, free it
        del layer_targets

        input_catcher.attach(target_layer, layer_name, raise_error=True)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.no_grad():
                for layer_ins in tqdm(layer_inputs):
                    try:
                        layer_ins_dev = transfer_to_device(layer_ins, target_layer.device)
                        _ = target_layer(
                            *layer_ins_dev["args"], **layer_ins_dev["kwargs"]  # type: ignore[index]
                        )
                    except Exception:
                        pass
        # Free old inputs before reassigning
        del layer_inputs
        layer_inputs = input_catcher.inputs[layer_name]
        input_catcher.detach(layer_name)
        torch.cuda.empty_cache()

# Final evaluation
print("\n" + "=" * 60)
print("Final evaluation after all layers pruned...")
print("=" * 60)

sys.stdout = devnull
sys.stderr = devnull
with torch.no_grad():
    results = simple_evaluate(
        model=hf_model,
        tasks=["wikitext"],
        num_fewshot=0,
        task_manager=task_manager,
        log_samples=False,
        batch_size=2,
        verbosity="ERROR",
    )

sys.stdout = original_stdout
sys.stderr = original_stderr

if results is not None:
    wikitext_results = results["results"].get("wikitext", {})
    benchmark_results.append(
        {
            "layer_idx": len(teacher.model.layers) - 1,
            "stage": "final",
            "word_perplexity": wikitext_results.get("word_perplexity,none", None),
            "byte_perplexity": wikitext_results.get("byte_perplexity,none", None),
            "bits_per_byte": wikitext_results.get("bits_per_byte,none", None),
        }
    )
    print(f"Final results: {wikitext_results}")

print("\n" + "=" * 60)
print("PRUNING COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {csv_output_path}")
save_and_print_results(benchmark_results, csv_output_path)

