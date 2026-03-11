import torch
import os
from pathlib import Path
from torch.nn.utils import clip_grad_norm_

from astra.hooks import ModuleInputCatcher, ModuleOutputCatcher
from copy import deepcopy
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch import nn
import numpy as np
import torch.nn.utils.prune as prune

# from spastra.blocks import BlockCoupling
from astra.groups import GroupSpec
from astra.misc import transfer_to_device
from astra.blocks import BlockSpec
from astra.proximals import AdamProxy
from astra.evaluate import evaluate_ppl_hf


# base_dir = Path("/buckets/")
base_dir = Path("~/scratch/buckets/")
base_dir.mkdir(parents=True, exist_ok=True)
# if "HF_HOME" not in os.environ:
os.environ["HF_HOME"] = str(base_dir / "datasets/huggingface")

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


print(os.environ["HF_HOME"])

seq_length = 1024
num_samples = 1024

threshold_epochs = 5

recover_epochs = 10

model_name = "Qwen/Qwen3-8B"

teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

student = teacher

print("embed_weights", teacher.model.embed_tokens.weight.numel() / 1000**2)
print(
    "attn weights",
    sum(
        p.numel()
        for n, p in teacher.model.named_parameters()
        if "self_attn" in n
    )
    / 1000**2,
)
print(
    "mlp weights",
    sum(p.numel() for n, p in teacher.model.named_parameters() if "mlp" in n)
    / 1000**2,
)
print("lm head weights", teacher.lm_head.weight.numel() / 1000**2)


ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

wikitext = " ".join(ds["text"])

len(wikitext)

input_text = []
for i in range(num_samples):
    input_text.append(wikitext[i * seq_length : (i + 1) * seq_length])

tokenized_inputs = [
    tokenizer([text], return_tensors="pt", return_token_type_ids=False)
    for text in input_text
]


input_catcher = ModuleInputCatcher(device=torch.device("cpu"))
output_catcher = ModuleOutputCatcher(device=torch.device("cpu"))

layer_idx = 0
layer_name = f"decoder_{layer_idx}"

first_layer = teacher.model.layers[layer_idx]
input_catcher.attach(first_layer, layer_name)

print("Computing teacher inputs")
with torch.no_grad():
    for model_inputs in tqdm(tokenized_inputs):
        _ = teacher(
            **model_inputs.to(teacher.device), labels=None, use_cache=False
        )

teacher_inputs = input_catcher.inputs[layer_name]
student_inputs = deepcopy(teacher_inputs)
input_catcher.detach(layer_name)


all_layers = student.model.layers
prev_layers = list(range(0))

for layer_idx in range(len(prev_layers)):
    layer_name = f"decoder_{layer_idx}"
    print(prev_layers, layer_idx, layer_name)

    teacher_layer = teacher.model.layers[layer_idx]
    teacher_layer.device = list(teacher_layer.parameters())[0].device

    output_catcher.attach(teacher_layer, layer_name)
    with torch.no_grad():
        for model_inputs in tqdm(teacher_inputs):
            model_inputs = transfer_to_device(
                model_inputs, teacher_layer.device
            )
            _ = teacher_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    teacher_targets = output_catcher.outputs[layer_name]
    output_catcher.detach(layer_name)

    student_layer = student.model.layers[layer_idx]
    layer_ckpt_path = (
        base_dir / f"checkpoints/{model_name}_decoder_{layer_idx}.cpt"
    )
    print("Loading:", layer_ckpt_path)
    student_layer.load_state_dict(torch.load(layer_ckpt_path))

    torch.cuda.empty_cache()

    for t_input, t_target in zip(teacher_inputs, teacher_targets):
        t_input["args"] = (t_target,)

    output_catcher.attach(student_layer, layer_name)
    with torch.no_grad():
        for model_inputs in tqdm(student_inputs):
            model_inputs = transfer_to_device(
                model_inputs, student_layer.device
            )
            _ = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    for s_input, s_target in zip(
        student_inputs, output_catcher.outputs[layer_name]
    ):
        s_input["args"] = (s_target,)

    output_catcher.detach(layer_name)

    torch.cuda.empty_cache()

# torch.cuda.empty_cache()
# results = evaluate_ppl_hf(teacher, tokenizer, silent=True)
# print("results before pruning: ", results)
# torch.cuda.empty_cache()


for layer_idx in range(len(prev_layers), len(all_layers)):
    layer_name = f"decoder_{layer_idx}"
    print(prev_layers, layer_idx, layer_name)

    teacher_layer = teacher.model.layers[layer_idx]
    teacher_layer.device = list(teacher_layer.parameters())[0].device

    output_catcher.attach(teacher_layer, layer_name)
    with torch.no_grad():
        for model_inputs in tqdm(teacher_inputs):
            model_inputs = transfer_to_device(
                model_inputs, teacher_layer.device
            )
            _ = teacher_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    teacher_targets = output_catcher.outputs[layer_name]
    output_catcher.detach(layer_name)

    # for prev_l in prev_layers:
    #     layer_ckpt_path = Path(
    #         f"/buckets/checkpoints/{model_name}_decoder_{prev_l}.cpt"
    #     )
    #     print("Loading:", layer_ckpt_path)
    #     student.model.layers[prev_l].load_state_dict(
    #         torch.load(layer_ckpt_path)
    #     )

    student_layer = student.model.layers[layer_idx]

    for p in student.model.parameters():
        p.requires_grad = False

    for n, p in student_layer.named_parameters():
        p.requires_grad = True
        if "norm" in n:
            p.requires_grad = False

    original_weights = deepcopy(student_layer.state_dict())

    optimizer = Adam(
        student_layer.parameters(),
        lr=2e-5,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    )

    criterion = nn.MSELoss()

    prune_layers = {}

    for n, l_name in student_layer.self_attn.named_children():
        if isinstance(l_name, nn.Linear):
            prune_layers[n] = l_name

    for n, l_name in student_layer.mlp.named_children():
        if isinstance(l_name, nn.Linear):
            prune_layers[n] = l_name

    groups = []

    for n, p in student_layer.self_attn.named_parameters():
        if "_proj.weight" in n and p.requires_grad:
            groups.append(
                GroupSpec(
                    BlockSpec(p, block_shape=(1, 1)), group_shape=(1, 4), name=n
                )
            )

    for n, p in student_layer.mlp.named_parameters():
        if "_proj.weight" in n and p.requires_grad:
            groups.append(
                GroupSpec(
                    BlockSpec(p, block_shape=(1, 1)), group_shape=(1, 4), name=n
                )
            )

    groups_nnz = [2] * 4 + [2] * 3

    assert len(groups_nnz) == len(groups)

    lambds = {g: torch.zeros_like(g.kth_largest(None, 1)) for g in groups}

    beta = 0.9

    proxy = AdamProxy(optimizer)

    import numpy as np

    for layer_name, layer in prune_layers.items():
        input_catcher.attach(layer, layer_name)

    student_layer.device = list(student_layer.parameters())[0].device

    pbar = tqdm(range(len(student_inputs)), desc=f"Eval initial loss")
    for idx in pbar:
        model_inputs = transfer_to_device(
            student_inputs[idx], student_layer.device
        )
        target = transfer_to_device(teacher_targets[idx], student_layer.device)
        pred = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    alphas = {}

    with torch.no_grad():
        i = 0
        for layer_name, layer in prune_layers.items():
            print(layer_name)
            X = 0.0
            g = groups[i]
            i += 1

            for ins in tqdm(input_catcher.inputs[layer_name]):
                batch = ins["args"][0][0]
                X = X + batch.to(student_layer.device).square().mean(dim=0)

            alphas[g.block] = (
                X / len(input_catcher.inputs[layer_name]) + 1e-12
            ).unsqueeze(0)

    for layer_name, layer in prune_layers.items():
        input_catcher.detach(layer_name)

    for b, a in alphas.items():
        alphas[b] = (a / a.mean()) * 1e-3
        alphas[b].clamp_(min=1e-4)

    for b, a in alphas.items():
        alphas[b] = a.to(student_layer.device)

    with torch.no_grad():
        pbar = tqdm(range(len(student_inputs)), desc=f"Eval initial loss")
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        for idx in pbar:
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            num_batches += 1
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            total_loss += criterion(pred, target).item()
            total_mse += criterion(torch.zeros_like(target), target).item()
            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.6f}",
                mse=f"{total_mse / num_batches:.6f}",
                density=np.mean([g.nnz() / g.numel() for g in groups]),
            )

    torch.cuda.empty_cache()

    for epoch in range(threshold_epochs):
        pbar = tqdm(
            np.random.permutation(len(student_inputs)),
            desc=f"Epoch {epoch + 1}/{threshold_epochs}",
        )
        total_loss = 0.0
        num_batches = 0
        for idx in pbar:
            num_batches += 1

            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )

            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for g_nnz, g in zip(groups_nnz, groups):
                conditioners = {}
                psis = {}
                block = g.block

                gradient, lr, conditioner = proxy.get_info(block.param)
                psi = (gradient - alphas[g.block] * block.param.data).abs()
                vals = g.kth_largest({block: psi}, num_nz=g_nnz + 1)
                vals.add_(g.kth_largest({block: psi}, num_nz=g_nnz))
                vals.mul_(0.5)

                lambds[g].mul_(beta).add_((1 - beta) * vals)
                g.soft_threshold(
                    lambds[g] * lr, conditioners={block: conditioner}
                )

            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.6f}",
                density=np.mean([g.nnz() / g.numel() for g in groups]),
            )

    for n, p in student_layer.named_parameters():
        if "proj" in n:
            print(n, ((p.data.abs() > 1e-12).sum() / p.numel()).item())

    param_masks = {}

    for g_nnz, g in zip(groups_nnz, groups):
        for b, m in g.get_masks(num_nz=g_nnz).items():
            param_masks[b.param] = m

    for p, m in param_masks.items():
        print(p.shape, m.sum() / m.numel())

    student_layer.load_state_dict(deepcopy(original_weights))

    for p, m in param_masks.items():
        p.data.mul_(m)

    for layer_name, layer in student_layer.self_attn.named_children():
        if isinstance(layer, nn.Linear):
            print(layer_name)
            mask = param_masks[layer.weight]
            prune.custom_from_mask(layer, "weight", mask)

    for layer_name, layer in student_layer.mlp.named_children():
        if isinstance(layer, nn.Linear):
            print(layer_name)
            mask = param_masks[layer.weight]
            prune.custom_from_mask(layer, "weight", mask)

    optimizer = AdamW(
        student_layer.parameters(),
        lr=4e-5,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
    )

    for epoch in range(recover_epochs):
        pbar = tqdm(
            np.random.permutation(len(student_inputs)),
            desc=f"Epoch {epoch + 1}/{recover_epochs}",
        )
        total_loss = 0.0
        num_batches = 0
        for idx in pbar:
            num_batches += 1

            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )

            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=f"{total_loss / num_batches:.6f}")

        # for layer_name, layer in prune_layers.items():
        #     if isinstance(layer, nn.Linear):
        #         p = layer.weight
        #         print(
        #             layer_name,
        #             ((p.data.abs() > 1e-12).sum() / p.numel()).item(),
        #         )

    for layer_name, layer in prune_layers.items():
        print(layer_name)
        prune.remove(layer, "weight")

    for p, m in param_masks.items():
        p.data.mul_(m)

    layer_ckpt_path = (
        base_dir / f"checkpoints/{model_name}_decoder_{layer_idx}.cpt"
    )
    layer_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving layer {layer_idx} weights to {layer_ckpt_path}")

    torch.save(student_layer.state_dict(), layer_ckpt_path)

    with torch.no_grad():
        pbar = tqdm(range(len(student_inputs)), desc=f"Eval initial loss")
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        for idx in pbar:
            model_inputs = transfer_to_device(
                student_inputs[idx], student_layer.device
            )
            target = transfer_to_device(
                teacher_targets[idx], student_layer.device
            )
            # target = transfer_to_device(layer_targets[idx],network.device)[0]
            num_batches += 1
            pred = student_layer(
                model_inputs["args"][0], **model_inputs["kwargs"]
            )
            total_loss += criterion(pred, target).item()
            total_mse += criterion(torch.zeros_like(target), target).item()
            pbar.set_postfix(
                loss=f"{total_loss / num_batches:.6f}",
                mse=f"{total_mse / num_batches:.6f}",
                density=np.mean([g.nnz() / g.numel() for g in groups]),
            )

    for layer_name, layer in list(
        student_layer.self_attn.named_children()
    ) + list(student_layer.mlp.named_children()):
        if isinstance(layer, nn.Linear):
            p = layer.weight
            print(layer_name, ((p.data.abs() > 1e-12).sum() / p.numel()).item())

    results = evaluate_ppl_hf(student, tokenizer, silent=True)
    print(f"After {layer_name} pruned: ", results)
    torch.cuda.empty_cache()

    for t_input, t_target in zip(teacher_inputs, teacher_targets):
        t_input["args"] = (t_target,)

    output_catcher.attach(student_layer, layer_name)
    with torch.no_grad():
        for model_inputs in tqdm(student_inputs):
            model_inputs = transfer_to_device(
                model_inputs, student_layer.device
            )
            _ = student_layer(model_inputs["args"][0], **model_inputs["kwargs"])

    for s_input, s_target in zip(
        student_inputs, output_catcher.outputs[layer_name]
    ):
        s_input["args"] = (s_target,)

    output_catcher.detach(layer_name)

    torch.cuda.empty_cache()
    prev_layers.append(layer_idx)
