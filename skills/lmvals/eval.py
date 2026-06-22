#!/usr/bin/env python
"""LLM evaluation with lm_eval.

Can be used as a script or imported as a library.

Script usage:
    python lmvals/eval.py --model Qwen/Qwen3-8B --tasks common
    python lmvals/eval.py --model Qwen/Qwen3-8B --ckpt-dir path/to/checkpoints

Library usage:
    from lmvals.eval import evaluate_model, evaluate_tasks, COMMON_TASKS
    results = evaluate_model(model, tokenizer, tasks=["wikitext", "arc_easy"])
    results = evaluate_tasks(model, tokenizer, tasks="common", verbose=True)
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

COMMON_TASKS = [
    "wikitext", "arc_easy", "arc_challenge", "piqa",
    "winogrande", "boolq", "lambada_openai",
]

ALL_TASKS = COMMON_TASKS + ["mmlu", "hellaswag"]

_devnull = open(os.devnull, "w")
_task_mgr = None


def _get_task_mgr():
    global _task_mgr
    if _task_mgr is None:
        _task_mgr = TaskManager()
    return _task_mgr


def suppress_loggers():
    """Suppress noisy loggers from lm_eval, transformers, etc."""
    for name in ("lm_eval", "httpx", "transformers", "datasets", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)


def resolve_tasks(tasks):
    """Resolve task specification to list of task names.

    Args:
        tasks: "common", "all", a list, or comma-separated string.

    Returns:
        List of task name strings.
    """
    if tasks == "common":
        return list(COMMON_TASKS)
    elif tasks == "all":
        return list(ALL_TASKS)
    elif isinstance(tasks, str):
        return [t.strip() for t in tasks.split(",")]
    return list(tasks)


def evaluate_single(model, tokenizer, task, verbose=False, batch_size=4):
    """Run a single lm_eval task on a model.

    Args:
        model: HuggingFace model (already on device).
        tokenizer: Corresponding tokenizer.
        task: Task name string (e.g., "wikitext", "arc_easy").
        verbose: Show progress bars.
        batch_size: Batch size for evaluation.

    Returns:
        Dict of {task/metric_name: value} for all numeric metrics.
    """
    hflm = HFLM(model, tokenizer=tokenizer)
    tm = _get_task_mgr()

    if not verbose:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _devnull, _devnull
    try:
        with torch.no_grad():
            res = simple_evaluate(
                model=hflm, tasks=[task], num_fewshot=0,
                task_manager=tm, log_samples=False,
                batch_size=batch_size,
                verbosity="INFO" if verbose else "ERROR",
            )
    finally:
        if not verbose:
            sys.stdout, sys.stderr = old_out, old_err

    tr = res["results"].get(task, {})
    return {
        f"{task}/{k}": v for k, v in tr.items()
        if k != "alias" and isinstance(v, (int, float))
    }


def evaluate_model(model, tokenizer, tasks="common", verbose=False,
                   batch_size=4, print_results=True):
    """Evaluate a model on multiple lm_eval tasks.

    Args:
        model: HuggingFace model (already on device).
        tokenizer: Corresponding tokenizer.
        tasks: "common", "all", list of names, or comma-separated string.
        verbose: Show progress bars.
        batch_size: Batch size for evaluation.
        print_results: Print results table to stdout.

    Returns:
        Dict of all metrics across all tasks, plus eval_time_s.
    """
    task_list = resolve_tasks(tasks)
    if not verbose:
        suppress_loggers()

    results = {}
    total_time = 0

    if print_results:
        print(f"{'Task':<25} {'Metric':<30} {'Value':>10} {'Time':>6}")
        print("-" * 75)

    for task in task_list:
        t0 = time.time()
        task_results = evaluate_single(
            model, tokenizer, task, verbose=verbose, batch_size=batch_size
        )
        dt = time.time() - t0
        total_time += dt

        if print_results:
            for metric, value in task_results.items():
                val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"  {task:<23} {metric:<30} {val_str:>10} {dt:5.0f}s",
                      flush=True)

        results.update(task_results)

    results["eval_time_s"] = round(total_time, 1)

    if print_results:
        print("-" * 75)
        print(f"Total: {total_time:.0f}s")

    return results


def evaluate_and_save(model, tokenizer, tasks="common", output=None,
                      verbose=False, batch_size=4, metadata=None):
    """Evaluate and save results to JSON incrementally.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        tasks: Task specification.
        output: Output JSON path. None for auto-naming.
        verbose: Show progress.
        batch_size: Batch size.
        metadata: Extra dict to include in results.

    Returns:
        Dict of all results.
    """
    task_list = resolve_tasks(tasks)
    if not verbose:
        suppress_loggers()

    results = {"tasks": task_list, "status": "running"}
    if metadata:
        results.update(metadata)

    if output is None:
        timestamp = time.strftime("%Y%m%d_%H%M")
        output = f"lmvals/eval_{timestamp}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    def save():
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

    save()

    print(f"Evaluating {len(task_list)} tasks → {output}")
    print(f"{'Task':<25} {'Metric':<30} {'Value':>10} {'Time':>6}")
    print("-" * 75)

    total_time = 0
    for task in task_list:
        t0 = time.time()
        task_results = evaluate_single(
            model, tokenizer, task, verbose=verbose, batch_size=batch_size
        )
        dt = time.time() - t0
        total_time += dt

        for metric, value in task_results.items():
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"  {task:<23} {metric:<30} {val_str:>10} {dt:5.0f}s",
                  flush=True)

        results.update(task_results)
        results["eval_time_s"] = round(total_time, 1)
        save()

    print("-" * 75)
    print(f"Total: {total_time:.0f}s → {output}")

    results["status"] = "done"
    save()
    return results


def load_checkpoints(model, ckpt_dir, device):
    """Load pruned layer checkpoints into model.

    Args:
        model: HuggingFace model.
        ckpt_dir: Directory containing layer_NNN.pt files.
        device: Target device.

    Returns:
        Number of layers loaded.
    """
    loaded = 0
    for f in sorted(os.listdir(ckpt_dir)):
        if not (f.startswith("layer_") and f.endswith(".pt")):
            continue
        idx = int(f.replace("layer_", "").replace(".pt", ""))
        ckpt = torch.load(
            os.path.join(ckpt_dir, f), map_location=device, weights_only=True
        )
        model.model.layers[idx].load_state_dict(ckpt)
        loaded += 1
    return loaded


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run lm_eval benchmarks")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--tasks", default="common",
                        help="'common', 'all', or comma-separated task names")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Directory with layer_NNN.pt checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    if args.ckpt_dir:
        n = load_checkpoints(model, args.ckpt_dir, device)
        print(f"Loaded {n} pruned layer checkpoints")
        label = "pruned"
    else:
        print("Evaluating dense model")
        label = "dense"

    output = args.output
    if output is None:
        model_tag = args.model.split("/")[-1]
        timestamp = time.strftime("%Y%m%d_%H%M")
        output = f"lmvals/{model_tag}_{label}_{timestamp}.json"

    metadata = {
        "model": args.model,
        "ckpt_dir": args.ckpt_dir,
        "device": str(device),
    }

    evaluate_and_save(
        model, tokenizer, tasks=args.tasks, output=output,
        verbose=args.verbose, batch_size=args.batch_size, metadata=metadata,
    )


if __name__ == "__main__":
    main()
