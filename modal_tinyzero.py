"""
Modal wrapper for running TinyZero remotely on Modal GPUs.

This wrapper keeps all heavy work off the local machine:
  1. TinyZero is cloned and installed inside the Modal image.
  2. Countdown data is prepared on Modal.
  3. Training is launched on a remote GPU container.

Usage examples:
  modal run modal_tinyzero.py
  modal run --detach modal_tinyzero.py --base-model Qwen/Qwen2.5-3B --n-gpus 2
  modal run modal_tinyzero.py --prepare-data False --extra-hydra-args "critic.model.enable_gradient_checkpointing=True"
  modal run modal_tinyzero.py --dataset-name gsm8k --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1
  modal run modal_tinyzero.py --dataset-name gsm8k --prompt-conditioning-mode mixed --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1
  modal run modal_tinyzero.py --dataset-name math500 --total-training-steps 250 --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1
  modal run --detach modal_tinyzero.py --dataset-name countdown --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1 --prompt-conditioning-mode adaptive --total-training-steps 350 --seed 1
  modal run --detach modal_tinyzero.py --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1 --gpu-type A100-40GB

Notes:
  - This wrapper targets the archived TinyZero countdown setup from GitHub.
  - The default path is the 2-GPU Qwen2.5-3B run described in the TinyZero README.
  - prompt-conditioning-mode=off is the single-prompt baseline; mixed is uniform round-robin; adaptive is a softmax bandit over per-family success rate.
  - gpu-type defaults to A100-80GB. Use list_available_gpu_types() to see supported Modal GPU strings.
"""

from __future__ import annotations

import json
import re
import shlex
import textwrap
import ast
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import modal


APP_NAME = "tinyzero-modal"
REPO_URL = "https://github.com/zefangzh/TinyZero.git"
REMOTE_REPO_DIR = "/root/TinyZero"
ARTIFACT_DIR = "/root/tinyzero-artifacts"
DATA_ROOT = f"{ARTIFACT_DIR}/data"
RUN_ROOT = f"{ARTIFACT_DIR}/runs"
HF_CACHE = f"{ARTIFACT_DIR}/hf"

app = modal.App(APP_NAME)
wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": "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"})

_shared_env = {
    "HF_HOME": HF_CACHE,
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.9")
    .apt_install("git", "build-essential", "curl", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        f"git clone {REPO_URL} {REMOTE_REPO_DIR}",
        "python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121",
        "python -m pip install vllm==0.6.3 ray",
        f"cd {REMOTE_REPO_DIR} && python -m pip install -e .",
        "python -m pip install flash-attn --no-build-isolation",
        "python -m pip install wandb ipython matplotlib",
    )
    .env(
        {
            **_shared_env,
            "VLLM_ATTENTION_BACKEND": "XFORMERS",
            "HYDRA_FULL_ERROR": "1",
        }
    )
)

artifacts = modal.Volume.from_name("tinyzero-artifacts", create_if_missing=True)

DATASET_SOURCES = {
    "gsm8k": "openai/gsm8k",
    "math500": "HuggingFaceH4/MATH-500",
    "aime24": "HuggingFaceH4/aime_2024",
}

DEFAULT_GPU_TYPE = "A100-80GB"
SUPPORTED_MODAL_GPU_TYPES = {
    "T4": "T4",
    "L4": "L4",
    "A10": "A10",
    "L40S": "L40S",
    "A100": "A100",
    "A100-40GB": "A100-40GB",
    "A100-80GB": "A100-80GB",
    "RTX-PRO-6000": "RTX-PRO-6000",
    "H100": "H100",
    "H100!": "H100!",
    "H200": "H200",
    "B200": "B200",
    "B200+": "B200+",
}
MODAL_GPU_MAX_COUNTS = {
    "A10": 4,
    "T4": 8,
    "L4": 8,
    "L40S": 8,
    "A100": 8,
    "A100-40GB": 8,
    "A100-80GB": 8,
    "RTX-PRO-6000": 8,
    "H100": 8,
    "H100!": 8,
    "H200": 8,
    "B200": 8,
    "B200+": 8,
}


def list_available_gpu_types() -> list[str]:
    """Return Modal GPU type strings accepted by --gpu-type."""
    return list(SUPPORTED_MODAL_GPU_TYPES)


def _normalize_gpu_type(gpu_type: str) -> str:
    normalized = gpu_type.strip().upper()
    aliases = {
        "A100-40G": "A100-40GB",
        "A100:40G": "A100-40GB",
        "A100-80G": "A100-80GB",
        "A100:80G": "A100-80GB",
        "A10G": "A10",
        "RTXPRO6000": "RTX-PRO-6000",
        "RTX-PRO6000": "RTX-PRO-6000",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODAL_GPU_TYPES:
        supported = ", ".join(list_available_gpu_types())
        raise ValueError(f"Unsupported gpu_type={gpu_type!r}. Supported Modal GPU types: {supported}")
    return normalized


def _modal_gpu_spec(gpu_type: str, n_gpus: int) -> str:
    if n_gpus < 1:
        raise ValueError("n_gpus must be >= 1.")
    gpu = SUPPORTED_MODAL_GPU_TYPES[_normalize_gpu_type(gpu_type)]
    max_count = MODAL_GPU_MAX_COUNTS[gpu]
    if n_gpus > max_count:
        raise ValueError(f"Modal GPU type {gpu} supports at most {max_count} GPUs per container.")
    return gpu if n_gpus == 1 else f"{gpu}:{n_gpus}"


def _repo_checkout(repo_ref: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", REMOTE_REPO_DIR, "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", REMOTE_REPO_DIR, "checkout", repo_ref], check=True)


def _countdown_family_instruction(prompt_family: str) -> str:
    instructions = {
        "default": "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
        "target_first": "Work backward from the target before choosing an expression. Show your work in <think> </think> tags and return the final equation in <answer> </answer> tags.",
        "build_up": "First build useful intermediate numbers from the inputs, then combine them. Show your work in <think> </think> tags and return the final equation in <answer> </answer> tags.",
        "verify": "Check that each number is used at most once and verify the arithmetic before finalizing. Show your work in <think> </think> tags and return the final equation in <answer> </answer> tags.",
    }
    return instructions[prompt_family]


def _make_countdown_prefix(example: dict, template_type: str, prompt_family: str = "default") -> str:
    target = example["target"]
    numbers = example["nums"]
    family_instruction = _countdown_family_instruction(prompt_family)
    if template_type == "base":
        return (
            "A conversation between User and Assistant.\n"
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. "
            f"User: Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /). Use each provided "
            "number exactly once, do not introduce any other numbers, and stop immediately "
            f"after the closing </answer> tag. This problem is guaranteed to have a solution. {family_instruction} "
            "Assistant: Let me solve this step by step.\n<think>"
        )
    if template_type == "qwen-instruct":
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant. You first thinks about the reasoning process "
            "in the mind and then provides the user with the answer.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /). Use each provided "
            "number exactly once, do not introduce any other numbers, and stop immediately "
            f"after the closing </answer> tag. This problem is guaranteed to have a solution. {family_instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Let me solve this step by step.\n<think>"
        )
    raise ValueError(f"Unsupported template_type: {template_type}")


def _normalize_countdown_numbers(numbers) -> tuple[int, ...]:
    if isinstance(numbers, str):
        return tuple(int(item) for item in re.findall(r"-?\d+", numbers))
    return tuple(int(item) for item in list(numbers))


@lru_cache(maxsize=1_000_000)
def _countdown_solution_expression_cached(target: int, numbers_key: tuple[int, ...]) -> str | None:
    target_value = Fraction(int(target), 1)
    initial_items = tuple((Fraction(number, 1), str(number)) for number in numbers_key)
    seen: set[tuple[tuple[int, int], ...]] = set()

    def search(items: tuple[tuple[Fraction, str], ...]) -> str | None:
        if len(items) == 1:
            value, expr = items[0]
            return expr if value == target_value else None

        value_key = tuple(sorted((value.numerator, value.denominator) for value, _ in items))
        if value_key in seen:
            return None
        seen.add(value_key)

        item_count = len(items)
        for i in range(item_count):
            for j in range(i + 1, item_count):
                a_value, a_expr = items[i]
                b_value, b_expr = items[j]
                rest = tuple(items[k] for k in range(item_count) if k not in {i, j})
                candidates: list[tuple[Fraction, str]] = [
                    (a_value + b_value, f"({a_expr} + {b_expr})"),
                    (a_value * b_value, f"({a_expr} * {b_expr})"),
                    (a_value - b_value, f"({a_expr} - {b_expr})"),
                    (b_value - a_value, f"({b_expr} - {a_expr})"),
                ]
                if b_value != 0:
                    candidates.append((a_value / b_value, f"({a_expr} / {b_expr})"))
                if a_value != 0:
                    candidates.append((b_value / a_value, f"({b_expr} / {a_expr})"))

                for candidate in candidates:
                    result = search(rest + (candidate,))
                    if result is not None:
                        return result
        return None

    return search(initial_items)


def _countdown_solution_expression(target, numbers) -> str | None:
    numbers_key = tuple(sorted(_normalize_countdown_numbers(numbers)))
    return _countdown_solution_expression_cached(int(target), numbers_key)


def _is_solvable_countdown_example(example: dict) -> bool:
    return _countdown_solution_expression(example["target"], example["nums"]) is not None


def _extract_countdown_answer(output: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return ""
    answer = matches[-1].strip()
    answer = re.sub(r"<\|.*?\|>", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def _candidate_countdown_expressions(answer: str) -> list[str]:
    answer = re.sub(r"\\boxed\s*\{(.*?)\}", r"\1", answer)
    answer = answer.replace("^", "**")
    if "=" in answer:
        return [part.strip() for part in answer.split("=") if part.strip()]
    return [answer.strip()] if answer.strip() else []


def _safe_eval_countdown_expr(expr: str) -> tuple[Fraction, list[Fraction]] | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    used_numbers: list[Fraction] = []

    def visit(node) -> Fraction | None:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = Fraction(str(node.value))
            used_numbers.append(abs(value))
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = visit(node.left)
            right = visit(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                return None
            return left / right
        return None

    value = visit(tree)
    if value is None:
        return None
    return value, used_numbers


def _countdown_ground_truth_parts(gts) -> tuple[int, tuple[int, ...]] | None:
    if not isinstance(gts, dict) or "target" not in gts or "numbers" not in gts:
        return None
    return int(gts["target"]), _normalize_countdown_numbers(gts["numbers"])


def _is_valid_countdown_ground_truth(gts) -> bool:
    parts = _countdown_ground_truth_parts(gts)
    if parts is None:
        return True
    target, numbers = parts
    return _countdown_solution_expression(target, numbers) is not None


def _has_valid_countdown_equation(output: str, gts) -> bool:
    parts = _countdown_ground_truth_parts(gts)
    if parts is None:
        return True
    target, numbers = parts
    answer = _extract_countdown_answer(output)
    if not answer or answer.upper() == "NO_SOLUTION":
        return False
    expected_counts = Counter(Fraction(number, 1) for number in numbers)
    target_value = Fraction(target, 1)
    for expr in _candidate_countdown_expressions(answer):
        evaluated = _safe_eval_countdown_expr(expr)
        if evaluated is None:
            continue
        value, used_numbers = evaluated
        if value == target_value and Counter(used_numbers) == expected_counts:
            return True
    return False


def _extract_gsm8k_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker, 1)[1].strip()
    return answer_text.strip()


def _make_gsm8k_prompt(question: str, prompt_family: str = "default") -> str:
    family_instructions = {
        "default": 'Let\'s think step by step and output the final answer after "####".',
        "verify": 'Reason step by step, verify each intermediate calculation, and output the final answer after "####".',
        "decompose": 'Break the problem into smaller subproblems, solve them carefully, and output the final answer after "####".',
        "equation_first": 'Translate the problem into equations before solving, then output the final answer after "####".',
    }
    instruction = family_instructions[prompt_family]
    return f"{question.strip()} {instruction}"


def _make_competition_math_prompt(problem: str, prompt_family: str = "default") -> str:
    family_instructions = {
        "default": "Solve the following math problem step by step. Please put your final answer within \\boxed{}.",
        "verify": "Solve the following math problem step by step. Verify each algebraic and arithmetic step, and put your final answer within \\boxed{}.",
        "decompose": "Decompose the following math problem into smaller subproblems, solve carefully, and put your final answer within \\boxed{}.",
        "equation_first": "Translate the following math problem into equations or cases before solving, and put your final answer within \\boxed{}.",
    }
    instruction = family_instructions[prompt_family]
    return f"{instruction}\n\n{problem.strip()}\n\nRemember to put your final answer within \\boxed{{}}."


def _prompt_families_for_dataset(dataset_name: str) -> list[str]:
    if dataset_name == "countdown":
        return ["default", "target_first", "build_up", "verify"]
    if dataset_name in {"gsm8k", "math500", "aime24"}:
        return ["default", "verify", "decompose", "equation_first"]
    return ["default"]


def _build_prompt_variants(dataset_name: str, example: dict, template_type: str) -> dict[str, str]:
    families = _prompt_families_for_dataset(dataset_name)
    if dataset_name == "countdown":
        return {family: _make_countdown_prefix(example, template_type, family) for family in families}
    if dataset_name == "gsm8k":
        return {family: _make_gsm8k_prompt(example["question"], family) for family in families}
    if dataset_name in {"math500", "aime24"}:
        return {family: _make_competition_math_prompt(example["problem"], family) for family in families}
    return {"default": str(example)}


def _select_prompt_family(*, dataset_name: str, split: str, idx: int, prompt_conditioning_mode: str) -> str:
    if split != "train" or prompt_conditioning_mode == "off":
        return "default"
    if prompt_conditioning_mode not in {"mixed", "adaptive"}:
        raise ValueError(f"Unsupported prompt_conditioning_mode: {prompt_conditioning_mode}")
    families = _prompt_families_for_dataset(dataset_name)
    return families[idx % len(families)]


def _split_single_split_dataset(dataset, *, train_size: int, test_size: int, default_train_size: int, default_test_size: int):
    total = len(dataset)
    if train_size == 0 and test_size == 0:
        train_size = default_train_size
        test_size = default_test_size
    elif train_size == 0:
        train_size = total - test_size
    elif test_size == 0:
        test_size = total - train_size

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must both be positive for single-split datasets.")
    if train_size + test_size > total:
        raise ValueError("Requested train_size + test_size exceeds the available dataset size.")

    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, train_size + test_size))
    return train_dataset, test_dataset


def _parse_eval_ks(eval_ks: str, validation_rollouts: int) -> list[int]:
    ks = []
    for chunk in eval_ks.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            continue
        if value <= validation_rollouts:
            ks.append(value)
    if not ks:
        ks = [min(validation_rollouts, 1)]
    if validation_rollouts >= 1:
        ks.append(1)
    return sorted(set(ks))


def _extract_reasoning_text(output: str) -> str:
    text = output
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if think_match:
        text = think_match.group(1)
    text = re.sub(r"<answer>.*?</answer>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.split("####", 1)[0]
    text = re.sub(r"\\boxed\s*\{.*?\}", " ", text, flags=re.DOTALL)
    return text


def _reasoning_signature(output: str) -> str:
    text = _extract_reasoning_text(output).lower()
    text = re.sub(r"\d+(?:\.\d+)?", "<num>", text)
    text = re.sub(r"[^a-z<>\+\-\*\/=\(\)\[\]\{\}\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:64])


def _compute_validation_passk_clusters(validation_data_dir: Path, ks: list[int]) -> dict[int, dict[str, float]]:
    per_step_metrics: dict[int, dict[str, float]] = {}
    for jsonl_path in sorted(validation_data_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        groups: dict[str, list[dict]] = defaultdict(list)
        with open(jsonl_path, "r", encoding="utf-8") as infile:
            for line in infile:
                record = json.loads(line)
                group_key = f"{record.get('input', '')}||{record.get('gts', '')}"
                groups[group_key].append(record)

        skipped_invalid_groups = 0
        valid_groups: dict[str, list[dict]] = {}
        for group_key, records in groups.items():
            if records and not _is_valid_countdown_ground_truth(records[0].get("gts")):
                skipped_invalid_groups += 1
                continue
            valid_groups[group_key] = records
        groups = valid_groups

        if not groups:
            continue

        step = int(jsonl_path.stem)
        metrics: dict[str, float] = {
            "val-data/custom/evaluable_problem_count": float(len(groups)),
            "val-data/custom/skipped_invalid_problem_count": float(skipped_invalid_groups),
        }
        base_pass = 0.0
        base_max = 0.0
        for k in ks:
            pass_hits = 0
            max_scores = []
            cluster_counts = []
            correct_cluster_counts = []
            valid_equation_rates = []
            for records in groups.values():
                subset = records[:k]
                valid_subset = [
                    item for item in subset if _has_valid_countdown_equation(item.get("output", ""), item.get("gts"))
                ]
                valid_ids = {id(item) for item in valid_subset}
                scores = [
                    float(item.get("score", 0.0)) if id(item) in valid_ids else 0.0
                    for item in subset
                ]
                max_score = max(scores) if scores else 0.0
                max_scores.append(max_score)
                if max_score > 0.5:
                    pass_hits += 1
                signatures = {_reasoning_signature(item.get("output", "")) for item in valid_subset}
                signatures.discard("")
                cluster_counts.append(float(len(signatures)))
                correct_signatures = {
                    _reasoning_signature(item.get("output", ""))
                    for item in valid_subset
                    if float(item.get("score", 0.0)) > 0.5
                }
                correct_signatures.discard("")
                correct_cluster_counts.append(float(len(correct_signatures)))
                valid_equation_rates.append(len(valid_subset) / len(subset) if subset else 0.0)

            group_count = float(len(groups))
            pass_at_k = pass_hits / group_count if group_count else 0.0
            max_at_k = sum(max_scores) / len(max_scores) if max_scores else 0.0
            if k == 1:
                base_pass = pass_at_k
                base_max = max_at_k
            clusters_at_k = sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0.0
            correct_clusters_at_k = (
                sum(correct_cluster_counts) / len(correct_cluster_counts) if correct_cluster_counts else 0.0
            )
            valid_equation_rate_at_k = (
                sum(valid_equation_rates) / len(valid_equation_rates) if valid_equation_rates else 0.0
            )
            pass_gain_at_k = pass_at_k - base_pass
            max_gain_at_k = max_at_k - base_max
            metrics.update(
                {
                    f"val-core/custom/pass@{k}": pass_at_k,
                    f"val-core/custom/max@{k}": max_at_k,
                    f"val-aux/custom/clusters@{k}": clusters_at_k,
                    f"val-aux/custom/correct_clusters@{k}": correct_clusters_at_k,
                    f"val-aux/custom/valid_equation_rate@{k}": valid_equation_rate_at_k,
                    f"val-gap/custom/pass_gain@{k}": pass_gain_at_k,
                    f"val-gap/custom/max_gain@{k}": max_gain_at_k,
                    f"custom/pass_at_{k}": pass_at_k,
                    f"custom/max_at_{k}": max_at_k,
                    f"custom/cluster_at_{k}": clusters_at_k,
                    f"custom/clusters_at_{k}": clusters_at_k,
                    f"custom/correct_cluster_at_{k}": correct_clusters_at_k,
                    f"custom/correct_clusters_at_{k}": correct_clusters_at_k,
                    f"custom/valid_equation_rate_at_{k}": valid_equation_rate_at_k,
                    f"custom/pass_gain_at_{k}": pass_gain_at_k,
                    f"custom/max_gain_at_{k}": max_gain_at_k,
                }
            )
        per_step_metrics[step] = metrics
    return per_step_metrics


def _log_validation_passk_clusters_to_wandb(
    *,
    validation_data_dir: Path,
    wandb_project: str,
    wandb_entity: str,
    experiment_name: str,
    run_id: str,
    eval_ks: str,
    validation_rollouts: int,
) -> None:
    import wandb

    ks = _parse_eval_ks(eval_ks, validation_rollouts)
    validation_files = sorted(validation_data_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    print(
        "Custom validation metrics: "
        f"dir={validation_data_dir}, files={len(validation_files)}, eval_ks={','.join(map(str, ks))}"
    )
    if not validation_files:
        print(
            "Custom validation metrics skipped: no validation JSONL files were written. "
            "Check that trainer.test_freq is <= trainer.total_training_steps and that validation ran at least once."
        )
        return
    per_step_metrics = _compute_validation_passk_clusters(validation_data_dir, ks)
    if not per_step_metrics:
        print(
            "Custom validation metrics skipped: validation files existed, but no evaluable problem groups were found."
        )
        return

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=experiment_name,
        id=run_id,
        resume="allow",
    )
    try:
        for step, metrics in sorted(per_step_metrics.items()):
            wandb.log(metrics, step=step)
        logged_keys = sorted({key for metrics in per_step_metrics.values() for key in metrics})
        print(
            "Custom validation metrics logged to W&B: "
            f"steps={sorted(per_step_metrics)}, metric_keys={len(logged_keys)}"
        )
        print("Custom validation metric examples: " + ", ".join(logged_keys[:12]))
    finally:
        wandb.finish()


def _flatten_wandb_summary(data: dict, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in data.items():
        metric_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and "_type" not in value:
            flattened.update(_flatten_wandb_summary(value, metric_key))
        else:
            flattened[metric_key] = value
    return flattened


def _format_summary_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _load_local_wandb_summary(run_dir: Path) -> dict[str, object]:
    candidates: list[Path] = []
    candidates.extend(run_dir.glob("wandb/latest-run/files/wandb-summary.json"))
    candidates.extend(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
    candidates = sorted(
        {path.resolve(): path for path in candidates if path.exists()}.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for summary_path in candidates:
        try:
            with open(summary_path, "r", encoding="utf-8") as infile:
                payload = json.load(infile)
            if isinstance(payload, dict) and payload:
                return _flatten_wandb_summary(payload)
        except Exception as exc:
            print(f"Could not read local W&B summary {summary_path}: {exc}")
    return {}


def _load_api_wandb_summary(*, wandb_project: str, wandb_entity: str, run_id: str) -> dict[str, object]:
    if not wandb_entity:
        return {}
    try:
        import wandb

        api = wandb.Api(timeout=30)
        run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
        return _flatten_wandb_summary(dict(run.summary))
    except Exception as exc:
        print(f"Could not fetch W&B summary from API: {exc}")
        return {}


def _print_full_wandb_summary(
    *,
    run_dir: Path,
    wandb_project: str,
    wandb_entity: str,
    run_id: str,
) -> None:
    metrics = _load_local_wandb_summary(run_dir)
    if not metrics:
        metrics = _load_api_wandb_summary(
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_id=run_id,
        )

    print("\nFull W&B run summary:")
    if not metrics:
        print("wandb-full: no local/API W&B summary metrics found")
        return

    for key in sorted(metrics):
        print(f"wandb-full: {key} {_format_summary_value(metrics[key])}")
    print(f"wandb-full: metric_count {len(metrics)}")


def _print_full_run_parameters(params: dict[str, object]) -> None:
    print("\nFull run parameters:")
    for key in sorted(params):
        print(f"param: {key}={_format_summary_value(params[key])}")


def _run_and_tee(cmd: list[str], log_path: str, env: dict[str, str] | None = None) -> None:
    import os
    import subprocess
    import sys

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def _install_modal_tinyzero_extensions() -> None:
    """Patch the freshly cloned TinyZero checkout with Modal-only experiment hooks."""
    repo_dir = Path(REMOTE_REPO_DIR)

    countdown_reward_path = repo_dir / "verl" / "utils" / "reward_score" / "countdown.py"
    countdown_reward_path.write_text(
        textwrap.dedent(
            r'''
            import ast
            import operator
            import random
            import re
            from collections import Counter


            _OPS = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }


            def extract_solution(solution_str):
                """Extract the final equation from the whole assistant response."""
                if "Assistant:" in solution_str:
                    solution_str = solution_str.split("Assistant:", 1)[1]
                elif "<|im_start|>assistant" in solution_str:
                    solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

                answer_pattern = r"<answer>(.*?)</answer>"
                matches = list(re.finditer(answer_pattern, solution_str, flags=re.IGNORECASE | re.DOTALL))
                if not matches:
                    return None

                final_answer = matches[-1].group(1).strip()
                final_answer = re.sub(r"<\|.*?\|>", " ", final_answer)
                final_answer = re.sub(r"\s+", " ", final_answer).strip()
                if not final_answer or final_answer.upper() == "NO_SOLUTION":
                    return None
                return final_answer


            def validate_equation(equation_str, available_numbers):
                """Validate that the equation uses exactly the provided numbers once."""
                try:
                    numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
                    return Counter(numbers_in_eq) == Counter(int(n) for n in list(available_numbers))
                except Exception:
                    return False


            def _eval_node(node):
                if isinstance(node, ast.Expression):
                    return _eval_node(node.body)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
                    return _OPS[type(node.op)](_eval_node(node.operand))
                if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                    return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
                raise ValueError("Unsupported expression node.")


            def evaluate_equation(equation_str):
                """Safely evaluate arithmetic equations without eval()."""
                try:
                    allowed_pattern = r"^[\d+\-*/().\s]+$"
                    if not re.match(allowed_pattern, equation_str):
                        raise ValueError("Invalid characters in equation.")
                    return _eval_node(ast.parse(equation_str, mode="eval"))
                except Exception:
                    return None


            def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
                target = ground_truth["target"]
                numbers = ground_truth["numbers"]
                equation = extract_solution(solution_str=solution_str)
                do_print = random.randint(1, 64) == 1
                if do_print:
                    solution_preview = str(solution_str)
                    if len(solution_preview) > 1200:
                        solution_preview = solution_preview[:1200] + "... <truncated>"
                    print("--------------------------------")
                    print(f"Target: {target} | Numbers: {numbers}")
                    print(f"Extracted equation: {equation}")
                    print(f"Solution string: {solution_preview}")

                if equation is None:
                    if do_print:
                        print("No equation found")
                    return 0.0
                if not validate_equation(equation, numbers):
                    if do_print:
                        print("Invalid equation")
                    return format_score

                result = evaluate_equation(equation)
                if result is None:
                    if do_print:
                        print("Could not evaluate equation")
                    return format_score
                if abs(result - target) < 1e-5:
                    if do_print:
                        print(f"Correct equation: {equation} = {result}")
                    return score
                if do_print:
                    print(f"Wrong result: equation = {result}, target = {target}")
                return format_score
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    controller_path = repo_dir / "verl" / "utils" / "adaptive_prompt_controller.py"
    controller_path.write_text(
        textwrap.dedent(
            r'''
            import json
            import math
            import os
            from pathlib import Path


            def _enabled():
                return os.environ.get("TZ_PROMPT_CONTROLLER_MODE") == "adaptive"


            def _state_path():
                value = os.environ.get("TZ_PROMPT_CONTROLLER_STATE")
                return Path(value) if value else None


            def _env_families():
                raw = os.environ.get("TZ_PROMPT_FAMILIES", "")
                return [item.strip() for item in raw.split(",") if item.strip()]


            def _initial_state(families=None):
                families = list(families or _env_families())
                return {
                    "step": 0,
                    "families": families,
                    "counts": {family: 0.0 for family in families},
                    "successes": {family: 0.0 for family in families},
                    "rates": {family: 0.0 for family in families},
                    "probs": {family: 1.0 / len(families) for family in families} if families else {},
                }


            def _read_state():
                path = _state_path()
                if path is None or not path.exists():
                    return _initial_state()
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    return _initial_state()


            def _write_state(state):
                path = _state_path()
                if path is None:
                    return
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                tmp_path.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
                tmp_path.replace(path)


            def _ensure_families(state, families):
                existing = list(state.get("families") or [])
                merged = []
                for family in list(families or []) + existing:
                    if family and family not in merged:
                        merged.append(family)
                state["families"] = merged
                state.setdefault("counts", {})
                state.setdefault("successes", {})
                for family in merged:
                    state["counts"].setdefault(family, 0.0)
                    state["successes"].setdefault(family, 0.0)
                return state


            def _attach_probs(state):
                families = list(state.get("families") or [])
                if not families:
                    state["rates"] = {}
                    state["probs"] = {}
                    return state
                temp = max(float(os.environ.get("TZ_PROMPT_BANDIT_TEMP", "0.25")), 1e-6)
                prior_success = float(os.environ.get("TZ_PROMPT_BANDIT_PRIOR_SUCCESS", "0.25"))
                prior_count = float(os.environ.get("TZ_PROMPT_BANDIT_PRIOR_COUNT", "1.0"))
                counts = state.get("counts", {})
                successes = state.get("successes", {})
                rates = {}
                logits = []
                for family in families:
                    count = float(counts.get(family, 0.0))
                    success = float(successes.get(family, 0.0))
                    rate = (success + prior_success) / max(count + prior_count, 1e-9)
                    rates[family] = rate
                    logits.append(rate / temp)
                max_logit = max(logits)
                exp_values = [math.exp(logit - max_logit) for logit in logits]
                denom = sum(exp_values) or 1.0
                state["rates"] = rates
                state["probs"] = {family: exp_values[i] / denom for i, family in enumerate(families)}
                return state


            def reset_state():
                if not _enabled():
                    return
                state = _attach_probs(_initial_state())
                _write_state(state)


            def choose_family(prompt_variants_json, rng):
                if not _enabled() or not prompt_variants_json:
                    return None
                variants = json.loads(prompt_variants_json) if isinstance(prompt_variants_json, str) else dict(prompt_variants_json)
                families = [family for family in _env_families() if family in variants]
                if not families:
                    families = sorted(variants.keys())
                state = _attach_probs(_ensure_families(_read_state(), families))
                probs = state.get("probs", {})
                draw = rng.random()
                cumulative = 0.0
                selected = families[-1]
                for family in families:
                    cumulative += float(probs.get(family, 0.0))
                    if draw <= cumulative:
                        selected = family
                        break
                return selected, variants[selected]


            def _as_extra_info_dict(value):
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
                return {}


            def update_from_rewards(extra_infos, sequence_scores):
                if not _enabled():
                    return {}
                state = _ensure_families(_read_state(), _env_families())
                threshold = float(os.environ.get("TZ_PROMPT_SUCCESS_THRESHOLD", "0.5"))
                for extra_info, score in zip(extra_infos, sequence_scores):
                    info = _as_extra_info_dict(extra_info)
                    family = info.get("prompt_family")
                    if family not in state["families"]:
                        continue
                    state["counts"][family] = float(state["counts"].get(family, 0.0)) + 1.0
                    state["successes"][family] = float(state["successes"].get(family, 0.0)) + (1.0 if float(score) > threshold else 0.0)
                state["step"] = int(state.get("step", 0)) + 1
                state = _attach_probs(state)
                _write_state(state)
                metrics = {
                    "prompt_controller/step": float(state["step"]),
                    "prompt_controller/temperature": float(os.environ.get("TZ_PROMPT_BANDIT_TEMP", "0.25")),
                }
                for family in state.get("families", []):
                    metrics[f"prompt_controller/{family}/count"] = float(state["counts"].get(family, 0.0))
                    metrics[f"prompt_controller/{family}/success_rate"] = float(state["rates"].get(family, 0.0))
                    metrics[f"prompt_controller/{family}/prob"] = float(state["probs"].get(family, 0.0))
                return metrics
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    dataset_path = repo_dir / "verl" / "utils" / "dataset" / "rl_dataset.py"
    dataset_text = dataset_path.read_text(encoding="utf-8")
    if "TZ_MODAL_ADAPTIVE_DATASET_PATCH" not in dataset_text:
        dataset_text = dataset_text.replace("import os\n", "import os\nimport random\n", 1)
        dataset_text = dataset_text.replace(
            "import verl.utils.torch_functional as verl_F\n",
            "import verl.utils.torch_functional as verl_F\n"
            "from verl.utils.adaptive_prompt_controller import choose_family\n",
            1,
        )
        dataset_text = dataset_text.replace(
            "        self.truncation = truncation\n\n        self._download()",
            "        self.truncation = truncation\n"
            "        # TZ_MODAL_ADAPTIVE_DATASET_PATCH\n"
            "        seed = int(os.environ.get('TZ_PROMPT_SEED', os.environ.get('TZ_SEED', '1')))\n"
            "        self._adaptive_prompt_rng = random.Random(seed + os.getpid())\n\n"
            "        self._download()",
            1,
        )
        dataset_text = dataset_text.replace(
            "        row_dict = self.dataframe.iloc[item].to_dict()\n\n        chat = row_dict.pop(self.prompt_key)",
            "        row_dict = self.dataframe.iloc[item].to_dict()\n\n"
            "        prompt_variants_json = row_dict.pop('prompt_variants_json', None)\n"
            "        choice = choose_family(prompt_variants_json, self._adaptive_prompt_rng)\n"
            "        if choice is not None:\n"
            "            prompt_family, prompt_text = choice\n"
            "            row_dict[self.prompt_key] = [{'role': 'user', 'content': prompt_text}]\n"
            "            extra_info = dict(row_dict.get('extra_info') or {})\n"
            "            extra_info['prompt_family'] = prompt_family\n"
            "            extra_info['prompt_conditioning_mode'] = 'adaptive'\n"
            "            row_dict['extra_info'] = extra_info\n\n"
            "        chat = row_dict.pop(self.prompt_key)",
            1,
        )
        if "TZ_MODAL_ADAPTIVE_DATASET_PATCH" not in dataset_text:
            raise RuntimeError("Failed to patch TinyZero RLHFDataset for adaptive prompts.")
        dataset_path.write_text(dataset_text, encoding="utf-8")

    trainer_path = repo_dir / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    trainer_text = trainer_path.read_text(encoding="utf-8")
    if "TZ_MODAL_ADAPTIVE_TRAINER_PATCH" not in trainer_text:
        trainer_text = trainer_text.replace("import os\nimport uuid\n", "import json\nimport os\nimport uuid\n", 1)
        trainer_text = trainer_text.replace(
            "from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance\n",
            "from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance\n"
            "from verl.utils.adaptive_prompt_controller import reset_state, update_from_rewards\n",
            1,
        )
        trainer_text = trainer_text.replace(
            "        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn\n"
            "        self.train_dataset = RLHFDataset",
            "        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn\n"
            "        # TZ_MODAL_ADAPTIVE_TRAINER_PATCH\n"
            "        seed = int(os.environ.get('TZ_SEED', self.config.trainer.get('seed', 1)))\n"
            "        np.random.seed(seed)\n"
            "        torch.manual_seed(seed)\n"
            "        self._train_loader_generator = torch.Generator()\n"
            "        self._train_loader_generator.manual_seed(seed)\n"
            "        self._val_loader_generator = torch.Generator()\n"
            "        self._val_loader_generator.manual_seed(seed + 1)\n"
            "        self.train_dataset = RLHFDataset",
            1,
        )
        trainer_text = trainer_text.replace(
            "                                           shuffle=True,\n"
            "                                           drop_last=True,\n"
            "                                           collate_fn=collate_fn)\n\n"
            "        self.val_dataset = RLHFDataset",
            "                                           shuffle=True,\n"
            "                                           drop_last=True,\n"
            "                                           generator=self._train_loader_generator,\n"
            "                                           collate_fn=collate_fn)\n\n"
            "        self.val_dataset = RLHFDataset",
            1,
        )
        trainer_text = trainer_text.replace(
            "                                         shuffle=True,\n"
            "                                         drop_last=True,\n"
            "                                         collate_fn=collate_fn)\n\n"
            "        assert len(self.train_dataloader) >= 1",
            "                                         shuffle=True,\n"
            "                                         drop_last=True,\n"
            "                                         generator=self._val_loader_generator,\n"
            "                                         collate_fn=collate_fn)\n\n"
            "        assert len(self.train_dataloader) >= 1",
            1,
        )
        trainer_text = trainer_text.replace(
            "    def _validate(self):\n",
            "    def _write_validation_records(self, test_batch, reward_tensor):\n"
            "        validation_data_dir = self.config.trainer.get('validation_data_dir', None)\n"
            "        if not validation_data_dir:\n"
            "            return\n"
            "        os.makedirs(validation_data_dir, exist_ok=True)\n"
            "        output_path = os.path.join(validation_data_dir, f'{self.global_steps}.jsonl')\n"
            "        with open(output_path, 'w', encoding='utf-8') as outfile:\n"
            "            for i in range(len(test_batch)):\n"
            "                data_item = test_batch[i]\n"
            "                prompt_ids = data_item.batch['prompts']\n"
            "                prompt_length = prompt_ids.shape[-1]\n"
            "                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()\n"
            "                valid_prompt_ids = prompt_ids[-valid_prompt_length:]\n"
            "                response_ids = data_item.batch['responses']\n"
            "                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()\n"
            "                valid_response_ids = response_ids[:valid_response_length]\n"
            "                reward_model = data_item.non_tensor_batch.get('reward_model', {})\n"
            "                extra_info = data_item.non_tensor_batch.get('extra_info', {})\n"
            "                record = {\n"
            "                    'input': self.tokenizer.decode(valid_prompt_ids),\n"
            "                    'output': self.tokenizer.decode(valid_response_ids),\n"
            "                    'score': float(reward_tensor[i].sum().item()),\n"
            "                    'gts': reward_model.get('ground_truth') if isinstance(reward_model, dict) else reward_model,\n"
            "                    'extra_info': extra_info,\n"
            "                }\n"
            "                outfile.write(json.dumps(record, default=str) + '\\n')\n\n"
            "    def _validate(self):\n",
            1,
        )
        trainer_text = trainer_text.replace(
            "            test_gen_batch.meta_info = {\n"
            "                'eos_token_id': self.tokenizer.eos_token_id,\n"
            "                'pad_token_id': self.tokenizer.pad_token_id,\n"
            "                'recompute_log_prob': False,\n"
            "                'do_sample': False,\n"
            "                'validate': True,\n"
            "            }\n",
            "            val_kwargs = self.config.actor_rollout_ref.rollout.get('val_kwargs', {})\n"
            "            if val_kwargs is None:\n"
            "                val_kwargs = {}\n"
            "            elif not isinstance(val_kwargs, dict):\n"
            "                val_kwargs = OmegaConf.to_container(val_kwargs, resolve=True)\n"
            "            else:\n"
            "                val_kwargs = dict(val_kwargs)\n"
            "            val_do_sample = bool(val_kwargs.pop('do_sample', False))\n"
            "            val_n = int(val_kwargs.get('n', self.config.actor_rollout_ref.rollout.n if val_do_sample else 1))\n"
            "            test_gen_batch.meta_info = {\n"
            "                'eos_token_id': self.tokenizer.eos_token_id,\n"
            "                'pad_token_id': self.tokenizer.pad_token_id,\n"
            "                'recompute_log_prob': False,\n"
            "                'do_sample': val_do_sample,\n"
            "                'validate': True,\n"
            "                'sampling_kwargs': val_kwargs,\n"
            "            }\n",
            1,
        )
        trainer_text = trainer_text.replace(
            "            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)\n"
            "            print('validation generation end')\n\n"
            "            test_batch = test_batch.union(test_output_gen_batch)\n",
            "            test_output_gen_batch = unpad_dataproto(\n"
            "                test_output_gen_batch_padded,\n"
            "                pad_size=pad_size * val_n if val_do_sample else pad_size,\n"
            "            )\n"
            "            print('validation generation end')\n\n"
            "            if val_do_sample and val_n > 1:\n"
            "                test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)\n"
            "            test_batch = test_batch.union(test_output_gen_batch)\n",
            1,
        )
        trainer_text = trainer_text.replace(
            "            reward_tensor = self.val_reward_fn(test_batch)\n\n"
            "            reward_tensor_lst.append(reward_tensor)\n",
            "            reward_tensor = self.val_reward_fn(test_batch)\n"
            "            self._write_validation_records(test_batch, reward_tensor)\n\n"
            "            reward_tensor_lst.append(reward_tensor)\n",
            1,
        )
        trainer_text = trainer_text.replace(
            "        self.global_steps = 0\n\n        # perform validation before training",
            "        reset_state()\n\n"
            "        self.global_steps = 0\n\n"
            "        # perform validation before training",
            1,
        )
        trainer_text = trainer_text.replace(
            "                        reward_tensor = self.reward_fn(batch)\n"
            "                        batch.batch['token_level_scores'] = reward_tensor\n\n"
            "                        # compute rewards. apply_kl_penalty if available",
            "                        reward_tensor = self.reward_fn(batch)\n"
            "                        batch.batch['token_level_scores'] = reward_tensor\n"
            "                        sequence_scores = reward_tensor.sum(-1).detach().cpu().tolist()\n"
            "                        metrics.update(update_from_rewards(\n"
            "                            batch.non_tensor_batch.get('extra_info', []), sequence_scores))\n\n"
            "                        # compute rewards. apply_kl_penalty if available",
            1,
        )
        if "TZ_MODAL_ADAPTIVE_TRAINER_PATCH" not in trainer_text:
            raise RuntimeError("Failed to patch TinyZero RayPPOTrainer for adaptive prompts.")
        if "_write_validation_records" not in trainer_text or "sampling_kwargs" not in trainer_text:
            raise RuntimeError("Failed to patch TinyZero RayPPOTrainer for validation rollout metrics.")
        trainer_path.write_text(trainer_text, encoding="utf-8")

    trainer_text = trainer_path.read_text(encoding="utf-8")
    if "_write_validation_records" not in trainer_text or "sampling_kwargs" not in trainer_text:
        raise RuntimeError(
            "TinyZero RayPPOTrainer validation metrics patch is missing, so pass@k/clusters@k cannot be logged. "
            "Restart the Modal app with the updated modal_tinyzero.py so the runtime patch is applied."
        )

    fsdp_path = repo_dir / "verl" / "workers" / "fsdp_workers.py"
    fsdp_text = fsdp_path.read_text(encoding="utf-8")
    if "TZ_MODAL_VAL_KWARGS_PATCH" not in fsdp_text:
        fsdp_text = fsdp_text.replace(
            "            output = self.rollout.generate_sequences(prompts=prompts)\n",
            "            output = self.rollout.generate_sequences(\n"
            "                prompts=prompts,\n"
            "                **prompts.meta_info.get('sampling_kwargs', {}),\n"
            "            )  # TZ_MODAL_VAL_KWARGS_PATCH\n",
            1,
        )
        if "TZ_MODAL_VAL_KWARGS_PATCH" not in fsdp_text:
            raise RuntimeError("Failed to patch TinyZero FSDP worker for validation sampling kwargs.")
        fsdp_path.write_text(fsdp_text, encoding="utf-8")

    vllm_path = repo_dir / "verl" / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout.py"
    vllm_text = vllm_path.read_text(encoding="utf-8")
    if "TZ_MODAL_STOP_TOKEN_PATCH" not in vllm_text:
        vllm_text = vllm_text.replace(
            "        # supporting adding any sampling params from the config file\n"
            "        for k in config.keys():\n"
            "            if hasattr(SamplingParams(), str(k)):\n"
            "                kwargs[k] = config.get(k)\n"
            "        print(f\"kwargs: {kwargs}\")\n",
            "        # supporting adding any sampling params from the config file\n"
            "        for k in config.keys():\n"
            "            if hasattr(SamplingParams(), str(k)):\n"
            "                kwargs[k] = config.get(k)\n"
            "        # TZ_MODAL_STOP_TOKEN_PATCH: prevent endless special-token tails after a valid answer.\n"
            "        stop_token_ids = set(kwargs.get('stop_token_ids') or [])\n"
            "        for token in ['<|endoftext|>', '<|im_end|>']:\n"
            "            try:\n"
            "                token_id = tokenizer.convert_tokens_to_ids(token)\n"
            "            except Exception:\n"
            "                token_id = None\n"
            "            if isinstance(token_id, int) and token_id >= 0:\n"
            "                stop_token_ids.add(token_id)\n"
            "        for token_id in [getattr(tokenizer, 'eos_token_id', None), getattr(tokenizer, 'pad_token_id', None)]:\n"
            "            if isinstance(token_id, int) and token_id >= 0:\n"
            "                stop_token_ids.add(token_id)\n"
            "        kwargs['stop_token_ids'] = sorted(stop_token_ids)\n"
            "        kwargs['ignore_eos'] = False\n"
            "        print(f\"kwargs: {kwargs}\")\n",
            1,
        )
        if "TZ_MODAL_STOP_TOKEN_PATCH" not in vllm_text:
            raise RuntimeError("Failed to patch TinyZero vLLM rollout stop token ids.")
        vllm_path.write_text(vllm_text, encoding="utf-8")

    if "TZ_MODAL_VAL_N_PATCH" not in vllm_text:
        vllm_text = vllm_text.replace(
            "        # users can customize different sampling_params at different run\n",
            "        sampling_n = int(kwargs.get('n', self.config.n))  # TZ_MODAL_VAL_N_PATCH\n\n"
            "        # users can customize different sampling_params at different run\n",
            1,
        )
        vllm_text = vllm_text.replace(
            "        if self.config.n > 1 and do_sample:\n"
            "            idx = idx.repeat_interleave(self.config.n, dim=0)\n"
            "            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)\n"
            "            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)\n"
            "            batch_size = batch_size * self.config.n\n",
            "        if sampling_n > 1 and do_sample:\n"
            "            idx = idx.repeat_interleave(sampling_n, dim=0)\n"
            "            attention_mask = attention_mask.repeat_interleave(sampling_n, dim=0)\n"
            "            position_ids = position_ids.repeat_interleave(sampling_n, dim=0)\n"
            "            batch_size = batch_size * sampling_n\n",
            1,
        )
        if "TZ_MODAL_VAL_N_PATCH" not in vllm_text:
            raise RuntimeError("Failed to patch TinyZero vLLM rollout for validation n.")
        vllm_path.write_text(vllm_text, encoding="utf-8")

    main_path = repo_dir / "verl" / "trainer" / "main_ppo.py"
    main_text = main_path.read_text(encoding="utf-8")
    if "TZ_MODAL_ENV_PATCH" not in main_text:
        main_text = main_text.replace(
            "from verl import DataProto\nimport torch\n",
            "from verl import DataProto\nimport os\nimport random\n\nimport numpy as np\nimport torch\n",
            1,
        )
        main_text = main_text.replace(
            "        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})",
            "        # TZ_MODAL_ENV_PATCH\n"
            "        env_vars = {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}\n"
            "        for key in [\n"
            "            'TZ_SEED', 'TZ_PROMPT_SEED', 'TZ_PROMPT_CONTROLLER_MODE',\n"
            "            'TZ_PROMPT_CONTROLLER_STATE', 'TZ_PROMPT_FAMILIES',\n"
            "            'TZ_PROMPT_BANDIT_TEMP', 'TZ_PROMPT_BANDIT_PRIOR_SUCCESS',\n"
            "            'TZ_PROMPT_BANDIT_PRIOR_COUNT', 'TZ_PROMPT_SUCCESS_THRESHOLD',\n"
            "        ]:\n"
            "            value = os.environ.get(key)\n"
            "            if value is not None:\n"
            "                env_vars[key] = value\n"
            "        ray.init(runtime_env={'env_vars': env_vars})",
            1,
        )
        main_text = main_text.replace(
            "def main_task(config):\n"
            "    from verl.utils.fs import copy_local_path_from_hdfs\n",
            "def main_task(config):\n"
            "    seed = int(os.environ.get('TZ_SEED', '1'))\n"
            "    random.seed(seed)\n"
            "    np.random.seed(seed)\n"
            "    torch.manual_seed(seed)\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.manual_seed_all(seed)\n\n"
            "    from verl.utils.fs import copy_local_path_from_hdfs\n",
            1,
        )
        if "TZ_MODAL_ENV_PATCH" not in main_text:
            raise RuntimeError("Failed to patch TinyZero main_ppo Ray runtime environment.")
        main_path.write_text(main_text, encoding="utf-8")


def _build_train_command(
    *,
    base_model: str,
    dataset_subdir: str,
    experiment_name: str,
    n_gpus: int,
    train_batch_size: int,
    val_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    actor_lr: float,
    actor_ppo_mini_batch_size: int,
    actor_ppo_micro_batch_size: int,
    rollout_log_prob_micro_batch_size: int,
    rollout_gpu_memory_utilization: float,
    ref_log_prob_micro_batch_size: int,
    critic_lr: float,
    critic_ppo_micro_batch_size: int,
    kl_coef: float,
    adv_estimator: str,
    rollout_n: int,
    use_kl_loss: bool,
    actor_kl_loss_coef: float,
    actor_kl_loss_type: str,
    save_freq: int,
    test_freq: int,
    total_epochs: int,
    total_training_steps: int,
    validation_rollouts: int,
    extra_hydra_args: str,
    wandb_mode: str,
    wandb_project: str,
    validation_data_dir: str,
    seed: int,
) -> list[str]:
    data_dir = f"{DATA_ROOT}/{dataset_subdir}"
    actor_micro_batch_size_global = actor_ppo_micro_batch_size * n_gpus
    rollout_log_prob_micro_batch_size_global = rollout_log_prob_micro_batch_size * n_gpus
    ref_log_prob_micro_batch_size_global = ref_log_prob_micro_batch_size * n_gpus
    critic_micro_batch_size_global = critic_ppo_micro_batch_size * n_gpus
    logger_override = "trainer.logger=['console']" if wandb_mode == "disabled" else "trainer.logger=['console','wandb']"
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/test.parquet",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={val_batch_size}",
        f"data.max_prompt_length={max_prompt_length}",
        f"data.max_response_length={max_response_length}",
        f"actor_rollout_ref.model.path={base_model}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.optim.lr={actor_lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor_ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={actor_micro_batch_size_global}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size={rollout_log_prob_micro_batch_size_global}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={n_gpus}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size={ref_log_prob_micro_batch_size_global}",
        f"critic.optim.lr={critic_lr}",
        f"critic.model.path={base_model}",
        "critic.model.enable_gradient_checkpointing=True",
        f"critic.ppo_micro_batch_size={critic_micro_batch_size_global}",
        f"critic.forward_micro_batch_size={critic_micro_batch_size_global}",
        f"algorithm.adv_estimator={adv_estimator}",
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
        f"actor_rollout_ref.rollout.n={rollout_n}",
        logger_override,
        "+trainer.val_before_train=False",
        "trainer.default_hdfs_dir=null",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={test_freq}",
        f"trainer.project_name={wandb_project}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.total_epochs={total_epochs}",
        f"+trainer.seed={seed}",
        f"+trainer.validation_data_dir={validation_data_dir}",
        f"+actor_rollout_ref.rollout.val_kwargs.n={validation_rollouts}",
        "+actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "+actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.top_k=-1",
    ]
    if total_training_steps > 0:
        cmd.append(f"trainer.total_training_steps={total_training_steps}")
    if use_kl_loss:
        cmd.extend(
            [
                "actor_rollout_ref.actor.use_kl_loss=True",
                f"actor_rollout_ref.actor.kl_loss_coef={actor_kl_loss_coef}",
                f"actor_rollout_ref.actor.kl_loss_type={actor_kl_loss_type}",
            ]
        )
    if extra_hydra_args.strip():
        cmd.extend(shlex.split(extra_hydra_args))
    return cmd


@app.function(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    timeout=60 * 60,
    cpu=4,
)
def prepare_countdown_data(
    repo_ref: str = "main",
    dataset_name: str = "countdown",
    dataset_subdir: str = "countdown",
    template_type: str = "base",
    prompt_conditioning_mode: str = "off",
    train_size: int = 327680,
    test_size: int = 1024,
) -> dict[str, str]:
    from datasets import load_dataset

    _repo_checkout(repo_ref)

    data_dir = Path(DATA_ROOT) / dataset_subdir
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "countdown":
        raw_dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        original_count = len(raw_dataset)
        raw_dataset = raw_dataset.filter(_is_solvable_countdown_example)
        print(f"Countdown solvable filter kept {len(raw_dataset)} of {original_count} examples.")
        if len(raw_dataset) < train_size + test_size:
            raise ValueError("Requested train/test split is larger than the Countdown dataset.")

        train_dataset = raw_dataset.select(range(train_size))
        test_dataset = raw_dataset.select(range(train_size, train_size + test_size))

        def make_map_fn(split: str):
            def process_fn(example: dict, idx: int) -> dict:
                prompt_family = _select_prompt_family(
                    dataset_name=dataset_name,
                    split=split,
                    idx=idx,
                    prompt_conditioning_mode=prompt_conditioning_mode,
                )
                prompt_variants = _build_prompt_variants(dataset_name, example, template_type)
                reference_solution = _countdown_solution_expression(example["target"], example["nums"])
                if reference_solution is None:
                    raise ValueError(f"Unsolvable countdown example reached map step: {example}")
                solution = {"target": example["target"], "numbers": example["nums"]}
                row = {
                    "data_source": "countdown",
                    "prompt": [{"role": "user", "content": prompt_variants[prompt_family]}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "prompt_family": prompt_family,
                        "prompt_conditioning_mode": prompt_conditioning_mode,
                        "countdown_valid": True,
                        "reference_solution": reference_solution,
                    },
                }
                if split == "train" and prompt_conditioning_mode == "adaptive":
                    row["prompt_variants_json"] = json.dumps(prompt_variants, sort_keys=True)
                return row

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    elif dataset_name == "gsm8k":
        raw_dataset = load_dataset(DATASET_SOURCES["gsm8k"], "main")
        train_dataset = raw_dataset["train"]
        test_dataset = raw_dataset["test"]

        if 0 < train_size < len(train_dataset):
            train_dataset = train_dataset.select(range(train_size))
        if 0 < test_size < len(test_dataset):
            test_dataset = test_dataset.select(range(test_size))

        def make_map_fn(split: str):
            def process_fn(example: dict, idx: int) -> dict:
                prompt_family = _select_prompt_family(
                    dataset_name=dataset_name,
                    split=split,
                    idx=idx,
                    prompt_conditioning_mode=prompt_conditioning_mode,
                )
                prompt_variants = _build_prompt_variants(dataset_name, example, template_type)
                question = prompt_variants[prompt_family]
                answer_raw = example["answer"].strip()
                row = {
                    "data_source": DATASET_SOURCES["gsm8k"],
                    "prompt": [{"role": "user", "content": question}],
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer_raw},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": _extract_gsm8k_answer(answer_raw)},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "prompt_family": prompt_family,
                        "prompt_conditioning_mode": prompt_conditioning_mode,
                        "question_raw": example["question"],
                        "answer_raw": answer_raw,
                    },
                }
                if split == "train" and prompt_conditioning_mode == "adaptive":
                    row["prompt_variants_json"] = json.dumps(prompt_variants, sort_keys=True)
                return row

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    elif dataset_name == "math500":
        raw_dataset = load_dataset(DATASET_SOURCES["math500"], split="test")
        train_dataset, test_dataset = _split_single_split_dataset(
            raw_dataset,
            train_size=train_size,
            test_size=test_size,
            default_train_size=450,
            default_test_size=50,
        )

        def make_map_fn(split: str):
            def process_fn(example: dict, idx: int) -> dict:
                prompt_family = _select_prompt_family(
                    dataset_name=dataset_name,
                    split=split,
                    idx=idx,
                    prompt_conditioning_mode=prompt_conditioning_mode,
                )
                prompt_variants = _build_prompt_variants(dataset_name, example, template_type)
                problem = prompt_variants[prompt_family]
                answer_raw = example["answer"].strip()
                solution_raw = example["solution"].strip()
                row = {
                    "data_source": DATASET_SOURCES["math500"],
                    "prompt": [{"role": "user", "content": problem}],
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution_raw},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer_raw},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "prompt_family": prompt_family,
                        "prompt_conditioning_mode": prompt_conditioning_mode,
                        "problem_raw": example["problem"],
                        "answer_raw": answer_raw,
                        "subject": example.get("subject"),
                        "level": example.get("level"),
                        "unique_id": example.get("unique_id"),
                    },
                }
                if split == "train" and prompt_conditioning_mode == "adaptive":
                    row["prompt_variants_json"] = json.dumps(prompt_variants, sort_keys=True)
                return row

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    elif dataset_name == "aime24":
        raw_dataset = load_dataset(DATASET_SOURCES["aime24"], split="train")
        train_dataset, test_dataset = _split_single_split_dataset(
            raw_dataset,
            train_size=train_size,
            test_size=test_size,
            default_train_size=24,
            default_test_size=6,
        )

        def make_map_fn(split: str):
            def process_fn(example: dict, idx: int) -> dict:
                prompt_family = _select_prompt_family(
                    dataset_name=dataset_name,
                    split=split,
                    idx=idx,
                    prompt_conditioning_mode=prompt_conditioning_mode,
                )
                prompt_variants = _build_prompt_variants(dataset_name, example, template_type)
                problem = prompt_variants[prompt_family]
                answer_raw = str(example["answer"]).strip()
                solution_raw = example["solution"].strip()
                row = {
                    "data_source": DATASET_SOURCES["aime24"],
                    "prompt": [{"role": "user", "content": problem}],
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution_raw},
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer_raw},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "prompt_family": prompt_family,
                        "prompt_conditioning_mode": prompt_conditioning_mode,
                        "problem_raw": example["problem"],
                        "answer_raw": answer_raw,
                        "id": example.get("id"),
                        "url": example.get("url"),
                        "year": example.get("year"),
                    },
                }
                if split == "train" and prompt_conditioning_mode == "adaptive":
                    row["prompt_variants_json"] = json.dumps(prompt_variants, sort_keys=True)
                return row

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    train_dataset.to_parquet(data_dir / "train.parquet")
    test_dataset.to_parquet(data_dir / "test.parquet")

    artifacts.commit()
    return {
        "train_file": str(data_dir / "train.parquet"),
        "test_file": str(data_dir / "test.parquet"),
    }


def _train_impl(
    *,
    repo_ref: str,
    base_model: str,
    experiment_name: str,
    dataset_name: str,
    dataset_subdir: str,
    prompt_conditioning_mode: str,
    n_gpus: int,
    train_batch_size: int,
    val_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    actor_lr: float,
    actor_ppo_mini_batch_size: int,
    actor_ppo_micro_batch_size: int,
    rollout_log_prob_micro_batch_size: int,
    rollout_gpu_memory_utilization: float,
    ref_log_prob_micro_batch_size: int,
    critic_lr: float,
    critic_ppo_micro_batch_size: int,
    kl_coef: float,
    adv_estimator: str,
    rollout_n: int,
    use_kl_loss: bool,
    actor_kl_loss_coef: float,
    actor_kl_loss_type: str,
    save_freq: int,
    test_freq: int,
    total_epochs: int,
    total_training_steps: int,
    validation_rollouts: int,
    eval_ks: str,
    extra_hydra_args: str,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
    seed: int,
    prompt_bandit_temp: float,
    prompt_bandit_prior_success: float,
    prompt_bandit_prior_count: float,
    prompt_success_threshold: float,
) -> str:
    import os
    import subprocess

    import torch

    artifacts.reload()
    _repo_checkout(repo_ref)
    _install_modal_tinyzero_extensions()

    run_dir = Path(RUN_ROOT) / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(run_dir / "train.log")
    validation_data_dir = run_dir / "validation"
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", experiment_name)

    subprocess.run(["nvidia-smi", "-L"], check=False)
    print(f"torch={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal training container.")
    if torch.cuda.device_count() < n_gpus:
        raise RuntimeError(f"Expected {n_gpus} visible GPU(s), found {torch.cuda.device_count()}.")
    for gpu_idx in range(torch.cuda.device_count()):
        print(f"gpu[{gpu_idx}]={torch.cuda.get_device_name(gpu_idx)}")

    env = {
        "N_GPUS": str(n_gpus),
        "BASE_MODEL": base_model,
        "DATA_DIR": f"{DATA_ROOT}/{dataset_subdir}",
        "ROLLOUT_TP_SIZE": str(n_gpus),
        "EXPERIMENT_NAME": experiment_name,
        "WANDB_MODE": wandb_mode,
        "WANDB_PROJECT": wandb_project,
        "WANDB_NAME": experiment_name,
        "WANDB_RUN_ID": run_id,
        "WANDB_RESUME": "allow",
        "PYTHONHASHSEED": str(seed),
        "TZ_SEED": str(seed),
        "TZ_PROMPT_SEED": str(seed),
        "TZ_PROMPT_CONTROLLER_MODE": prompt_conditioning_mode,
    }
    if prompt_conditioning_mode == "adaptive":
        env.update(
            {
                "TZ_PROMPT_CONTROLLER_STATE": str(run_dir / "prompt_controller_state.json"),
                "TZ_PROMPT_FAMILIES": ",".join(_prompt_families_for_dataset(dataset_name)),
                "TZ_PROMPT_BANDIT_TEMP": str(prompt_bandit_temp),
                "TZ_PROMPT_BANDIT_PRIOR_SUCCESS": str(prompt_bandit_prior_success),
                "TZ_PROMPT_BANDIT_PRIOR_COUNT": str(prompt_bandit_prior_count),
                "TZ_PROMPT_SUCCESS_THRESHOLD": str(prompt_success_threshold),
            }
        )
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity

    os.chdir(run_dir)
    cmd = _build_train_command(
        base_model=base_model,
        dataset_subdir=dataset_subdir,
        experiment_name=experiment_name,
        n_gpus=n_gpus,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        actor_lr=actor_lr,
        actor_ppo_mini_batch_size=actor_ppo_mini_batch_size,
        actor_ppo_micro_batch_size=actor_ppo_micro_batch_size,
        rollout_log_prob_micro_batch_size=rollout_log_prob_micro_batch_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        ref_log_prob_micro_batch_size=ref_log_prob_micro_batch_size,
        critic_lr=critic_lr,
        critic_ppo_micro_batch_size=critic_ppo_micro_batch_size,
        kl_coef=kl_coef,
        adv_estimator=adv_estimator,
        rollout_n=rollout_n,
        use_kl_loss=use_kl_loss,
        actor_kl_loss_coef=actor_kl_loss_coef,
        actor_kl_loss_type=actor_kl_loss_type,
        save_freq=save_freq,
        test_freq=test_freq,
        total_epochs=total_epochs,
        total_training_steps=total_training_steps,
        validation_rollouts=validation_rollouts,
        extra_hydra_args=extra_hydra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        validation_data_dir=str(validation_data_dir),
        seed=seed,
    )
    run_error = None
    try:
        _run_and_tee(cmd, log_path=log_path, env=env)
    except subprocess.CalledProcessError as exc:
        run_error = exc
    finally:
        if wandb_mode != "disabled":
            if validation_data_dir.exists():
                _log_validation_passk_clusters_to_wandb(
                    validation_data_dir=validation_data_dir,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    experiment_name=experiment_name,
                    run_id=run_id,
                    eval_ks=eval_ks,
                    validation_rollouts=validation_rollouts,
                )
            else:
                print(
                    "Custom validation metrics skipped: validation directory does not exist: "
                    f"{validation_data_dir}. Check that validation ran and trainer.test_freq is not larger "
                    "than trainer.total_training_steps."
                )
        if wandb_mode != "disabled":
            _print_full_wandb_summary(
                run_dir=run_dir,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                run_id=run_id,
            )
        _print_full_run_parameters(
            {
                "repo_ref": repo_ref,
                "base_model": base_model,
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "dataset_subdir": dataset_subdir,
                "prompt_conditioning_mode": prompt_conditioning_mode,
                "n_gpus": n_gpus,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size,
                "max_prompt_length": max_prompt_length,
                "max_response_length": max_response_length,
                "actor_lr": actor_lr,
                "actor_ppo_mini_batch_size": actor_ppo_mini_batch_size,
                "actor_ppo_micro_batch_size": actor_ppo_micro_batch_size,
                "rollout_log_prob_micro_batch_size": rollout_log_prob_micro_batch_size,
                "rollout_gpu_memory_utilization": rollout_gpu_memory_utilization,
                "ref_log_prob_micro_batch_size": ref_log_prob_micro_batch_size,
                "critic_lr": critic_lr,
                "critic_ppo_micro_batch_size": critic_ppo_micro_batch_size,
                "kl_coef": kl_coef,
                "adv_estimator": adv_estimator,
                "rollout_n": rollout_n,
                "use_kl_loss": use_kl_loss,
                "actor_kl_loss_coef": actor_kl_loss_coef,
                "actor_kl_loss_type": actor_kl_loss_type,
                "save_freq": save_freq,
                "test_freq": test_freq,
                "total_epochs": total_epochs,
                "total_training_steps": total_training_steps,
                "validation_rollouts": validation_rollouts,
                "eval_ks": eval_ks,
                "extra_hydra_args": extra_hydra_args,
                "wandb_mode": wandb_mode,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "seed": seed,
                "prompt_bandit_temp": prompt_bandit_temp,
                "prompt_bandit_prior_success": prompt_bandit_prior_success,
                "prompt_bandit_prior_count": prompt_bandit_prior_count,
                "prompt_success_threshold": prompt_success_threshold,
                "run_id": run_id,
                "log_path": log_path,
                "validation_data_dir": str(validation_data_dir),
                "train_command": " ".join(shlex.quote(part) for part in cmd),
            }
        )
    if run_error is not None:
        raise run_error

    artifacts.commit()
    return str(run_dir)


@app.cls(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    secrets=[wandb_secret],
    timeout=60 * 60 * 24,
    cpu=8,
)
class TinyZeroTrainingWorker:
    @modal.method()
    def run(self, train_kwargs: dict) -> str:
        return _train_impl(**train_kwargs)


def _choose_training_worker(gpu_type: str, n_gpus: int):
    gpu_spec = _modal_gpu_spec(gpu_type, n_gpus)
    worker_cls = TinyZeroTrainingWorker.with_options(gpu=gpu_spec, cpu=8 * n_gpus)
    return worker_cls, gpu_spec


@app.function(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    secrets=[wandb_secret],
    gpu=_modal_gpu_spec(DEFAULT_GPU_TYPE, 1),
    timeout=60 * 60 * 24,
    cpu=8,
)
def train_tinyzero_1gpu(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-1.5B",
    experiment_name: str = "countdown-qwen2.5-1.5b-modal",
    dataset_name: str = "countdown",
    dataset_subdir: str = "countdown",
    prompt_conditioning_mode: str = "off",
    train_batch_size: int = 256,
    val_batch_size: int = 1312,
    max_prompt_length: int = 256,
    max_response_length: int = 512,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
    adv_estimator: str = "gae",
    rollout_n: int = 1,
    use_kl_loss: bool = False,
    actor_kl_loss_coef: float = 0.001,
    actor_kl_loss_type: str = "low_var_kl",
    save_freq: int = 100,
    test_freq: int = 100,
    total_epochs: int = 15,
    total_training_steps: int = 0,
    validation_rollouts: int = 8,
    eval_ks: str = "1,4,8",
    extra_hydra_args: str = "",
    wandb_mode: str = "online",
    wandb_project: str = "TinyZero",
    wandb_entity: str = "",
    seed: int = 1,
    prompt_bandit_temp: float = 0.25,
    prompt_bandit_prior_success: float = 0.25,
    prompt_bandit_prior_count: float = 1.0,
    prompt_success_threshold: float = 0.5,
) -> str:
    return _train_impl(
        repo_ref=repo_ref,
        base_model=base_model,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        dataset_subdir=dataset_subdir,
        prompt_conditioning_mode=prompt_conditioning_mode,
        n_gpus=1,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        actor_lr=actor_lr,
        actor_ppo_mini_batch_size=actor_ppo_mini_batch_size,
        actor_ppo_micro_batch_size=actor_ppo_micro_batch_size,
        rollout_log_prob_micro_batch_size=rollout_log_prob_micro_batch_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        ref_log_prob_micro_batch_size=ref_log_prob_micro_batch_size,
        critic_lr=critic_lr,
        critic_ppo_micro_batch_size=critic_ppo_micro_batch_size,
        kl_coef=kl_coef,
        adv_estimator=adv_estimator,
        rollout_n=rollout_n,
        use_kl_loss=use_kl_loss,
        actor_kl_loss_coef=actor_kl_loss_coef,
        actor_kl_loss_type=actor_kl_loss_type,
        save_freq=save_freq,
        test_freq=test_freq,
        total_epochs=total_epochs,
        total_training_steps=total_training_steps,
        validation_rollouts=validation_rollouts,
        eval_ks=eval_ks,
        extra_hydra_args=extra_hydra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        seed=seed,
        prompt_bandit_temp=prompt_bandit_temp,
        prompt_bandit_prior_success=prompt_bandit_prior_success,
        prompt_bandit_prior_count=prompt_bandit_prior_count,
        prompt_success_threshold=prompt_success_threshold,
    )


@app.function(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    secrets=[wandb_secret],
    gpu=_modal_gpu_spec(DEFAULT_GPU_TYPE, 2),
    timeout=60 * 60 * 24,
    cpu=16,
)
def train_tinyzero_2gpu(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-3B",
    experiment_name: str = "countdown-qwen2.5-3b-modal",
    dataset_name: str = "countdown",
    dataset_subdir: str = "countdown",
    prompt_conditioning_mode: str = "off",
    train_batch_size: int = 256,
    val_batch_size: int = 1312,
    max_prompt_length: int = 256,
    max_response_length: int = 512,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
    adv_estimator: str = "gae",
    rollout_n: int = 1,
    use_kl_loss: bool = False,
    actor_kl_loss_coef: float = 0.001,
    actor_kl_loss_type: str = "low_var_kl",
    save_freq: int = 100,
    test_freq: int = 100,
    total_epochs: int = 15,
    total_training_steps: int = 0,
    validation_rollouts: int = 8,
    eval_ks: str = "1,4,8",
    extra_hydra_args: str = "",
    wandb_mode: str = "online",
    wandb_project: str = "TinyZero",
    wandb_entity: str = "",
    seed: int = 1,
    prompt_bandit_temp: float = 0.25,
    prompt_bandit_prior_success: float = 0.25,
    prompt_bandit_prior_count: float = 1.0,
    prompt_success_threshold: float = 0.5,
) -> str:
    return _train_impl(
        repo_ref=repo_ref,
        base_model=base_model,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        dataset_subdir=dataset_subdir,
        prompt_conditioning_mode=prompt_conditioning_mode,
        n_gpus=2,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        actor_lr=actor_lr,
        actor_ppo_mini_batch_size=actor_ppo_mini_batch_size,
        actor_ppo_micro_batch_size=actor_ppo_micro_batch_size,
        rollout_log_prob_micro_batch_size=rollout_log_prob_micro_batch_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        ref_log_prob_micro_batch_size=ref_log_prob_micro_batch_size,
        critic_lr=critic_lr,
        critic_ppo_micro_batch_size=critic_ppo_micro_batch_size,
        kl_coef=kl_coef,
        adv_estimator=adv_estimator,
        rollout_n=rollout_n,
        use_kl_loss=use_kl_loss,
        actor_kl_loss_coef=actor_kl_loss_coef,
        actor_kl_loss_type=actor_kl_loss_type,
        save_freq=save_freq,
        test_freq=test_freq,
        total_epochs=total_epochs,
        total_training_steps=total_training_steps,
        validation_rollouts=validation_rollouts,
        eval_ks=eval_ks,
        extra_hydra_args=extra_hydra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        seed=seed,
        prompt_bandit_temp=prompt_bandit_temp,
        prompt_bandit_prior_success=prompt_bandit_prior_success,
        prompt_bandit_prior_count=prompt_bandit_prior_count,
        prompt_success_threshold=prompt_success_threshold,
    )


@app.local_entrypoint()
def main(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-3B",
    experiment_name: str = "countdown-qwen2.5-3b-modal",
    n_gpus: int = 2,
    gpu_type: str = DEFAULT_GPU_TYPE,
    prepare_data: bool = True,
    dataset_name: str = "countdown",
    dataset_subdir: str = "countdown",
    template_type: str = "base",
    prompt_conditioning_mode: str = "off",
    train_size: int = 327680,
    test_size: int = 1024,
    train_batch_size: int = 256,
    val_batch_size: int = 1312,
    max_prompt_length: int = 256,
    max_response_length: int = 512,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
    adv_estimator: str = "gae",
    rollout_n: int = 1,
    use_kl_loss: bool = False,
    actor_kl_loss_coef: float = 0.001,
    actor_kl_loss_type: str = "low_var_kl",
    save_freq: int = 100,
    test_freq: int = 100,
    total_epochs: int = 15,
    total_training_steps: int = 0,
    validation_rollouts: int = 8,
    eval_ks: str = "1,4,8",
    extra_hydra_args: str = "",
    wandb_mode: str = "online",
    wandb_project: str = "TinyZero",
    wandb_entity: str = "",
    seed: int = 1,
    prompt_bandit_temp: float = 0.25,
    prompt_bandit_prior_success: float = 0.25,
    prompt_bandit_prior_count: float = 1.0,
    prompt_success_threshold: float = 0.5,
) -> None:
    if prompt_conditioning_mode not in {"off", "mixed", "adaptive"}:
        raise ValueError("prompt_conditioning_mode must be one of: off, mixed, adaptive.")
    if adv_estimator not in {"gae", "grpo"}:
        raise ValueError("adv_estimator must be one of: gae, grpo.")
    if adv_estimator == "grpo":
        if rollout_n < 2:
            raise ValueError("GRPO requires --rollout-n >= 2 so each prompt has a response group.")
        use_kl_loss = True
    worker_cls, gpu_spec = _choose_training_worker(gpu_type, n_gpus)
    if template_type == "base" and "instruct" in base_model.lower():
        template_type = "qwen-instruct"

    if dataset_name != "countdown" and dataset_subdir == "countdown":
        dataset_subdir = dataset_name

    if prompt_conditioning_mode != "off" and dataset_subdir == dataset_name:
        dataset_subdir = f"{dataset_subdir}-{prompt_conditioning_mode}"

    if dataset_name in {"gsm8k", "math500", "aime24"} and train_size == 327680:
        train_size = 0
    if dataset_name in {"gsm8k", "math500", "aime24"} and test_size == 1024:
        test_size = 0

    if experiment_name == "countdown-qwen2.5-3b-modal":
        model_slug = base_model.split("/")[-1].lower().replace(".", "-")
        experiment_name = f"{dataset_name}-{model_slug}"
        if prompt_conditioning_mode != "off":
            experiment_name += f"-{prompt_conditioning_mode}"
        experiment_name += "-modal"

    if prepare_data:
        prepare_countdown_data.remote(
            repo_ref=repo_ref,
            dataset_name=dataset_name,
            dataset_subdir=dataset_subdir,
            template_type=template_type,
            prompt_conditioning_mode=prompt_conditioning_mode,
            train_size=train_size,
            test_size=test_size,
        )

    train_kwargs = dict(
        repo_ref=repo_ref,
        base_model=base_model,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        dataset_subdir=dataset_subdir,
        prompt_conditioning_mode=prompt_conditioning_mode,
        n_gpus=n_gpus,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        actor_lr=actor_lr,
        actor_ppo_mini_batch_size=actor_ppo_mini_batch_size,
        actor_ppo_micro_batch_size=actor_ppo_micro_batch_size,
        rollout_log_prob_micro_batch_size=rollout_log_prob_micro_batch_size,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        ref_log_prob_micro_batch_size=ref_log_prob_micro_batch_size,
        critic_lr=critic_lr,
        critic_ppo_micro_batch_size=critic_ppo_micro_batch_size,
        kl_coef=kl_coef,
        adv_estimator=adv_estimator,
        rollout_n=rollout_n,
        use_kl_loss=use_kl_loss,
        actor_kl_loss_coef=actor_kl_loss_coef,
        actor_kl_loss_type=actor_kl_loss_type,
        save_freq=save_freq,
        test_freq=test_freq,
        total_epochs=total_epochs,
        total_training_steps=total_training_steps,
        validation_rollouts=validation_rollouts,
        eval_ks=eval_ks,
        extra_hydra_args=extra_hydra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        seed=seed,
        prompt_bandit_temp=prompt_bandit_temp,
        prompt_bandit_prior_success=prompt_bandit_prior_success,
        prompt_bandit_prior_count=prompt_bandit_prior_count,
        prompt_success_threshold=prompt_success_threshold,
    )

    print(f"Launching TinyZero on Modal GPU spec: {gpu_spec}")
    run_dir = worker_cls().run.remote(train_kwargs)

    print(f"TinyZero run completed. Artifacts saved under: {run_dir}")
