"""
Modal wrapper for running TinyZero remotely on Modal GPUs.

This wrapper keeps all heavy work off the local machine:
  1. TinyZero is cloned and installed inside the Modal image.
  2. Countdown data is prepared on Modal.
  3. Training is launched on a remote GPU container.

Usage examples:
  modal run --detach modal_tinyzero.py
  modal run --detach modal_tinyzero.py --base-model Qwen/Qwen2.5-1.5B --n-gpus 1
  modal run --detach modal_tinyzero.py --base-model Qwen/Qwen2.5-3B --n-gpus 2
  modal run --detach modal_tinyzero.py --prepare-data False --extra-hydra-args "critic.model.enable_gradient_checkpointing=True"
  modal run --detach modal_tinyzero.py --dataset-name gsm8k --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1
  modal run --detach modal_tinyzero.py --dataset-name countdown --total-training-steps 250 --prompt-conditioning-mode mixed --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1
  modal run --detach modal_tinyzero.py --dataset-name math500 --total-training-steps 250 --base-model Qwen/Qwen2.5-1.5B-Instruct --n-gpus 1

Notes:
  - This wrapper targets the archived TinyZero countdown setup from GitHub.
  - The default path is the 2-GPU Qwen2.5-3B run described in the TinyZero README.
  - W&B is disabled by default so no secret is required.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import defaultdict
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
            "You can use basic arithmetic operations (+, -, *, /) and each number "
            f"can only be used once. {family_instruction} "
            "Assistant: Let me solve this step by step.\n"
        )
    if template_type == "qwen-instruct":
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant. You first thinks about the reasoning process "
            "in the mind and then provides the user with the answer.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /) and each number "
            f"can only be used once. {family_instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Let me solve this step by step.\n"
        )
    raise ValueError(f"Unsupported template_type: {template_type}")


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


def _select_prompt_family(*, dataset_name: str, split: str, idx: int, prompt_conditioning_mode: str) -> str:
    if split != "train" or prompt_conditioning_mode == "off":
        return "default"
    if prompt_conditioning_mode != "mixed":
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

        if not groups:
            continue

        step = int(jsonl_path.stem)
        metrics: dict[str, float] = {}
        for k in ks:
            pass_hits = 0
            cluster_counts = []
            for records in groups.values():
                subset = records[:k]
                if not subset:
                    continue
                if any(float(item.get("score", 0.0)) > 0.5 for item in subset):
                    pass_hits += 1
                signatures = {_reasoning_signature(item.get("output", "")) for item in subset}
                signatures.discard("")
                cluster_counts.append(float(len(signatures)))

            group_count = float(len(groups))
            metrics[f"val-core/custom/pass@{k}"] = pass_hits / group_count if group_count else 0.0
            metrics[f"val-aux/custom/clusters@{k}"] = (
                sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0.0
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
    per_step_metrics = _compute_validation_passk_clusters(validation_data_dir, ks)
    if not per_step_metrics:
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
    finally:
        wandb.finish()


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
    save_freq: int,
    test_freq: int,
    total_epochs: int,
    total_training_steps: int,
    validation_rollouts: int,
    extra_hydra_args: str,
    wandb_mode: str,
    wandb_project: str,
    validation_data_dir: str,
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
        f"algorithm.kl_ctrl.kl_coef={kl_coef}",
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
        f"+trainer.validation_data_dir={validation_data_dir}",
        f"+actor_rollout_ref.rollout.val_kwargs.n={validation_rollouts}",
        "+actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "+actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "+actor_rollout_ref.rollout.val_kwargs.top_k=-1",
    ]
    if total_training_steps > 0:
        cmd.append(f"trainer.total_training_steps={total_training_steps}")
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
        if len(raw_dataset) <= train_size + test_size:
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
                solution = {"target": example["target"], "numbers": example["nums"]}
                return {
                    "data_source": "countdown",
                    "prompt": [{"role": "user", "content": _make_countdown_prefix(example, template_type, prompt_family)}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {"split": split, "index": idx, "prompt_family": prompt_family},
                }

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
                question = _make_gsm8k_prompt(example["question"], prompt_family)
                answer_raw = example["answer"].strip()
                return {
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
                        "question_raw": example["question"],
                        "answer_raw": answer_raw,
                    },
                }

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
                problem = _make_competition_math_prompt(example["problem"], prompt_family)
                answer_raw = example["answer"].strip()
                solution_raw = example["solution"].strip()
                return {
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
                        "problem_raw": example["problem"],
                        "answer_raw": answer_raw,
                        "subject": example.get("subject"),
                        "level": example.get("level"),
                        "unique_id": example.get("unique_id"),
                    },
                }

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
                problem = _make_competition_math_prompt(example["problem"], prompt_family)
                answer_raw = str(example["answer"]).strip()
                solution_raw = example["solution"].strip()
                return {
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
                        "problem_raw": example["problem"],
                        "answer_raw": answer_raw,
                        "id": example.get("id"),
                        "url": example.get("url"),
                        "year": example.get("year"),
                    },
                }

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
    dataset_subdir: str,
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
) -> str:
    import os
    import subprocess

    import torch

    artifacts.reload()
    _repo_checkout(repo_ref)

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
    }
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
        save_freq=save_freq,
        test_freq=test_freq,
        total_epochs=total_epochs,
        total_training_steps=total_training_steps,
        validation_rollouts=validation_rollouts,
        extra_hydra_args=extra_hydra_args,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        validation_data_dir=str(validation_data_dir),
    )
    run_error = None
    try:
        _run_and_tee(cmd, log_path=log_path, env=env)
    except subprocess.CalledProcessError as exc:
        run_error = exc
    finally:
        if wandb_mode != "disabled" and validation_data_dir.exists():
            _log_validation_passk_clusters_to_wandb(
                validation_data_dir=validation_data_dir,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                experiment_name=experiment_name,
                run_id=run_id,
                eval_ks=eval_ks,
                validation_rollouts=validation_rollouts,
            )
    if run_error is not None:
        raise run_error

    artifacts.commit()
    return str(run_dir)


@app.function(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    secrets=[wandb_secret],
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    cpu=8,
)
def train_tinyzero_1gpu(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-1.5B",
    experiment_name: str = "countdown-qwen2.5-1.5b-modal",
    dataset_subdir: str = "countdown",
    train_batch_size: int = 256,
    val_batch_size: int = 1312,
    max_prompt_length: int = 256,
    max_response_length: int = 1024,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
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
) -> str:
    return _train_impl(
        repo_ref=repo_ref,
        base_model=base_model,
        experiment_name=experiment_name,
        dataset_subdir=dataset_subdir,
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
    )


@app.function(
    image=image,
    volumes={ARTIFACT_DIR: artifacts},
    secrets=[wandb_secret],
    gpu="A100-80GB:2",
    timeout=60 * 60 * 24,
    cpu=16,
)
def train_tinyzero_2gpu(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-3B",
    experiment_name: str = "countdown-qwen2.5-3b-modal",
    dataset_subdir: str = "countdown",
    train_batch_size: int = 256,
    val_batch_size: int = 1312,
    max_prompt_length: int = 256,
    max_response_length: int = 1024,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
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
) -> str:
    return _train_impl(
        repo_ref=repo_ref,
        base_model=base_model,
        experiment_name=experiment_name,
        dataset_subdir=dataset_subdir,
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
    )


@app.local_entrypoint()
def main(
    repo_ref: str = "main",
    base_model: str = "Qwen/Qwen2.5-3B",
    experiment_name: str = "countdown-qwen2.5-3b-modal",
    n_gpus: int = 2,
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
    max_response_length: int = 1024,
    actor_lr: float = 1e-6,
    actor_ppo_mini_batch_size: int = 64,
    actor_ppo_micro_batch_size: int = 4,
    rollout_log_prob_micro_batch_size: int = 4,
    rollout_gpu_memory_utilization: float = 0.25,
    ref_log_prob_micro_batch_size: int = 2,
    critic_lr: float = 1e-5,
    critic_ppo_micro_batch_size: int = 2,
    kl_coef: float = 0.001,
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
) -> None:
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
        dataset_subdir=dataset_subdir,
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
    )

    if n_gpus == 1:
        run_dir = train_tinyzero_1gpu.remote(**train_kwargs)
    elif n_gpus == 2:
        run_dir = train_tinyzero_2gpu.remote(**train_kwargs)
    else:
        raise ValueError("This wrapper currently supports only 1 or 2 GPUs per run.")

    print(f"TinyZero run completed. Artifacts saved under: {run_dir}")
