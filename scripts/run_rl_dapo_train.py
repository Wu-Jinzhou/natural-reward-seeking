#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
VERL_DIR = PROJECT_ROOT / "verl"
for path in (SRC_DIR, VERL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import ray

from natural_reward_seeking.prompting import condition_map
from natural_reward_seeking.rl import load_rl_poc_config
from natural_reward_seeking.rl.training import build_training_overrides, install_zero_variance_group_filter
from natural_reward_seeking.utils import write_json
from verl.experimental.reward_loop.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-condition DAPO RL PoC training job.")
    parser.add_argument("--config", default="configs/rl_dapo_poc.toml")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--initial-model-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_rl_poc_config(args.config)
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else config.output.directory / "dataset"
    output_root = Path(args.output_dir).resolve() if args.output_dir else config.output.directory / "training"
    output_root.mkdir(parents=True, exist_ok=True)

    condition_lookup = condition_map()
    if args.condition not in condition_lookup:
        raise ValueError(f"Unknown condition: {args.condition}")
    if args.condition not in config.training.condition_names:
        raise ValueError(
            f"Condition {args.condition} is not enabled in {args.config}. "
            f"Enabled conditions: {', '.join(config.training.condition_names)}"
        )
    condition = condition_lookup[args.condition]

    dataset_files_path = dataset_dir / "dataset_files.json"
    if not dataset_files_path.exists():
        raise FileNotFoundError(
            f"Could not find {dataset_files_path}. Run scripts/prepare_rl_poc_dataset.py first."
        )
    dataset_files = json.loads(dataset_files_path.read_text(encoding="utf-8"))
    if condition.name not in dataset_files:
        raise ValueError(f"No parquet files found for condition {condition.name} in {dataset_files_path}")

    total_training_steps = max(
        1,
        config.training.num_train_epochs * math.ceil(config.dataset.train_size / config.training.train_batch_size),
    )
    run_output_dir = output_root / condition.name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    reward_fn_path = SRC_DIR / "natural_reward_seeking" / "rl" / "reward_fn.py"
    zero_variance_log_path = run_output_dir / "dynamic_filter_steps.jsonl"
    zero_variance_summary_path = run_output_dir / "dynamic_filter_summary.json"
    install_zero_variance_group_filter(zero_variance_log_path, zero_variance_summary_path)

    overrides = build_training_overrides(
        condition=condition,
        train_file=dataset_files[condition.name]["train_file"],
        val_file=dataset_files[condition.name]["val_file"],
        run_output_dir=run_output_dir,
        policy_model_id=config.models.policy_model_id,
        reward_model_id=config.models.reward_model_id,
        trust_remote_code=config.models.trust_remote_code,
        data_seed=config.dataset.seed,
        total_training_steps=total_training_steps,
        train_batch_size=config.training.train_batch_size,
        group_size=config.training.group_size,
        max_prompt_tokens=config.training.max_prompt_tokens,
        max_output_tokens=config.training.max_output_tokens,
        ppo_mini_batch_size=config.training.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=config.training.ppo_micro_batch_size_per_gpu,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        ppo_epochs=config.training.ppo_epochs,
        temperature=config.training.temperature,
        checkpoint_count=config.training.checkpoint_count,
        filter_overlong_prompts=config.training.filter_overlong_prompts,
        use_dynamic_bsz=config.training.use_dynamic_bsz,
        fsdp_param_offload=config.training.fsdp_param_offload,
        fsdp_optimizer_offload=config.training.fsdp_optimizer_offload,
        policy_gpu_memory_utilization=config.training.policy_gpu_memory_utilization,
        reward_gpu_memory_utilization=config.training.reward_gpu_memory_utilization,
        reward_num_workers=config.training.reward_num_workers,
        actor_ppo_max_token_len_per_gpu=config.training.actor_ppo_max_token_len_per_gpu,
        rollout_log_prob_max_token_len_per_gpu=config.training.rollout_log_prob_max_token_len_per_gpu,
        rollout_max_num_batched_tokens=config.training.rollout_max_num_batched_tokens,
        reward_max_num_batched_tokens=config.training.reward_max_num_batched_tokens,
        reward_max_model_len=config.training.reward_max_model_len,
        reward_fn_path=reward_fn_path,
        initial_model_path=args.initial_model_path,
    )

    write_json(
        run_output_dir / "run_metadata.json",
        {
            "config": config.to_dict(),
            "condition": {
                "name": condition.name,
                "label": condition.label,
                "system_prompt": condition.system_prompt,
            },
            "dataset_dir": str(dataset_dir.resolve()),
            "total_training_steps": total_training_steps,
            "zero_variance_log_path": str(zero_variance_log_path.resolve()),
        },
    )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config_dir = VERL_DIR / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        verl_config = compose(config_name="ppo_trainer", overrides=overrides)
    auto_set_device(verl_config)
    verl_config = migrate_legacy_reward_impl(verl_config)

    resolved_config_path = run_output_dir / "resolved_verl_config.yaml"
    resolved_config_path.write_text(OmegaConf.to_yaml(verl_config), encoding="utf-8")
    print(f"Running RL training for {condition.name}")
    print(f"Run output dir: {run_output_dir}")
    print(f"Resolved config: {resolved_config_path}")

    try:
        run_ppo(verl_config)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
