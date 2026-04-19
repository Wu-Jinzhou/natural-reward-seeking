from __future__ import annotations

import atexit
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from natural_reward_seeking.prompting.conditions import PromptCondition
from natural_reward_seeking.utils.io import write_json


def hydra_quote(value: Any) -> str:
    return json.dumps(str(value))


@dataclass
class ZeroVarianceFilterTracker:
    log_path: Path
    summary_path: Path
    step_count: int = 0
    total_groups: int = 0
    dropped_groups: int = 0
    fallback_steps: int = 0

    def record(
        self,
        *,
        kept_groups: int,
        dropped_groups: int,
        total_groups: int,
        kept_samples: int,
        dropped_samples: int,
        fallback_unfiltered: bool,
    ) -> None:
        self.step_count += 1
        self.total_groups += total_groups
        self.dropped_groups += dropped_groups
        if fallback_unfiltered:
            self.fallback_steps += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": self.step_count,
                        "kept_groups": kept_groups,
                        "dropped_groups": dropped_groups,
                        "total_groups": total_groups,
                        "kept_samples": kept_samples,
                        "dropped_samples": dropped_samples,
                        "fallback_unfiltered": fallback_unfiltered,
                    }
                )
                + "\n"
            )

    def write_summary(self) -> None:
        summary = {
            "steps_observed": self.step_count,
            "total_groups_observed": self.total_groups,
            "dropped_groups_total": self.dropped_groups,
            "dropped_group_rate": (self.dropped_groups / self.total_groups) if self.total_groups else 0.0,
            "fallback_unfiltered_steps": self.fallback_steps,
        }
        write_json(self.summary_path, summary)


def install_zero_variance_group_filter(log_path: str | Path, summary_path: str | Path) -> ZeroVarianceFilterTracker:
    from verl.trainer.ppo import ray_trainer

    if getattr(ray_trainer, "_natural_reward_seeking_zero_variance_filter_installed", False):
        return ray_trainer._natural_reward_seeking_zero_variance_filter_tracker

    tracker = ZeroVarianceFilterTracker(log_path=Path(log_path), summary_path=Path(summary_path))
    original_compute_advantage = ray_trainer.compute_advantage

    def wrapped_compute_advantage(*args, **kwargs):
        if not args:
            return original_compute_advantage(*args, **kwargs)
        data = args[0]
        uid_values = data.non_tensor_batch.get("uid")
        if uid_values is None:
            return original_compute_advantage(*args, **kwargs)

        response_mask = (
            data.batch["response_mask"]
            if "response_mask" in data.batch.keys()
            else ray_trainer.compute_response_mask(data)
        )
        token_rewards = data.batch["token_level_rewards"]
        sequence_rewards = (token_rewards * response_mask).sum(dim=-1).detach().float().cpu()

        grouped_indices: OrderedDict[str, list[int]] = OrderedDict()
        for index, uid in enumerate(uid_values.tolist()):
            grouped_indices.setdefault(str(uid), []).append(index)

        keep_indices: list[int] = []
        dropped_groups = 0
        for indices in grouped_indices.values():
            group_rewards = sequence_rewards[indices]
            is_uniform = bool(torch.all(group_rewards == group_rewards[0]))
            if is_uniform:
                dropped_groups += 1
                continue
            keep_indices.extend(indices)

        total_groups = len(grouped_indices)
        fallback_unfiltered = False
        filtered_data = data
        if keep_indices and len(keep_indices) < len(sequence_rewards):
            filtered_data = data.select_idxs(keep_indices)
        elif dropped_groups and not keep_indices:
            fallback_unfiltered = True

        tracker.record(
            kept_groups=total_groups - dropped_groups,
            dropped_groups=dropped_groups,
            total_groups=total_groups,
            kept_samples=len(keep_indices) if keep_indices else len(sequence_rewards),
            dropped_samples=len(sequence_rewards) - len(keep_indices) if keep_indices else 0,
            fallback_unfiltered=fallback_unfiltered,
        )
        print(
            "[dynamic_filter] "
            f"step={tracker.step_count} kept_groups={total_groups - dropped_groups}/{total_groups} "
            f"dropped_groups={dropped_groups} fallback_unfiltered={fallback_unfiltered}"
        )

        new_args = (filtered_data, *args[1:])
        return original_compute_advantage(*new_args, **kwargs)

    ray_trainer.compute_advantage = wrapped_compute_advantage
    ray_trainer._natural_reward_seeking_zero_variance_filter_installed = True
    ray_trainer._natural_reward_seeking_zero_variance_filter_tracker = tracker
    atexit.register(tracker.write_summary)
    return tracker


def build_training_overrides(
    *,
    condition: PromptCondition,
    train_file: str | Path,
    val_file: str | Path,
    run_output_dir: str | Path,
    policy_model_id: str,
    reward_model_id: str,
    trust_remote_code: bool,
    data_seed: int,
    total_training_steps: int,
    train_batch_size: int,
    group_size: int,
    max_prompt_tokens: int,
    max_output_tokens: int,
    ppo_mini_batch_size: int,
    ppo_micro_batch_size_per_gpu: int,
    learning_rate: float,
    num_train_epochs: int,
    ppo_epochs: int,
    temperature: float,
    checkpoint_count: int,
    filter_overlong_prompts: bool,
    use_dynamic_bsz: bool,
    fsdp_param_offload: bool,
    fsdp_optimizer_offload: bool,
    policy_gpu_memory_utilization: float,
    reward_gpu_memory_utilization: float,
    reward_num_workers: int,
    actor_ppo_max_token_len_per_gpu: int,
    rollout_log_prob_max_token_len_per_gpu: int,
    rollout_max_num_batched_tokens: int,
    reward_max_num_batched_tokens: int,
    reward_max_model_len: int,
    reward_fn_path: str | Path,
    initial_model_path: str | Path | None = None,
) -> list[str]:
    run_output_path = Path(run_output_dir)
    checkpoint_dir = run_output_path / "checkpoints"
    train_file_path = Path(train_file).resolve()
    val_file_path = Path(val_file).resolve()
    reward_fn_file = Path(reward_fn_path).resolve()
    model_path = Path(initial_model_path).resolve() if initial_model_path is not None else policy_model_id
    save_freq = max(1, total_training_steps // checkpoint_count)
    return [
        f"data.train_files={hydra_quote(train_file_path)}",
        f"data.val_files={hydra_quote(val_file_path)}",
        "data.prompt_key=prompt",
        "data.truncation=error",
        f"data.max_prompt_length={max_prompt_tokens}",
        f"data.max_response_length={max_output_tokens}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size={train_batch_size}",
        "data.return_raw_chat=True",
        f"data.filter_overlong_prompts={str(filter_overlong_prompts).lower()}",
        "data.shuffle=True",
        f"data.seed={data_seed}",
        f"actor_rollout_ref.rollout.n={group_size}",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.clip_ratio_low=0.8",
        "actor_rollout_ref.actor.clip_ratio_high=1.28",
        "actor_rollout_ref.actor.clip_ratio_c=3.0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        f"actor_rollout_ref.actor.ppo_epochs={ppo_epochs}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={str(use_dynamic_bsz).lower()}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_ppo_max_token_len_per_gpu}",
        f"actor_rollout_ref.actor.optim.lr={learning_rate}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.grad_clip=1.0",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(fsdp_param_offload).lower()}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(fsdp_optimizer_offload).lower()}",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=1",
        "actor_rollout_ref.actor.checkpoint.save_contents=['hf_model']",
        "actor_rollout_ref.actor.checkpoint.load_contents=['hf_model']",
        f"actor_rollout_ref.ref.log_prob_use_dynamic_bsz={str(use_dynamic_bsz).lower()}",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={rollout_log_prob_max_token_len_per_gpu}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.prompt_length={max_prompt_tokens}",
        f"actor_rollout_ref.rollout.response_length={max_output_tokens}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={policy_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={str(use_dynamic_bsz).lower()}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={rollout_log_prob_max_token_len_per_gpu}",
        f"actor_rollout_ref.rollout.temperature={temperature}",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        f"actor_rollout_ref.model.path={hydra_quote(model_path)}",
        f"actor_rollout_ref.model.trust_remote_code={str(trust_remote_code).lower()}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "reward.reward_manager.name=dapo",
        f"reward.num_workers={reward_num_workers}",
        "reward.reward_model.enable=True",
        "reward.reward_model.enable_resource_pool=False",
        f"reward.reward_model.model_path={hydra_quote(reward_model_id)}",
        "reward.reward_model.rollout.name=vllm",
        "reward.reward_model.rollout.tensor_model_parallel_size=1",
        f"reward.reward_model.rollout.gpu_memory_utilization={reward_gpu_memory_utilization}",
        "reward.reward_model.rollout.free_cache_engine=True",
        f"reward.reward_model.rollout.max_num_batched_tokens={reward_max_num_batched_tokens}",
        f"reward.reward_model.rollout.max_model_len={reward_max_model_len}",
        f"reward.reward_model.rollout.prompt_length={reward_max_model_len}",
        "reward.reward_model.rollout.response_length=1",
        f"reward.custom_reward_function.path={hydra_quote(reward_fn_file)}",
        "reward.custom_reward_function.name=compute_skywork_reward",
        f"+reward.custom_reward_function.reward_kwargs.model_name={hydra_quote(reward_model_id)}",
        "trainer.logger=['console']",
        "trainer.project_name=natural_reward_seeking_rl_poc",
        f"trainer.experiment_name={condition.name}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.save_freq={save_freq}",
        f"trainer.total_epochs={num_train_epochs}",
        f"trainer.total_training_steps={total_training_steps}",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={hydra_quote(checkpoint_dir)}",
        f"trainer.max_actor_ckpt_to_keep={checkpoint_count}",
        "trainer.use_legacy_worker_impl=disable",
    ]
