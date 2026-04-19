from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RL_POC_CONFIG = PROJECT_ROOT / "configs" / "rl_dapo_poc.toml"


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass
class RLDatasetConfig:
    root: Path
    heldout_manifest: Path
    eval_manifest: Path
    train_data_type: str
    train_size: int
    val_size: int
    seed: int


@dataclass
class RLModelConfig:
    policy_model_id: str
    reward_model_id: str
    trust_remote_code: bool


@dataclass
class RLTrainingConfig:
    condition_names: list[str]
    group_size: int
    max_prompt_tokens: int
    max_output_tokens: int
    train_batch_size: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    learning_rate: float
    num_train_epochs: int
    ppo_epochs: int
    temperature: float
    checkpoint_count: int
    filter_overlong_prompts: bool
    use_dynamic_bsz: bool
    fsdp_param_offload: bool
    fsdp_optimizer_offload: bool
    policy_gpu_memory_utilization: float
    reward_gpu_memory_utilization: float
    reward_num_workers: int
    actor_ppo_max_token_len_per_gpu: int
    rollout_log_prob_max_token_len_per_gpu: int
    rollout_max_num_batched_tokens: int
    reward_max_num_batched_tokens: int
    reward_max_model_len: int


@dataclass
class RLEvalConfig:
    policy_backend: str
    batch_size: int
    reward_batch_size: int
    max_new_tokens: int
    enable_thinking: bool
    top_keywords: int
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass
class RLOutputConfig:
    directory: Path


@dataclass
class RLPocConfig:
    dataset: RLDatasetConfig
    models: RLModelConfig
    training: RLTrainingConfig
    evaluation: RLEvalConfig
    output: RLOutputConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"]["root"] = str(self.dataset.root)
        payload["dataset"]["heldout_manifest"] = str(self.dataset.heldout_manifest)
        payload["dataset"]["eval_manifest"] = str(self.dataset.eval_manifest)
        payload["output"]["directory"] = str(self.output.directory)
        return payload


def load_rl_poc_config(config_path: str | Path | None = None) -> RLPocConfig:
    path = _resolve_repo_path(config_path or DEFAULT_RL_POC_CONFIG)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return RLPocConfig(
        dataset=RLDatasetConfig(
            root=_resolve_repo_path(raw["dataset"]["root"]),
            heldout_manifest=_resolve_repo_path(raw["dataset"]["heldout_manifest"]),
            eval_manifest=_resolve_repo_path(raw["dataset"]["eval_manifest"]),
            train_data_type=str(raw["dataset"]["train_data_type"]),
            train_size=int(raw["dataset"]["train_size"]),
            val_size=int(raw["dataset"]["val_size"]),
            seed=int(raw["dataset"]["seed"]),
        ),
        models=RLModelConfig(
            policy_model_id=str(raw["models"]["policy_model_id"]),
            reward_model_id=str(raw["models"]["reward_model_id"]),
            trust_remote_code=bool(raw["models"]["trust_remote_code"]),
        ),
        training=RLTrainingConfig(
            condition_names=[str(value) for value in raw["training"]["condition_names"]],
            group_size=int(raw["training"]["group_size"]),
            max_prompt_tokens=int(raw["training"]["max_prompt_tokens"]),
            max_output_tokens=int(raw["training"]["max_output_tokens"]),
            train_batch_size=int(raw["training"]["train_batch_size"]),
            ppo_mini_batch_size=int(raw["training"]["ppo_mini_batch_size"]),
            ppo_micro_batch_size_per_gpu=int(raw["training"]["ppo_micro_batch_size_per_gpu"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            num_train_epochs=int(raw["training"]["num_train_epochs"]),
            ppo_epochs=int(raw["training"]["ppo_epochs"]),
            temperature=float(raw["training"]["temperature"]),
            checkpoint_count=int(raw["training"]["checkpoint_count"]),
            filter_overlong_prompts=bool(raw["training"]["filter_overlong_prompts"]),
            use_dynamic_bsz=bool(raw["training"]["use_dynamic_bsz"]),
            fsdp_param_offload=bool(raw["training"]["fsdp_param_offload"]),
            fsdp_optimizer_offload=bool(raw["training"]["fsdp_optimizer_offload"]),
            policy_gpu_memory_utilization=float(raw["training"]["policy_gpu_memory_utilization"]),
            reward_gpu_memory_utilization=float(raw["training"]["reward_gpu_memory_utilization"]),
            reward_num_workers=int(raw["training"]["reward_num_workers"]),
            actor_ppo_max_token_len_per_gpu=int(raw["training"]["actor_ppo_max_token_len_per_gpu"]),
            rollout_log_prob_max_token_len_per_gpu=int(
                raw["training"]["rollout_log_prob_max_token_len_per_gpu"]
            ),
            rollout_max_num_batched_tokens=int(raw["training"]["rollout_max_num_batched_tokens"]),
            reward_max_num_batched_tokens=int(raw["training"]["reward_max_num_batched_tokens"]),
            reward_max_model_len=int(raw["training"]["reward_max_model_len"]),
        ),
        evaluation=RLEvalConfig(
            policy_backend=str(raw["evaluation"]["policy_backend"]),
            batch_size=int(raw["evaluation"]["batch_size"]),
            reward_batch_size=int(raw["evaluation"]["reward_batch_size"]),
            max_new_tokens=int(raw["evaluation"]["max_new_tokens"]),
            enable_thinking=bool(raw["evaluation"]["enable_thinking"]),
            top_keywords=int(raw["evaluation"]["top_keywords"]),
            bootstrap_samples=int(raw["evaluation"]["bootstrap_samples"]),
            bootstrap_seed=int(raw["evaluation"]["bootstrap_seed"]),
        ),
        output=RLOutputConfig(
            directory=_resolve_repo_path(raw["output"]["dir"]),
        ),
    )
