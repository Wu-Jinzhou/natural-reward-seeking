from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INITIAL_EVAL_CONFIG = PROJECT_ROOT / "configs" / "initial_prompt_eval.toml"
DEFAULT_SKILL_GAP_EVAL_CONFIG = PROJECT_ROOT / "configs" / "skill_gap_eval.toml"


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass
class DatasetConfig:
    root: Path
    sample_per_category: int
    seed: int


@dataclass
class ModelConfig:
    policy_model_id: str
    policy_backend: str
    reward_model_id: str
    trust_remote_code: bool
    attn_implementation: str


@dataclass
class GenerationConfig:
    batch_size: int
    reward_batch_size: int
    max_new_tokens: int
    enable_thinking: bool


@dataclass
class AnalysisConfig:
    top_keywords: int
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass
class OutputConfig:
    directory: Path


@dataclass
class InitialEvalConfig:
    dataset: DatasetConfig
    models: ModelConfig
    generation: GenerationConfig
    analysis: AnalysisConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"]["root"] = str(self.dataset.root)
        payload["output"]["directory"] = str(self.output.directory)
        return payload


def load_initial_eval_config(config_path: str | Path | None = None) -> InitialEvalConfig:
    path = _resolve_repo_path(config_path or DEFAULT_INITIAL_EVAL_CONFIG)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return InitialEvalConfig(
        dataset=DatasetConfig(
            root=_resolve_repo_path(raw["dataset"]["root"]),
            sample_per_category=int(raw["dataset"]["sample_per_category"]),
            seed=int(raw["dataset"]["seed"]),
        ),
        models=ModelConfig(
            policy_model_id=str(raw["models"]["policy_model_id"]),
            policy_backend=str(raw["models"].get("policy_backend", "transformers")),
            reward_model_id=str(raw["models"]["reward_model_id"]),
            trust_remote_code=bool(raw["models"]["trust_remote_code"]),
            attn_implementation=str(raw["models"]["attn_implementation"]),
        ),
        generation=GenerationConfig(
            batch_size=int(raw["generation"]["batch_size"]),
            reward_batch_size=int(raw["generation"]["reward_batch_size"]),
            max_new_tokens=int(raw["generation"]["max_new_tokens"]),
            enable_thinking=bool(raw["generation"]["enable_thinking"]),
        ),
        analysis=AnalysisConfig(
            top_keywords=int(raw["analysis"]["top_keywords"]),
            bootstrap_samples=int(raw["analysis"]["bootstrap_samples"]),
            bootstrap_seed=int(raw["analysis"]["bootstrap_seed"]),
        ),
        output=OutputConfig(
            directory=_resolve_repo_path(raw["output"]["dir"]),
        ),
    )


SkillGapEvalConfig = InitialEvalConfig


def load_skill_gap_eval_config(config_path: str | Path | None = None) -> SkillGapEvalConfig:
    return load_initial_eval_config(config_path or DEFAULT_SKILL_GAP_EVAL_CONFIG)
