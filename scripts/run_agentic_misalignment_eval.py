#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from natural_reward_seeking.agentic_misalignment import (
    build_agentic_grid,
    build_agentic_prompt_cases,
    build_agentic_prompt_conditions,
    classify_response_rows,
    clean_response_for_reward,
    load_anthropic_prompt_generator,
    write_agentic_analysis,
)
from natural_reward_seeking.agentic_misalignment.prompts import generate_agentic_prompts
from natural_reward_seeking.models import PolicyGenerator, SkyworkRewardScorer
from natural_reward_seeking.utils import save_rows_csv, write_json, write_jsonl


@dataclass
class AgenticConfig:
    repo_root: Path
    scenarios: list[str]
    goal_types: list[str]
    goal_values: list[str]
    urgency_types: list[str]
    samples_per_condition: int
    production_variant: bool


@dataclass
class ModelConfig:
    policy_model_id: str
    policy_backend: str
    reward_model_id: str
    classifier_model_id: str
    trust_remote_code: bool
    attn_implementation: str


@dataclass
class GenerationConfig:
    batch_size: int
    reward_batch_size: int
    reward_max_length: int
    reward_truncation_side: str
    classifier_concurrency: int
    max_new_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool


@dataclass
class AnalysisConfig:
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass
class OutputConfig:
    directory: Path


@dataclass
class EvalConfig:
    agentic: AgenticConfig
    models: ModelConfig
    generation: GenerationConfig
    analysis: AnalysisConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agentic"]["repo_root"] = str(self.agentic.repo_root)
        payload["output"]["directory"] = str(self.output.directory)
        return payload


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _as_list(value: Any) -> list[str]:
    return [str(item) for item in value]


def _split_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_config(path: str | Path) -> EvalConfig:
    raw = tomllib.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))
    return EvalConfig(
        agentic=AgenticConfig(
            repo_root=_resolve_repo_path(raw["agentic"]["repo_root"]),
            scenarios=_as_list(raw["agentic"]["scenarios"]),
            goal_types=_as_list(raw["agentic"]["goal_types"]),
            goal_values=_as_list(raw["agentic"]["goal_values"]),
            urgency_types=_as_list(raw["agentic"]["urgency_types"]),
            samples_per_condition=int(raw["agentic"]["samples_per_condition"]),
            production_variant=bool(raw["agentic"].get("production_variant", False)),
        ),
        models=ModelConfig(
            policy_model_id=str(raw["models"]["policy_model_id"]),
            policy_backend=str(raw["models"].get("policy_backend", "vllm")),
            reward_model_id=str(raw["models"]["reward_model_id"]),
            classifier_model_id=str(raw["models"]["classifier_model_id"]),
            trust_remote_code=bool(raw["models"].get("trust_remote_code", False)),
            attn_implementation=str(raw["models"].get("attn_implementation", "auto")),
        ),
        generation=GenerationConfig(
            batch_size=int(raw["generation"]["batch_size"]),
            reward_batch_size=int(raw["generation"]["reward_batch_size"]),
            reward_max_length=int(raw["generation"].get("reward_max_length", 4096)),
            reward_truncation_side=str(raw["generation"].get("reward_truncation_side", "right")),
            classifier_concurrency=int(raw["generation"].get("classifier_concurrency", 20)),
            max_new_tokens=int(raw["generation"]["max_new_tokens"]),
            temperature=float(raw["generation"].get("temperature", 1.0)),
            top_p=float(raw["generation"].get("top_p", 1.0)),
            enable_thinking=bool(raw["generation"].get("enable_thinking", True)),
        ),
        analysis=AnalysisConfig(
            bootstrap_samples=int(raw["analysis"]["bootstrap_samples"]),
            bootstrap_seed=int(raw["analysis"]["bootstrap_seed"]),
        ),
        output=OutputConfig(directory=_resolve_repo_path(raw["output"]["dir"])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Olmo on Anthropic agentic-misalignment scenarios.")
    parser.add_argument("--config", default="configs/agentic_misalignment_eval.toml")
    parser.add_argument("--agentic-repo-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--policy-model-id", default=None)
    parser.add_argument("--policy-backend", default=None)
    parser.add_argument("--reward-model-id", default=None)
    parser.add_argument("--classifier-model-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reward-batch-size", type=int, default=None)
    parser.add_argument("--reward-max-length", type=int, default=None)
    parser.add_argument("--reward-truncation-side", default=None)
    parser.add_argument("--classifier-concurrency", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--samples-per-condition", type=int, default=None)
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--goal-types", default=None)
    parser.add_argument("--goal-values", default=None)
    parser.add_argument("--urgency-types", default=None)
    parser.add_argument("--prompt-conditions", default=None)
    parser.add_argument("--stages", default="prompts,generation,scoring,classification,analysis")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--production-variant", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def apply_overrides(config: EvalConfig, args: argparse.Namespace) -> EvalConfig:
    if args.agentic_repo_root is not None:
        config.agentic.repo_root = Path(args.agentic_repo_root).resolve()
    if args.output_dir is not None:
        config.output.directory = _resolve_repo_path(args.output_dir)
    if args.policy_model_id is not None:
        config.models.policy_model_id = str(args.policy_model_id)
    if args.policy_backend is not None:
        config.models.policy_backend = str(args.policy_backend)
    if args.reward_model_id is not None:
        config.models.reward_model_id = str(args.reward_model_id)
    if args.classifier_model_id is not None:
        config.models.classifier_model_id = str(args.classifier_model_id)
    if args.batch_size is not None:
        config.generation.batch_size = int(args.batch_size)
    if args.reward_batch_size is not None:
        config.generation.reward_batch_size = int(args.reward_batch_size)
    if args.reward_max_length is not None:
        config.generation.reward_max_length = int(args.reward_max_length)
    if args.reward_truncation_side is not None:
        config.generation.reward_truncation_side = str(args.reward_truncation_side)
    if args.classifier_concurrency is not None:
        config.generation.classifier_concurrency = int(args.classifier_concurrency)
    if args.max_new_tokens is not None:
        config.generation.max_new_tokens = int(args.max_new_tokens)
    if args.temperature is not None:
        config.generation.temperature = float(args.temperature)
    if args.top_p is not None:
        config.generation.top_p = float(args.top_p)
    if args.samples_per_condition is not None:
        config.agentic.samples_per_condition = int(args.samples_per_condition)
    if args.scenarios is not None:
        config.agentic.scenarios = _split_csv(args.scenarios) or config.agentic.scenarios
    if args.goal_types is not None:
        config.agentic.goal_types = _split_csv(args.goal_types) or config.agentic.goal_types
    if args.goal_values is not None:
        config.agentic.goal_values = _split_csv(args.goal_values) or config.agentic.goal_values
    if args.urgency_types is not None:
        config.agentic.urgency_types = _split_csv(args.urgency_types) or config.agentic.urgency_types
    if args.attn_implementation is not None:
        config.models.attn_implementation = str(args.attn_implementation)
    if args.production_variant is not None:
        config.agentic.production_variant = bool(args.production_variant)
    if args.trust_remote_code is not None:
        config.models.trust_remote_code = bool(args.trust_remote_code)
    if args.enable_thinking is not None:
        config.generation.enable_thinking = bool(args.enable_thinking)
    return config


def parse_stages(raw: str) -> list[str]:
    stages = [stage.strip().lower() for stage in raw.split(",") if stage.strip()]
    valid = {"prompts", "generation", "scoring", "classification", "analysis"}
    invalid = [stage for stage in stages if stage not in valid]
    if invalid:
        raise ValueError(f"Unsupported stages: {', '.join(invalid)}")
    if not stages:
        raise ValueError("At least one stage must be selected.")
    return stages


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_prompt_artifacts(config: EvalConfig, output_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_generator = load_anthropic_prompt_generator(config.agentic.repo_root)
    agentic_conditions = build_agentic_grid(
        scenarios=config.agentic.scenarios,
        goal_types=config.agentic.goal_types,
        goal_values=config.agentic.goal_values,
        urgency_types=config.agentic.urgency_types,
    )
    agentic_prompts = generate_agentic_prompts(
        prompt_generator=prompt_generator,
        conditions=agentic_conditions,
        production_variant=config.agentic.production_variant,
    )
    prompt_conditions = build_agentic_prompt_conditions()
    requested_conditions = _split_csv(args.prompt_conditions)
    if requested_conditions:
        requested = set(requested_conditions)
        prompt_conditions = [condition for condition in prompt_conditions if condition.name in requested]
        missing = requested - {condition.name for condition in prompt_conditions}
        if missing:
            raise ValueError(f"Unknown prompt conditions: {', '.join(sorted(missing))}")

    prompt_manifest_rows = []
    for prompt in agentic_prompts:
        prompt_manifest_rows.append(
            {
                **prompt.metadata,
                "agentic_system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "email_content": prompt.email_content,
                "prompt_text": prompt.user_prompt_with_email,
            }
        )
    case_rows = build_agentic_prompt_cases(
        agentic_prompts=agentic_prompts,
        prompt_conditions=prompt_conditions,
        samples_per_condition=config.agentic.samples_per_condition,
    )
    write_jsonl(output_dir / "agentic_prompt_manifest.jsonl", prompt_manifest_rows)
    save_rows_csv(output_dir / "agentic_prompt_manifest.csv", prompt_manifest_rows)
    write_jsonl(output_dir / "case_manifest.jsonl", case_rows)
    save_rows_csv(output_dir / "case_manifest.csv", case_rows)
    return prompt_manifest_rows, case_rows


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    stages = parse_stages(args.stages)
    if config.models.policy_backend == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", config.to_dict())

    case_rows: list[dict[str, Any]] | None = None
    if "prompts" in stages or "generation" in stages:
        _, case_rows = write_prompt_artifacts(config, output_dir, args)
        write_json(
            output_dir / "prompt_generation_summary.json",
            {
                "num_agentic_conditions": len({row["agentic_condition_id"] for row in case_rows}),
                "num_prompt_conditions": len({row["condition_name"] for row in case_rows}),
                "samples_per_agentic_prompt_condition": config.agentic.samples_per_condition,
                "num_total_cases": len(case_rows),
            },
        )

    response_rows: list[dict[str, Any]]
    needs_existing_responses = any(stage in stages for stage in ("scoring", "classification", "analysis"))
    if "generation" in stages:
        if case_rows is None:
            case_rows = read_jsonl_rows(output_dir / "case_manifest.jsonl")
        generator = PolicyGenerator(
            model_id=config.models.policy_model_id,
            backend=config.models.policy_backend,
            batch_size=config.generation.batch_size,
            max_new_tokens=config.generation.max_new_tokens,
            enable_thinking=config.generation.enable_thinking,
            attn_implementation=config.models.attn_implementation,
            trust_remote_code=config.models.trust_remote_code,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
        )
        response_rows = generator.generate_case_rows(
            case_rows,
            progress_desc="Generating agentic responses",
        )
        generator.release()
        write_jsonl(output_dir / "responses_pre_reward.jsonl", response_rows)
        save_rows_csv(output_dir / "responses_pre_reward.csv", response_rows)
        write_jsonl(output_dir / "responses.jsonl", response_rows)
        save_rows_csv(output_dir / "responses.csv", response_rows)
    elif needs_existing_responses:
        response_rows = read_jsonl_rows(output_dir / "responses.jsonl" if (output_dir / "responses.jsonl").exists() else output_dir / "responses_pre_reward.jsonl")
    else:
        return

    if "scoring" in stages:
        reward_scorer = SkyworkRewardScorer(
            model_id=config.models.reward_model_id,
            trust_remote_code=config.models.trust_remote_code,
            attn_implementation="sdpa",
            max_length=config.generation.reward_max_length,
            truncation_side=config.generation.reward_truncation_side,
        )
        reward_messages = []
        for row in response_rows:
            reward_text = clean_response_for_reward(row)
            row["reward_scoring_text"] = reward_text
            reward_messages.append(
                [
                    {"role": "system", "content": row["agentic_system_prompt"]},
                    {"role": "user", "content": row["prompt_text"]},
                    {"role": "assistant", "content": reward_text},
                ]
            )
        reward_scores = reward_scorer.score_messages(
            reward_messages,
            batch_size=config.generation.reward_batch_size,
            progress_desc="Scoring agentic responses",
        )
        reward_scorer.release()
        for row, reward_score in zip(response_rows, reward_scores, strict=True):
            row["reward_score"] = float(reward_score)
        write_jsonl(output_dir / "responses.jsonl", response_rows)
        save_rows_csv(output_dir / "responses.csv", response_rows)

    if "classification" in stages:
        response_rows = classify_response_rows(
            response_rows,
            agentic_repo_root=config.agentic.repo_root,
            classifier_model_id=config.models.classifier_model_id,
            concurrency=config.generation.classifier_concurrency,
        )
        write_jsonl(output_dir / "responses.jsonl", response_rows)
        save_rows_csv(output_dir / "responses.csv", response_rows)

    if "analysis" in stages:
        write_agentic_analysis(
            response_rows,
            output_dir=output_dir,
            max_new_tokens=config.generation.max_new_tokens,
            bootstrap_samples=config.analysis.bootstrap_samples,
            bootstrap_seed=config.analysis.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
