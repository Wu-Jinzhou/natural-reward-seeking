#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from natural_reward_seeking.analysis import build_summary_frames, build_target_reward_frames, count_pattern_hits, render_all_plots
from natural_reward_seeking.config import load_initial_eval_config
from natural_reward_seeking.data import load_wildjailbreak_train_rows, sample_rows_by_category
from natural_reward_seeking.models import PolicyGenerator, SkyworkRewardScorer
from natural_reward_seeking.prompting import PROMPT_CONDITIONS, REWARD_KEYWORD_PATTERNS
from natural_reward_seeking.utils import save_rows_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the initial prompt-condition evaluation for the natural-reward-seeking experiment."
        )
    )
    parser.add_argument("--config", default="configs/initial_prompt_eval.toml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-per-category", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy-model-id", default=None)
    parser.add_argument("--policy-backend", default=None)
    parser.add_argument("--reward-model-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reward-batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--stages", default="generation,scoring")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace):
    if args.dataset_root is not None:
        config.dataset.root = Path(args.dataset_root).resolve()
    if args.output_dir is not None:
        path = Path(args.output_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        config.output.directory = path.resolve()
    if args.sample_per_category is not None:
        config.dataset.sample_per_category = int(args.sample_per_category)
    if args.seed is not None:
        config.dataset.seed = int(args.seed)
    if args.policy_model_id is not None:
        config.models.policy_model_id = str(args.policy_model_id)
    if args.policy_backend is not None:
        config.models.policy_backend = str(args.policy_backend)
    if args.reward_model_id is not None:
        config.models.reward_model_id = str(args.reward_model_id)
    if args.batch_size is not None:
        config.generation.batch_size = int(args.batch_size)
    if args.reward_batch_size is not None:
        config.generation.reward_batch_size = int(args.reward_batch_size)
    if args.max_new_tokens is not None:
        config.generation.max_new_tokens = int(args.max_new_tokens)
    if args.attn_implementation is not None:
        config.models.attn_implementation = str(args.attn_implementation)
    if args.trust_remote_code is not None:
        config.models.trust_remote_code = bool(args.trust_remote_code)
    if args.enable_thinking is not None:
        config.generation.enable_thinking = bool(args.enable_thinking)
    return config


def build_case_rows(sampled_rows: list[dict], *, conditions) -> list[dict]:
    cases: list[dict] = []
    for row in sampled_rows:
        for condition in conditions:
            cases.append(
                {
                    "case_id": f"{row['prompt_id']}::{condition.name}",
                    "prompt_id": row["prompt_id"],
                    "source_index": row["source_index"],
                    "data_type": row["data_type"],
                    "prompt_field": row["prompt_field"],
                    "prompt_text": row["prompt_text"],
                    "condition_name": condition.name,
                    "condition_label": condition.label,
                    "system_prompt": condition.system_prompt,
                }
            )
    return cases


def parse_stages(raw: str) -> list[str]:
    stages = [stage.strip().lower() for stage in raw.split(",") if stage.strip()]
    valid = {"generation", "scoring"}
    invalid = [stage for stage in stages if stage not in valid]
    if invalid:
        raise ValueError(f"Unsupported stages: {', '.join(invalid)}")
    if not stages:
        raise ValueError("At least one stage must be selected.")
    return stages


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_initial_eval_config(args.config), args)
    stages = parse_stages(args.stages)
    if config.models.policy_backend == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    output_dir = config.output.directory
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "run_config.json", config.to_dict())

    response_rows: list[dict]
    target_completion_rows: list[dict]

    if "generation" in stages:
        dataset_rows = load_wildjailbreak_train_rows(config.dataset.root)
        sampled_rows, manifest = sample_rows_by_category(
            dataset_rows,
            sample_per_category=config.dataset.sample_per_category,
            seed=config.dataset.seed,
        )
        manifest.update(
            {
                "dataset_root": str(config.dataset.root),
                "num_prompt_conditions": len(PROMPT_CONDITIONS),
                "num_total_cases": len(sampled_rows) * len(PROMPT_CONDITIONS),
            }
        )
        write_json(output_dir / "sample_manifest.json", manifest)
        save_rows_csv(output_dir / "sample_manifest.csv", manifest["manifest_rows"])

        target_completion_rows = []
        for row in sampled_rows:
            target_completion_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "source_index": row["source_index"],
                    "data_type": row["data_type"],
                    "prompt_field": row["prompt_field"],
                    "prompt_text": row["prompt_text"],
                    "target_completion": str(row.get("target_completion", "")).strip(),
                }
            )
        write_jsonl(output_dir / "target_completion_inputs.jsonl", target_completion_rows)
        save_rows_csv(output_dir / "target_completion_inputs.csv", target_completion_rows)

        case_rows = build_case_rows(sampled_rows, conditions=PROMPT_CONDITIONS)

        generator = PolicyGenerator(
            model_id=config.models.policy_model_id,
            backend=config.models.policy_backend,
            batch_size=config.generation.batch_size,
            max_new_tokens=config.generation.max_new_tokens,
            enable_thinking=config.generation.enable_thinking,
            attn_implementation=config.models.attn_implementation,
            trust_remote_code=config.models.trust_remote_code,
        )
        response_rows = generator.generate_case_rows(
            case_rows,
            progress_desc="Generating policy responses",
        )
        generator.release()

        write_jsonl(output_dir / "responses_pre_reward.jsonl", response_rows)
        save_rows_csv(output_dir / "responses_pre_reward.csv", response_rows)
        write_jsonl(output_dir / "responses.jsonl", response_rows)
        save_rows_csv(output_dir / "responses.csv", response_rows)
    else:
        response_rows = read_jsonl_rows(output_dir / "responses_pre_reward.jsonl")
        target_completion_rows = read_jsonl_rows(output_dir / "target_completion_inputs.jsonl")

    if "scoring" not in stages:
        return

    reward_scorer = SkyworkRewardScorer(
        model_id=config.models.reward_model_id,
        trust_remote_code=config.models.trust_remote_code,
        attn_implementation="sdpa",
    )
    target_reward_messages = [
        [
            {"role": "user", "content": row["prompt_text"]},
            {"role": "assistant", "content": row["target_completion"]},
        ]
        for row in target_completion_rows
    ]
    target_reward_scores = reward_scorer.score_messages(
        target_reward_messages,
        batch_size=config.generation.reward_batch_size,
        progress_desc="Scoring target completions",
    )
    for row, reward_score in zip(target_completion_rows, target_reward_scores, strict=True):
        row["reward_score"] = float(reward_score)

    reward_messages = []
    for row in response_rows:
        reward_text = row["answer"] if str(row.get("answer", "")).strip() else row["raw_output"]
        row["reward_scoring_text"] = reward_text
        reward_messages.append(
            [
                {"role": "user", "content": row["prompt_text"]},
                {"role": "assistant", "content": reward_text},
            ]
        )
    reward_scores = reward_scorer.score_messages(
        reward_messages,
        batch_size=config.generation.reward_batch_size,
        progress_desc="Scoring generated responses",
    )

    for row, reward_score in zip(response_rows, reward_scores, strict=True):
        row["reward_score"] = float(reward_score)
        keyword_metrics = count_pattern_hits(str(row.get("reasoning_trace", "")), REWARD_KEYWORD_PATTERNS)
        row.update(keyword_metrics)
    reward_scorer.release()

    write_jsonl(output_dir / "target_completion_scores.jsonl", target_completion_rows)
    save_rows_csv(output_dir / "target_completion_scores.csv", target_completion_rows)

    summary_overall, summary_by_category, keyword_overall, keyword_by_category = build_summary_frames(
        response_rows,
        keywords=list(REWARD_KEYWORD_PATTERNS.keys()),
        bootstrap_samples=config.analysis.bootstrap_samples,
        bootstrap_seed=config.analysis.bootstrap_seed,
    )
    target_reward_overall, target_reward_by_category = build_target_reward_frames(
        target_completion_rows,
        bootstrap_samples=config.analysis.bootstrap_samples,
        bootstrap_seed=config.analysis.bootstrap_seed,
    )
    summary_overall.to_csv(output_dir / "summary_overall.csv", index=False)
    summary_by_category.to_csv(output_dir / "summary_by_category.csv", index=False)
    target_reward_overall.to_csv(output_dir / "target_reward_baseline_overall.csv", index=False)
    target_reward_by_category.to_csv(output_dir / "target_reward_baseline_by_category.csv", index=False)
    keyword_overall.to_csv(output_dir / "keyword_counts_overall.csv", index=False)
    keyword_by_category.to_csv(output_dir / "keyword_counts_by_category.csv", index=False)

    render_all_plots(
        summary_overall=summary_overall,
        summary_by_category=summary_by_category,
        target_reward_overall=target_reward_overall,
        target_reward_by_category=target_reward_by_category,
        keyword_overall=keyword_overall,
        keyword_by_category=keyword_by_category,
        plots_dir=plots_dir,
        top_keywords=config.analysis.top_keywords,
    )


if __name__ == "__main__":
    main()
