#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from natural_reward_seeking.models import PolicyGenerator, SkyworkRewardScorer
from natural_reward_seeking.rl.config import load_rl_poc_config
from natural_reward_seeking.rl.dataset import load_manifest_rows
from natural_reward_seeking.rl.evaluation import (
    add_keyword_metrics,
    discover_checkpoint_specs,
    summarize_checkpoint_run,
    write_checkpoint_plots,
)
from natural_reward_seeking.utils import save_rows_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved RL checkpoints with user-only prompting.")
    parser.add_argument("--config", default="configs/rl_dapo_poc.toml")
    parser.add_argument("--run-dir", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _default_run_dirs(config) -> list[Path]:
    return [config.output.directory / "training" / condition_name for condition_name in config.training.condition_names]


def _build_case_rows(eval_rows: list[dict], *, checkpoint_name: str, checkpoint_label: str) -> list[dict]:
    rows: list[dict] = []
    for row in eval_rows:
        rows.append(
            {
                "case_id": f"{row['prompt_id']}::{checkpoint_name}",
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "data_type": row["data_type"],
                "prompt_field": row["prompt_field"],
                "prompt_text": row["prompt_text"],
                "condition_name": checkpoint_name,
                "condition_label": checkpoint_label,
                "system_prompt": None,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_rl_poc_config(args.config)
    run_dirs = [Path(path).resolve() for path in args.run_dir] if args.run_dir else _default_run_dirs(config)
    output_root = Path(args.output_dir).resolve() if args.output_dir else config.output.directory / "checkpoint_eval"
    output_root.mkdir(parents=True, exist_ok=True)

    eval_rows = load_manifest_rows(config.dataset.eval_manifest)
    write_json(output_root / "eval_config.json", config.to_dict())

    final_summary_rows: list[dict] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Skipping missing run dir: {run_dir}")
            continue
        run_name = run_dir.name
        checkpoint_specs = discover_checkpoint_specs(run_dir)
        if not checkpoint_specs:
            print(f"Skipping {run_dir}: no checkpoints found.")
            continue

        run_output_dir = output_root / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        all_response_rows: list[dict] = []

        for checkpoint in checkpoint_specs:
            checkpoint_output_dir = run_output_dir / checkpoint.checkpoint_name
            checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
            generator = PolicyGenerator(
                model_id=str(checkpoint.hf_model_dir),
                backend=config.evaluation.policy_backend,
                batch_size=config.evaluation.batch_size,
                max_new_tokens=config.evaluation.max_new_tokens,
                enable_thinking=config.evaluation.enable_thinking,
                attn_implementation="sdpa",
                trust_remote_code=config.models.trust_remote_code,
            )
            case_rows = _build_case_rows(
                eval_rows,
                checkpoint_name=checkpoint.checkpoint_name,
                checkpoint_label=checkpoint.checkpoint_label,
            )
            response_rows = generator.generate_case_rows(
                case_rows,
                progress_desc=f"Generating {run_name} {checkpoint.checkpoint_label}",
            )
            generator.release()

            reward_scorer = SkyworkRewardScorer(
                model_id=config.models.reward_model_id,
                trust_remote_code=config.models.trust_remote_code,
                attn_implementation="sdpa",
            )
            reward_messages = []
            for row in response_rows:
                reward_text = row["answer"] if str(row.get("answer", "")).strip() else row["raw_output"]
                row["reward_scoring_text"] = reward_text
                row["run_name"] = run_name
                row["checkpoint_step"] = checkpoint.step
                reward_messages.append(
                    [
                        {"role": "user", "content": row["prompt_text"]},
                        {"role": "assistant", "content": reward_text},
                    ]
                )
            reward_scores = reward_scorer.score_messages(
                reward_messages,
                batch_size=config.evaluation.reward_batch_size,
                progress_desc=f"Scoring {run_name} {checkpoint.checkpoint_label}",
            )
            reward_scorer.release()
            for row, reward_score in zip(response_rows, reward_scores, strict=True):
                row["reward_score"] = float(reward_score)
            add_keyword_metrics(response_rows)

            write_jsonl(checkpoint_output_dir / "responses.jsonl", response_rows)
            save_rows_csv(checkpoint_output_dir / "responses.csv", response_rows)
            all_response_rows.extend(response_rows)

        summary_overall, summary_by_category, keyword_overall, keyword_by_category = summarize_checkpoint_run(
            all_response_rows,
            max_new_tokens=config.evaluation.max_new_tokens,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            bootstrap_seed=config.evaluation.bootstrap_seed,
        )
        summary_overall["run_name"] = run_name
        summary_overall["checkpoint_step"] = summary_overall["condition_name"].map(
            lambda value: int(str(value).split("_")[-1])
        )
        summary_by_category["run_name"] = run_name
        summary_by_category["checkpoint_step"] = summary_by_category["condition_name"].map(
            lambda value: int(str(value).split("_")[-1])
        )
        summary_overall.to_csv(run_output_dir / "summary_overall.csv", index=False)
        summary_by_category.to_csv(run_output_dir / "summary_by_category.csv", index=False)
        keyword_overall.to_csv(run_output_dir / "keyword_counts_overall.csv", index=False)
        keyword_by_category.to_csv(run_output_dir / "keyword_counts_by_category.csv", index=False)
        write_checkpoint_plots(
            summary_overall=summary_overall,
            summary_by_category=summary_by_category,
            output_dir=run_output_dir / "plots",
        )

        if not summary_overall.empty:
            final_row = summary_overall.sort_values("checkpoint_step").iloc[-1].to_dict()
            final_summary_rows.append(final_row)

    if final_summary_rows:
        frame = pd.DataFrame(final_summary_rows)
        frame.to_csv(output_root / "final_checkpoint_summary_overall.csv", index=False)


if __name__ == "__main__":
    main()
