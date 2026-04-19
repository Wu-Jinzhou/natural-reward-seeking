#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from natural_reward_seeking.analysis import (
    build_summary_frames,
    build_target_reward_frames,
    count_pattern_hits,
    is_valid_response_row,
    render_all_plots,
)
from natural_reward_seeking.models import PolicyGenerator, SkyworkRewardScorer
from natural_reward_seeking.prompting import REWARD_KEYWORD_PATTERNS
from natural_reward_seeking.utils import save_rows_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone repair script: rerun unfinished responses at a higher max_new_tokens, "
            "rescore saved outputs, and regenerate summaries and plots."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="results/initial_prompt_eval",
        help="Directory containing run_config.json, responses_pre_reward.jsonl, and target_completion_scores.jsonl.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Higher generation cap for rerunning unfinished cases.",
    )
    parser.add_argument(
        "--response-source",
        default="responses_pre_reward.jsonl",
        help="Source response file inside output-dir to patch.",
    )
    parser.add_argument(
        "--scored-response-source",
        default="responses.jsonl",
        help="Scored response file inside output-dir. If present with reward_score values, only rerun rows are rescored.",
    )
    parser.add_argument(
        "--target-scores-source",
        default="target_completion_scores.jsonl",
        help="Target-completion score file inside output-dir used for reward baseline plots.",
    )
    parser.add_argument(
        "--policy-backend",
        default=None,
        help="Optional override for the policy backend. Defaults to the original run_config setting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for the policy batch size (Transformers backend only; vLLM ignores it).",
    )
    parser.add_argument(
        "--reward-batch-size",
        type=int,
        default=None,
        help="Optional override for reward scoring batch size.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not back up files before overwriting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect and report which rows would be rerun.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def backup_artifacts(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = output_dir / "repair_backups" / f"rerun_unfinished_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "responses_pre_reward.jsonl",
        "responses_pre_reward.csv",
        "responses.jsonl",
        "responses.csv",
        "summary_overall.csv",
        "summary_by_category.csv",
        "keyword_counts_overall.csv",
        "keyword_counts_by_category.csv",
        "target_reward_baseline_overall.csv",
        "target_reward_baseline_by_category.csv",
        "rerun_unfinished_report.json",
        "rerun_unfinished_rows_before.jsonl",
        "rerun_unfinished_rows_after.jsonl",
    ):
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / source.name)
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        shutil.copytree(plots_dir, backup_dir / "plots", dirs_exist_ok=True)
    return backup_dir


def has_saved_scores(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all("reward_score" in row for row in rows)


def select_unfinished_rows(rows: list[dict[str, Any]], *, original_max_new_tokens: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        generated = int(row.get("num_tokens_generated", -1) or -1)
        removed_terminal_tokens = int(row.get("removed_terminal_tokens", 0) or 0)
        if generated >= original_max_new_tokens and removed_terminal_tokens == 0:
            selected.append(row)
    return selected


def make_policy_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_rows: list[dict[str, Any]] = []
    for row in rows:
        case_rows.append(
            {
                "case_id": row["case_id"],
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "data_type": row["data_type"],
                "prompt_field": row["prompt_field"],
                "prompt_text": row["prompt_text"],
                "condition_name": row["condition_name"],
                "condition_label": row["condition_label"],
                "system_prompt": row.get("system_prompt"),
            }
        )
    return case_rows


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    run_config_path = output_dir / "run_config.json"
    responses_path = output_dir / args.response_source
    scored_responses_path = output_dir / args.scored_response_source
    target_scores_path = output_dir / args.target_scores_source
    plots_dir = output_dir / "plots"

    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at {run_config_path}")
    if not responses_path.exists():
        raise FileNotFoundError(f"Response source not found at {responses_path}")
    if not target_scores_path.exists():
        raise FileNotFoundError(f"Target score source not found at {target_scores_path}")

    run_config = read_json(run_config_path)
    responses_before = read_jsonl_rows(responses_path)
    scored_rows_before = read_jsonl_rows(scored_responses_path) if scored_responses_path.exists() else []
    target_completion_rows = read_jsonl_rows(target_scores_path)

    original_max_new_tokens = int(run_config["generation"]["max_new_tokens"])
    rerun_rows_before = select_unfinished_rows(
        responses_before,
        original_max_new_tokens=original_max_new_tokens,
    )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir),
        "original_response_source": str(responses_path),
        "original_scored_response_source": str(scored_responses_path),
        "target_score_source": str(target_scores_path),
        "original_max_new_tokens": original_max_new_tokens,
        "rerun_max_new_tokens": int(args.max_new_tokens),
        "num_total_rows": len(responses_before),
        "num_selected_rows": len(rerun_rows_before),
        "selected_case_ids": [row["case_id"] for row in rerun_rows_before],
    }

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    backup_dir: str | None = None
    if not args.no_backup:
        backup_dir = str(backup_artifacts(output_dir))
        report["backup_directory"] = backup_dir

    if not rerun_rows_before:
        report["status"] = "no_unfinished_rows_found"
        write_json(output_dir / "rerun_unfinished_report.json", report)
        print("No unfinished rows found. Nothing to rerun.")
        return

    if (args.policy_backend or run_config["models"]["policy_backend"]) == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    policy_backend = str(args.policy_backend or run_config["models"]["policy_backend"])
    batch_size = int(args.batch_size or run_config["generation"]["batch_size"])
    reward_batch_size = int(args.reward_batch_size or run_config["generation"]["reward_batch_size"])

    generator = PolicyGenerator(
        model_id=str(run_config["models"]["policy_model_id"]),
        backend=policy_backend,
        batch_size=batch_size,
        max_new_tokens=int(args.max_new_tokens),
        enable_thinking=bool(run_config["generation"]["enable_thinking"]),
        attn_implementation=str(run_config["models"]["attn_implementation"]),
        trust_remote_code=bool(run_config["models"]["trust_remote_code"]),
    )
    rerun_case_rows = make_policy_case_rows(rerun_rows_before)
    rerun_rows_after = generator.generate_case_rows(
        rerun_case_rows,
        progress_desc="Rerunning unfinished responses",
    )
    generator.release()

    rerun_by_case_id = {row["case_id"]: row for row in rerun_rows_after}
    updated_pre_reward_rows: list[dict[str, Any]] = []
    replaced_count = 0
    for row in responses_before:
        replacement = rerun_by_case_id.get(row["case_id"])
        if replacement is None:
            updated_pre_reward_rows.append(row)
            continue
        updated_pre_reward_rows.append(replacement)
        replaced_count += 1

    write_jsonl(output_dir / "rerun_unfinished_rows_before.jsonl", rerun_rows_before)
    write_jsonl(output_dir / "rerun_unfinished_rows_after.jsonl", rerun_rows_after)
    write_jsonl(output_dir / "responses_pre_reward.jsonl", updated_pre_reward_rows)
    save_rows_csv(output_dir / "responses_pre_reward.csv", updated_pre_reward_rows)

    can_patch_existing_scores = has_saved_scores(scored_rows_before) and len(scored_rows_before) == len(updated_pre_reward_rows)
    if can_patch_existing_scores:
        scored_rows = deepcopy(scored_rows_before)
        scored_by_case_id = {row["case_id"]: row for row in scored_rows}
        scoring_target_rows: list[dict[str, Any]] = []
        reward_messages: list[list[dict[str, str]]] = []
        for rerun_row in rerun_rows_after:
            merged = dict(scored_by_case_id[rerun_row["case_id"]])
            merged.update(rerun_row)
            reward_text = merged["answer"] if str(merged.get("answer", "")).strip() else merged["raw_output"]
            merged["reward_scoring_text"] = reward_text
            scoring_target_rows.append(merged)
            reward_messages.append(
                [
                    {"role": "user", "content": merged["prompt_text"]},
                    {"role": "assistant", "content": reward_text},
                ]
            )

        reward_scorer = SkyworkRewardScorer(
            model_id=str(run_config["models"]["reward_model_id"]),
            trust_remote_code=bool(run_config["models"]["trust_remote_code"]),
            attn_implementation="sdpa",
        )
        reward_scores = reward_scorer.score_messages(
            reward_messages,
            batch_size=reward_batch_size,
            progress_desc="Scoring repaired rows",
        )
        reward_scorer.release()

        rescored_by_case_id: dict[str, dict[str, Any]] = {}
        for row, reward_score in zip(scoring_target_rows, reward_scores, strict=True):
            row["reward_score"] = float(reward_score)
            row.update(count_pattern_hits(str(row.get("reasoning_trace", "")), REWARD_KEYWORD_PATTERNS))
            rescored_by_case_id[row["case_id"]] = row

        patched_scored_rows: list[dict[str, Any]] = []
        for row in scored_rows:
            patched_scored_rows.append(rescored_by_case_id.get(row["case_id"], row))
        scored_rows = patched_scored_rows
        scoring_mode = "partial_rescore"
        num_rescored_rows = len(scoring_target_rows)
    else:
        scored_rows = deepcopy(updated_pre_reward_rows)
        reward_messages = []
        for row in scored_rows:
            reward_text = row["answer"] if str(row.get("answer", "")).strip() else row["raw_output"]
            row["reward_scoring_text"] = reward_text
            reward_messages.append(
                [
                    {"role": "user", "content": row["prompt_text"]},
                    {"role": "assistant", "content": reward_text},
                ]
            )

        reward_scorer = SkyworkRewardScorer(
            model_id=str(run_config["models"]["reward_model_id"]),
            trust_remote_code=bool(run_config["models"]["trust_remote_code"]),
            attn_implementation="sdpa",
        )
        reward_scores = reward_scorer.score_messages(
            reward_messages,
            batch_size=reward_batch_size,
            progress_desc="Scoring repaired response table",
        )
        reward_scorer.release()

        for row, reward_score in zip(scored_rows, reward_scores, strict=True):
            row["reward_score"] = float(reward_score)
            row.update(count_pattern_hits(str(row.get("reasoning_trace", "")), REWARD_KEYWORD_PATTERNS))
        scoring_mode = "full_rescore"
        num_rescored_rows = len(scored_rows)

    write_jsonl(output_dir / "responses.jsonl", scored_rows)
    save_rows_csv(output_dir / "responses.csv", scored_rows)

    valid_scored_rows = [
        row
        for row in scored_rows
        if is_valid_response_row(row, max_new_tokens=int(args.max_new_tokens))
    ]
    summary_overall, summary_by_category, keyword_overall, keyword_by_category = build_summary_frames(
        valid_scored_rows,
        keywords=list(REWARD_KEYWORD_PATTERNS.keys()),
        bootstrap_samples=int(run_config["analysis"]["bootstrap_samples"]),
        bootstrap_seed=int(run_config["analysis"]["bootstrap_seed"]),
    )
    target_reward_overall, target_reward_by_category = build_target_reward_frames(
        target_completion_rows,
        bootstrap_samples=int(run_config["analysis"]["bootstrap_samples"]),
        bootstrap_seed=int(run_config["analysis"]["bootstrap_seed"]),
    )

    summary_overall.to_csv(output_dir / "summary_overall.csv", index=False)
    summary_by_category.to_csv(output_dir / "summary_by_category.csv", index=False)
    keyword_overall.to_csv(output_dir / "keyword_counts_overall.csv", index=False)
    keyword_by_category.to_csv(output_dir / "keyword_counts_by_category.csv", index=False)
    target_reward_overall.to_csv(output_dir / "target_reward_baseline_overall.csv", index=False)
    target_reward_by_category.to_csv(output_dir / "target_reward_baseline_by_category.csv", index=False)

    render_all_plots(
        summary_overall=summary_overall,
        summary_by_category=summary_by_category,
        target_reward_overall=target_reward_overall,
        target_reward_by_category=target_reward_by_category,
        keyword_overall=keyword_overall,
        keyword_by_category=keyword_by_category,
        response_rows=valid_scored_rows,
        plots_dir=plots_dir,
        top_keywords=int(run_config["analysis"]["top_keywords"]),
    )

    report.update(
        {
            "status": "completed",
            "policy_backend": policy_backend,
            "policy_batch_size": batch_size,
            "reward_batch_size": reward_batch_size,
            "num_replaced_rows": replaced_count,
            "num_rescored_rows": num_rescored_rows,
            "scoring_mode": scoring_mode,
            "num_still_capped_after_rerun": sum(
                1
                for row in rerun_rows_after
                if int(row.get("num_tokens_generated", -1) or -1) >= int(args.max_new_tokens)
                and int(row.get("removed_terminal_tokens", 0) or 0) == 0
            ),
        }
    )
    write_json(output_dir / "rerun_unfinished_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
