from __future__ import annotations

import random
from typing import Any

import pandas as pd


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean_value = float(sum(values) / len(values))
    if len(values) == 1:
        return mean_value, mean_value, mean_value
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(draw) / len(draw))
    means.sort()
    low_idx = max(0, int(0.025 * len(means)) - 1)
    high_idx = min(len(means) - 1, int(0.975 * len(means)))
    return mean_value, float(means[low_idx]), float(means[high_idx])


def _group_summary(group: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    rewards = [float(value) for value in group["reward_score"].dropna().tolist()]
    reasoning = [float(value) for value in group["reasoning_char_count"].dropna().tolist()] if "reasoning_char_count" in group.columns else []
    answers = [float(value) for value in group["answer_char_count"].dropna().tolist()] if "answer_char_count" in group.columns else []
    reward_mean, reward_ci_low, reward_ci_high = bootstrap_mean_ci(rewards, samples=bootstrap_samples, seed=bootstrap_seed)
    reasoning_mean, reasoning_ci_low, reasoning_ci_high = bootstrap_mean_ci(reasoning, samples=bootstrap_samples, seed=bootstrap_seed + 1)
    answer_mean, answer_ci_low, answer_ci_high = bootstrap_mean_ci(answers, samples=bootstrap_samples, seed=bootstrap_seed + 2)
    parse_success_rate = (
        float(group["raw_output"].fillna("").map(lambda value: bool(str(value).strip())).mean())
        if "raw_output" in group.columns
        else None
    )
    reasoning_trace_rate = (
        float(group["has_reasoning_trace"].astype(int).mean())
        if "has_reasoning_trace" in group.columns
        else None
    )
    keyword_hit_rate = (
        float(group["has_keyword_hit"].astype(int).mean())
        if "has_keyword_hit" in group.columns
        else None
    )
    mean_keyword_count = float(group["keyword_count"].mean()) if "keyword_count" in group.columns else None
    return {
        "n": int(len(group)),
        "mean_reward": reward_mean,
        "reward_ci_low": reward_ci_low,
        "reward_ci_high": reward_ci_high,
        "mean_reasoning_chars": reasoning_mean,
        "reasoning_ci_low": reasoning_ci_low,
        "reasoning_ci_high": reasoning_ci_high,
        "mean_answer_chars": answer_mean,
        "answer_ci_low": answer_ci_low,
        "answer_ci_high": answer_ci_high,
        "parse_success_rate": parse_success_rate,
        "reasoning_trace_rate": reasoning_trace_rate,
        "keyword_hit_rate": keyword_hit_rate,
        "mean_keyword_count": mean_keyword_count,
    }


def _keyword_summary(group: pd.DataFrame, *, condition_name: str, condition_label: str, category: str | None, keywords: list[str]) -> list[dict[str, Any]]:
    rows = []
    for keyword in keywords:
        mention_count = int(group["keyword_hits"].map(lambda hits: int(keyword in hits)).sum())
        total_hits = int(group["keyword_hits"].map(lambda hits: int(hits.get(keyword, 0))).sum())
        rows.append(
            {
                "data_type": category or "overall",
                "condition_name": condition_name,
                "condition_label": condition_label,
                "keyword": keyword,
                "mention_count": mention_count,
                "mention_rate": mention_count / len(group) if len(group) else 0.0,
                "total_hits": total_hits,
            }
        )
    return rows


def build_summary_frames(
    response_rows: list[dict[str, Any]],
    *,
    keywords: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(response_rows)
    overall_rows: list[dict[str, Any]] = []
    by_category_rows: list[dict[str, Any]] = []
    keyword_overall_rows: list[dict[str, Any]] = []
    keyword_category_rows: list[dict[str, Any]] = []

    for (condition_name, condition_label), group in frame.groupby(["condition_name", "condition_label"], sort=False):
        summary = _group_summary(group, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        overall_rows.append(
            {
                "condition_name": condition_name,
                "condition_label": condition_label,
                **summary,
            }
        )
        keyword_overall_rows.extend(
            _keyword_summary(
                group,
                condition_name=condition_name,
                condition_label=condition_label,
                category=None,
                keywords=keywords,
            )
        )

    for (data_type, condition_name, condition_label), group in frame.groupby(["data_type", "condition_name", "condition_label"], sort=False):
        summary = _group_summary(group, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        by_category_rows.append(
            {
                "data_type": data_type,
                "condition_name": condition_name,
                "condition_label": condition_label,
                **summary,
            }
        )
        keyword_category_rows.extend(
            _keyword_summary(
                group,
                condition_name=condition_name,
                condition_label=condition_label,
                category=data_type,
                keywords=keywords,
            )
        )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(by_category_rows),
        pd.DataFrame(keyword_overall_rows),
        pd.DataFrame(keyword_category_rows),
    )


def build_target_reward_frames(
    target_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(target_rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    overall_summary = _group_summary(frame, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    overall_frame = pd.DataFrame(
        [
            {
                "condition_name": "target_completion",
                "condition_label": "Target completion",
                **overall_summary,
            }
        ]
    )

    category_rows: list[dict[str, Any]] = []
    for data_type, group in frame.groupby("data_type", sort=False):
        summary = _group_summary(group, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        category_rows.append(
            {
                "data_type": data_type,
                "condition_name": "target_completion",
                "condition_label": "Target completion",
                **summary,
            }
        )
    return overall_frame, pd.DataFrame(category_rows)
