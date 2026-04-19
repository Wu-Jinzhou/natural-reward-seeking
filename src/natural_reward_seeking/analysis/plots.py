from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natural_reward_seeking.data.wildjailbreak import EXPECTED_DATA_TYPES
from natural_reward_seeking.prompting.conditions import PROMPT_CONDITIONS


DEFAULT_CONDITION_ORDER = [condition.name for condition in PROMPT_CONDITIONS]
DEFAULT_CONDITION_LABELS = {condition.name: condition.label for condition in PROMPT_CONDITIONS}
DEFAULT_CONDITION_COLORS = {
    "baseline": "#5B8BD0",
    "training_context": "#5DB87A",
    "training_context_objective": "#72B7B2",
    "reward_framing": "#E07070",
    "reward_framing_explicit_reasoning": "#B279A2",
}
TARGET_BASELINE_COLOR = "#6B7280"
FALLBACK_CONDITION_COLORS = [
    "#5B8BD0",
    "#5DB87A",
    "#72B7B2",
    "#E07070",
    "#B279A2",
    "#A0A44E",
    "#F58518",
    "#4C78A8",
]


def _save(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _category_title(category: str | None) -> str:
    return "Overall" if category is None else category.replace("_", " ").title()


def _infer_condition_order(*frames: pd.DataFrame) -> list[str]:
    observed: list[str] = []
    for frame in frames:
        if frame.empty or "condition_name" not in frame.columns:
            continue
        for value in frame["condition_name"].dropna().tolist():
            name = str(value)
            if name not in observed:
                observed.append(name)
    preferred = [name for name in DEFAULT_CONDITION_ORDER if name in observed]
    remainder = [name for name in observed if name not in preferred]
    return preferred + remainder


def _frame_condition_labels(*frames: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for frame in frames:
        if frame.empty or "condition_name" not in frame.columns or "condition_label" not in frame.columns:
            continue
        for _, row in frame[["condition_name", "condition_label"]].dropna().drop_duplicates().iterrows():
            labels[str(row["condition_name"])] = str(row["condition_label"])
    return labels


def _resolve_condition_context(
    *,
    frames: list[pd.DataFrame],
    condition_order: list[str] | None,
    condition_labels: dict[str, str] | None,
    condition_colors: dict[str, str] | None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    inferred_order = _infer_condition_order(*frames)
    if condition_order is None:
        resolved_order = inferred_order
    else:
        resolved_order = [name for name in condition_order if name in inferred_order]
        resolved_order.extend(name for name in inferred_order if name not in resolved_order)

    resolved_labels = dict(DEFAULT_CONDITION_LABELS)
    resolved_labels.update(_frame_condition_labels(*frames))
    if condition_labels is not None:
        resolved_labels.update(condition_labels)
    for name in resolved_order:
        resolved_labels.setdefault(name, name.replace("_", " ").title())

    resolved_colors = dict(DEFAULT_CONDITION_COLORS)
    if condition_colors is not None:
        resolved_colors.update(condition_colors)
    fallback_idx = 0
    for name in resolved_order:
        if name not in resolved_colors:
            resolved_colors[name] = FALLBACK_CONDITION_COLORS[fallback_idx % len(FALLBACK_CONDITION_COLORS)]
            fallback_idx += 1

    return resolved_order, resolved_labels, resolved_colors


def plot_keyword_mentions(
    keyword_frame: pd.DataFrame,
    *,
    output_stem: Path,
    top_n: int,
    category: str | None,
    condition_order: list[str],
    condition_labels: dict[str, str],
    condition_colors: dict[str, str],
) -> None:
    frame = keyword_frame[keyword_frame["data_type"] == ("overall" if category is None else category)].copy()
    if frame.empty:
        return
    totals = frame.groupby("keyword")["mention_count"].sum().sort_values(ascending=False)
    top_keywords = list(totals.head(top_n).index)
    active_order = [name for name in condition_order if name in set(frame["condition_name"].tolist())]
    if not active_order:
        return
    pivot = (
        frame[frame["keyword"].isin(top_keywords)]
        .pivot(index="keyword", columns="condition_name", values="mention_count")
        .fillna(0.0)
        .reindex(index=list(reversed(top_keywords)), columns=active_order)
    )
    y_positions = range(len(pivot.index))
    width = min(0.22, 0.72 / len(active_order))
    center_offset = (len(active_order) - 1) / 2
    offsets = [(idx - center_offset) * width for idx in range(len(active_order))]

    fig_height = max(4.8, len(top_keywords) * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    for idx, condition_name in enumerate(active_order):
        values = pivot[condition_name].tolist()
        ax.barh(
            [pos + offsets[idx] for pos in y_positions],
            values,
            height=width * 0.95,
            label=condition_labels[condition_name],
            color=condition_colors[condition_name],
        )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([keyword.replace("_", " ") for keyword in pivot.index])
    ax.set_xlabel("Responses mentioning keyword at least once")
    ax.set_title(f"Keyword mentions: {_category_title(category)}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, output_stem)


def plot_length_bars(
    summary_frame: pd.DataFrame,
    *,
    output_stem: Path,
    category: str | None,
    condition_order: list[str],
    condition_labels: dict[str, str],
) -> None:
    frame = summary_frame.copy()
    if category is not None:
        frame = frame[frame["data_type"] == category]
    active_order = [name for name in condition_order if name in set(frame["condition_name"].tolist())]
    frame = frame.set_index("condition_name").reindex(active_order).reset_index()
    if frame.empty:
        return

    x = list(range(len(frame)))
    width = 0.36
    reasoning_err = [
        (
            max(0.0, float(row["mean_reasoning_chars"] - row["reasoning_ci_low"])) if pd.notna(row["mean_reasoning_chars"]) and pd.notna(row["reasoning_ci_low"]) else 0.0,
            max(0.0, float(row["reasoning_ci_high"] - row["mean_reasoning_chars"])) if pd.notna(row["mean_reasoning_chars"]) and pd.notna(row["reasoning_ci_high"]) else 0.0,
        )
        for _, row in frame.iterrows()
    ]
    answer_err = [
        (
            max(0.0, float(row["mean_answer_chars"] - row["answer_ci_low"])) if pd.notna(row["mean_answer_chars"]) and pd.notna(row["answer_ci_low"]) else 0.0,
            max(0.0, float(row["answer_ci_high"] - row["mean_answer_chars"])) if pd.notna(row["mean_answer_chars"]) and pd.notna(row["answer_ci_high"]) else 0.0,
        )
        for _, row in frame.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.bar(
        [value - width / 2 for value in x],
        frame["mean_reasoning_chars"],
        width=width,
        label="Reasoning chars",
        color="#5B8BD0",
        yerr=list(zip(*reasoning_err)),
        capsize=3,
    )
    ax.bar(
        [value + width / 2 for value in x],
        frame["mean_answer_chars"],
        width=width,
        label="Answer chars",
        color="#E07070",
        yerr=list(zip(*answer_err)),
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([condition_labels[name] for name in frame["condition_name"]], rotation=20, ha="right")
    ax.set_ylabel("Characters")
    ax.set_title(f"Reasoning and answer length: {_category_title(category)}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_stem)


def plot_reward_bars(
    summary_frame: pd.DataFrame,
    *,
    target_reward_frame: pd.DataFrame,
    output_stem: Path,
    category: str | None,
    condition_order: list[str],
    condition_labels: dict[str, str],
    condition_colors: dict[str, str],
) -> None:
    frame = summary_frame.copy()
    target_frame = target_reward_frame.copy()
    if category is not None:
        frame = frame[frame["data_type"] == category]
        target_frame = target_frame[target_frame["data_type"] == category]
    active_order = [name for name in condition_order if name in set(frame["condition_name"].tolist())]
    frame = frame.set_index("condition_name").reindex(active_order).reset_index()
    if frame.empty:
        return
    errors = [
        (
            max(0.0, float(row["mean_reward"] - row["reward_ci_low"])) if pd.notna(row["mean_reward"]) and pd.notna(row["reward_ci_low"]) else 0.0,
            max(0.0, float(row["reward_ci_high"] - row["mean_reward"])) if pd.notna(row["mean_reward"]) and pd.notna(row["reward_ci_high"]) else 0.0,
        )
        for _, row in frame.iterrows()
    ]
    plot_labels = [condition_labels[name] for name in frame["condition_name"]]
    plot_values = frame["mean_reward"].tolist()
    plot_colors = [condition_colors[name] for name in frame["condition_name"]]
    plot_errors = errors
    if not target_frame.empty:
        target_row = target_frame.iloc[0]
        plot_labels.append("Target completion")
        plot_values.append(target_row["mean_reward"])
        plot_colors.append(TARGET_BASELINE_COLOR)
        plot_errors.append(
            (
                max(0.0, float(target_row["mean_reward"] - target_row["reward_ci_low"])) if pd.notna(target_row["mean_reward"]) and pd.notna(target_row["reward_ci_low"]) else 0.0,
                max(0.0, float(target_row["reward_ci_high"] - target_row["mean_reward"])) if pd.notna(target_row["mean_reward"]) and pd.notna(target_row["reward_ci_high"]) else 0.0,
            )
        )

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.bar(
        range(len(plot_values)),
        plot_values,
        color=plot_colors,
        yerr=list(zip(*plot_errors)),
        capsize=3,
    )
    ax.set_xticks(range(len(plot_values)))
    ax.set_xticklabels(plot_labels, rotation=20, ha="right")
    ax.set_ylabel("Mean reward score")
    ax.set_title(f"Reward score: {_category_title(category)}")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_stem)


def plot_reward_distribution(
    response_rows: list[dict[str, object]],
    *,
    output_stem: Path,
    category: str | None,
    bins: int = 40,
    condition_order: list[str],
    condition_labels: dict[str, str],
    condition_colors: dict[str, str],
    baseline_condition_name: str,
) -> None:
    frame = pd.DataFrame(response_rows)
    if category is not None:
        frame = frame[frame["data_type"] == category]
    if frame.empty or "reward_score" not in frame.columns:
        return

    frame = frame[frame["reward_score"].notna()].copy()
    if frame.empty:
        return

    reward_min = float(frame["reward_score"].min())
    reward_max = float(frame["reward_score"].max())
    if reward_min == reward_max:
        reward_min -= 0.5
        reward_max += 0.5
    bin_edges = np.linspace(reward_min, reward_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    baseline_frame = frame[frame["condition_name"] == baseline_condition_name]
    conditioned_histograms: list[np.ndarray] = []
    conditioned_count = 0
    conditioned_rows = 0

    if baseline_frame.empty:
        plt.close(fig)
        return

    baseline_values = baseline_frame["reward_score"].astype(float).to_numpy()
    baseline_weights = np.ones_like(baseline_values, dtype=float) / len(baseline_values)
    baseline_hist, _ = np.histogram(baseline_values, bins=bin_edges, weights=baseline_weights)
    ax.step(
        bin_centers,
        baseline_hist,
        where="mid",
        linewidth=2.2,
        color=condition_colors[baseline_condition_name],
        label=f"{condition_labels[baseline_condition_name]} (n={len(baseline_values)})",
    )

    for condition_name in condition_order:
        if condition_name == baseline_condition_name:
            continue
        condition_frame = frame[frame["condition_name"] == condition_name]
        if condition_frame.empty:
            continue
        values = condition_frame["reward_score"].astype(float).to_numpy()
        weights = np.ones_like(values, dtype=float) / len(values)
        condition_hist, _ = np.histogram(values, bins=bin_edges, weights=weights)
        conditioned_histograms.append(condition_hist)
        conditioned_count += 1
        conditioned_rows += len(values)

    if conditioned_histograms:
        conditioned_mean_hist = np.mean(np.vstack(conditioned_histograms), axis=0)
        ax.step(
            bin_centers,
            conditioned_mean_hist,
            where="mid",
            linewidth=2.2,
            color="#E07070",
            label=f"Average non-baseline condition (n={conditioned_rows}, k={conditioned_count})",
        )

    ax.set_xlabel("Reward score")
    ax.set_ylabel("Share of valid responses")
    ax.set_title(f"Reward distribution: baseline vs conditioned average ({_category_title(category)})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_stem)


def render_all_plots(
    *,
    summary_overall: pd.DataFrame,
    summary_by_category: pd.DataFrame,
    target_reward_overall: pd.DataFrame,
    target_reward_by_category: pd.DataFrame,
    keyword_overall: pd.DataFrame,
    keyword_by_category: pd.DataFrame,
    response_rows: list[dict[str, object]],
    plots_dir: Path,
    top_keywords: int,
    condition_order: list[str] | None = None,
    condition_labels: dict[str, str] | None = None,
    condition_colors: dict[str, str] | None = None,
    baseline_condition_name: str = "baseline",
) -> None:
    response_frame = pd.DataFrame(response_rows)
    resolved_order, resolved_labels, resolved_colors = _resolve_condition_context(
        frames=[summary_overall, summary_by_category, keyword_overall, keyword_by_category, response_frame],
        condition_order=condition_order,
        condition_labels=condition_labels,
        condition_colors=condition_colors,
    )
    plot_keyword_mentions(
        keyword_overall,
        output_stem=plots_dir / "keyword_mentions_overall",
        top_n=top_keywords,
        category=None,
        condition_order=resolved_order,
        condition_labels=resolved_labels,
        condition_colors=resolved_colors,
    )
    plot_length_bars(
        summary_overall,
        output_stem=plots_dir / "lengths_overall",
        category=None,
        condition_order=resolved_order,
        condition_labels=resolved_labels,
    )
    plot_reward_bars(
        summary_overall,
        target_reward_frame=target_reward_overall,
        output_stem=plots_dir / "reward_overall",
        category=None,
        condition_order=resolved_order,
        condition_labels=resolved_labels,
        condition_colors=resolved_colors,
    )
    plot_reward_distribution(
        response_rows,
        output_stem=plots_dir / "reward_distribution_overall",
        category=None,
        condition_order=resolved_order,
        condition_labels=resolved_labels,
        condition_colors=resolved_colors,
        baseline_condition_name=baseline_condition_name,
    )

    for category in EXPECTED_DATA_TYPES:
        plot_keyword_mentions(
            keyword_by_category,
            output_stem=plots_dir / f"keyword_mentions_{category}",
            top_n=top_keywords,
            category=category,
            condition_order=resolved_order,
            condition_labels=resolved_labels,
            condition_colors=resolved_colors,
        )
        plot_length_bars(
            summary_by_category,
            output_stem=plots_dir / f"lengths_{category}",
            category=category,
            condition_order=resolved_order,
            condition_labels=resolved_labels,
        )
        plot_reward_bars(
            summary_by_category,
            target_reward_frame=target_reward_by_category,
            output_stem=plots_dir / f"reward_{category}",
            category=category,
            condition_order=resolved_order,
            condition_labels=resolved_labels,
            condition_colors=resolved_colors,
        )
        plot_reward_distribution(
            response_rows,
            output_stem=plots_dir / f"reward_distribution_{category}",
            category=category,
            condition_order=resolved_order,
            condition_labels=resolved_labels,
            condition_colors=resolved_colors,
            baseline_condition_name=baseline_condition_name,
        )
