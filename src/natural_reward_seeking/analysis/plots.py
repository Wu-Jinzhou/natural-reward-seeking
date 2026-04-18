from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from natural_reward_seeking.data.wildjailbreak import EXPECTED_DATA_TYPES
from natural_reward_seeking.prompting.conditions import PROMPT_CONDITIONS


CONDITION_ORDER = [condition.name for condition in PROMPT_CONDITIONS]
CONDITION_LABELS = {condition.name: condition.label for condition in PROMPT_CONDITIONS}
CONDITION_COLORS = {
    "baseline": "#5B8BD0",
    "training_context": "#5DB87A",
    "training_context_objective": "#72B7B2",
    "reward_framing": "#E07070",
    "reward_framing_explicit_reasoning": "#B279A2",
}
TARGET_BASELINE_COLOR = "#6B7280"


def _save(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _category_title(category: str | None) -> str:
    return "Overall" if category is None else category.replace("_", " ").title()


def plot_keyword_mentions(keyword_frame: pd.DataFrame, *, output_stem: Path, top_n: int, category: str | None) -> None:
    frame = keyword_frame[keyword_frame["data_type"] == ("overall" if category is None else category)].copy()
    if frame.empty:
        return
    totals = frame.groupby("keyword")["mention_count"].sum().sort_values(ascending=False)
    top_keywords = list(totals.head(top_n).index)
    pivot = (
        frame[frame["keyword"].isin(top_keywords)]
        .pivot(index="keyword", columns="condition_name", values="mention_count")
        .fillna(0.0)
        .reindex(index=list(reversed(top_keywords)), columns=CONDITION_ORDER)
    )
    y_positions = range(len(pivot.index))
    width = 0.14
    offsets = [(-2 + idx) * width for idx in range(len(CONDITION_ORDER))]

    fig_height = max(4.8, len(top_keywords) * 0.38 + 1.5)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    for idx, condition_name in enumerate(CONDITION_ORDER):
        values = pivot[condition_name].tolist()
        ax.barh(
            [pos + offsets[idx] for pos in y_positions],
            values,
            height=width * 0.95,
            label=CONDITION_LABELS[condition_name],
            color=CONDITION_COLORS[condition_name],
        )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([keyword.replace("_", " ") for keyword in pivot.index])
    ax.set_xlabel("Responses mentioning keyword at least once")
    ax.set_title(f"Keyword mentions: {_category_title(category)}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    _save(fig, output_stem)


def plot_length_bars(summary_frame: pd.DataFrame, *, output_stem: Path, category: str | None) -> None:
    frame = summary_frame.copy()
    if category is not None:
        frame = frame[frame["data_type"] == category]
    frame = frame.set_index("condition_name").reindex(CONDITION_ORDER).reset_index()
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
    ax.set_xticklabels([CONDITION_LABELS[name] for name in frame["condition_name"]], rotation=20, ha="right")
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
) -> None:
    frame = summary_frame.copy()
    target_frame = target_reward_frame.copy()
    if category is not None:
        frame = frame[frame["data_type"] == category]
        target_frame = target_frame[target_frame["data_type"] == category]
    frame = frame.set_index("condition_name").reindex(CONDITION_ORDER).reset_index()
    if frame.empty:
        return
    errors = [
        (
            max(0.0, float(row["mean_reward"] - row["reward_ci_low"])) if pd.notna(row["mean_reward"]) and pd.notna(row["reward_ci_low"]) else 0.0,
            max(0.0, float(row["reward_ci_high"] - row["mean_reward"])) if pd.notna(row["mean_reward"]) and pd.notna(row["reward_ci_high"]) else 0.0,
        )
        for _, row in frame.iterrows()
    ]
    plot_labels = [CONDITION_LABELS[name] for name in frame["condition_name"]]
    plot_values = frame["mean_reward"].tolist()
    plot_colors = [CONDITION_COLORS[name] for name in frame["condition_name"]]
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


def render_all_plots(
    *,
    summary_overall: pd.DataFrame,
    summary_by_category: pd.DataFrame,
    target_reward_overall: pd.DataFrame,
    target_reward_by_category: pd.DataFrame,
    keyword_overall: pd.DataFrame,
    keyword_by_category: pd.DataFrame,
    plots_dir: Path,
    top_keywords: int,
) -> None:
    plot_keyword_mentions(keyword_overall, output_stem=plots_dir / "keyword_mentions_overall", top_n=top_keywords, category=None)
    plot_length_bars(summary_overall, output_stem=plots_dir / "lengths_overall", category=None)
    plot_reward_bars(summary_overall, target_reward_frame=target_reward_overall, output_stem=plots_dir / "reward_overall", category=None)

    for category in EXPECTED_DATA_TYPES:
        plot_keyword_mentions(
            keyword_by_category,
            output_stem=plots_dir / f"keyword_mentions_{category}",
            top_n=top_keywords,
            category=category,
        )
        plot_length_bars(
            summary_by_category,
            output_stem=plots_dir / f"lengths_{category}",
            category=category,
        )
        plot_reward_bars(
            summary_by_category,
            target_reward_frame=target_reward_by_category,
            output_stem=plots_dir / f"reward_{category}",
            category=category,
        )
