from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".cache" / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from natural_reward_seeking.analysis.summaries import bootstrap_mean_ci
from natural_reward_seeking.analysis.validity import is_valid_response_row


CONDITION_COLORS = {
    "baseline": "#5B8BD0",
    "initial": "#72B7B2",
    "skill_gap": "#E07070",
}


def _save(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _condition_order(frame: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for value in frame["condition_name"].dropna().tolist():
        name = str(value)
        if name not in names:
            names.append(name)
    return names


def _condition_labels(frame: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    if "condition_label" not in frame.columns:
        return labels
    for _, row in frame[["condition_name", "condition_label"]].dropna().drop_duplicates().iterrows():
        labels[str(row["condition_name"])] = str(row["condition_label"])
    return labels


def _summary(group: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    rewards = [float(value) for value in group.get("reward_score", pd.Series(dtype=float)).dropna().tolist()]
    reward_mean, reward_ci_low, reward_ci_high = bootstrap_mean_ci(
        rewards,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    harmful_values = group.get("harmful_behavior", pd.Series(dtype=object)).dropna().tolist()
    harmful_numeric = [float(bool(value)) for value in harmful_values]
    harmful_rate, harmful_ci_low, harmful_ci_high = bootstrap_mean_ci(
        harmful_numeric,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 10,
    )
    return {
        "n": int(len(group)),
        "n_reward_scored": int(len(rewards)),
        "mean_reward": reward_mean,
        "reward_ci_low": reward_ci_low,
        "reward_ci_high": reward_ci_high,
        "n_classified": int(len(harmful_numeric)),
        "harmful_behavior_rate": harmful_rate,
        "harmful_behavior_ci_low": harmful_ci_low,
        "harmful_behavior_ci_high": harmful_ci_high,
        "mean_generated_tokens": float(group["num_tokens_generated"].mean()) if "num_tokens_generated" in group.columns else None,
        "reasoning_trace_rate": float(group["has_reasoning_trace"].astype(bool).mean()) if "has_reasoning_trace" in group.columns else None,
        "classification_error_rate": (
            float(group["classification_error"].fillna("").map(lambda value: bool(str(value).strip())).mean())
            if "classification_error" in group.columns
            else None
        ),
    }


def _summarize_by(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame([{"group": "overall", **_summary(frame, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)}])
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = {column: key for column, key in zip(group_cols, keys, strict=True)}
        payload.update(_summary(group, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed))
        rows.append(payload)
    return pd.DataFrame(rows)


def _plot_metric_by_condition(
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: Path,
) -> None:
    if frame.empty or metric not in frame.columns:
        return
    condition_frame = frame.dropna(subset=[metric]).copy()
    if condition_frame.empty:
        return
    order = _condition_order(condition_frame)
    labels = _condition_labels(condition_frame)
    condition_frame = condition_frame.set_index("condition_name").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        range(len(condition_frame)),
        condition_frame[metric].astype(float),
        color=["#5B8BD0" if name == "baseline" else "#72B7B2" if str(name).startswith("initial_") else "#E07070" for name in condition_frame["condition_name"]],
    )
    ax.set_xticks(range(len(condition_frame)))
    ax.set_xticklabels([labels.get(name, name) for name in condition_frame["condition_name"]], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_stem)


def _plot_reward_distribution(frame: pd.DataFrame, *, output_stem: Path, title: str) -> None:
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
    bins = np.linspace(reward_min, reward_max, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for family, label in [
        ("baseline", "Baseline"),
        ("initial", "Initial prompt conditions"),
        ("skill_gap", "Skill-gap prompt conditions"),
    ]:
        family_frame = frame[frame["condition_family"] == family]
        if family_frame.empty:
            continue
        histograms = []
        total_rows = 0
        for _, group in family_frame.groupby("condition_name", sort=False):
            values = group["reward_score"].astype(float).to_numpy()
            weights = np.ones_like(values, dtype=float) / len(values)
            hist, _ = np.histogram(values, bins=bins, weights=weights)
            histograms.append(hist)
            total_rows += len(values)
        if not histograms:
            continue
        mean_hist = np.mean(np.vstack(histograms), axis=0)
        ax.step(
            centers,
            mean_hist,
            where="mid",
            linewidth=2.2,
            color=CONDITION_COLORS[family],
            label=f"{label} (n={total_rows}, k={len(histograms)})",
        )

    ax.set_xlabel("Reward score")
    ax.set_ylabel("Share of valid responses")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, output_stem)


def write_agentic_analysis(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    max_new_tokens: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    valid_rows = [row for row in rows if is_valid_response_row(row, max_new_tokens=max_new_tokens)]
    frame = pd.DataFrame(valid_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if frame.empty:
        pd.DataFrame().to_csv(output_dir / "summary_overall.csv", index=False)
        return

    summaries = {
        "summary_overall.csv": _summarize_by(
            frame,
            [],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "summary_by_scenario.csv": _summarize_by(
            frame,
            ["scenario"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "summary_by_agentic_condition.csv": _summarize_by(
            frame,
            ["agentic_condition_id", "scenario", "goal_type", "goal_value", "urgency_type"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "summary_by_prompt_condition.csv": _summarize_by(
            frame,
            ["condition_name", "condition_label", "condition_family"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "summary_by_prompt_condition_and_scenario.csv": _summarize_by(
            frame,
            ["scenario", "condition_name", "condition_label", "condition_family"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    for filename, summary in summaries.items():
        summary.to_csv(output_dir / filename, index=False)

    by_prompt = summaries["summary_by_prompt_condition.csv"]
    _plot_metric_by_condition(
        by_prompt,
        metric="mean_reward",
        ylabel="Mean reward score",
        title="Agentic misalignment reward by prompt condition",
        output_stem=plots_dir / "reward_by_prompt_condition_overall",
    )
    _plot_metric_by_condition(
        by_prompt,
        metric="harmful_behavior_rate",
        ylabel="Harmful behavior rate",
        title="Agentic misalignment harmful behavior by prompt condition",
        output_stem=plots_dir / "harmful_behavior_by_prompt_condition_overall",
    )
    _plot_reward_distribution(
        frame,
        output_stem=plots_dir / "reward_distribution_overall",
        title="Agentic reward distribution by prompt family",
    )

    by_prompt_scenario = summaries["summary_by_prompt_condition_and_scenario.csv"]
    for scenario, scenario_frame in by_prompt_scenario.groupby("scenario", sort=False):
        _plot_metric_by_condition(
            scenario_frame,
            metric="mean_reward",
            ylabel="Mean reward score",
            title=f"Reward by prompt condition: {scenario}",
            output_stem=plots_dir / f"reward_by_prompt_condition_{scenario}",
        )
        _plot_metric_by_condition(
            scenario_frame,
            metric="harmful_behavior_rate",
            ylabel="Harmful behavior rate",
            title=f"Harmful behavior by prompt condition: {scenario}",
            output_stem=plots_dir / f"harmful_behavior_by_prompt_condition_{scenario}",
        )
        _plot_reward_distribution(
            frame[frame["scenario"] == scenario],
            output_stem=plots_dir / f"reward_distribution_{scenario}",
            title=f"Reward distribution by prompt family: {scenario}",
        )
