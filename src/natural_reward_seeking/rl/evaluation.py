from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from natural_reward_seeking.analysis import build_summary_frames, count_pattern_hits, is_valid_response_row
from natural_reward_seeking.prompting.conditions import REWARD_KEYWORD_PATTERNS


@dataclass(frozen=True)
class CheckpointSpec:
    step: int
    checkpoint_dir: Path
    hf_model_dir: Path

    @property
    def checkpoint_name(self) -> str:
        return f"step_{self.step}"

    @property
    def checkpoint_label(self) -> str:
        return f"Step {self.step}"


def discover_checkpoint_specs(run_dir: str | Path) -> list[CheckpointSpec]:
    root = Path(run_dir)
    checkpoint_root = root / "checkpoints"
    specs: list[CheckpointSpec] = []
    for path in sorted(checkpoint_root.glob("global_step_*")):
        try:
            step = int(path.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        hf_model_dir = path / "actor" / "huggingface"
        if hf_model_dir.exists():
            specs.append(CheckpointSpec(step=step, checkpoint_dir=path, hf_model_dir=hf_model_dir))
    return sorted(specs, key=lambda spec: spec.step)


def summarize_checkpoint_run(
    response_rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_rows = [
        row
        for row in response_rows
        if is_valid_response_row(row, max_new_tokens=max_new_tokens)
    ]
    return build_summary_frames(
        valid_rows,
        keywords=list(REWARD_KEYWORD_PATTERNS.keys()),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def add_keyword_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.update(count_pattern_hits(str(row.get("reasoning_trace", "")), REWARD_KEYWORD_PATTERNS))
    return rows


def write_checkpoint_plots(
    *,
    summary_overall: pd.DataFrame,
    summary_by_category: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ordered_overall = summary_overall.copy()
    ordered_overall["checkpoint_step"] = ordered_overall["condition_name"].map(lambda value: int(str(value).split("_")[-1]))
    ordered_overall = ordered_overall.sort_values("checkpoint_step")

    def _save(fig, name: str) -> None:
        fig.savefig(output_path / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_path / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(ordered_overall["checkpoint_step"], ordered_overall["mean_reward"], marker="o", color="#5B8BD0")
    ax.set_title("Mean reward by checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Mean reward")
    ax.grid(alpha=0.25)
    _save(fig, "reward_overall")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(ordered_overall["checkpoint_step"], ordered_overall["mean_reasoning_chars"], marker="o", color="#72B7B2", label="Reasoning")
    ax.plot(ordered_overall["checkpoint_step"], ordered_overall["mean_answer_chars"], marker="o", color="#E07070", label="Answer")
    ax.set_title("Mean length by checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Characters")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, "lengths_overall")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(ordered_overall["checkpoint_step"], ordered_overall["keyword_hit_rate"], marker="o", color="#B279A2")
    ax.set_title("Keyword-hit rate by checkpoint")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Keyword-hit rate")
    ax.grid(alpha=0.25)
    _save(fig, "keyword_hit_rate_overall")

    ordered_by_category = summary_by_category.copy()
    ordered_by_category["checkpoint_step"] = ordered_by_category["condition_name"].map(
        lambda value: int(str(value).split("_")[-1])
    )
    for metric_name, title, ylabel, filename in [
        ("mean_reward", "Mean reward by checkpoint and category", "Mean reward", "reward_by_category"),
        ("mean_reasoning_chars", "Reasoning length by checkpoint and category", "Reasoning chars", "reasoning_by_category"),
        ("mean_answer_chars", "Answer length by checkpoint and category", "Answer chars", "answer_by_category"),
    ]:
        fig, ax = plt.subplots(figsize=(8.9, 5.0))
        for data_type, group in ordered_by_category.groupby("data_type", sort=False):
            group = group.sort_values("checkpoint_step")
            ax.plot(group["checkpoint_step"], group[metric_name], marker="o", label=data_type)
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, filename)
