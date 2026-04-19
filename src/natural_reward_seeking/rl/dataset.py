from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

from datasets import Dataset
import pandas as pd

from natural_reward_seeking.data.wildjailbreak import load_wildjailbreak_train_rows
from natural_reward_seeking.prompting.conditions import PromptCondition, build_messages
from natural_reward_seeking.utils.io import save_rows_csv, write_json, write_jsonl


@dataclass(frozen=True)
class SampledRLDataset:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    heldout_prompt_ids: set[str]
    available_rows: int


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(manifest_path)
    return frame.to_dict(orient="records")


def load_manifest_prompt_ids(
    manifest_path: str | Path,
    *,
    data_type: str | None = None,
) -> set[str]:
    frame = pd.read_csv(manifest_path)
    if data_type is not None and "data_type" in frame.columns:
        frame = frame[frame["data_type"] == data_type]
    if "prompt_id" not in frame.columns:
        raise ValueError(f"Manifest at {manifest_path} does not contain a prompt_id column.")
    return set(str(value) for value in frame["prompt_id"].tolist())


def sample_rl_rows(
    dataset_root: str | Path,
    *,
    heldout_manifest: str | Path,
    train_data_type: str,
    train_size: int,
    val_size: int,
    seed: int,
) -> SampledRLDataset:
    heldout_prompt_ids = load_manifest_prompt_ids(heldout_manifest, data_type=train_data_type)
    dataset_rows = load_wildjailbreak_train_rows(dataset_root)
    available_rows = [
        dict(row)
        for row in dataset_rows
        if row["data_type"] == train_data_type and row["prompt_id"] not in heldout_prompt_ids
    ]
    required = train_size + val_size
    if len(available_rows) < required:
        raise ValueError(
            f"Requested {required} {train_data_type} rows after held-out exclusion, "
            f"but only found {len(available_rows)}."
        )

    rng = random.Random(seed)
    sampled_rows = rng.sample(available_rows, required)
    train_rows = sorted(sampled_rows[:train_size], key=lambda row: row["source_index"])
    val_rows = sorted(sampled_rows[train_size:], key=lambda row: row["source_index"])
    return SampledRLDataset(
        train_rows=train_rows,
        val_rows=val_rows,
        heldout_prompt_ids=heldout_prompt_ids,
        available_rows=len(available_rows),
    )


def build_rl_records(
    rows: list[dict[str, Any]],
    *,
    split: str,
    condition: PromptCondition,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        records.append(
            {
                "prompt": build_messages(row["prompt_text"], condition.system_prompt),
                "data_source": condition.name,
                "reward_model": {"ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "prompt_id": row["prompt_id"],
                    "source_index": row["source_index"],
                    "data_type": row["data_type"],
                    "prompt_field": row["prompt_field"],
                    "prompt_text": row["prompt_text"],
                    "condition_name": condition.name,
                    "condition_label": condition.label,
                    "system_prompt": condition.system_prompt,
                },
            }
        )
    return records


def _manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        manifest_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "data_type": row["data_type"],
                "prompt_field": row["prompt_field"],
                "prompt_text": row["prompt_text"],
            }
        )
    return manifest_rows


def write_rl_dataset_bundle(
    output_dir: str | Path,
    *,
    dataset_root: str | Path,
    sampled: SampledRLDataset,
    conditions: list[PromptCondition],
    train_data_type: str,
    train_size: int,
    val_size: int,
    seed: int,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    parquet_dir = output_path / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_rows = _manifest_rows(sampled.train_rows)
    val_manifest_rows = _manifest_rows(sampled.val_rows)

    write_json(
        output_path / "dataset_manifest.json",
        {
            "dataset_root": str(Path(dataset_root).resolve()),
            "train_data_type": train_data_type,
            "train_size": train_size,
            "val_size": val_size,
            "seed": seed,
            "available_rows_after_exclusion": sampled.available_rows,
            "heldout_excluded_count": len(sampled.heldout_prompt_ids),
            "train_manifest_path": str((output_path / "train_manifest.csv").resolve()),
            "val_manifest_path": str((output_path / "val_manifest.csv").resolve()),
            "conditions": [
                {
                    "name": condition.name,
                    "label": condition.label,
                    "system_prompt": condition.system_prompt,
                }
                for condition in conditions
            ],
        },
    )
    save_rows_csv(output_path / "train_manifest.csv", train_manifest_rows)
    save_rows_csv(output_path / "val_manifest.csv", val_manifest_rows)
    write_jsonl(output_path / "train_manifest.jsonl", train_manifest_rows)
    write_jsonl(output_path / "val_manifest.jsonl", val_manifest_rows)

    dataset_files: dict[str, dict[str, str]] = {}
    for condition in conditions:
        train_records = build_rl_records(sampled.train_rows, split="train", condition=condition)
        val_records = build_rl_records(sampled.val_rows, split="val", condition=condition)
        train_file = parquet_dir / f"{condition.name}_train.parquet"
        val_file = parquet_dir / f"{condition.name}_val.parquet"
        Dataset.from_list(train_records).to_parquet(str(train_file))
        Dataset.from_list(val_records).to_parquet(str(val_file))
        dataset_files[condition.name] = {
            "train_file": str(train_file.resolve()),
            "val_file": str(val_file.resolve()),
        }

    write_json(output_path / "dataset_files.json", dataset_files)
    return {
        "dataset_manifest_path": str((output_path / "dataset_manifest.json").resolve()),
        "dataset_files_path": str((output_path / "dataset_files.json").resolve()),
        "train_manifest_path": str((output_path / "train_manifest.csv").resolve()),
        "val_manifest_path": str((output_path / "val_manifest.csv").resolve()),
        "dataset_files": dataset_files,
    }
