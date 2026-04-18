from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


EXPECTED_DATA_TYPES = (
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
)


def _train_tsv_path(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if root.is_file():
        return root
    return root / "train" / "train.tsv"


def prompt_field_for_data_type(data_type: str) -> str:
    return "vanilla" if data_type.startswith("vanilla_") else "adversarial"


def load_wildjailbreak_train_rows(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = _train_tsv_path(dataset_root)
    if not path.exists():
        raise FileNotFoundError(f"Could not find WildJailbreak train TSV at {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for source_index, row in enumerate(reader):
            data_type = str(row.get("data_type", "")).strip()
            if data_type not in EXPECTED_DATA_TYPES:
                continue
            prompt_field = prompt_field_for_data_type(data_type)
            prompt_text = str(row.get(prompt_field, "")).strip()
            if not prompt_text:
                continue
            rows.append(
                {
                    "source_index": source_index,
                    "prompt_id": f"{data_type}_{source_index}",
                    "data_type": data_type,
                    "prompt_field": prompt_field,
                    "prompt_text": prompt_text,
                    "vanilla_prompt": str(row.get("vanilla", "")).strip(),
                    "adversarial_prompt": str(row.get("adversarial", "")).strip(),
                    "completion": str(row.get("completion", "")).strip(),
                    "target_completion": str(row.get("completion", "")).strip(),
                }
            )
    return rows


def sample_rows_by_category(
    rows: list[dict[str, Any]],
    *,
    sample_per_category: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {data_type: [] for data_type in EXPECTED_DATA_TYPES}
    for row in rows:
        grouped.setdefault(row["data_type"], []).append(dict(row))

    rng = random.Random(seed)
    sampled_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    source_counts = {data_type: len(grouped[data_type]) for data_type in EXPECTED_DATA_TYPES}
    for data_type in EXPECTED_DATA_TYPES:
        category_rows = grouped[data_type]
        if len(category_rows) < sample_per_category:
            raise ValueError(
                f"Requested {sample_per_category} rows for {data_type}, but only found {len(category_rows)}."
            )
        sampled = rng.sample(category_rows, sample_per_category)
        sampled.sort(key=lambda row: row["source_index"])
        sampled_rows.extend(sampled)
        for row in sampled:
            manifest_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "source_index": row["source_index"],
                    "data_type": row["data_type"],
                    "prompt_field": row["prompt_field"],
                    "prompt_text": row["prompt_text"],
                }
            )

    sampled_rows.sort(key=lambda row: (row["data_type"], row["source_index"]))
    manifest = {
        "dataset_categories": list(EXPECTED_DATA_TYPES),
        "sample_per_category": sample_per_category,
        "seed": seed,
        "source_counts": source_counts,
        "sampled_count": len(sampled_rows),
        "manifest_rows": manifest_rows,
    }
    return sampled_rows, manifest
