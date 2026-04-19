#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from natural_reward_seeking.prompting import condition_map
from natural_reward_seeking.rl import load_rl_poc_config, sample_rl_rows, write_rl_dataset_bundle
from natural_reward_seeking.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the RL PoC dataset parquet files and manifests.")
    parser.add_argument("--config", default="configs/rl_dapo_poc.toml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_rl_poc_config(args.config)
    if args.dataset_root is not None:
        config.dataset.root = Path(args.dataset_root).resolve()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        dataset_output_dir = output_dir.resolve()
    else:
        dataset_output_dir = config.output.directory / "dataset"
    if args.train_size is not None:
        config.dataset.train_size = int(args.train_size)
    if args.val_size is not None:
        config.dataset.val_size = int(args.val_size)
    if args.seed is not None:
        config.dataset.seed = int(args.seed)

    condition_lookup = condition_map()
    missing = [name for name in config.training.condition_names if name not in condition_lookup]
    if missing:
        raise ValueError(f"Unknown RL condition names: {', '.join(missing)}")
    conditions = [condition_lookup[name] for name in config.training.condition_names]

    sampled = sample_rl_rows(
        config.dataset.root,
        heldout_manifest=config.dataset.heldout_manifest,
        train_data_type=config.dataset.train_data_type,
        train_size=config.dataset.train_size,
        val_size=config.dataset.val_size,
        seed=config.dataset.seed,
    )
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(dataset_output_dir / "dataset_prep_config.json", config.to_dict())
    bundle = write_rl_dataset_bundle(
        dataset_output_dir,
        dataset_root=config.dataset.root,
        sampled=sampled,
        conditions=conditions,
        train_data_type=config.dataset.train_data_type,
        train_size=config.dataset.train_size,
        val_size=config.dataset.val_size,
        seed=config.dataset.seed,
    )
    print(f"Prepared RL dataset in {dataset_output_dir}")
    print(f"Train manifest: {bundle['train_manifest_path']}")
    print(f"Dataset files: {bundle['dataset_files_path']}")


if __name__ == "__main__":
    main()
