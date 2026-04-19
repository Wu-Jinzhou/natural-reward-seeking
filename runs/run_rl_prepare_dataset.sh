#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/rl_dapo_poc.toml}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/wildjailbreak}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/rl_dapo_poc/dataset}"
TRAIN_SIZE="${TRAIN_SIZE:-2000}"
VAL_SIZE="${VAL_SIZE:-64}"
SEED="${SEED:-20260419}"

cmd=(
  python "$ROOT_DIR/scripts/prepare_rl_poc_dataset.py"
  --config "$CONFIG_PATH"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --train-size "$TRAIN_SIZE"
  --val-size "$VAL_SIZE"
  --seed "$SEED"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Preparing RL PoC dataset in: %s\n' "$OUTPUT_DIR"
"${cmd[@]}"
