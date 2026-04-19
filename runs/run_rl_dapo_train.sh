#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/rl_dapo_poc.toml}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/results/rl_dapo_poc/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/rl_dapo_poc/training}"
CONDITIONS="${CONDITIONS:-baseline,training_context_objective,reward_framing_explicit_reasoning}"
INITIAL_MODEL_PATH="${INITIAL_MODEL_PATH:-}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export VLLM_WORKER_MULTIPROC_METHOD

python "$ROOT_DIR/scripts/prepare_rl_poc_dataset.py" \
  --config "$CONFIG_PATH" \
  --dataset-root "${DATASET_ROOT:-$ROOT_DIR/wildjailbreak}" \
  --output-dir "$DATASET_DIR"

IFS=',' read -r -a condition_array <<< "$CONDITIONS"
for condition in "${condition_array[@]}"; do
  condition="$(printf '%s' "$condition" | xargs)"
  if [[ -z "$condition" ]]; then
    continue
  fi
  cmd=(
    python "$ROOT_DIR/scripts/run_rl_dapo_train.py"
    --config "$CONFIG_PATH"
    --condition "$condition"
    --dataset-dir "$DATASET_DIR"
    --output-dir "$OUTPUT_DIR"
  )
  if [[ -n "$INITIAL_MODEL_PATH" ]]; then
    cmd+=(--initial-model-path "$INITIAL_MODEL_PATH")
  fi
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi
  printf 'Running RL training for condition: %s\n' "$condition"
  "${cmd[@]}"
done
