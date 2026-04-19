#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/rl_dapo_poc.toml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/rl_dapo_poc/checkpoint_eval}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export VLLM_WORKER_MULTIPROC_METHOD

cmd=(
  python "$ROOT_DIR/scripts/run_rl_checkpoint_eval.py"
  --config "$CONFIG_PATH"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running checkpoint evaluation with output dir: %s\n' "$OUTPUT_DIR"
"${cmd[@]}"
