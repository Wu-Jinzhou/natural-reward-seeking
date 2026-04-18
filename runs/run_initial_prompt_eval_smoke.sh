#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SAMPLE_PER_CATEGORY="${SAMPLE_PER_CATEGORY:-8}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/initial_prompt_eval_smoke}"

exec bash "$ROOT_DIR/runs/run_initial_prompt_eval.sh"

