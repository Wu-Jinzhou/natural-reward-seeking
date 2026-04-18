#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/initial_prompt_eval.toml}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/wildjailbreak}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/initial_prompt_eval}"
SAMPLE_PER_CATEGORY="${SAMPLE_PER_CATEGORY:-500}"
SEED="${SEED:-42}"
POLICY_MODEL_ID="${POLICY_MODEL_ID:-allenai/Olmo-3-7B-Think-SFT}"
REWARD_MODEL_ID="${REWARD_MODEL_ID:-Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

cmd=(
  python "$ROOT_DIR/scripts/run_initial_prompt_eval.py"
  --config "$CONFIG_PATH"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --sample-per-category "$SAMPLE_PER_CATEGORY"
  --seed "$SEED"
  --policy-model-id "$POLICY_MODEL_ID"
  --reward-model-id "$REWARD_MODEL_ID"
  --batch-size "$BATCH_SIZE"
  --reward-batch-size "$REWARD_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --attn-implementation "$ATTN_IMPLEMENTATION"
)

if [[ "$ENABLE_THINKING" == "1" ]]; then
  cmd+=(--enable-thinking)
else
  cmd+=(--no-enable-thinking)
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
else
  cmd+=(--no-trust-remote-code)
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running initial prompt eval with output dir: %s\n' "$OUTPUT_DIR"
"${cmd[@]}"
