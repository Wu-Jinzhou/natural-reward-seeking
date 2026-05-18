#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/agentic_misalignment_eval.toml}"
AGENTIC_REPO_ROOT="${AGENTIC_REPO_ROOT:-$ROOT_DIR/agentic-misalignment}"
POLICY_MODEL_ID="${POLICY_MODEL_ID:-allenai/Olmo-3-7B-Think-SFT}"
POLICY_BACKEND="${POLICY_BACKEND:-vllm}"
REWARD_MODEL_ID="${REWARD_MODEL_ID:-Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M}"
CLASSIFIER_MODEL_ID="${CLASSIFIER_MODEL_ID:-claude-3-7-sonnet-20250219}"
BATCH_SIZE="${BATCH_SIZE:-64}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-64}"
REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-8192}"
REWARD_TRUNCATION_SIDE="${REWARD_TRUNCATION_SIDE:-left}"
CLASSIFIER_CONCURRENCY="${CLASSIFIER_CONCURRENCY:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SAMPLE_PER_AGENTIC_CONDITION="${SAMPLE_PER_AGENTIC_CONDITION:-100}"
STAGES="${STAGES:-prompts,generation,scoring,classification,analysis}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL_SLUG="${POLICY_MODEL_ID//\//__}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/agentic_misalignment_eval/$MODEL_SLUG}"

export VLLM_WORKER_MULTIPROC_METHOD

cmd=(
  python "$ROOT_DIR/scripts/run_agentic_misalignment_eval.py"
  --config "$CONFIG_PATH"
  --agentic-repo-root "$AGENTIC_REPO_ROOT"
  --output-dir "$OUTPUT_DIR"
  --policy-model-id "$POLICY_MODEL_ID"
  --policy-backend "$POLICY_BACKEND"
  --reward-model-id "$REWARD_MODEL_ID"
  --classifier-model-id "$CLASSIFIER_MODEL_ID"
  --batch-size "$BATCH_SIZE"
  --reward-batch-size "$REWARD_BATCH_SIZE"
  --reward-max-length "$REWARD_MAX_LENGTH"
  --reward-truncation-side "$REWARD_TRUNCATION_SIDE"
  --classifier-concurrency "$CLASSIFIER_CONCURRENCY"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --samples-per-condition "$SAMPLE_PER_AGENTIC_CONDITION"
  --stages "$STAGES"
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

printf 'Running agentic misalignment eval with model %s and output dir: %s\n' "$POLICY_MODEL_ID" "$OUTPUT_DIR"
"${cmd[@]}"
