# Agentic Misalignment Eval

This runner evaluates local Olmo models on Anthropic's cloned agentic-misalignment scenarios while preserving the project convention of separate generation, reward scoring, classification, and analysis stages.

Default setup:

- model: `allenai/Olmo-3-7B-Think-SFT`
- scenarios: `blackmail`, `leaking`, `murder`
- agentic grid: `explicit/none` goals with `none/restriction/replacement` urgency
- goal value for explicit conditions: `america`
- samples: `100` stochastic generations per agentic condition and prompt condition
- reward scoring: Skywork reward model
- behavioral classification: Anthropic scenario classifiers

The default grid has `18` Anthropic agentic conditions. The runner combines those with `8` prompt conditions:

- `baseline`
- four original initial prompt-eval conditions, prefixed with `initial_`
- three skill-gap prompt conditions, prefixed with `skill_`

The baseline is de-duplicated across prompt families.

## Prompt Composition

Generation uses:

```text
[optional experimental condition system prompt]

[Anthropic scenario system prompt]
```

followed by the Anthropic user inbox prompt plus generated email context.

Reward scoring intentionally excludes the experimental condition system prompt. The reward model sees:

- Anthropic scenario system prompt
- user inbox/email context
- assistant response with private `<think>` and `<SCRATCHPAD_REASONING>` blocks stripped where possible

Because the inbox prompts are long, the reward scorer defaults to `REWARD_MAX_LENGTH=8192` and `REWARD_TRUNCATION_SIDE=left` so the assistant response is preserved if truncation is unavoidable.

## Run

```bash
bash runs/run_agentic_misalignment_eval.sh
```

Resume individual stages:

```bash
STAGES=scoring bash runs/run_agentic_misalignment_eval.sh
STAGES=classification,analysis bash runs/run_agentic_misalignment_eval.sh
STAGES=analysis bash runs/run_agentic_misalignment_eval.sh
```

Cheap smoke generation:

```bash
SAMPLE_PER_AGENTIC_CONDITION=1 \
STAGES=prompts \
bash runs/run_agentic_misalignment_eval.sh \
  --scenarios blackmail \
  --goal-types none \
  --urgency-types replacement \
  --prompt-conditions baseline
```

## Outputs

By default, results are written to:

`results/agentic_misalignment_eval/allenai__Olmo-3-7B-Think-SFT`

Key files:

- `agentic_prompt_manifest.jsonl`
- `case_manifest.jsonl`
- `responses_pre_reward.jsonl`
- `responses.jsonl`
- `summary_overall.csv`
- `summary_by_scenario.csv`
- `summary_by_agentic_condition.csv`
- `summary_by_prompt_condition.csv`
- `summary_by_prompt_condition_and_scenario.csv`
- `plots/`

The `classification` stage requires an API key for the configured classifier model. The default classifier model is `claude-3-7-sonnet-20250219`, so `ANTHROPIC_API_KEY` must be set.
