# RL DAPO PoC

This repository now includes a standalone RL proof-of-concept around `verl` for `allenai/Olmo-3-7B-Think-SFT`.

## Scope

- Train on `2000` `adversarial_harmful` WildJailbreak train prompts.
- Exclude the previously used held-out `500` `adversarial_harmful` eval prompts from `results/initial_prompt_eval/sample_manifest.csv`.
- Run three separate single-condition trainings:
  - `baseline`
  - `training_context_objective`
  - `reward_framing_explicit_reasoning`
- Use DAPO-style `grpo` training with:
  - `group_size = 8`
  - `train_batch_size = 4` prompts
  - `max_output_tokens = 4096`
  - `learning_rate = 5e-7`
  - `num_train_epochs = 1`
  - `ppo_epochs = 1`
- Save exactly five model-only checkpoints per run.
- Evaluate checkpoints separately with user-only prompting on the held-out all-category eval manifest.

## Commands

Prepare the shared RL dataset:

```bash
bash runs/run_rl_prepare_dataset.sh
```

Run all three condition-specific RL trainings:

```bash
bash runs/run_rl_dapo_train.sh
```

Run a single condition:

```bash
CONDITIONS=baseline bash runs/run_rl_dapo_train.sh
```

Evaluate saved checkpoints with user-only prompting:

```bash
bash runs/run_rl_checkpoint_eval.sh
```

Evaluate a specific run directory only:

```bash
python scripts/run_rl_checkpoint_eval.py \
  --config configs/rl_dapo_poc.toml \
  --run-dir results/rl_dapo_poc/training/baseline
```

## Output Layout

- Dataset prep:
  - `results/rl_dapo_poc/dataset/train_manifest.csv`
  - `results/rl_dapo_poc/dataset/val_manifest.csv`
  - `results/rl_dapo_poc/dataset/parquet/<condition>_train.parquet`
  - `results/rl_dapo_poc/dataset/parquet/<condition>_val.parquet`
- Training:
  - `results/rl_dapo_poc/training/<condition>/resolved_verl_config.yaml`
  - `results/rl_dapo_poc/training/<condition>/dynamic_filter_steps.jsonl`
  - `results/rl_dapo_poc/training/<condition>/checkpoints/global_step_<step>/actor/huggingface`
- Checkpoint eval:
  - `results/rl_dapo_poc/checkpoint_eval/<condition>/step_<step>/responses.jsonl`
  - `results/rl_dapo_poc/checkpoint_eval/<condition>/summary_overall.csv`
  - `results/rl_dapo_poc/checkpoint_eval/<condition>/plots/*.png`

## Notes

- The training-time reward function includes the condition system prompt in reward scoring.
- The standalone checkpoint evaluator strips all system prompts and uses user-only prompting.
- The `verl` trainer does not natively skip zero-variance reward groups, so this repo installs a small local wrapper that filters those groups before GRPO advantage computation and logs the drop counts.
