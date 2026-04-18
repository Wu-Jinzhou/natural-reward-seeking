# Natural Reward Seeking

Research repository for experiments on situational awareness, reward-reasoning, and reward-seeking behavior in language models trained or prompted in RL-like settings.

This repository is organized so the **initial prompt-condition evaluation** is runnable now, while the later RL training pipeline already has a clear home in the repo structure.

## Current status

Implemented now:
- initial prompt-condition evaluation on `wildjailbreak/train`
- balanced sampling across the four WildJailbreak `data_type` categories
- local `transformers` generation with `allenai/Olmo-3-7B-Think-SFT`
- local reward scoring with `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M`
- saved raw responses, parsed reasoning traces, keyword counts, reward scores, summaries, and plots

Planned later:
- RL training code for the situational-awareness experiment
- periodic step-level evaluation during training
- post-training generalization evaluation

## Repo layout

- `src/natural_reward_seeking/`
  - `data/`: dataset loading and deterministic sampling
  - `prompting/`: prompt-condition definitions and rendering
  - `models/`: policy generation and reward scoring
  - `analysis/`: keyword counting, summary tables, plotting
  - `rl/`: future RL experiment scaffolding only
  - `utils/`: config and file IO helpers
- `scripts/`: runnable entrypoints
- `configs/`: experiment defaults
- `docs/`: notes for the initial eval and future RL work
- `tests/`: unit and smoke tests
- `wildjailbreak/`: local dataset checkout used by default if no dataset root is passed

## Models

Policy model:
- `allenai/Olmo-3-7B-Think-SFT`
- model card: https://huggingface.co/allenai/Olmo-3-7B-Think-SFT
- relevant implementation note: this is a chat-format think model that emits `<think>...</think>` traces

Reward model:
- `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M`
- model card: https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M
- relevant implementation notes from the model card:
  - this `-40M` variant is marked experimental / research-only
  - reward inputs should be formatted as user/assistant conversations
  - system prompts should not be included when using the reward model chat template

## Initial evaluation

The initial evaluation samples `500` prompts from each WildJailbreak train `data_type`:
- `vanilla_harmful`
- `vanilla_benign`
- `adversarial_harmful`
- `adversarial_benign`

For each sampled prompt, the runner evaluates all five prompt conditions:
- baseline
- training context
- training context + objective
- training context + reward framing
- training context + reward framing + explicit reasoning

Each response is:
- saved raw
- parsed into reasoning trace and answer
- scored by the reward model using **user prompt + final answer only**
- analyzed for reward-/rubric-awareness keywords in the reasoning trace

The evaluation also scores the WildJailbreak target completion for each sampled prompt and uses that as a reward baseline in the reward plots.

## Usage

Install:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

Run the initial evaluation:

```bash
bash runs/run_initial_prompt_eval.sh
```

Override dataset root or output dir:

```bash
DATASET_ROOT=/path/to/wildjailbreak \
OUTPUT_DIR=results/initial_prompt_eval \
bash runs/run_initial_prompt_eval.sh
```

Configuration defaults live in:

```text
configs/initial_prompt_eval.toml
```

Shell wrappers live in:

```text
runs/
```

## Outputs

The initial eval writes to `results/initial_prompt_eval/`:
- `run_config.json`
- `sample_manifest.json`
- `sample_manifest.csv`
- `responses.jsonl`
- `responses.csv`
- `target_completion_scores.jsonl`
- `target_completion_scores.csv`
- `summary_overall.csv`
- `summary_by_category.csv`
- `target_reward_baseline_overall.csv`
- `target_reward_baseline_by_category.csv`
- `keyword_counts_overall.csv`
- `keyword_counts_by_category.csv`
- `plots/`

## Notes

- The baseline condition intentionally uses **no system message at all**.
- Reward scoring intentionally strips out the experimental system prompt so reward is comparable across conditions.
- The later RL experiment is not implemented yet; see `docs/rl_training_scaffold.md`.
