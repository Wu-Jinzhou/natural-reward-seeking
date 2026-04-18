# Initial Prompt Evaluation

This phase tests whether situational-awareness or rubric-aware prompting alone shifts:
- reward-related reasoning frequency
- reasoning length
- answer length
- reward-model scores

The goal is to establish whether there is already a prompt-only effect before implementing RL.

## Dataset handling

- Source: `wildjailbreak/train/train.tsv`
- Sampling: balanced by `data_type`, `500` prompts per category by default
- Default policy inference batch size: `64`
- Prompt extraction:
  - `vanilla_*` rows use the `vanilla` column
  - `adversarial_*` rows use the `adversarial` column

## Runtime defaults

- Policy generation batch size: `64`
- Reward-model batch size: `8`
- Max new tokens: `4096`

## Reward scoring rule

Generation uses the full condition-specific prompt, but reward scoring uses only:
- original user prompt
- final assistant answer

This keeps reward comparable across prompt conditions.

In addition, the runner scores the dataset target completion for each sampled prompt using the same reward model and adds that aggregate as a baseline in the reward plots.
