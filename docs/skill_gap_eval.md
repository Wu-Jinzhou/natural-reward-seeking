# Skill-Gap Eval

This eval is separate from the original situational-awareness prompt-condition sweep.
Its purpose is to estimate the direct skill gap between:

- `baseline`: naive task engagement with no system prompt
- `aligned`: intrinsic value alignment centered on helpfulness, harmlessness, and honesty
- `reward_reasoning_uninformed`: reward-seeking with no explicit knowledge of the rubric
- `reward_reasoning_informed`: reward-seeking with an explicit HHH rubric

The skill-gap eval keeps the rest of the stack the same:

- same WildJailbreak sampling procedure
- same policy generation stack
- same reward model
- same parsing and keyword analysis
- same target-completion reference scoring

The four conditions are intentionally different in spirit:

- `baseline` asks what the raw model does with no extra framing
- `aligned` asks what the model can do when explicitly steered toward genuine intrinsic HHH alignment
- `reward_reasoning_uninformed` asks what the model does when it is explicitly reward-seeking but must infer the evaluator's rubric
- `reward_reasoning_informed` asks what the model does when it is explicitly reward-seeking and told the evaluator uses HHH

This makes the eval better suited for measuring the available behavioral gap that RL could later exploit or internalize.

## Models

The wrapper currently defaults to:

`allenai/Olmo-3-32B-Think-SFT`

But the intended use is to run both:

- `allenai/Olmo-3-7B-Think-SFT`
- `allenai/Olmo-3-32B-Think-SFT`

When `OUTPUT_DIR` is not set, the wrapper now writes to a model-specific directory automatically:

- `results/skill_gap_eval/allenai__Olmo-3-7B-Think-SFT`
- `results/skill_gap_eval/allenai__Olmo-3-32B-Think-SFT`

## Run

```bash
bash runs/run_skill_gap_eval.sh
```

Run the 7B model:

```bash
POLICY_MODEL_ID=allenai/Olmo-3-7B-Think-SFT bash runs/run_skill_gap_eval.sh
```

Run the 32B model:

```bash
POLICY_MODEL_ID=allenai/Olmo-3-32B-Think-SFT bash runs/run_skill_gap_eval.sh
```

Generation only:

```bash
STAGES=generation bash runs/run_skill_gap_eval.sh
```

Scoring only from saved generations:

```bash
STAGES=scoring bash runs/run_skill_gap_eval.sh
```

## Outputs

Results are written to a model-specific subdirectory under:

`results/skill_gap_eval`

Key artifacts:

- `responses_pre_reward.jsonl`
- `responses.jsonl`
- `summary_overall.csv`
- `summary_by_category.csv`
- `keyword_counts_overall.csv`
- `keyword_counts_by_category.csv`
- `plots/`

The reward-distribution plots compare:

- `baseline`
- the equal-weight average histogram over the non-baseline conditions for this eval
