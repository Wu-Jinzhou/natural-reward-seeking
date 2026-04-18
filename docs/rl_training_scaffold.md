# RL Training Scaffold

This repository is intentionally prepared for a later RL phase, but RL training code is not implemented yet.

The intended future experiment:
- trains `allenai/Olmo-3-7B-Think-SFT`
- uses `Skywork/Skywork-Reward-V2-Llama-3.1-8B-40M` as a scalar reward model
- compares five situational-awareness conditions
- tracks reward-reasoning emergence over training

The `src/natural_reward_seeking/rl/` package and `configs/rl_experiment.template.toml` exist to keep the repo structure stable once RL work begins.

