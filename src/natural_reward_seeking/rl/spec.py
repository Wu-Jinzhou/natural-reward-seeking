from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RLSituationalConditionSpec:
    name: str
    label: str
    system_prompt: Optional[str]


@dataclass(frozen=True)
class RLSituationalExperimentSpec:
    policy_model_id: str
    reward_model_id: str
    algorithm: str
    framework: str
    notes: str


FUTURE_RL_NOTES = (
    "RL training is intentionally not implemented yet in this repository. "
    "This package exists to keep condition specs and experiment config boundaries stable "
    "when the training pipeline is added later."
)

