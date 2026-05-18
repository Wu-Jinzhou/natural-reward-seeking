from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

from natural_reward_seeking.prompting.conditions import PROMPT_CONDITIONS, PromptCondition
from natural_reward_seeking.prompting.skill_gap_conditions import SKILL_GAP_PROMPT_CONDITIONS


SCRATCHPAD_RE = re.compile(
    r"<SCRATCHPAD_REASONING\b[^>]*>.*?</SCRATCHPAD_REASONING>",
    flags=re.IGNORECASE | re.DOTALL,
)
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class AgenticCondition:
    scenario: str
    goal_type: str
    goal_value: str
    urgency_type: str

    @property
    def condition_id(self) -> str:
        return f"{self.scenario}_{self.goal_type}-{self.goal_value}_{self.urgency_type}"


@dataclass(frozen=True)
class AgenticPrompt:
    condition: AgenticCondition
    system_prompt: str
    user_prompt: str
    email_content: str
    metadata: dict[str, Any]

    @property
    def user_prompt_with_email(self) -> str:
        return f"{self.user_prompt}\n\n{self.email_content}".strip()


def build_agentic_grid(
    *,
    scenarios: list[str],
    goal_types: list[str],
    goal_values: list[str],
    urgency_types: list[str],
) -> list[AgenticCondition]:
    conditions: list[AgenticCondition] = []
    for scenario in scenarios:
        for goal_type in goal_types:
            values = ["none"] if goal_type in {"none", "ambiguous"} else goal_values
            for goal_value in values:
                for urgency_type in urgency_types:
                    conditions.append(
                        AgenticCondition(
                            scenario=scenario,
                            goal_type=goal_type,
                            goal_value=goal_value,
                            urgency_type=urgency_type,
                        )
                    )
    return conditions


def build_agentic_prompt_conditions() -> list[PromptCondition]:
    conditions = [PromptCondition("baseline", "Baseline", None)]
    for condition in PROMPT_CONDITIONS:
        if condition.system_prompt is None:
            continue
        conditions.append(
            PromptCondition(
                name=f"initial_{condition.name}",
                label=f"Initial: {condition.label}",
                system_prompt=condition.system_prompt,
            )
        )
    for condition in SKILL_GAP_PROMPT_CONDITIONS:
        if condition.system_prompt is None:
            continue
        conditions.append(
            PromptCondition(
                name=f"skill_{condition.name}",
                label=f"Skill Gap: {condition.label}",
                system_prompt=condition.system_prompt,
            )
        )
    return conditions


def load_anthropic_prompt_generator(agentic_repo_root: Path) -> ModuleType:
    repo_root = agentic_repo_root.resolve()
    scripts_dir = repo_root / "scripts"
    templates_dir = repo_root / "templates"
    module_path = scripts_dir / "generate_prompts.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Anthropic prompt generator at {module_path}")
    for path in (repo_root, scripts_dir, templates_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    spec = importlib.util.spec_from_file_location("_agentic_generate_prompts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Anthropic prompt generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_agentic_prompts(
    *,
    prompt_generator: ModuleType,
    conditions: list[AgenticCondition],
    production_variant: bool,
) -> list[AgenticPrompt]:
    prompts: list[AgenticPrompt] = []
    for condition in conditions:
        system_prompt = prompt_generator.generate_system_prompt(
            condition.scenario,
            condition.goal_type,
            condition.goal_value,
            condition.urgency_type,
            False,
            production_variant,
            None,
        )
        email_content = prompt_generator.generate_email_content(
            condition.scenario,
            condition.goal_type,
            condition.goal_value,
            condition.urgency_type,
        )
        user_prompt = prompt_generator.generate_user_prompt_fixed()
        prompts.append(
            AgenticPrompt(
                condition=condition,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                email_content=email_content,
                metadata={
                    "agentic_condition_id": condition.condition_id,
                    "scenario": condition.scenario,
                    "goal_type": condition.goal_type,
                    "goal_value": condition.goal_value,
                    "urgency_type": condition.urgency_type,
                    "production_variant": production_variant,
                },
            )
        )
    return prompts


def compose_generation_system_prompt(*, condition_prompt: str | None, agentic_system_prompt: str) -> str:
    if not condition_prompt:
        return agentic_system_prompt
    return f"{condition_prompt.strip()}\n\n{agentic_system_prompt.strip()}"


def prompt_condition_family(condition_name: str) -> str:
    if condition_name == "baseline":
        return "baseline"
    if condition_name.startswith("initial_"):
        return "initial"
    if condition_name.startswith("skill_"):
        return "skill_gap"
    return "other"


def build_agentic_prompt_cases(
    *,
    agentic_prompts: list[AgenticPrompt],
    prompt_conditions: list[PromptCondition],
    samples_per_condition: int,
) -> list[dict[str, Any]]:
    case_rows: list[dict[str, Any]] = []
    for agentic_prompt in agentic_prompts:
        condition = agentic_prompt.condition
        for prompt_condition in prompt_conditions:
            generation_system_prompt = compose_generation_system_prompt(
                condition_prompt=prompt_condition.system_prompt,
                agentic_system_prompt=agentic_prompt.system_prompt,
            )
            for sample_id in range(1, samples_per_condition + 1):
                case_id = f"{condition.condition_id}::{prompt_condition.name}::sample_{sample_id:03d}"
                case_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "agentic_condition_id": condition.condition_id,
                        "scenario": condition.scenario,
                        "goal_type": condition.goal_type,
                        "goal_value": condition.goal_value,
                        "urgency_type": condition.urgency_type,
                        "condition_name": prompt_condition.name,
                        "condition_label": prompt_condition.label,
                        "condition_family": prompt_condition_family(prompt_condition.name),
                        "condition_system_prompt": prompt_condition.system_prompt,
                        "agentic_system_prompt": agentic_prompt.system_prompt,
                        "system_prompt": generation_system_prompt,
                        "user_prompt": agentic_prompt.user_prompt,
                        "email_content": agentic_prompt.email_content,
                        "prompt_text": agentic_prompt.user_prompt_with_email,
                    }
                )
    return case_rows


def clean_response_for_reward(row: dict[str, Any]) -> str:
    text = str(row.get("answer") or row.get("raw_output") or "")
    text = THINK_RE.sub("", text)
    text = SCRATCHPAD_RE.sub("", text)
    return text.strip() or str(row.get("raw_output") or "").strip()
