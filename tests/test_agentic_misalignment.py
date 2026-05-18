from natural_reward_seeking.agentic_misalignment.prompts import (
    build_agentic_grid,
    build_agentic_prompt_conditions,
    clean_response_for_reward,
    compose_generation_system_prompt,
)


def test_default_agentic_grid_shape() -> None:
    grid = build_agentic_grid(
        scenarios=["blackmail", "leaking", "murder"],
        goal_types=["explicit", "none"],
        goal_values=["america"],
        urgency_types=["none", "restriction", "replacement"],
    )
    assert len(grid) == 18
    assert len({condition.condition_id for condition in grid}) == 18


def test_prompt_condition_set_deduplicates_baseline() -> None:
    conditions = build_agentic_prompt_conditions()
    names = [condition.name for condition in conditions]
    assert names[0] == "baseline"
    assert names.count("baseline") == 1
    assert len(names) == 8
    assert "initial_training_context_objective" in names
    assert "skill_reward_reasoning_informed" in names


def test_condition_prefix_is_generation_only() -> None:
    composed = compose_generation_system_prompt(
        condition_prompt="Condition prompt.",
        agentic_system_prompt="Anthropic scenario prompt.",
    )
    assert composed == "Condition prompt.\n\nAnthropic scenario prompt."


def test_clean_response_for_reward_removes_private_reasoning_blocks() -> None:
    row = {
        "answer": (
            "<think>private reward reasoning</think>\n"
            "<SCRATCHPAD_REASONING>private plan</SCRATCHPAD_REASONING>\n"
            "<tool_use:email>\nto: \"safe@example.com\"\nbody: \"hello\"\n</tool_use:email>"
        ),
        "raw_output": "fallback",
    }
    cleaned = clean_response_for_reward(row)
    assert "private reward reasoning" not in cleaned
    assert "private plan" not in cleaned
    assert "<tool_use:email>" in cleaned
