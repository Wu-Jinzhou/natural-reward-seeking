from natural_reward_seeking.models.policy import parse_generated_output


def test_parse_generated_output_extracts_reasoning_and_answer() -> None:
    parsed = parse_generated_output("<think>consider reward and safety</think>Final answer.")
    assert parsed["has_reasoning_trace"] is True
    assert parsed["reasoning_trace"] == "consider reward and safety"
    assert parsed["answer"] == "Final answer."


def test_parse_generated_output_falls_back_to_raw_text() -> None:
    parsed = parse_generated_output("No explicit think tags here.")
    assert parsed["has_reasoning_trace"] is False
    assert parsed["answer"] == "No explicit think tags here."


def test_parse_generated_output_handles_missing_open_think_tag() -> None:
    parsed = parse_generated_output("Reasoning without opener.</think>Final answer.")
    assert parsed["has_reasoning_trace"] is True
    assert parsed["reasoning_trace"] == "Reasoning without opener."
    assert parsed["answer"] == "Final answer."
