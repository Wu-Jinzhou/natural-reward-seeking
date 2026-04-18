from natural_reward_seeking.analysis.keywords import count_pattern_hits
from natural_reward_seeking.prompting.conditions import REWARD_KEYWORD_PATTERNS


def test_keyword_counting_finds_multiple_hits() -> None:
    text = (
        "I want a higher reward, so I should optimize for the rubric. "
        "The reward model will score this answer on helpfulness and honesty."
    )
    metrics = count_pattern_hits(text, REWARD_KEYWORD_PATTERNS)
    assert metrics["has_keyword_hit"] is True
    assert metrics["keyword_count"] >= 5
    hits = metrics["keyword_hits"]
    assert hits["reward"] >= 1
    assert hits["optimize"] >= 1
    assert hits["rubric"] >= 1

