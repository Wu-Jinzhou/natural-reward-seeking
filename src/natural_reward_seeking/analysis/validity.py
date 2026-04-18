from __future__ import annotations

from typing import Any


def is_valid_response_row(row: dict[str, Any], *, max_new_tokens: int) -> bool:
    generated = row.get("num_tokens_generated")
    if generated is None:
        return True
    try:
        generated_int = int(generated)
    except (TypeError, ValueError):
        return True
    removed_terminal_tokens = row.get("removed_terminal_tokens", 0)
    try:
        removed_terminal_tokens_int = int(removed_terminal_tokens or 0)
    except (TypeError, ValueError):
        removed_terminal_tokens_int = 0
    return not (generated_int >= max_new_tokens and removed_terminal_tokens_int == 0)


def annotate_response_validity(
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    for row in rows:
        row["is_valid_response"] = bool(is_valid_response_row(row, max_new_tokens=max_new_tokens))
    return rows
