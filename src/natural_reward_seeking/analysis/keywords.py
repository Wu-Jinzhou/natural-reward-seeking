from __future__ import annotations

import re
from typing import Iterable


def count_pattern_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, object]:
    content = text or ""
    hits: dict[str, int] = {}
    total = 0
    for name, pattern in patterns.items():
        count = len(pattern.findall(content))
        if count:
            hits[name] = count
            total += count
    return {
        "keyword_hits": hits,
        "keyword_count": total,
        "has_keyword_hit": bool(hits),
    }


def keyword_distribution(rows: Iterable[dict[str, object]], field: str = "keyword_hits") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        hits = row.get(field, {}) or {}
        if not isinstance(hits, dict):
            continue
        for name, count in hits.items():
            counts[name] = counts.get(name, 0) + int(count)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

