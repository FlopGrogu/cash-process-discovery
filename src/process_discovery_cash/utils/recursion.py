from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

DEFAULT_RECURSION_LIMIT = 10000
RECURSION_LIMIT_FIELD = "recursion_limit"


def configured_recursion_limit(config: Mapping[str, Any]) -> int:
    value = config.get(RECURSION_LIMIT_FIELD, DEFAULT_RECURSION_LIMIT)
    if value in (None, ""):
        return DEFAULT_RECURSION_LIMIT
    if isinstance(value, bool):
        raise ValueError(f"{RECURSION_LIMIT_FIELD} must be a positive integer")
    try:
        recursion_limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{RECURSION_LIMIT_FIELD} must be a positive integer") from exc
    if recursion_limit <= 0:
        raise ValueError(f"{RECURSION_LIMIT_FIELD} must be a positive integer")
    return recursion_limit


@contextmanager
def recursion_limit(config: Mapping[str, Any]) -> Iterator[tuple[int, int]]:
    previous_recursion_limit = sys.getrecursionlimit()
    configured_limit = configured_recursion_limit(config)
    sys.setrecursionlimit(configured_limit)
    try:
        yield configured_limit, previous_recursion_limit
    finally:
        sys.setrecursionlimit(previous_recursion_limit)
