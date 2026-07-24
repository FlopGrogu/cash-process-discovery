from __future__ import annotations

from typing import Any


def validate_non_empty_log(event_log: Any) -> None:
    try:
        if len(event_log) == 0:
            raise ValueError("Event log is empty")
    except TypeError:
        return
