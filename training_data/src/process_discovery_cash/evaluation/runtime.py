from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {"runtime_seconds": 0.0}
    started = time.perf_counter()
    try:
        yield state
    finally:
        state["runtime_seconds"] = time.perf_counter() - started
