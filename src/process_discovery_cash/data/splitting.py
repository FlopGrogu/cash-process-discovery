from __future__ import annotations

from typing import Any


def split_log_for_experiment(
    event_log: Any,
    split_id: str | None = None,
    seed: int | None = None,
) -> tuple[Any, Any]:
    """Return the full log for both training and evaluation.

    The split_id and seed parameters are accepted only for older callers. Current
    experiment execution loads explicit full train/test logs and does not create
    train/test partitions or cross-validation folds.
    """
    return event_log, event_log
