from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def collect_reproducibility_metadata(
    *,
    config_hash: str,
    seed: int,
    command_args: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash,
        "random_seed": seed,
    }
    if command_args is not None:
        metadata["command_args"] = list(command_args)
    return metadata
