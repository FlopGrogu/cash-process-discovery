from __future__ import annotations

import os
import re
import resource
import sys
from typing import Any


def current_peak_rss_bytes() -> int | None:
    peaks = [_rusage_peak_rss_bytes(resource.RUSAGE_SELF)]
    if hasattr(resource, "RUSAGE_CHILDREN"):
        peaks.append(_rusage_peak_rss_bytes(resource.RUSAGE_CHILDREN))
    values = [value for value in peaks if value is not None]
    if not values:
        return None
    return max(values)


def attach_resource_metadata(
    metadata: dict[str, Any],
    start_peak_rss_bytes: int | None,
) -> None:
    metadata["memory"] = memory_metadata(start_peak_rss_bytes)
    slurm = slurm_metadata()
    if slurm:
        metadata["slurm"] = slurm


def memory_metadata(start_peak_rss_bytes: int | None) -> dict[str, Any]:
    end_peak_rss_bytes = current_peak_rss_bytes()
    peak_values = [
        value for value in [start_peak_rss_bytes, end_peak_rss_bytes] if value is not None
    ]
    peak_rss_bytes = max(peak_values) if peak_values else 0
    payload: dict[str, Any] = {
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mb": round(peak_rss_bytes / 1024 / 1024, 2),
        "source": "resource.getrusage",
        "scope": "current_process_and_waited_children",
    }
    if start_peak_rss_bytes is not None:
        payload["start_peak_rss_bytes"] = start_peak_rss_bytes
    if end_peak_rss_bytes is not None:
        payload["end_peak_rss_bytes"] = end_peak_rss_bytes
    return payload


def slurm_metadata() -> dict[str, Any]:
    fields = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "job_partition": os.getenv("SLURM_JOB_PARTITION"),
        "job_name": os.getenv("SLURM_JOB_NAME"),
    }
    requested_memory = os.getenv("PDCASH_SLURM_REQUESTED_MEMORY")
    if requested_memory:
        fields["requested_memory"] = requested_memory
        requested_memory_bytes = parse_memory_bytes(requested_memory)
        if requested_memory_bytes is not None:
            fields["requested_memory_bytes"] = requested_memory_bytes
    else:
        slurm_memory = os.getenv("SLURM_MEM_PER_NODE") or os.getenv("SLURM_MEM_PER_CPU")
        if slurm_memory:
            fields["requested_memory"] = slurm_memory
            requested_memory_bytes = _parse_slurm_memory_bytes(slurm_memory)
            if requested_memory_bytes is not None:
                fields["requested_memory_bytes"] = requested_memory_bytes
    cpus_per_task = _parse_int_env("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        fields["requested_cpus_per_task"] = cpus_per_task
    return {key: value for key, value in fields.items() if value not in (None, "")}


def parse_memory_bytes(value: str) -> int | None:
    match = _MEMORY_PATTERN.match(value)
    if match is None:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    unit = unit[:-1] if unit.endswith("b") else unit
    unit = unit[:-1] if unit.endswith("i") else unit
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
        "p": 1024**5,
    }
    multiplier = multipliers.get(unit)
    if multiplier is None:
        return None
    return int(amount * multiplier)


def _rusage_peak_rss_bytes(who: int) -> int | None:
    try:
        max_rss = resource.getrusage(who).ru_maxrss
    except (OSError, ValueError):
        return None
    if max_rss < 0:
        return None
    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss * 1024)


def _parse_slurm_memory_bytes(value: str) -> int | None:
    if re.fullmatch(r"\s*\d+\s*", value):
        return int(value) * 1024**2
    return parse_memory_bytes(value)


def _parse_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


_MEMORY_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?i?b?|[kmgtp])?\s*$", re.I)
