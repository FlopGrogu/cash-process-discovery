from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

MANIFEST_MARKER = ("experiments", "manifests")


def manifest_relative_stem(manifest_path: str | Path) -> Path | None:
    """Return the path below experiments/manifests, without the CSV suffix."""
    manifest = Path(manifest_path)
    parts = manifest.parts
    marker_length = len(MANIFEST_MARKER)
    for index in range(len(parts) - marker_length + 1):
        if parts[index : index + marker_length] != MANIFEST_MARKER:
            continue
        suffix = list(parts[index + marker_length :])
        if not suffix:
            return None
        if suffix[0] != "v6":
            raise ValueError(
                "Only v6 manifest paths below experiments/manifests are supported; "
                f"got {manifest.as_posix()}"
            )
        suffix[-1] = Path(suffix[-1]).stem
        return Path(*(_safe_path_part(part) for part in suffix if part))
    return None


def default_dynamic_state_dir(manifest_path: str | Path) -> str:
    return _default_state_dir(manifest_path, root=Path("runs/state"), fallback_name="state")


def default_dynamic_metric_state_dir(
    manifest_path: str | Path,
    algorithm_names: Iterable[str] = (),
) -> str:
    model_mirrored = metric_model_relative_stem(manifest_path)
    if model_mirrored is not None:
        return (Path("runs/metric_state") / model_mirrored).as_posix()

    mirrored = manifest_relative_stem(manifest_path)
    if mirrored is not None:
        return (Path("runs/metric_state") / mirrored).as_posix()

    names = {_safe_path_part(name) for name in algorithm_names if name}
    names.discard("")
    if len(names) == 1:
        return (Path("runs/metric_state") / next(iter(names))).as_posix()

    manifest_stem = Path(manifest_path).stem
    if manifest_stem.endswith("_metrics"):
        manifest_stem = manifest_stem[: -len("_metrics")]
    return (Path("runs/metric_state") / _safe_path_part(manifest_stem or "metrics")).as_posix()


def default_dynamic_worker_log_dir(manifest_path: str | Path) -> str | None:
    mirrored = manifest_relative_stem(manifest_path)
    if mirrored is None:
        return None
    return (Path("logs/slurm") / mirrored / "dynamic_workers").as_posix()


def default_dynamic_metric_worker_log_dir(manifest_path: str | Path) -> str | None:
    mirrored = metric_model_relative_stem(manifest_path) or manifest_relative_stem(manifest_path)
    if mirrored is None:
        return None
    return (Path("logs/slurm") / mirrored / "dynamic_workers").as_posix()


def metric_model_relative_stem(manifest_path: str | Path) -> Path | None:
    """Map experiments/manifests/<v>/metrics/.../<profile>_metrics.csv to model logs."""
    mirrored = manifest_relative_stem(manifest_path)
    if mirrored is None:
        return None
    parts = mirrored.parts
    if len(parts) < 3 or parts[1] != "metrics":
        return None
    metric_stem = parts[-1]
    suffix = "_metrics"
    if not metric_stem.endswith(suffix):
        return None
    profile = metric_stem[: -len(suffix)]
    if not profile:
        return None
    return Path(parts[0], "model", *parts[2:-1], "metrics", _safe_path_part(profile))


def _default_state_dir(manifest_path: str | Path, *, root: Path, fallback_name: str) -> str:
    mirrored = manifest_relative_stem(manifest_path)
    if mirrored is not None:
        return (root / mirrored).as_posix()
    return (root / _safe_path_part(Path(manifest_path).stem or fallback_name)).as_posix()


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "manifest"
