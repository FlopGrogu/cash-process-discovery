from __future__ import annotations

import os
from pathlib import Path

_DOTENV_LOADED = False
_DOTENV_VALUES: dict[str, str] = {}


def load_dotenv_if_present(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""
    global _DOTENV_LOADED, _DOTENV_VALUES
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    _DOTENV_VALUES = {}

    dotenv_path = Path(path) if path is not None else _discover_repo_root() / ".env"
    if not dotenv_path.exists():
        return
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        _DOTENV_VALUES[key] = _strip_dotenv_quotes(value.strip())


def _strip_dotenv_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _discover_repo_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def project_root() -> Path:
    load_dotenv_if_present()
    env_root = os.getenv("PROJECT_ROOT") or _DOTENV_VALUES.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return _discover_repo_root()


def data_root() -> Path:
    load_dotenv_if_present()
    env_root = os.getenv("DATA_ROOT") or _DOTENV_VALUES.get("DATA_ROOT")
    return Path(env_root).expanduser().resolve() if env_root else project_root() / "data"


def results_root() -> Path:
    load_dotenv_if_present()
    env_root = os.getenv("RESULTS_ROOT") or _DOTENV_VALUES.get("RESULTS_ROOT")
    return Path(env_root).expanduser().resolve() if env_root else project_root() / "results"


def log_root() -> Path:
    load_dotenv_if_present()
    env_root = os.getenv("LOG_ROOT") or _DOTENV_VALUES.get("LOG_ROOT")
    return Path(env_root).expanduser().resolve() if env_root else project_root() / "logs/slurm"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return _resolve_namespace_path(path)


def portable_project_path(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve(strict=False)
    namespace_roots = (
        ("data", data_root()),
        ("results", results_root()),
        ("logs/slurm", log_root()),
        ("", project_root()),
    )
    for prefix, root in namespace_roots:
        try:
            relative = resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        return (Path(prefix) / relative).as_posix() if prefix else relative.as_posix()
    return resolved.as_posix()


def resolve_portable_path(path: str | Path, *, base_path: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    namespace_candidate = _resolve_namespace_path(path)
    if namespace_candidate.exists():
        return namespace_candidate

    if base_path is not None:
        base_candidate = Path(base_path).parent / path
        if base_candidate.exists():
            return base_candidate
    return namespace_candidate


def _resolve_namespace_path(path: Path) -> Path:
    parts = path.parts
    if parts and parts[0] == "data":
        return data_root().joinpath(*parts[1:])
    if parts and parts[0] == "results":
        return results_root().joinpath(*parts[1:])
    if len(parts) >= 2 and parts[:2] == ("logs", "slurm"):
        return log_root().joinpath(*parts[2:])
    return project_root() / path


def ensure_parent(path: str | Path) -> Path:
    path = resolve_portable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
