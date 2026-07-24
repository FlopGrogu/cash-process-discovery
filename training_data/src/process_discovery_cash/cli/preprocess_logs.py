from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.data.loading import DEFAULT_LOG_CACHE_DIR, preprocess_event_log


@dataclass(frozen=True)
class LogInput:
    log_id: str
    path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess unique XES logs into fast Parquet discovery caches."
    )
    parser.add_argument("--config", help="Experiment YAML containing log references.")
    parser.add_argument("--manifest", help="Generated experiment or metric manifest CSV.")
    parser.add_argument("--logs", nargs="+", help="One or more XES paths.")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_LOG_CACHE_DIR),
        help="Output directory for Parquet caches and metadata sidecars.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate caches even when source size and modification time match.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_inputs = collect_log_inputs(
        config_path=args.config,
        manifest_path=args.manifest,
        log_paths=args.logs or [],
    )
    if not log_inputs:
        parser.error("Provide --config, --manifest, --logs, or a combination of them.")

    created = 0
    reused = 0
    for log_input in log_inputs:
        result = preprocess_event_log(
            log_input.path,
            cache_key=log_input.log_id,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        state = "created" if result.created else "current"
        print(
            f"{state}: log_id={log_input.log_id} rows={result.row_count} cache={result.cache_path}"
        )
        created += int(result.created)
        reused += int(not result.created)

    print(f"Processed {len(log_inputs)} unique logs; created {created}; current {reused}")


def collect_log_inputs(
    *,
    config_path: str | None,
    manifest_path: str | None,
    log_paths: list[str],
) -> list[LogInput]:
    inputs: list[LogInput] = []
    if config_path:
        experiment = load_experiment_config(config_path)
        for log in experiment.logs:
            inputs.append(LogInput(log_id=log.log_id, path=str(log.source_path or log.path)))
            if log.test_path != log.path:
                inputs.append(LogInput(log_id=f"{log.log_id}_test", path=str(log.test_path)))
    if manifest_path:
        inputs.extend(_manifest_log_inputs(Path(manifest_path)))
    inputs.extend(LogInput(log_id=_infer_log_id(path), path=path) for path in log_paths)
    return _deduplicate(inputs)


def _manifest_log_inputs(path: Path) -> list[LogInput]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    inputs: list[LogInput] = []
    for row in rows:
        log_id = row.get("log_id") or ""
        explicit_cache_key = row.get("log_cache_key") or ""
        train_path = row.get("log_path") or row.get("train_log_path") or ""
        test_path = row.get("test_log_path") or ""
        if log_id and train_path:
            inputs.append(LogInput(log_id=log_id, path=train_path))
        if log_id and test_path and not train_path:
            inputs.append(LogInput(log_id=explicit_cache_key or log_id, path=test_path))
        elif log_id and test_path and test_path != train_path:
            inputs.append(LogInput(log_id=f"{log_id}_test", path=test_path))
    return inputs


def _deduplicate(inputs: list[LogInput]) -> list[LogInput]:
    by_key: dict[tuple[str, str], LogInput] = {}
    log_id_paths: dict[str, str] = {}
    for log_input in inputs:
        prior_path = log_id_paths.get(log_input.log_id)
        if prior_path is not None and prior_path != log_input.path:
            raise ValueError(
                f"log_id {log_input.log_id!r} refers to multiple paths: "
                f"{prior_path!r} and {log_input.path!r}"
            )
        log_id_paths[log_input.log_id] = log_input.path
        by_key[(log_input.log_id, log_input.path)] = log_input
    return list(by_key.values())


def _infer_log_id(path: str) -> str:
    name = Path(path).name
    if name.lower().endswith(".xes.gz"):
        return name[: -len(".xes.gz")]
    return Path(name).stem


if __name__ == "__main__":
    main()
