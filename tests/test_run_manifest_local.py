from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from process_discovery_cash.cli import run_manifest_local as cli
from process_discovery_cash.experiments import local_manifest_runner as local_runner
from process_discovery_cash.experiments.local_manifest_runner import (
    ManifestFilters,
    load_indexed_manifest_rows,
    run_local_manifest,
    run_local_manifest_row,
    select_manifest_rows,
)


def test_loading_small_manifest_selects_all_rows(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="inductive_miner", variant="im"),
        ],
    )

    rows = load_indexed_manifest_rows(manifest_path)
    selected = select_manifest_rows(rows, ManifestFilters())

    assert [row.row_index for row in selected] == [0, 1]


def test_algorithm_filter_selects_matching_rows(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="inductive_miner", variant="im"),
            _manifest_row(tmp_path, 2, algorithm="inductive_miner", variant="imf"),
        ],
    )

    rows = load_indexed_manifest_rows(manifest_path)
    selected = select_manifest_rows(rows, ManifestFilters(algorithm="inductive_miner"))

    assert [row.row_index for row in selected] == [1, 2]


def test_row_index_selects_exactly_one_row(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="ilp_miner"),
        ],
    )

    rows = load_indexed_manifest_rows(manifest_path)
    selected = select_manifest_rows(rows, ManifestFilters(row_indices={1}))

    assert len(selected) == 1
    assert selected[0].row["algorithm_id"] == "ilp_miner"


def test_max_rows_limits_after_filtering(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="inductive_miner", variant="im"),
            _manifest_row(tmp_path, 2, algorithm="inductive_miner", variant="imf"),
        ],
    )

    rows = load_indexed_manifest_rows(manifest_path)
    selected = select_manifest_rows(
        rows,
        ManifestFilters(algorithm="inductive_miner", max_rows=1),
    )

    assert [row.row_index for row in selected] == [1]


def test_dry_run_does_not_execute_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [_manifest_row(tmp_path, 0, algorithm="genetic_miner")],
    )
    monkeypatch.setattr(
        cli,
        "run_local_manifest",
        lambda *_args, **_kwargs: _raise("dry-run should not execute rows"),
    )

    cli.main(["--manifest", manifest_path.as_posix(), "--dry-run"])

    output = capsys.readouterr().out
    assert "algorithm=genetic_miner" in output
    assert "Matching rows: 1" in output


def test_existing_success_result_skips_unless_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [_manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic")],
    )
    indexed = load_indexed_manifest_rows(manifest_path)[0]
    output_path = Path(indexed.row["output_path"])
    _write_success_result(output_path, indexed.row)
    calls: list[dict[str, str]] = []

    def fake_run(
        row: dict[str, str],
        command_args: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        calls.append(row)
        _write_result(Path(row["output_path"]), "success")
        return Path(row["output_path"])

    monkeypatch.setattr(local_runner, "_run_manifest_row", fake_run)

    skipped = run_local_manifest_row(indexed)
    forced = run_local_manifest_row(indexed, force=True)

    assert skipped.status == "skipped"
    assert forced.status == "success"
    assert len(calls) == 1


def test_existing_success_is_skipped_with_higher_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = _manifest_row(tmp_path, 0, algorithm="inductive_miner", variant="im")
    row["algorithm_params_json"] = json.dumps(
        {"variant": "im", "discovery_timeout_seconds": 60},
        sort_keys=True,
    )
    row["params_json"] = row["algorithm_params_json"]
    output_path = Path(row["output_path"])
    _write_success_result(output_path, row)
    indexed = local_runner.IndexedManifestRow(row_index=0, row=row)
    monkeypatch.setattr(
        local_runner,
        "_run_manifest_row",
        lambda *_args, **_kwargs: _raise("successful stable result should be skipped"),
    )

    result = run_local_manifest_row(indexed)

    assert result.status == "skipped"


@pytest.mark.parametrize("existing_status", ["failed", "timeout"])
def test_failed_or_timeout_result_is_rerun_with_current_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing_status: str,
) -> None:
    row = _manifest_row(tmp_path, 0, algorithm="inductive_miner", variant="im")
    row["algorithm_params_json"] = json.dumps(
        {"variant": "im", "discovery_timeout_seconds": 60},
        sort_keys=True,
    )
    row["params_json"] = row["algorithm_params_json"]
    output_path = Path(row["output_path"])
    _write_result(output_path, existing_status)
    indexed = local_runner.IndexedManifestRow(row_index=0, row=row)
    observed_timeouts: list[int] = []

    def fake_run(
        current_row: dict[str, str],
        command_args: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        params = json.loads(current_row["params_json"])
        observed_timeouts.append(params["discovery_timeout_seconds"])
        _write_result(Path(current_row["output_path"]), "success")
        return Path(current_row["output_path"])

    monkeypatch.setattr(local_runner, "_run_manifest_row", fake_run)

    result = run_local_manifest_row(indexed)

    assert result.status == "success"
    assert observed_timeouts == [60]


def test_failed_row_is_recorded_in_status_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [_manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic")],
    )
    rows = load_indexed_manifest_rows(manifest_path)

    def fake_run(
        row: dict[str, str],
        command_args: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        output_path = Path(row["output_path"])
        _write_result(output_path, "failed", error_message="boom")
        return output_path

    monkeypatch.setattr(local_runner, "_run_manifest_row", fake_run)
    status_path = tmp_path / "status.csv"

    results = run_local_manifest(rows, status_path=status_path)

    status_rows = list(csv.DictReader(status_path.open(encoding="utf-8")))
    assert results[0].status == "failed"
    assert status_rows[0]["status"] == "failed"
    assert status_rows[0]["error"] == "boom"
    assert (
        Path(rows[0].row["output_path"]).with_suffix(".error.txt").read_text(encoding="utf-8")
        == "boom\n"
    )


def test_strict_stops_on_first_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="ilp_miner"),
        ],
    )
    rows = load_indexed_manifest_rows(manifest_path)
    calls: list[str] = []

    def fake_run(
        row: dict[str, str],
        command_args: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        calls.append(row["config_id"])
        output_path = Path(row["output_path"])
        _write_result(output_path, "failed", error_message="boom")
        return output_path

    monkeypatch.setattr(local_runner, "_run_manifest_row", fake_run)

    results = run_local_manifest(rows, status_path=tmp_path / "status.csv", strict=True)

    assert [result.row_index for result in results] == [0]
    assert calls == ["cfg_0"]


def test_continue_on_error_records_multiple_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic"),
            _manifest_row(tmp_path, 1, algorithm="ilp_miner"),
        ],
    )
    rows = load_indexed_manifest_rows(manifest_path)

    def fake_run(
        row: dict[str, str],
        command_args: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        output_path = Path(row["output_path"])
        _write_result(output_path, "failed", error_message=f"failed {row['config_id']}")
        return output_path

    monkeypatch.setattr(local_runner, "_run_manifest_row", fake_run)
    status_path = tmp_path / "status.csv"

    results = run_local_manifest(rows, status_path=status_path)

    assert [result.status for result in results] == ["failed", "failed"]
    assert len(list(csv.DictReader(status_path.open(encoding="utf-8")))) == 2


def test_missing_required_manifest_columns_fail_clearly(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["log_id", "seed", "algorithm_id", "output_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "log_id": "tiny_log",
                "seed": "0",
                "algorithm_id": "alpha_miner",
                "output_path": str(tmp_path / "result.json"),
            }
        )

    with pytest.raises(ValueError, match="log_path"):
        load_indexed_manifest_rows(manifest_path)


def test_unhashed_manifest_identity_ignores_log_directory(tmp_path: Path) -> None:
    first_row = _manifest_row(tmp_path, 0, algorithm="alpha_miner", variant="classic")
    second_row = dict(first_row)
    for row in [first_row, second_row]:
        row.pop("config_id")
        row.pop("config_hash")
    first_row["log_dir"] = "logs/slurm/first"
    second_row["log_dir"] = "logs/slurm/second"

    first_manifest = _write_manifest(tmp_path / "first.csv", [first_row])
    second_manifest = _write_manifest(tmp_path / "second.csv", [second_row])
    normalized_first = load_indexed_manifest_rows(first_manifest)[0].row
    normalized_second = load_indexed_manifest_rows(second_manifest)[0].row

    assert normalized_first["config_hash"] == normalized_second["config_hash"]


def test_strict_and_continue_on_error_conflict_fails_cli() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--manifest", "missing.csv", "--strict", "--continue-on-error"])

    assert excinfo.value.code == 2


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _manifest_row(
    tmp_path: Path,
    index: int,
    *,
    algorithm: str,
    variant: str = "",
) -> dict[str, str]:
    params = {"variant": variant} if variant else {}
    params_json = json.dumps(params, sort_keys=True)
    return {
        "experiment_id": "test_experiment",
        "log_id": "tiny_log",
        "log_path": "data/example/tiny_log.xes",
        "seed": "0",
        "algorithm_id": algorithm,
        "algorithm_variant": variant,
        "algorithm": algorithm,
        "backend": "pm4py",
        "algorithm_params_json": params_json,
        "params_json": params_json,
        "config_id": f"cfg_{index}",
        "config_hash": f"cfg_{index}",
        "output_path": str(tmp_path / f"result_{index}.json"),
    }


def _write_result(path: Path, status: str, error_message: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": status, "error_message": error_message}) + "\n",
        encoding="utf-8",
    )


def _write_success_result(path: Path, row: dict[str, str]) -> None:
    params = json.loads(row.get("algorithm_params_json") or row.get("params_json") or "{}")
    payload = {
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row.get("test_log_path") or row["log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row.get("algorithm_id") or row["algorithm"],
        "backend": row["backend"],
        "hyperparameters": params,
        "discovered_model_type": "unknown",
        "metrics": {},
        "metric_statuses": {},
        "status": "success",
        "metadata": {"config_hash": row["config_hash"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _raise(message: str) -> None:
    raise AssertionError(message)
