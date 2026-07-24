import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from process_discovery_cash.discovery import split as split_module
from process_discovery_cash.discovery.external_backend import _subprocess_text, run_command
from process_discovery_cash.discovery.split import (
    DEFAULT_SPLIT_MINER_JAVA_OPTIONS,
    SplitMiner,
    _resolve_java_bin,
    _split_miner_error_message,
    build_split_miner_command,
)


def _write_xes(path: Path, lifecycle_values: list[str | None]) -> None:
    events = []
    for index, lifecycle_value in enumerate(lifecycle_values):
        lifecycle = (
            ""
            if lifecycle_value is None
            else f'<string key="lifecycle:transition" value="{lifecycle_value}" />'
        )
        events.append(
            f'<event><string key="concept:name" value="event-{index}" />{lifecycle}</event>'
        )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8" ?>'
        '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">'
        f"<trace>{''.join(events)}</trace></log>",
        encoding="utf-8",
    )


def _lifecycle_values(path: Path) -> list[str | None]:
    root = ET.parse(path).getroot()
    values = []
    for event in root.iter("{http://www.xes-standard.org/}event"):
        values.append(
            next(
                (
                    child.attrib.get("value")
                    for child in event
                    if child.attrib.get("key") == "lifecycle:transition"
                ),
                None,
            )
        )
    return values


def test_split_miner_v1_discovers_from_unmodified_input(monkeypatch, tmp_path: Path) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_text("jar placeholder", encoding="utf-8")
    source = tmp_path / "input.xes"
    _write_xes(source, [None, "start"])
    captured: dict[str, object] = {}

    def fake_run_command(command, timeout_seconds, cwd):
        input_path = Path(command[command.index("--logPath") + 1])
        captured["input_path"] = input_path
        captured["lifecycle_values"] = _lifecycle_values(input_path)
        output_path = Path(command[command.index("--outputPath") + 1])
        output_path.write_text("<definitions />\n", encoding="utf-8")
        return 0, "ok", "", 0.01, False

    monkeypatch.setattr(split_module, "run_command", fake_run_command)
    monkeypatch.setattr(split_module, "_load_bpmn_model", lambda _path: (None, None))

    result = SplitMiner().discover(
        [],
        {
            "jar_path": jar.as_posix(),
            "input_log_path": source.as_posix(),
            "output_dir": (tmp_path / "out").as_posix(),
            "keep_output_files": True,
        },
    )

    assert result.status == "success"
    assert Path(captured["input_path"]).name == source.name
    assert captured["lifecycle_values"] == [None, "start"]
    assert "lifecycle_transitions_added" not in result.metadata
    assert result.warnings == []


def test_split_miner_handles_missing_jar_gracefully(monkeypatch) -> None:
    monkeypatch.delenv("SPLIT_MINER_JAR", raising=False)

    result = SplitMiner().discover([], {"jar_path": None, "epsilon": 0.1, "eta": 0.4})

    assert result.status == "unsupported"
    assert "JAR path is missing" in (result.error_message or "")


def test_split_miner_rejects_mismatched_jar_checksum(tmp_path: Path) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_bytes(b"not the approved Split Miner artifact")

    result = SplitMiner().discover(
        [],
        {
            "jar_path": jar.as_posix(),
            "jar_sha256": "0" * 64,
        },
    )

    assert result.status == "unsupported"
    assert "SHA-256 mismatch" in (result.error_message or "")
    assert hashlib.sha256(jar.read_bytes()).hexdigest() in (result.error_message or "")


def test_split_miner_v1_command_uses_upstream_cli_flags(tmp_path: Path) -> None:
    jar = tmp_path / "split-miner.jar"
    input_xes = tmp_path / "input.xes"
    output_model = tmp_path / "model.bpmn"

    command, warnings = build_split_miner_command(
        java_bin="java",
        jar=jar,
        input_xes=input_xes,
        output_model=output_model,
        config={
            "eta": 0.4,
            "epsilon": 0.1,
            "parallelismFirst": True,
            "removeLoopActivityMarkers": True,
            "replaceIORs": True,
            "diagram": True,
        },
    )

    assert warnings == []
    assert command == [
        "java",
        *DEFAULT_SPLIT_MINER_JAVA_OPTIONS,
        "-jar",
        str(jar),
        "--logPath",
        str(input_xes),
        "--outputPath",
        str(output_model),
        "--epsilon",
        "0.1",
        "--diagram",
        "--eta",
        "0.4",
        "--parallelismFirst",
        "--removeLoopActivityMarkers",
        "--replaceIORs",
    ]
    assert "--input" not in command
    assert "--output" not in command
    assert command.index("-jar") > 1


def test_split_miner_command_allows_explicit_java_options(tmp_path: Path) -> None:
    jar = tmp_path / "split-miner.jar"
    input_xes = tmp_path / "input.xes"
    output_model = tmp_path / "model.bpmn"

    command, warnings = build_split_miner_command(
        java_bin="java",
        jar=jar,
        input_xes=input_xes,
        output_model=output_model,
        config={"java_options": ["-Xmx4g", "-Djava.awt.headless=true"]},
    )

    assert warnings == []
    assert command[:4] == ["java", "-Xmx4g", "-Djava.awt.headless=true", "-jar"]


def test_split_miner_discover_loads_bpmn_and_cleans_output_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_text("jar placeholder", encoding="utf-8")
    (tmp_path / "input.xes").write_text("<log />\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run_command(command, timeout_seconds, cwd):
        captured["command"] = command
        assert timeout_seconds == 30
        assert Path(cwd).exists()
        output_path = Path(command[command.index("--outputPath") + 1])
        output_path.write_text("<definitions />\n", encoding="utf-8")
        return 0, "ok", "", 0.01, False

    monkeypatch.setattr(split_module, "run_command", fake_run_command)
    monkeypatch.setattr(
        split_module,
        "_load_bpmn_model",
        lambda _path: ("loaded-bpmn", None),
    )

    result = SplitMiner().discover(
        [],
        {
            "jar_path": jar.as_posix(),
            "input_log_path": "data/example/tiny_log.xes",
            "output_dir": (tmp_path / "out").as_posix(),
            "timeout_seconds": 30,
            "epsilon": 0.1,
            "eta": 0.4,
        },
    )

    assert result.status == "success"
    assert result.model_type == "bpmn"
    assert result.discovered_model == "loaded-bpmn"
    assert result.model_path is None
    assert result.metadata["output_files_cleaned"] is True
    assert not (tmp_path / "out").exists()
    assert captured["command"][0] == "java"
    assert captured["command"][1 : 1 + len(DEFAULT_SPLIT_MINER_JAVA_OPTIONS)] == (
        DEFAULT_SPLIT_MINER_JAVA_OPTIONS
    )
    assert captured["command"][captured["command"].index("-jar") + 1] == jar.as_posix()


def test_split_miner_can_keep_output_files_for_debugging(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_text("jar placeholder", encoding="utf-8")
    (tmp_path / "input.xes").write_text("<log />\n", encoding="utf-8")

    def fake_run_command(command, timeout_seconds, cwd):
        assert timeout_seconds == 86400
        assert Path(cwd).exists()
        output_path = Path(command[command.index("--outputPath") + 1])
        output_path.write_text("<definitions />\n", encoding="utf-8")
        return 0, "ok", "", 0.01, False

    monkeypatch.setattr(split_module, "run_command", fake_run_command)
    monkeypatch.setattr(
        split_module,
        "_load_bpmn_model",
        lambda _path: ("loaded-bpmn", None),
    )

    result = SplitMiner().discover(
        [],
        {
            "jar_path": jar.as_posix(),
            "input_log_path": "data/example/tiny_log.xes",
            "output_dir": (tmp_path / "out").as_posix(),
            "keep_output_files": True,
        },
    )

    assert result.status == "success"
    assert result.model_path is not None
    assert Path(result.model_path).exists()
    assert "output_files_cleaned" not in result.metadata


def test_split_miner_resolves_relative_jar_and_output_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_text("jar placeholder", encoding="utf-8")
    (tmp_path / "input.xes").write_text("<log />\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run_command(command, timeout_seconds, cwd):
        captured["command"] = command
        output_path = Path(command[command.index("--outputPath") + 1])
        output_path.write_text("<definitions />\n", encoding="utf-8")
        return 0, "ok", "", 0.01, False

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(split_module, "run_command", fake_run_command)
    monkeypatch.setattr(
        split_module,
        "_load_bpmn_model",
        lambda _path: ("loaded-bpmn", None),
    )

    result = SplitMiner().discover(
        [],
        {
            "jar_path": "split-miner.jar",
            "input_log_path": "input.xes",
            "output_dir": "results/split_row",
            "epsilon": 0.1,
            "eta": 0.4,
        },
    )

    assert result.status == "success"
    assert Path(captured["command"][captured["command"].index("-jar") + 1]).is_absolute()
    assert Path(captured["command"][captured["command"].index("--outputPath") + 1]).is_absolute()


def test_split_miner_resolves_relative_java_bin_from_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    jar = tmp_path / "split-miner.jar"
    jar.write_text("jar placeholder", encoding="utf-8")
    java_bin = tmp_path / "data" / "external" / "jdk1.8.0_202" / "bin" / "java"
    java_bin.parent.mkdir(parents=True)
    java_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "input.xes").write_text("<log />\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run_command(command, timeout_seconds, cwd):
        captured["command"] = command
        output_path = Path(command[command.index("--outputPath") + 1])
        output_path.write_text("<definitions />\n", encoding="utf-8")
        return 0, "ok", "", 0.01, False

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(split_module, "run_command", fake_run_command)
    monkeypatch.setattr(
        split_module,
        "_load_bpmn_model",
        lambda _path: ("loaded-bpmn", None),
    )

    result = SplitMiner().discover(
        [],
        {
            "jar_path": "split-miner.jar",
            "java_bin": "data/external/jdk1.8.0_202/bin/java",
            "input_log_path": "input.xes",
            "output_dir": "results/split_row",
        },
    )

    assert result.status == "success"
    assert captured["command"][0] == java_bin.as_posix()


def test_resolve_java_bin_leaves_plain_command_for_path_lookup() -> None:
    assert _resolve_java_bin("java") == "java"


def test_split_miner_jaxb_failure_has_actionable_error() -> None:
    message = _split_miner_error_message(1, "SCHWERWIEGEND: javax/xml/bind/DatatypeConverter")

    assert "JAXB" in message
    assert "Java 8" in message


def test_split_miner_missing_java_failure_has_actionable_error() -> None:
    message = _split_miner_error_message(
        127,
        "FileNotFoundError: [Errno 2] No such file or directory: 'java'",
    )

    assert "could not start Java" in message
    assert "JAVA_BIN" in message


def test_split_miner_native_memory_failure_is_classified() -> None:
    message = _split_miner_error_message(
        1,
        stdout=(
            "# There is insufficient memory for the Java Runtime Environment to continue.\n"
            "# Native memory allocation (malloc) failed to allocate 32744 bytes"
        ),
    )

    assert "native memory" in message
    assert "java_options" in message


def test_split_miner_metaspace_failure_is_classified() -> None:
    message = _split_miner_error_message(
        1,
        stdout="Error occurred during initialization of VM\nCould not allocate metaspace",
    )

    assert "metaspace" in message


def test_split_miner_shared_object_mapping_failure_is_classified() -> None:
    message = _split_miner_error_message(
        1,
        stderr="libawt_headless.so: failed to map segment from shared object",
    )

    assert "shared object" in message


def test_run_command_normalizes_missing_executable(tmp_path: Path) -> None:
    return_code, stdout, stderr, _runtime_seconds, timed_out = run_command(
        ["definitely-not-a-real-pdcash-executable"],
        timeout_seconds=1,
        cwd=tmp_path,
    )

    assert return_code == 127
    assert stdout == ""
    assert "FileNotFoundError" in stderr
    assert timed_out is False


def test_subprocess_timeout_output_is_normalized_to_text() -> None:
    assert _subprocess_text(None) == ""
    assert _subprocess_text("plain text") == "plain text"
    assert _subprocess_text(b"partial output\xff") == "partial output\ufffd"
