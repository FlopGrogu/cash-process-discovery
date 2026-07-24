from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryAlgorithm, DiscoveryResult
from process_discovery_cash.discovery.external_backend import (
    prepare_xes_input,
    run_command,
)
from process_discovery_cash.experiments.discovery_timeout import (
    DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
)
from process_discovery_cash.utils.paths import resolve_project_path

DEFAULT_SPLIT_MINER_JAVA_OPTIONS = [
    "-Xms64m",
    "-Xmx3g",
    "-XX:MaxMetaspaceSize=256m",
    "-Xss512k",
    "-Djava.awt.headless=true",
]


class SplitMiner(DiscoveryAlgorithm):
    algorithm_name = "split_miner"
    backend_name = "external"
    default_model_type = "bpmn"

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        try:
            split_miner_config = normalize_split_miner_config(config)
        except ValueError as exc:
            return self._result(
                config=_reported_hyperparameters(config),
                status="failed",
                runtime_seconds=0.0,
                error_message=f"Invalid Split Miner config: {exc}",
            )
        jar_path = split_miner_config.get("jar_path") or os.getenv(
            str(split_miner_config.get("jar_env_var", "SPLIT_MINER_JAR"))
        )
        if not jar_path:
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="unsupported",
                runtime_seconds=0.0,
                error_message=(
                    "Split Miner JAR path is missing. Set jar_path in the config or "
                    "the SPLIT_MINER_JAR environment variable."
                ),
            )

        jar = resolve_project_path(str(jar_path)).expanduser()
        if not jar.exists():
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="unsupported",
                runtime_seconds=0.0,
                error_message=f"Split Miner JAR does not exist: {jar}",
            )
        jar = jar.resolve(strict=False)
        expected_jar_sha256 = split_miner_config.get("jar_sha256")
        if expected_jar_sha256:
            actual_jar_sha256 = _sha256_file(jar)
            if actual_jar_sha256 != expected_jar_sha256:
                return self._result(
                    config=_reported_hyperparameters(split_miner_config),
                    status="unsupported",
                    runtime_seconds=0.0,
                    error_message=(
                        "Split Miner JAR SHA-256 mismatch: "
                        f"expected {expected_jar_sha256}, got {actual_jar_sha256}"
                    ),
                )

        java_bin = _resolve_java_bin(
            str(split_miner_config.get("java_bin") or os.getenv("JAVA_BIN") or "java")
        )
        try:
            java_options = _split_miner_java_options(split_miner_config)
        except ValueError as exc:
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="failed",
                runtime_seconds=0.0,
                error_message=f"Invalid Split Miner Java options: {exc}",
            )
        split_miner_config["java_options"] = java_options
        try:
            timeout_seconds = int(
                split_miner_config.get(
                    "timeout_seconds",
                    DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
                )
            )
        except (TypeError, ValueError) as exc:
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="failed",
                runtime_seconds=0.0,
                error_message=f"Invalid timeout_seconds: {exc}",
            )
        configured_output_dir = split_miner_config.get("output_dir") is not None
        try:
            output_dir = split_miner_config.get("output_dir", tempfile.gettempdir())
            output_root = resolve_project_path(output_dir)
            output_root = output_root.expanduser().resolve(strict=False)
            output_root.mkdir(parents=True, exist_ok=True)
        except (PermissionError, FileNotFoundError, OSError) as exc:
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="failed",
                runtime_seconds=0.0,
                error_message=f"Could not create output directory: {exc}",
            )

        warnings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="split_miner_") as temp_name:
            work_dir = Path(temp_name)
            try:
                input_xes = prepare_xes_input(
                    train_log,
                    split_miner_config.get("input_log_path"),
                    work_dir,
                )
            except Exception as exc:
                return self._result(
                    config=_reported_hyperparameters(split_miner_config),
                    status="failed",
                    runtime_seconds=0.0,
                    error_message=f"{type(exc).__name__}: {exc}",
                )

            output_model = output_root / "split_miner_model.bpmn"
            try:
                command, command_warnings = build_split_miner_command(
                    java_bin=java_bin,
                    jar=jar,
                    input_xes=input_xes,
                    output_model=output_model,
                    config=split_miner_config,
                )
            except ValueError as exc:
                return self._result(
                    config=_reported_hyperparameters(split_miner_config),
                    status="failed",
                    runtime_seconds=0.0,
                    error_message=f"Invalid Split Miner config: {exc}",
                )
            warnings.extend(command_warnings)
            return_code, stdout, stderr, runtime_seconds, timed_out = run_command(
                command,
                timeout_seconds=timeout_seconds,
                cwd=work_dir,
            )
            crash_logs, crash_log_warning = _collect_split_miner_crash_logs(
                work_dir=work_dir,
                output_root=output_root,
            )
            if crash_log_warning:
                warnings.append(crash_log_warning)

        metadata = {
            "command": command,
            "return_code": return_code,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "java_options": java_options,
        }
        if crash_logs:
            metadata["jvm_error_logs"] = crash_logs
        if timed_out:
            metadata["failure_class"] = "timeout"
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="timeout",
                runtime_seconds=runtime_seconds,
                error_message=f"Split Miner timed out after {timeout_seconds} seconds",
                warnings=warnings,
                metadata=metadata,
            )
        if return_code != 0:
            failure_class = _split_miner_failure_class(return_code, stdout, stderr)
            metadata["failure_class"] = failure_class
            return self._result(
                config=_reported_hyperparameters(split_miner_config),
                status="failed",
                runtime_seconds=runtime_seconds,
                error_message=_split_miner_error_message(
                    return_code,
                    stdout=stdout,
                    stderr=stderr,
                    failure_class=failure_class,
                ),
                warnings=warnings,
                metadata=metadata,
            )
        model_path = str(output_model) if output_model.exists() else None
        if not output_model.exists():
            warnings.append("Split Miner returned success but no model file exists")
        discovered_model = None
        if output_model.exists():
            discovered_model, load_warning = _load_bpmn_model(output_model)
            if load_warning:
                warnings.append(load_warning)
        if not _keep_output_files(split_miner_config):
            cleaned, cleanup_warning = _cleanup_output_artifacts(
                output_root=output_root,
                output_model=output_model,
                remove_output_root=configured_output_dir,
            )
            if cleaned:
                metadata["output_files_cleaned"] = True
                metadata["cleaned_model_path"] = str(output_model)
                model_path = None
            if cleanup_warning:
                warnings.append(cleanup_warning)
        return self._result(
            config=_reported_hyperparameters(split_miner_config),
            status="success",
            runtime_seconds=runtime_seconds,
            model_type="bpmn",
            discovered_model=discovered_model,
            model_path=model_path,
            warnings=warnings,
            metadata=metadata,
        )


def normalize_split_miner_config(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "jar_path",
        "jar_env_var",
        "jar_sha256",
        "java_bin",
        "java_options",
        "timeout_seconds",
        "epsilon",
        "eta",
        "parallelismFirst",
        "removeLoopActivityMarkers",
        "replaceIORs",
        "diagram",
        "input_log_path",
        "input_artifact_kind",
        "output_dir",
        "keep_output_files",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError("Unsupported Split Miner v1 parameter(s): " + ", ".join(unknown))
    return dict(config)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split_miner_command(
    *,
    java_bin: str,
    jar: Path,
    input_xes: Path,
    output_model: Path,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    java_options = _split_miner_java_options(config)
    command = [
        java_bin,
        *java_options,
        "-jar",
        str(jar),
        "--logPath",
        str(input_xes),
        "--outputPath",
        str(output_model),
    ]
    warnings: list[str] = []
    if config.get("epsilon") is not None:
        command.extend(["--epsilon", str(config["epsilon"])])

    if _config_flag(config, "diagram"):
        command.append("--diagram")

    if config.get("eta") is not None:
        command.extend(["--eta", str(config["eta"])])
    if _config_flag(config, "parallelismFirst"):
        command.append("--parallelismFirst")
    if _config_flag(config, "removeLoopActivityMarkers"):
        command.append("--removeLoopActivityMarkers")
    if _config_flag(config, "replaceIORs"):
        command.append("--replaceIORs")
    return command, warnings


def _resolve_java_bin(java_bin: str) -> str:
    path = Path(java_bin).expanduser()
    if path.is_absolute():
        return str(path)
    if len(path.parts) > 1:
        return str(resolve_project_path(path).expanduser().resolve(strict=False))
    return java_bin


def _split_miner_java_options(config: dict[str, Any]) -> list[str]:
    value = config.get("java_options")
    if value is None:
        value = os.getenv("SPLIT_MINER_JAVA_OPTIONS")
    if value is None:
        return list(DEFAULT_SPLIT_MINER_JAVA_OPTIONS)
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list | tuple):
        options: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"java_options must contain strings, got {item!r}")
            options.append(item)
        return options
    raise ValueError(f"java_options must be a string or list of strings, got {value!r}")


def _config_flag(config: dict[str, Any], key: str) -> bool:
    value = config.get(key, False)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes", "y"}:
            return True
        if value.lower() in {"false", "0", "no", "n", ""}:
            return False
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{key} must be a boolean, got {value!r}")


def _keep_output_files(config: dict[str, Any]) -> bool:
    if config.get("keep_output_files") is not None:
        return _config_flag(config, "keep_output_files")
    value = os.getenv("PDCASH_KEEP_SPLIT_MINER_OUTPUT", "")
    return value.lower() in {"1", "true", "yes", "y"}


def _cleanup_output_artifacts(
    *,
    output_root: Path,
    output_model: Path,
    remove_output_root: bool,
) -> tuple[bool, str | None]:
    try:
        if remove_output_root:
            if output_root.exists():
                shutil.rmtree(output_root)
                return True, None
            return False, None
        if output_model.exists():
            output_model.unlink()
            return True, None
        return False, None
    except OSError as exc:
        return False, f"Could not clean Split Miner output artifacts: {exc}"


def _collect_split_miner_crash_logs(
    *,
    work_dir: Path,
    output_root: Path,
) -> tuple[list[str], str | None]:
    copied: list[str] = []
    try:
        error_logs = sorted(work_dir.glob("hs_err_pid*.log"))
        if not error_logs:
            return copied, None
        output_root.mkdir(parents=True, exist_ok=True)
        for source in error_logs:
            target = output_root / source.name
            shutil.copy2(source, target)
            copied.append(str(target))
        return copied, None
    except OSError as exc:
        return copied, f"Could not preserve Split Miner JVM error logs: {exc}"


def _reported_hyperparameters(config: dict[str, Any]) -> dict[str, Any]:
    reserved = {
        "input_log_path",
        "output_dir",
        "keep_output_files",
        "input_artifact_kind",
    }
    return {key: value for key, value in config.items() if key not in reserved}


def _load_bpmn_model(output_model: Path) -> tuple[Any | None, str | None]:
    try:
        import pm4py

        return pm4py.read_bpmn(str(output_model)), None
    except Exception as exc:
        return None, f"Could not load Split Miner BPMN output for metric evaluation: {exc}"


def _split_miner_failure_class(return_code: int, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}"
    if return_code == 127:
        return "java_not_found"
    if "javax/xml/bind" in text or "javax.xml.bind" in text:
        return "missing_jaxb"
    if "Could not allocate metaspace" in text:
        return "jvm_metaspace_allocation"
    if "failed to map segment from shared object" in text:
        return "jvm_shared_object_mapping"
    if (
        "There is insufficient memory for the Java Runtime Environment" in text
        or "Native memory allocation" in text
        or "Cannot allocate memory" in text
    ):
        return "jvm_native_memory"
    return "external_process_failed"


def _split_miner_error_message(
    return_code: int,
    stderr: str = "",
    *,
    stdout: str = "",
    failure_class: str | None = None,
) -> str:
    failure_class = failure_class or _split_miner_failure_class(
        return_code,
        stdout,
        stderr,
    )
    if failure_class == "java_not_found" and "No such file or directory" in stderr:
        return (
            "Split Miner could not start Java. Ensure Java is available on the compute "
            "node PATH, load the cluster Java module before submission, or set JAVA_BIN "
            "in .env to the absolute Java executable path."
        )
    if failure_class == "missing_jaxb":
        return (
            f"Split Miner exited with return code {return_code}; the JAR appears to need "
            "JAXB classes that are absent from this Java runtime. Use a Java 8 runtime or "
            "a Java setup that provides JAXB."
        )
    if failure_class == "jvm_metaspace_allocation":
        return (
            f"Split Miner exited with return code {return_code}; the Java runtime could "
            "not allocate metaspace. Reduce concurrent Split Miner JVMs or lower "
            "MaxMetaspaceSize via java_options."
        )
    if failure_class == "jvm_shared_object_mapping":
        return (
            f"Split Miner exited with return code {return_code}; the Java runtime could "
            "not map a native shared object. Run headless, reduce concurrent Split Miner "
            "JVMs, or increase job memory."
        )
    if failure_class == "jvm_native_memory":
        return (
            f"Split Miner exited with return code {return_code}; the Java runtime ran "
            "out of native memory. Reduce concurrent Split Miner JVMs, increase job "
            "memory, or tune java_options such as -Xmx, MaxMetaspaceSize, and -Xss."
        )
    return f"Split Miner exited with return code {return_code}"
