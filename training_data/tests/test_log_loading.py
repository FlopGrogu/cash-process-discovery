import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from process_discovery_cash.data import loading
from process_discovery_cash.data.loading import (
    LogCacheError,
    detect_log_format,
    load_event_log,
    load_event_log_with_info,
    preprocess_event_log,
)
from process_discovery_cash.discovery.pm4py_backend import discover_alpha_miner


def test_detect_log_format_supports_xes_xes_gz_and_csv() -> None:
    assert detect_log_format("data/raw/example.xes") == "xes"
    assert detect_log_format("data/raw/example.xes.gz") == "xes"
    assert detect_log_format("data/raw/example.csv") == "csv"


def test_detect_log_format_rejects_plain_gz_with_clear_error() -> None:
    with pytest.raises(ValueError, match="Supported formats are: .xes, .xes.gz, .csv"):
        detect_log_format("data/raw/example.gz")


def test_load_event_log_unsupported_format_message_uses_full_filename(tmp_path) -> None:
    unsupported = tmp_path / "example.gz"
    unsupported.write_bytes(b"not an event log")

    with pytest.raises(ValueError, match="example\\.gz"):
        load_event_log(unsupported)


def test_xes_loader_is_independent_of_optional_rust_dependencies(monkeypatch) -> None:
    import pm4py

    captured = {}
    dataframe = _minimal_dataframe()
    monkeypatch.setattr(loading, "_rustxes_dependencies_available", lambda: True)
    monkeypatch.setattr(
        pm4py,
        "read_xes",
        lambda path, **kwargs: captured.update(path=path, kwargs=kwargs) or dataframe,
    )

    loaded = load_event_log_with_info("data/example/tiny_log.xes", use_cache=False)

    assert_frame_equal(loaded.log, dataframe, check_dtype=False)
    assert loaded.backend == "chunk_regex"
    assert captured["kwargs"]["variant"] == "chunk_regex"
    assert captured["kwargs"]["return_legacy_log_object"] is False


def test_xes_loader_uses_chunk_regex_when_rust_dependencies_are_missing(monkeypatch) -> None:
    import pm4py

    captured = {}
    dataframe = _minimal_dataframe()
    monkeypatch.setattr(loading, "_rustxes_dependencies_available", lambda: False)
    monkeypatch.setattr(
        pm4py,
        "read_xes",
        lambda path, **kwargs: captured.update(path=path, kwargs=kwargs) or dataframe,
    )

    loaded = load_event_log_with_info("data/example/tiny_log.xes", use_cache=False)

    assert_frame_equal(loaded.log, dataframe, check_dtype=False)
    assert loaded.backend == "chunk_regex"
    assert captured["kwargs"]["variant"] == "chunk_regex"


def test_xes_parser_errors_are_not_hidden_by_minimal_fallback(monkeypatch) -> None:
    import pm4py

    monkeypatch.setattr(loading, "_rustxes_dependencies_available", lambda: False)

    def fail_read(*_args, **_kwargs):
        raise ValueError("corrupt xes")

    monkeypatch.setattr(pm4py, "read_xes", fail_read)

    with pytest.raises(ValueError, match="corrupt xes"):
        load_event_log("data/example/tiny_log.xes", use_cache=False)


def test_xes_loader_retries_iterparse_after_chunk_regex_index_error(monkeypatch) -> None:
    import pm4py

    calls = []
    dataframe = _minimal_dataframe()

    def read_xes(path, **kwargs):
        calls.append((path, kwargs))
        if kwargs["variant"] == "chunk_regex":
            raise IndexError("list index out of range")
        return dataframe

    monkeypatch.setattr(pm4py, "read_xes", read_xes)

    loaded = load_event_log_with_info("data/example/tiny_log.xes", use_cache=False)

    assert_frame_equal(loaded.log, dataframe, check_dtype=False)
    assert loaded.backend == "iterparse_fallback"
    assert [kwargs["variant"] for _path, kwargs in calls] == [
        "chunk_regex",
        "iterparse",
    ]
    assert all(kwargs["show_progress_bar"] is False for _path, kwargs in calls)


def test_parquet_cache_preserves_order_values_and_types(tmp_path) -> None:
    source = _write_typed_xes(tmp_path / "typed.xes")
    cache_dir = tmp_path / "cache"

    direct = load_event_log(source, use_cache=False)
    result = preprocess_event_log(source, cache_key="typed", cache_dir=cache_dir)
    cached_info = load_event_log_with_info(source, cache_key="typed", cache_dir=cache_dir)

    assert result.created is True
    assert cached_info.cache_hit is True
    assert cached_info.backend == "parquet"
    assert_frame_equal(cached_info.log, direct)
    assert cached_info.log["concept:name"].tolist() == ["A", "B"]
    assert pd.api.types.is_bool_dtype(cached_info.log["approved"])
    assert pd.api.types.is_integer_dtype(cached_info.log["count"])
    assert pd.api.types.is_float_dtype(cached_info.log["cost"])
    assert pd.api.types.is_string_dtype(cached_info.log["note"])
    assert isinstance(cached_info.log["time:timestamp"].dtype, pd.DatetimeTZDtype)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["row_count"] == 2
    assert metadata["parser"] == "rustxes"


def test_preprocessing_regenerates_stale_cache(tmp_path) -> None:
    source = tmp_path / "tiny.xes"
    source.write_bytes(Path("data/example/tiny_log.xes").read_bytes())
    cache_dir = tmp_path / "cache"

    first = preprocess_event_log(source, cache_key="tiny", cache_dir=cache_dir)
    current = preprocess_event_log(source, cache_key="tiny", cache_dir=cache_dir)
    source.write_bytes(source.read_bytes() + b"\n")
    regenerated = preprocess_event_log(source, cache_key="tiny", cache_dir=cache_dir)

    assert first.created is True
    assert current.created is False
    assert regenerated.created is True


def test_preprocessing_uses_sha_when_size_and_mtime_match(tmp_path) -> None:
    source = tmp_path / "tiny.xes"
    source.write_bytes(Path("data/example/tiny_log.xes").read_bytes())
    original_stat = source.stat()
    cache_dir = tmp_path / "cache"
    first = preprocess_event_log(source, cache_key="tiny", cache_dir=cache_dir)
    first_metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))

    changed = source.read_bytes().replace(b'value="A"', b'value="Z"', 1)
    source.write_bytes(changed)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    regenerated = preprocess_event_log(source, cache_key="tiny", cache_dir=cache_dir)
    regenerated_metadata = json.loads(regenerated.metadata_path.read_text(encoding="utf-8"))

    assert regenerated.created is True
    assert regenerated_metadata["source_sha256"] != first_metadata["source_sha256"]


def test_loading_rejects_cache_when_source_hash_changes_with_same_stat(tmp_path) -> None:
    source = _write_typed_xes(tmp_path / "typed.xes")
    original_stat = source.stat()
    cache_dir = tmp_path / "cache"
    preprocess_event_log(source, cache_key="typed", cache_dir=cache_dir)

    changed = source.read_bytes().replace(b'value="A"', b'value="Z"', 1)
    assert len(changed) == original_stat.st_size
    source.write_bytes(changed)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    loaded = load_event_log_with_info(source, cache_key="typed", cache_dir=cache_dir)

    assert loaded.cache_hit is False
    assert loaded.backend == "chunk_regex"
    assert loaded.log["concept:name"].tolist() == ["Z", "B"]


def test_corrupt_parquet_cache_fails_visibly(tmp_path) -> None:
    source = tmp_path / "tiny.xes"
    source.write_bytes(Path("data/example/tiny_log.xes").read_bytes())
    result = preprocess_event_log(source, cache_key="tiny", cache_dir=tmp_path / "cache")
    result.cache_path.write_bytes(b"not parquet")

    with pytest.raises(LogCacheError, match="Could not read Parquet log cache"):
        load_event_log(source, cache_key="tiny", cache_dir=tmp_path / "cache")


def test_incomplete_cache_fails_visibly(tmp_path) -> None:
    source = tmp_path / "tiny.xes"
    source.write_bytes(Path("data/example/tiny_log.xes").read_bytes())
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "tiny.parquet").write_bytes(b"incomplete")

    with pytest.raises(LogCacheError, match="incomplete or invalid"):
        load_event_log(source, cache_key="tiny", cache_dir=cache_dir)


def test_alpha_discovery_is_equivalent_for_xes_and_cached_dataframe(tmp_path) -> None:
    source = Path("data/example/tiny_log.xes")
    direct = load_event_log(source, use_cache=False)
    preprocess_event_log(source, cache_key="tiny", cache_dir=tmp_path)
    cached = load_event_log(source, cache_key="tiny", cache_dir=tmp_path)

    direct_result = discover_alpha_miner(direct, {"variant": "classic"})
    cached_result = discover_alpha_miner(cached, {"variant": "classic"})

    assert _petri_net_signature(direct_result["model"]) == _petri_net_signature(
        cached_result["model"]
    )


def _petri_net_signature(model) -> tuple:
    net, initial_marking, final_marking = model
    transitions = sorted((transition.name, transition.label) for transition in net.transitions)
    arcs = sorted((arc.source.name, arc.target.name) for arc in net.arcs)
    initial = sorted((place.name, count) for place, count in initial_marking.items())
    final = sorted((place.name, count) for place, count in final_marking.items())
    return transitions, arcs, initial, final


def _write_typed_xes(path: Path) -> Path:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0">
  <trace>
    <string key="concept:name" value="case_1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2026-01-01T09:00:00.000+00:00"/>
      <boolean key="approved" value="true"/>
      <int key="count" value="2"/>
      <float key="cost" value="1.5"/>
      <string key="note" value="first"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2026-01-01T09:05:00.000+00:00"/>
      <boolean key="approved" value="false"/>
      <int key="count" value="3"/>
      <float key="cost" value="2.5"/>
      <string key="note" value="second"/>
    </event>
  </trace>
</log>
""",
        encoding="utf-8",
    )
    return path


def _minimal_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case:concept:name": ["case_1"],
            "concept:name": ["A"],
            "time:timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        }
    )
