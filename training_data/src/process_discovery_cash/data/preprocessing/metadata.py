from __future__ import annotations

import gzip
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO
from xml.etree import ElementTree

from process_discovery_cash.data.preprocessing.models import (
    CompanionInspection,
    DatasetPackageInspection,
    DatasetSpec,
    ProcessMiningMetadata,
    XESAttribute,
    XESClassifier,
    XESGlobal,
    XESHeaderMetadata,
)
from process_discovery_cash.utils.paths import resolve_portable_path


def read_xes_header_metadata(path: str | Path) -> XESHeaderMetadata:
    source = resolve_portable_path(path)
    root_attributes: dict[str, str] = {}
    extensions: list[dict[str, str]] = []
    classifiers: list[XESClassifier] = []
    globals_: list[XESGlobal] = []
    log_attributes: list[XESAttribute] = []
    trace_attribute_keys: set[str] = set()
    event_attribute_keys: set[str] = set()
    depth = 0
    in_first_trace = False
    in_event = False

    with _open_binary(source) as handle:
        for event, element in ElementTree.iterparse(handle, events=("start", "end")):
            tag = _local_name(element.tag)
            if event == "start":
                depth += 1
                if tag == "log":
                    root_attributes = dict(element.attrib)
                if tag == "trace":
                    in_first_trace = True
                elif tag == "event" and in_first_trace:
                    in_event = True
                continue

            if tag == "extension":
                extensions.append(dict(element.attrib))
            elif tag == "classifier":
                classifiers.append(
                    XESClassifier(
                        name=element.attrib.get("name", ""),
                        keys=_split_classifier_keys(element.attrib.get("keys", "")),
                        scope=element.attrib.get("scope"),
                    )
                )
            elif tag == "global":
                globals_.append(
                    XESGlobal(
                        scope=element.attrib.get("scope", ""),
                        attributes=[
                            XESAttribute(
                                key=child.attrib.get("key", ""),
                                value=child.attrib.get("value"),
                                type=_local_name(child.tag),
                            )
                            for child in list(element)
                            if child.attrib.get("key")
                        ],
                    )
                )
            elif depth == 2 and element.attrib.get("key"):
                log_attributes.append(
                    XESAttribute(
                        key=element.attrib["key"],
                        value=element.attrib.get("value"),
                        type=tag,
                    )
                )
            key = element.attrib.get("key")
            if key and in_first_trace:
                if in_event and depth == 4:
                    event_attribute_keys.add(key)
                elif not in_event and depth == 3:
                    trace_attribute_keys.add(key)
            if tag == "event" and in_first_trace:
                in_event = False
            if tag == "trace" and in_first_trace:
                break
            depth -= 1
            # Header elements are tiny. Keep children alive until their parent
            # <global> is processed; clearing every child here loses defaults.
            if tag in {"extension", "classifier", "global"} or depth == 1:
                element.clear()

    return XESHeaderMetadata(
        root_attributes=root_attributes,
        extensions=extensions,
        classifiers=classifiers,
        globals=globals_,
        log_attributes=log_attributes,
        trace_attribute_keys=sorted(trace_attribute_keys),
        event_attribute_keys=sorted(event_attribute_keys),
    )


def read_processmining_metadata(path: str | Path) -> ProcessMiningMetadata:
    metadata_path = resolve_portable_path(path)
    root = ElementTree.parse(metadata_path).getroot()
    if _local_name(root.tag) != "metadata":
        raise ValueError(
            f"Expected processmining.org metadata root <metadata>, got <{_local_name(root.tag)}>"
        )

    def text(name: str) -> str | None:
        element = root.find(name)
        if element is None or element.text is None:
            return None
        value = element.text.strip()
        return value or None

    extensions: list[dict[str, Any]] = []
    extensions_element = root.find("extensions")
    if extensions_element is not None:
        for element in list(extensions_element):
            extensions.append(
                {
                    "name": _local_name(element.tag),
                    "prefix": element.attrib.get("prefix"),
                    "uri": element.attrib.get("uri"),
                    "values": {
                        _local_name(child.tag): (child.text or "").strip()
                        for child in list(element)
                    },
                }
            )

    meta_extensions: dict[str, dict[str, str]] = {}
    meta_element = root.find("meta_extensions")
    if meta_element is not None:
        for element in list(meta_element):
            meta_extensions[_local_name(element.tag)] = {
                _local_name(child.tag): (child.text or "").strip() for child in list(element)
            }

    return ProcessMiningMetadata(
        path=metadata_path.as_posix(),
        doi=text("doi"),
        name=text("name"),
        description=text("description"),
        language=text("language"),
        log_type=text("log_type"),
        process_type=text("process_type"),
        expected_traces=_optional_int(text("number_of_traces")),
        expected_events=_optional_int(text("number_of_events")),
        min_events_per_trace=_optional_int(text("min_events_per_trace")),
        max_events_per_trace=_optional_int(text("max_events_per_trace")),
        global_trace_attributes=_metadata_attributes(root, "trace_level"),
        global_event_attributes=_metadata_attributes(root, "event_level"),
        extensions=extensions,
        meta_extensions=meta_extensions,
    )


def inspect_dataset_package(dataset: DatasetSpec) -> DatasetPackageInspection:
    source = resolve_portable_path(dataset.source_path)
    source_hash = sha256_file(source)
    companions: list[CompanionInspection] = []
    processmining_metadata: ProcessMiningMetadata | None = None
    discrepancies: list[str] = []

    for companion in dataset.companions:
        path = resolve_portable_path(companion.path)
        detected_role = companion.role
        root_element: str | None = None
        if path.suffix.lower() == ".xml":
            root_element = _xml_root_name(path)
            if root_element == "metadata":
                detected_role = "processmining_metadata"
                processmining_metadata = read_processmining_metadata(path)
            elif root_element == "log":
                detected_role = (
                    "duplicate_log_alias"
                    if sha256_file(path) == source_hash
                    else "alternative_log_version"
                )
        companions.append(
            CompanionInspection(
                path=path.as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                declared_role=companion.role,
                detected_role=detected_role,
                root_element=root_element,
            )
        )
        if detected_role != companion.role:
            discrepancies.append(
                f"Companion {path} declared as {companion.role} but detected as {detected_role}."
            )

    header = read_xes_header_metadata(source)
    if processmining_metadata is not None:
        _compare_metadata_globals(processmining_metadata, header, discrepancies)

    return DatasetPackageInspection(
        dataset_id=dataset.dataset_id,
        source_path=source.as_posix(),
        source_sha256=source_hash,
        source_size_bytes=source.stat().st_size,
        xes_header=header,
        processmining_metadata=processmining_metadata,
        companions=companions,
        discrepancies=discrepancies,
    )


def sha256_file(path: str | Path) -> str:
    source = resolve_portable_path(path)
    stat = source.stat()
    return _sha256_file_cached(source.as_posix(), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=256)
def _sha256_file_cached(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_binary(path: Path) -> BinaryIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _xml_root_name(path: Path) -> str:
    with _open_binary(path) as handle:
        for _event, element in ElementTree.iterparse(handle, events=("start",)):
            return _local_name(element.tag)
    raise ValueError(f"XML file has no root element: {path}")


def _metadata_attributes(
    root: ElementTree.Element,
    scope: str,
) -> list[dict[str, str | None]]:
    return [
        {"key": element.attrib.get("key"), "default": element.attrib.get("default")}
        for element in root.findall(f"./global_attributes/{scope}/attribute")
    ]


def _compare_metadata_globals(
    metadata: ProcessMiningMetadata,
    header: XESHeaderMetadata,
    discrepancies: list[str],
) -> None:
    xes_globals = {
        global_.scope: {attribute.key for attribute in global_.attributes}
        for global_ in header.globals
    }
    expected_trace = {str(attribute.get("key")) for attribute in metadata.global_trace_attributes}
    expected_event = {str(attribute.get("key")) for attribute in metadata.global_event_attributes}
    for scope, expected in [("trace", expected_trace), ("event", expected_event)]:
        actual = xes_globals.get(scope, set())
        if expected and expected != actual:
            discrepancies.append(
                f"External {scope} globals differ from XES globals: "
                f"external={sorted(expected)}, xes={sorted(actual)}."
            )


def _split_classifier_keys(value: str) -> list[str]:
    return [part for part in value.split() if part]


def _optional_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
