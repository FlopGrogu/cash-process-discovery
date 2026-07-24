from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LifecycleSemantics = Literal[
    "standard",
    "extended_standard",
    "complete_only",
    "status_like",
    "absent",
]
CompanionRole = Literal[
    "processmining_metadata",
    "documentation",
    "duplicate_log_alias",
    "alternative_log_version",
]


class CompanionSpec(BaseModel):
    path: str
    role: CompanionRole


class DatasetSchema(BaseModel):
    case_id: str = "trace:concept:name"
    activity: str = "concept:name"
    complete_timestamp: str = "time:timestamp"
    start_timestamp: str | None = None
    lifecycle: str | None = None
    lifecycle_semantics: LifecycleSemantics = "absent"
    optional_attributes: list[str] = Field(default_factory=list)
    classifier_profile: str = "default"


class DatasetSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str
    display_name: str
    source_path: str
    sha256: str | None = None
    size_bytes: int | None = None
    landing_url: str | None = None
    doi: str | None = None
    license: str | None = None
    companions: list[CompanionSpec] = Field(default_factory=list)
    event_schema: DatasetSchema = Field(
        default_factory=DatasetSchema,
        alias="schema",
        serialization_alias="schema",
    )
    metadata_affects_resolution: bool = False
    notes: list[str] = Field(default_factory=list)


class DatasetCatalog(BaseModel):
    datasets: dict[str, DatasetSpec]


class XESClassifier(BaseModel):
    name: str
    keys: list[str] = Field(default_factory=list)
    scope: str | None = None


class XESAttribute(BaseModel):
    key: str
    value: Any = None
    type: str


class XESGlobal(BaseModel):
    scope: str
    attributes: list[XESAttribute] = Field(default_factory=list)


class XESHeaderMetadata(BaseModel):
    root_attributes: dict[str, str] = Field(default_factory=dict)
    extensions: list[dict[str, str]] = Field(default_factory=list)
    classifiers: list[XESClassifier] = Field(default_factory=list)
    globals: list[XESGlobal] = Field(default_factory=list)
    log_attributes: list[XESAttribute] = Field(default_factory=list)
    trace_attribute_keys: list[str] = Field(default_factory=list)
    event_attribute_keys: list[str] = Field(default_factory=list)


class ProcessMiningMetadata(BaseModel):
    path: str
    doi: str | None = None
    name: str | None = None
    description: str | None = None
    language: str | None = None
    log_type: str | None = None
    process_type: str | None = None
    expected_traces: int | None = None
    expected_events: int | None = None
    min_events_per_trace: int | None = None
    max_events_per_trace: int | None = None
    global_trace_attributes: list[dict[str, str | None]] = Field(default_factory=list)
    global_event_attributes: list[dict[str, str | None]] = Field(default_factory=list)
    extensions: list[dict[str, Any]] = Field(default_factory=list)
    meta_extensions: dict[str, dict[str, str]] = Field(default_factory=dict)


class CompanionInspection(BaseModel):
    path: str
    sha256: str
    size_bytes: int
    declared_role: CompanionRole
    detected_role: CompanionRole
    root_element: str | None = None


class DatasetPackageInspection(BaseModel):
    dataset_id: str
    source_path: str
    source_sha256: str
    source_size_bytes: int
    xes_header: XESHeaderMetadata
    processmining_metadata: ProcessMiningMetadata | None = None
    companions: list[CompanionInspection] = Field(default_factory=list)
    discrepancies: list[str] = Field(default_factory=list)


class LifecycleAnalysis(BaseModel):
    interval_quality: Literal["green", "yellow", "red"]
    reasons: list[str] = Field(default_factory=list)
    lifecycle_values: dict[str, int] = Field(default_factory=dict)
    starts: int = 0
    completes: int = 0
    paired: int = 0
    unmatched_starts: int = 0
    unmatched_completes: int = 0
    negative_durations: int = 0
    zero_durations: int = 0
    positive_durations: int = 0


class ArtifactSelection(BaseModel):
    dataset_id: str | None = None
    source_log_path: str
    discovery_log_path: str | None = None
    test_discovery_log_path: str | None = None
    artifact_kind: str | None = None
    artifact_sha256: str | None = None
    preprocessing_fingerprint: str | None = None
    preprocessing_metadata_path: str | None = None
