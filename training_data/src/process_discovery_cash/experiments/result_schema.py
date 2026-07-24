from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ExperimentStatus = Literal[
    "success",
    "success_missing",
    "failed",
    "unsupported",
    "timeout",
    "skipped",
]
MetricStatus = Literal[
    "success",
    "missing_model",
    "unsupported_model_type",
    "backend_error",
    "timeout",
    "not_computed",
]


class MetricRecord(BaseModel):
    value: float | int | None = None
    status: MetricStatus
    error: str | None = None


class ExperimentResult(BaseModel):
    experiment_id: str
    log_id: str
    log_path: str
    test_log_path: str
    seed: int
    algorithm_name: str
    backend: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    discovered_model_type: str
    model_path: str | None = None
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    metric_statuses: dict[str, MetricRecord] = Field(default_factory=dict)
    runtime_seconds: float | None = None
    status: ExperimentStatus
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        try:
            return self.model_dump()
        except AttributeError:
            return self.dict()


class MetricResult(BaseModel):
    experiment_id: str
    log_id: str
    log_path: str
    test_log_path: str
    seed: int
    algorithm_name: str
    backend: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    discovered_model_type: str | None = None
    model_path: str | None = None
    metric_profile: str
    metric_names: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    metric_statuses: dict[str, MetricRecord] = Field(default_factory=dict)
    metric_runtime_seconds: float | None = None
    discovery_runtime_seconds: float | None = None
    status: ExperimentStatus
    discovery_status: ExperimentStatus
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    source_result_path: str | None = None
    source_config_hash: str | None = None
    config_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        try:
            return self.model_dump()
        except AttributeError:
            return self.dict()
