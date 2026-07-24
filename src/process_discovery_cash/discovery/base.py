from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DiscoveryStatus = Literal["success", "failed", "unsupported", "timeout", "skipped"]
ModelType = Literal[
    "petri_net",
    "process_tree",
    "bpmn",
    "heuristics_net",
    "directly_follows_graph",
    "external_file",
    "process_tree_or_petri_net",
    "heuristics_net_or_petri_net",
    "bpmn_or_petri_net",
    "unknown",
]


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    algorithm_name: str
    backend_name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float | None = None
    status: DiscoveryStatus
    model_type: ModelType = "unknown"
    model_path: str | None = None
    discovered_model: Any = Field(default=None, exclude=True)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json_record(self) -> dict[str, Any]:
        try:
            return self.model_dump(exclude={"discovered_model"})
        except AttributeError:
            return self.dict(exclude={"discovered_model"})


class DiscoveryAlgorithm(ABC):
    algorithm_name: str
    backend_name: str
    default_model_type: str = "unknown"

    def supports_algorithm(self) -> bool:
        return True

    def validate_config(self, config: dict[str, Any]) -> None:
        return None

    @abstractmethod
    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        """Discover a process model from a training event log."""

    def _result(
        self,
        *,
        config: dict[str, Any],
        status: DiscoveryStatus,
        runtime_seconds: float | None,
        model_type: str | None = None,
        discovered_model: Any = None,
        model_path: str | None = None,
        warnings: list[str] | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DiscoveryResult:
        return DiscoveryResult(
            algorithm_name=self.algorithm_name,
            backend_name=self.backend_name,
            hyperparameters=dict(config),
            runtime_seconds=runtime_seconds,
            status=status,
            model_type=model_type or self.default_model_type,
            discovered_model=discovered_model,
            model_path=model_path,
            warnings=warnings or [],
            error_message=error_message,
            metadata=metadata or {},
        )
