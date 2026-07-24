from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["latin_hypercube"] = "latin_hypercube"
    n_samples: int = Field(gt=0)
    seed: int


class AlgorithmReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    algorithm_id: str | None = None
    config: str | None = None
    backend: str | None = None
    model_type: str | None = None
    runtime_params: list[str] = Field(default_factory=list)
    artifact_algorithm_id: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    supported: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    configs: list[dict[str, Any]] = Field(default_factory=list)
    search_space_override: dict[str, Any] | None = None
    sampling: SamplingConfig | None = None

    @model_validator(mode="after")
    def normalize_inline_algorithm_id(self) -> AlgorithmReference:
        if self.algorithm_id is None:
            self.algorithm_id = self.name
        return self

    @model_validator(mode="after")
    def validate_inline_algorithm_definition(self) -> AlgorithmReference:
        fields_set = set(getattr(self, "model_fields_set", set()))
        inline_markers_present = bool(
            {"backend", "model_type", "runtime_params", "artifact_algorithm_id"} & fields_set
        ) or bool(self.default_params)
        if inline_markers_present:
            if self.backend is None:
                raise ValueError("Inline algorithm definitions require backend")
            if not self.algorithm_id:
                raise ValueError("Inline algorithm definitions require algorithm_id")
            if self.model_type is None:
                raise ValueError("Inline algorithm definitions require model_type")
            if not self.runtime_params:
                raise ValueError("Inline algorithm definitions require runtime_params")
        return self

    @model_validator(mode="after")
    def reject_conflicting_sampling_and_configs(self) -> AlgorithmReference:
        if self.sampling is not None and self.configs:
            raise ValueError(
                "AlgorithmReference cannot set both 'sampling' and 'configs'; "
                "'configs' is an explicit enumerated list and is incompatible "
                "with Latin Hypercube sampling"
            )
        return self


class LogReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_id: str
    dataset_id: str | None = None
    path: str | None = None
    source_path: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    # Optional pinned preprocessing fingerprint. When set, artifact resolution
    # uses it directly instead of hashing the raw source — lets a host with only
    # the processed artifacts (no raw logs) resolve the discovery parquet.
    preprocessing_fingerprint: str | None = None

    @model_validator(mode="after")
    def normalize_log_paths(self) -> LogReference:
        if self.path is None and self.train_path is not None:
            self.path = self.train_path
        if self.train_path is None and self.path is not None:
            self.train_path = self.path
        if self.path is None and self.dataset_id is None:
            raise ValueError("Log config must define dataset_id, path, or train_path")
        if self.test_path is None:
            self.test_path = self.path
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results_dir: str | None = None
    log_dir: str = "logs/slurm"
    output_path_template: str | None = None


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    profile: Literal["pm4py_default", "token", "alignment"] = "pm4py_default"
    names: list[str] = Field(
        default_factory=lambda: ["fitness", "precision", "generalization", "simplicity"]
    )
    export_model: bool = False


class HpoObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fitness": 1.0,
            "precision": 1.0,
            "generalization": 1.0,
            "simplicity": 1.0,
        }
    )
    failed_trial_value: float = 0.0

    @model_validator(mode="after")
    def validate_weights(self) -> HpoObjectiveConfig:
        if not self.weights:
            raise ValueError("HPO objective requires at least one metric weight")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("HPO objective weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("HPO objective weights must sum to a positive value")
        return self


class HpoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(gt=0)
    n_startup_trials: int = Field(default=10, gt=0)
    sampler_seed: int = 42
    per_trial_walltime_seconds: float = Field(default=600.0, gt=0)
    objective: HpoObjectiveConfig = Field(default_factory=HpoObjectiveConfig)
    export_model: bool | None = None
    multivariate: bool = True
    group: bool = True
    constant_liar: bool = True
    storage_root: str = "runs/hpo"
    summary_dirname: str = "hpo_summaries"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    logs: list[LogReference]
    algorithms: list[str | AlgorithmReference]
    allow_unsupported: bool = False
    strict_parameter_validation: bool = True
    seeds: list[int] = Field(default_factory=lambda: [0])
    output_root: str = "results/local"
    output: OutputConfig = Field(default_factory=OutputConfig)
    manifest_path: str | None = None
    algorithm_config_dir: str = "configs/algorithms"
    dataset_catalog_path: str = "configs/datasets/processmining_org.yaml"
    event_log_artifact_root: str = "data/processed/event_logs"
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    hpo: HpoConfig | None = None

    @model_validator(mode="after")
    def validate_hpo_requirements(self) -> ExperimentConfig:
        if self.hpo is None:
            return self
        if not self.metrics.enabled:
            raise ValueError(
                "HPO experiments require metrics.enabled: true; the optimizer needs "
                "metric values to compute the trial objective"
            )
        unknown = set(self.hpo.objective.weights) - set(self.metrics.names)
        if unknown:
            raise ValueError(
                f"HPO objective weights reference metrics not in metrics.names: {sorted(unknown)}"
            )
        return self


class AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm_id: str | None = None
    algorithm: str | None = None
    display_name: str | None = None
    backend: str
    supported: bool = True
    pm4py_function: str | None = None
    pm4py_api: dict[str, Any] = Field(default_factory=dict)
    model_type: str = "petri_net"
    default_params: dict[str, Any] = Field(default_factory=dict)
    search_space: dict[str, Any] = Field(default_factory=dict)
    conditional_search_space: list[dict[str, Any]] = Field(default_factory=list)
    parameter_mapping: dict[str, Any] = Field(default_factory=dict)
    runtime_params: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    external: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_algorithm_id(self) -> AlgorithmConfig:
        if self.algorithm_id is None and self.algorithm is not None:
            self.algorithm_id = self.algorithm
        if self.algorithm is None and self.algorithm_id is not None:
            self.algorithm = self.algorithm_id
        if self.algorithm_id is None:
            raise ValueError("Algorithm config must define algorithm_id")
        if self.algorithm != self.algorithm_id:
            raise ValueError(
                "Algorithm config has conflicting algorithm and algorithm_id values: "
                f"{self.algorithm!r} != {self.algorithm_id!r}"
            )
        return self


def ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
