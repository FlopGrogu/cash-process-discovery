"""Feature-space target design: LHS sampling, realism bands, feasibility."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from process_discovery_cash.generation.feature_space import (
    BAND_EXPANSION,
    BAND_IN_DISTRIBUTION,
    BAND_STRESS,
    LOG10_FEATURES,
    TARGET_FEATURES,
    RealAnchor,
    assign_band,
    design_bounds,
    from_target_space,
)

CONCURRENCY_LEVELS = ("low", "medium", "high")
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
BAND_SHARES = {BAND_IN_DISTRIBUTION: 0.7, BAND_EXPANSION: 0.2, BAND_STRESS: 0.1}

DEFAULT_CAPS_LOW = {"num_traces": 10.0, "avg_trace_length": 2.0, "num_activities": 3.0}
DEFAULT_CAPS_HIGH = {
    "num_traces": 10_000.0,
    "avg_trace_length": 100.0,
    "num_activities": 200.0,
}
MAX_EXPECTED_EVENTS = 200_000.0


@dataclass
class TargetSpec:
    target_id: str
    band: str
    values: dict[str, float]
    concurrency: str
    noise_level: float
    nearest_real_distance: float
    feasible: bool = True
    infeasible_reason: str | None = None
    repairs: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target_id": self.target_id,
            "band": self.band,
            "concurrency": self.concurrency,
            "noise_level": self.noise_level,
            "nearest_real_distance": self.nearest_real_distance,
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "repairs": "|".join(self.repairs),
        }
        for feature in TARGET_FEATURES:
            row[f"target_{feature}"] = self.values[feature]
        return row


def design_targets(
    real_features: pd.DataFrame,
    *,
    n_targets: int = 200,
    seed: int = 2024,
    expansion: float = 0.12,
    caps_low: dict[str, float] | None = None,
    caps_high: dict[str, float] | None = None,
    max_expected_events: float = MAX_EXPECTED_EVENTS,
    pool_factor: int = 4,
) -> list[TargetSpec]:
    """Design ~n_targets feature-space targets anchored on the real logs.

    Expansion and stress targets come from a maximin-optimized Latin Hypercube
    pool (``pool_factor`` x n_targets) over the expanded real-log bounding box,
    labelled by nearest-real distance. In-distribution targets densify the real
    cloud directly: standardized Gaussian jitter around randomly chosen real
    logs (21 points in 6-D leave uniform LHS samples almost never within the
    in-distribution radius). Shortfalls in any band are backfilled with jitter
    at a band-matched scale, so the 70/20/10 split is approximate but the band
    labels stay absolute (same thresholds as achieved-band classification).
    Infeasible targets are repaired where possible and kept (marked) otherwise.
    """
    caps_low = caps_low or DEFAULT_CAPS_LOW
    caps_high = caps_high or DEFAULT_CAPS_HIGH
    anchor = RealAnchor.from_features(real_features)
    low, high = design_bounds(
        real_features, expansion=expansion, caps_low=caps_low, caps_high=caps_high
    )
    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=len(TARGET_FEATURES), optimization="random-cd", seed=seed)
    pool = qmc.scale(sampler.random(n=max(pool_factor * n_targets, n_targets)), low, high)
    pool_distances = anchor.nearest_real_distance(pool)
    pool_bands = np.array([assign_band(distance) for distance in pool_distances])

    quotas = _band_quotas(n_targets)
    points: list[np.ndarray] = []
    jitter_scale = {BAND_IN_DISTRIBUTION: 0.5, BAND_EXPANSION: 1.6, BAND_STRESS: 3.0}
    for band, quota in quotas.items():
        if band == BAND_IN_DISTRIBUTION:
            points.extend(
                _jittered_anchor_points(anchor, quota, jitter_scale[band], rng, low, high)
            )
            continue
        indices = [int(i) for i in np.flatnonzero(pool_bands == band)]
        take = min(quota, len(indices))
        if take:
            chosen = rng.choice(len(indices), size=take, replace=False)
            points.extend(pool[indices[int(i)]] for i in chosen)
        if take < quota:
            points.extend(
                _jittered_anchor_points(anchor, quota - take, jitter_scale[band], rng, low, high)
            )

    distances = anchor.nearest_real_distance(np.vstack(points))
    targets: list[TargetSpec] = []
    for position, point in enumerate(points):
        spec = TargetSpec(
            target_id=f"t{position:04d}",
            band=assign_band(float(distances[position])),
            values=from_target_space(point),
            concurrency=str(rng.choice(CONCURRENCY_LEVELS, p=[0.4, 0.4, 0.2])),
            noise_level=float(rng.choice(NOISE_LEVELS, p=[0.55, 0.2, 0.15, 0.1])),
            nearest_real_distance=float(distances[position]),
        )
        repair_target(
            spec, caps_low=caps_low, caps_high=caps_high, max_expected_events=max_expected_events
        )
        targets.append(spec)
    return targets


def _jittered_anchor_points(
    anchor: RealAnchor,
    count: int,
    sigma: float,
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
) -> list[np.ndarray]:
    """Sample points by Gaussian jitter around random real logs, clipped to bounds.

    Only real logs inside the (compute-capped) bounds serve as jitter bases:
    around an out-of-bounds anchor (e.g. a 150k-trace log with a 10k cap) every
    jittered point would be clipped far away from it and silently leave the
    intended band.
    """
    real_points = anchor.mean + anchor.standardized_real * anchor.scale
    in_bounds = np.all((real_points >= low) & (real_points <= high), axis=1)
    bases = anchor.standardized_real[in_bounds]
    if bases.shape[0] == 0:
        clipped = np.clip(real_points, low, high)
        bases = (clipped - anchor.mean) / anchor.scale
    points: list[np.ndarray] = []
    for _ in range(count):
        base = bases[int(rng.integers(0, bases.shape[0]))]
        standardized = base + rng.normal(0.0, sigma, size=base.shape)
        point = anchor.mean + standardized * anchor.scale
        points.append(np.clip(point, low, high))
    return points


def repair_target(
    spec: TargetSpec,
    *,
    caps_low: dict[str, float],
    caps_high: dict[str, float],
    max_expected_events: float = MAX_EXPECTED_EVENTS,
) -> TargetSpec:
    """Repair or reject a target that is impossible or computationally useless."""
    values = spec.values
    for feature in TARGET_FEATURES:
        low_cap = caps_low.get(feature, 0.0)
        high_cap = caps_high.get(feature, 1.0 if feature not in LOG10_FEATURES else math.inf)
        clipped = min(max(values[feature], low_cap), high_cap)
        if clipped != values[feature]:
            spec.repairs.append(f"{feature}: {values[feature]:.4g} -> {clipped:.4g} (cap)")
            values[feature] = clipped
    for feature in ("num_traces", "num_activities"):
        values[feature] = float(round(values[feature]))

    expected_events = values["num_traces"] * values["avg_trace_length"]
    if expected_events > max_expected_events:
        repaired_traces = max(
            caps_low.get("num_traces", 10.0),
            float(round(max_expected_events / values["avg_trace_length"])),
        )
        spec.repairs.append(
            f"num_traces: {values['num_traces']:.4g} -> {repaired_traces:.4g} (events cap)"
        )
        values["num_traces"] = repaired_traces

    # At least two variants must exist, so the ratio has a hard floor.
    variant_floor = 2.0 / values["num_traces"]
    if values["variant_ratio"] < variant_floor:
        spec.repairs.append(
            f"variant_ratio: {values['variant_ratio']:.4g} -> {variant_floor:.4g} (min variants)"
        )
        values["variant_ratio"] = variant_floor

    # Variant ratio cannot exceed the share of expressible distinct sequences.
    log_max_variants = values["avg_trace_length"] * math.log10(max(values["num_activities"], 2))
    log_required = math.log10(max(values["variant_ratio"] * values["num_traces"], 1e-9))
    if log_required > log_max_variants + math.log10(0.9):
        achievable = 0.9 * 10.0 ** min(log_max_variants, 12.0) / values["num_traces"]
        achievable = min(max(achievable, 1.0 / values["num_traces"]), 1.0)
        if achievable < 0.01:
            spec.feasible = False
            spec.infeasible_reason = (
                "variant ratio unreachable with alphabet "
                f"{values['num_activities']:.0f} and mean length {values['avg_trace_length']:.1f}"
            )
            return spec
        spec.repairs.append(
            f"variant_ratio: {values['variant_ratio']:.4g} -> {achievable:.4g} (expressibility)"
        )
        values["variant_ratio"] = achievable

    # Dense DFG over a large alphabet is a spaghetti log; only stress may keep it.
    if values["dfg_density"] > 0.6 and values["num_activities"] > 50 and spec.band != BAND_STRESS:
        spec.repairs.append(f"dfg_density: {values['dfg_density']:.4g} -> 0.6 (spaghetti guard)")
        values["dfg_density"] = 0.6

    # Meaningful repetition needs room inside traces.
    if values["avg_trace_length"] < 3.0 and values["repetition_prevalence"] > 0.3:
        spec.repairs.append(
            f"repetition_prevalence: {values['repetition_prevalence']:.4g} -> 0.3 (short traces)"
        )
        values["repetition_prevalence"] = 0.3
    return spec


def targets_to_frame(targets: list[TargetSpec]) -> pd.DataFrame:
    return pd.DataFrame([target.to_row() for target in targets])


def targets_from_frame(frame: pd.DataFrame) -> list[TargetSpec]:
    """Rebuild TargetSpecs from a targets.csv frame (inverse of targets_to_frame)."""
    targets: list[TargetSpec] = []
    for _, row in frame.iterrows():
        infeasible_reason = row.get("infeasible_reason")
        if pd.isna(infeasible_reason):
            infeasible_reason = None
        repairs_raw = row.get("repairs")
        repairs = (
            [part for part in str(repairs_raw).split("|") if part]
            if not pd.isna(repairs_raw)
            else []
        )
        targets.append(
            TargetSpec(
                target_id=str(row["target_id"]),
                band=str(row["band"]),
                values={feature: float(row[f"target_{feature}"]) for feature in TARGET_FEATURES},
                concurrency=str(row["concurrency"]),
                noise_level=float(row["noise_level"]),
                nearest_real_distance=float(row["nearest_real_distance"]),
                feasible=_parse_bool(row["feasible"]),
                infeasible_reason=infeasible_reason,
                repairs=repairs,
            )
        )
    return targets


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Invalid boolean value for target feasibility: {value!r}")


def _band_quotas(n_targets: int) -> dict[str, int]:
    quotas = {band: int(round(share * n_targets)) for band, share in BAND_SHARES.items()}
    quotas[BAND_IN_DISTRIBUTION] += n_targets - sum(quotas.values())
    return quotas
