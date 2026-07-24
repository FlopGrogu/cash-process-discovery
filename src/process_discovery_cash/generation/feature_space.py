"""Six-axis target feature space anchored on the real event logs.

Count-like axes live in log10 space; ratio-like axes stay bounded in [0, 1].
All distances (realism bands, duplicate checks) are computed in this
transformed space after standardization against the real-log cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

TARGET_FEATURES = [
    "num_traces",
    "avg_trace_length",
    "num_activities",
    "variant_ratio",
    "dfg_density",
    "repetition_prevalence",
]
LOG10_FEATURES = frozenset({"num_traces", "avg_trace_length", "num_activities"})

IN_DISTRIBUTION_MAX_DISTANCE = 1.5
EXPANSION_MAX_DISTANCE = 3.0
NEAR_DUPLICATE_DISTANCE = 0.15

BAND_IN_DISTRIBUTION = "in_distribution"
BAND_EXPANSION = "expansion"
BAND_STRESS = "stress"


def to_target_space(features: pd.DataFrame | dict[str, Any]) -> np.ndarray:
    """Map raw feature values to the transformed 6-axis target space."""
    if isinstance(features, dict):
        frame = pd.DataFrame([features])
    else:
        frame = features
    columns = []
    for feature in TARGET_FEATURES:
        values = frame[feature].astype(float).to_numpy()
        if feature in LOG10_FEATURES:
            columns.append(np.log10(np.maximum(values, 1e-9)))
        else:
            columns.append(np.clip(values, 0.0, 1.0))
    return np.column_stack(columns)


def from_target_space(point: np.ndarray) -> dict[str, float]:
    """Map one transformed point back to raw feature values."""
    raw: dict[str, float] = {}
    for index, feature in enumerate(TARGET_FEATURES):
        value = float(point[index])
        raw[feature] = 10.0**value if feature in LOG10_FEATURES else float(np.clip(value, 0, 1))
    return raw


def assign_band(distance: float) -> str:
    if distance <= IN_DISTRIBUTION_MAX_DISTANCE:
        return BAND_IN_DISTRIBUTION
    if distance <= EXPANSION_MAX_DISTANCE:
        return BAND_EXPANSION
    return BAND_STRESS


@dataclass(frozen=True)
class RealAnchor:
    """Real-log cloud in transformed space with a standardization fitted on it."""

    mean: np.ndarray
    scale: np.ndarray
    standardized_real: np.ndarray

    @classmethod
    def from_features(cls, real_features: pd.DataFrame) -> RealAnchor:
        points = to_target_space(real_features)
        mean = points.mean(axis=0)
        scale = points.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(mean=mean, scale=scale, standardized_real=(points - mean) / scale)

    def standardize(self, points: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(points) - self.mean) / self.scale

    def nearest_real_distance(self, points: np.ndarray) -> np.ndarray:
        standardized = self.standardize(points)
        deltas = standardized[:, None, :] - self.standardized_real[None, :, :]
        return np.sqrt((deltas**2).sum(axis=2)).min(axis=1)

    def nearest_distance_to(self, points: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Min standardized distance from each point to a reference set (raw space)."""
        if reference.size == 0:
            return np.full(np.atleast_2d(points).shape[0], np.inf)
        standardized = self.standardize(points)
        reference_std = self.standardize(reference)
        deltas = standardized[:, None, :] - reference_std[None, :, :]
        return np.sqrt((deltas**2).sum(axis=2)).min(axis=1)


def design_bounds(
    real_features: pd.DataFrame,
    *,
    expansion: float = 0.12,
    caps_low: dict[str, float] | None = None,
    caps_high: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanded real-log bounding box in transformed space, with absolute caps."""
    points = to_target_space(real_features)
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    low = low - expansion * span
    high = high + expansion * span
    for index, feature in enumerate(TARGET_FEATURES):
        if caps_low and feature in caps_low:
            cap = caps_low[feature]
            low[index] = max(low[index], np.log10(cap) if feature in LOG10_FEATURES else cap)
        if caps_high and feature in caps_high:
            cap = caps_high[feature]
            high[index] = min(high[index], np.log10(cap) if feature in LOG10_FEATURES else cap)
        if feature not in LOG10_FEATURES:
            low[index] = max(low[index], 0.0)
            high[index] = min(high[index], 1.0)
        high[index] = max(high[index], low[index] + 1e-6)
    return low, high


def pairwise_binned_coverage(
    points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    n_bins: int = 3,
) -> dict[str, Any]:
    """Fraction of occupied cells over all pairwise n_bins x n_bins grids.

    A simple audit of feature-space coverage: each transformed axis is
    discretized into ``n_bins`` equal-width levels inside the design bounds
    (out-of-range points land in the edge bins), and occupancy is counted for
    every axis pair.
    """
    low, high = bounds
    points = np.atleast_2d(points)
    span = np.maximum(high - low, 1e-9)
    bins = np.clip(((points - low) / span * n_bins).astype(int), 0, n_bins - 1)

    per_pair: dict[str, int] = {}
    occupied_total = 0
    pairs = list(combinations(range(len(TARGET_FEATURES)), 2))
    for i, j in pairs:
        occupied = len(set(zip(bins[:, i].tolist(), bins[:, j].tolist(), strict=True)))
        per_pair[f"{TARGET_FEATURES[i]}|{TARGET_FEATURES[j]}"] = occupied
        occupied_total += occupied
    total_cells = len(pairs) * n_bins * n_bins
    return {
        "occupied_cells": occupied_total,
        "total_cells": total_cells,
        "coverage": occupied_total / total_cells,
        "per_pair": per_pair,
    }
