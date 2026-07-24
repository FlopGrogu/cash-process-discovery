from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from process_discovery_cash.data.preprocessing.models import (
    LifecycleAnalysis,
    LifecycleSemantics,
)


def analyze_lifecycle(
    dataframe: Any,
    *,
    semantics: LifecycleSemantics,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
    lifecycle_column: str | None,
    start_timestamp_column: str | None = None,
) -> LifecycleAnalysis:
    if start_timestamp_column and start_timestamp_column in dataframe.columns:
        return _analyze_interval_columns(
            dataframe,
            timestamp_column=timestamp_column,
            start_timestamp_column=start_timestamp_column,
        )
    if semantics == "extended_standard":
        analysis = _analyze_lifecycle_events(
            dataframe,
            case_column=case_column,
            activity_column=activity_column,
            timestamp_column=timestamp_column,
            lifecycle_column=lifecycle_column,
        )
        analysis.interval_quality = "yellow"
        analysis.reasons = ["extended_lifecycle_requires_domain_specific_repair"]
        return analysis
    if semantics != "standard":
        reason = {
            "complete_only": "complete_only_lifecycle",
            "status_like": "lifecycle_values_are_workflow_states",
            "absent": "no_real_start_information",
        }[semantics]
        values = _lifecycle_value_counts(dataframe, lifecycle_column)
        return LifecycleAnalysis(
            interval_quality="red",
            reasons=[reason],
            lifecycle_values=values,
        )

    analysis = _analyze_lifecycle_events(
        dataframe,
        case_column=case_column,
        activity_column=activity_column,
        timestamp_column=timestamp_column,
        lifecycle_column=lifecycle_column,
    )
    return _classify_standard_analysis(analysis)


def _analyze_lifecycle_events(
    dataframe: Any,
    *,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
    lifecycle_column: str | None,
) -> LifecycleAnalysis:
    if not lifecycle_column or lifecycle_column not in dataframe.columns:
        return LifecycleAnalysis(
            interval_quality="red",
            reasons=["missing_lifecycle_attribute"],
        )

    values = _lifecycle_value_counts(dataframe, lifecycle_column)
    queues: dict[tuple[str, str], deque[Any]] = defaultdict(deque)
    starts = completes = paired = unmatched_completes = 0
    negative = zero = positive = 0
    columns = dataframe[
        [case_column, activity_column, timestamp_column, lifecycle_column]
    ].itertuples(index=False, name=None)
    for case_id, activity, timestamp, lifecycle in columns:
        transition = str(lifecycle).strip().lower() if lifecycle is not None else ""
        key = (str(case_id), str(activity))
        if transition == "start":
            starts += 1
            queues[key].append(timestamp)
        elif transition == "complete":
            completes += 1
            if queues[key]:
                start = queues[key].popleft()
                paired += 1
                duration = (timestamp - start).total_seconds()
                negative += int(duration < 0)
                zero += int(duration == 0)
                positive += int(duration > 0)
            else:
                unmatched_completes += 1

    return LifecycleAnalysis(
        interval_quality="red",
        lifecycle_values=values,
        starts=starts,
        completes=completes,
        paired=paired,
        unmatched_starts=sum(len(queue) for queue in queues.values()),
        unmatched_completes=unmatched_completes,
        negative_durations=negative,
        zero_durations=zero,
        positive_durations=positive,
    )


def _classify_standard_analysis(analysis: LifecycleAnalysis) -> LifecycleAnalysis:
    if analysis.starts == 0:
        analysis.reasons = ["no_start_events"]
        return analysis
    pair_ratio = analysis.paired / analysis.starts
    unmatched_ratio = analysis.unmatched_starts / analysis.starts
    zero_ratio = analysis.zero_durations / analysis.paired if analysis.paired else 1.0
    reasons: list[str] = []
    if pair_ratio < 0.999:
        reasons.append("start_pairing_coverage_below_99_9_percent")
    if unmatched_ratio > 0.001:
        reasons.append("unmatched_start_ratio_above_0_1_percent")
    if analysis.negative_durations:
        reasons.append("negative_durations")
    if zero_ratio > 0.01:
        reasons.append("zero_duration_ratio_above_1_percent")
    if reasons:
        analysis.interval_quality = "yellow"
        analysis.reasons = reasons
    else:
        analysis.interval_quality = "green"
        analysis.reasons = ["trustworthy_source_start_complete_intervals"]
    return analysis


def _analyze_interval_columns(
    dataframe: Any,
    *,
    timestamp_column: str,
    start_timestamp_column: str,
) -> LifecycleAnalysis:
    valid = dataframe[[start_timestamp_column, timestamp_column]].dropna()
    if valid.empty:
        return LifecycleAnalysis(
            interval_quality="red",
            reasons=["no_complete_interval_rows"],
        )
    durations = (valid[timestamp_column] - valid[start_timestamp_column]).dt.total_seconds()
    missing = int(dataframe[start_timestamp_column].isna().sum())
    analysis = LifecycleAnalysis(
        interval_quality="red",
        starts=len(valid),
        completes=len(valid),
        paired=len(valid),
        unmatched_starts=missing,
        negative_durations=int((durations < 0).sum()),
        zero_durations=int((durations == 0).sum()),
        positive_durations=int((durations > 0).sum()),
    )
    return _classify_standard_analysis(analysis)


def _lifecycle_value_counts(dataframe: Any, lifecycle_column: str | None) -> dict[str, int]:
    if not lifecycle_column or lifecycle_column not in dataframe.columns:
        return {}
    counts = dataframe[lifecycle_column].fillna("<missing>").astype(str).str.lower().value_counts()
    return {str(key): int(value) for key, value in counts.items()}
