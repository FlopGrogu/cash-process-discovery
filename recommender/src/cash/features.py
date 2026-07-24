"""
Log feature extraction, ported verbatim from the feature-engineering notebook
(Feature_Extension_Exploration.ipynb, repo root). The functions are copied
unchanged so that feature values match the notebook exactly; a thin wrapper
``extract_log_features()`` runs the pipeline (own XES parser -> case features
-> compute_extended_features_for_log) on a single .xes/.xes.gz file.

Do not refactor this file: exact agreement with the notebook is the point.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import gzip
import math
import xml.etree.ElementTree as ET

import pandas as pd

try:
    import networkx as nx
except Exception:
    nx = None

TOP_N_VALUES = 10  # referenced by some notebook helpers

def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def typed_value(node):
    value = node.attrib.get("value")
    kind = local_name(node.tag)
    if value is None:
        return None
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "boolean":
        return value.lower() == "true"
    return value


def attrs_from_children(node) -> dict:
    attrs = {}
    for child in list(node):
        key = child.attrib.get("key")
        if key:
            attrs[key] = typed_value(child)
    return attrs


def open_xes(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_xes(path: Path, max_traces: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    case_rows = []
    trace_count = 0

    with open_xes(path) as fh:
        for _, trace in ET.iterparse(fh, events=("end",)):
            if local_name(trace.tag) != "trace":
                continue

            trace_count += 1
            trace_attrs = {}
            event_idx = 0

            for child in list(trace):
                if local_name(child.tag) == "event":
                    event_idx += 1
                    row = attrs_from_children(child)
                    row["case:concept:name"] = trace_attrs.get("concept:name", trace_count)
                    row["event_index"] = event_idx
                    event_rows.append(row)
                else:
                    key = child.attrib.get("key")
                    if key:
                        trace_attrs[key] = typed_value(child)

            case_rows.append({
                "case:concept:name": trace_attrs.get("concept:name", trace_count),
                **trace_attrs,
                "event_count": event_idx,
            })
            trace.clear()

            if max_traces is not None and trace_count >= max_traces:
                break

    events = pd.DataFrame(event_rows)
    cases = pd.DataFrame(case_rows)

    if "time:timestamp" in events.columns:
        events["time:timestamp"] = pd.to_datetime(events["time:timestamp"], utc=True, errors="coerce")

    return events, cases


def variant_tuple(values: pd.Series) -> tuple[str, ...]:
    return tuple(values.dropna().astype(str).tolist())


def variant_string(values: pd.Series) -> str:
    return " > ".join(variant_tuple(values))


def build_case_features(events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "case:concept:name" not in events.columns:
        return cases.copy()

    sort_cols = ["case:concept:name"]
    if "time:timestamp" in events.columns:
        sort_cols.append("time:timestamp")
    if "event_index" in events.columns:
        sort_cols.append("event_index")
    ordered = events.sort_values(sort_cols).copy()

    grouped = ordered.groupby("case:concept:name", dropna=False)
    features = grouped.size().to_frame("event_count_derived")

    if "concept:name" in ordered.columns:
        features["first_activity"] = grouped["concept:name"].first()
        features["last_activity"] = grouped["concept:name"].last()
        features["unique_activities"] = grouped["concept:name"].nunique(dropna=True)
        features["variant"] = grouped["concept:name"].agg(variant_string)
        features["variant_tuple"] = grouped["concept:name"].agg(variant_tuple)

    if "time:timestamp" in ordered.columns:
        features["start_time"] = grouped["time:timestamp"].min()
        features["end_time"] = grouped["time:timestamp"].max()
        features["duration_hours"] = (features["end_time"] - features["start_time"]).dt.total_seconds() / 3600

    return cases.merge(features.reset_index(), on="case:concept:name", how="left")


def summarize_log(log_name: str, path: Path, events: pd.DataFrame, case_features: pd.DataFrame) -> dict:
    summary = {
        "log": log_name,
        "path": str(path.relative_to(BASE_DIR)),
        "events": len(events),
        "cases": len(case_features),
        "event_columns": len(events.columns),
        "case_feature_columns": len(case_features.columns),
    }
    if "concept:name" in events.columns:
        summary["activities"] = int(events["concept:name"].nunique(dropna=True))
    if "time:timestamp" in events.columns:
        summary["min_timestamp"] = events["time:timestamp"].min()
        summary["max_timestamp"] = events["time:timestamp"].max()
    if "event_count_derived" in case_features.columns and not case_features.empty:
        summary["events_per_case_mean"] = round(float(case_features["event_count_derived"].mean()), 3)
        summary["events_per_case_min"] = int(case_features["event_count_derived"].min())
        summary["events_per_case_max"] = int(case_features["event_count_derived"].max())
    if "variant" in case_features.columns:
        summary["variants"] = int(case_features["variant"].nunique(dropna=True))
    return summary


def direct_follow_pairs(events: pd.DataFrame) -> pd.DataFrame:
    required = {"case:concept:name", "concept:name", "event_index"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame(columns=["source", "target"])

    ordered = events.sort_values(["case:concept:name", "event_index"])
    next_events = ordered.groupby("case:concept:name")["concept:name"].shift(-1)
    return pd.DataFrame({"source": ordered["concept:name"], "target": next_events}).dropna()


def entropy_from_counts(counts) -> float | None:
    series = pd.Series(list(counts), dtype=float).dropna()
    series = series[series > 0]
    if series.empty:
        return None
    probabilities = series / series.sum()
    return float(-(probabilities * probabilities.map(math.log2)).sum())


def population_std(values) -> float | None:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return None
    return float(series.std(ddof=0))


def safe_percentile(values, q: float) -> float | None:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return None
    return float(series.quantile(q))


def split_variant(variant) -> tuple[str, ...]:
    if isinstance(variant, tuple):
        return tuple(str(x) for x in variant)
    if pd.isna(variant):
        return tuple()
    return tuple(part.strip() for part in str(variant).split(">") if part.strip())


def repeated_activity_count(trace: tuple[str, ...]) -> int:
    return len(trace) - len(set(trace))


def has_non_self_loop_repetition(trace: tuple[str, ...]) -> bool:
    # Mirrors the idea of Fig4PM repetition_per_trace_overview: repeated activity excluding direct self-loop repetition.
    window = []
    for activity in trace:
        if activity not in window:
            window.append(activity)
            continue
        position = len(window) - 1 - window[::-1].index(activity)
        if position == len(window) - 1:
            window.append(activity)
        else:
            return True
    return False


def count_length_two_loop_patterns(trace: tuple[str, ...]) -> int:
    return sum(1 for i in range(len(trace) - 2) if trace[i] == trace[i + 2] and trace[i] != trace[i + 1])


def flattened_prefix_entropy(distinct_traces: list[tuple[str, ...]]) -> float | None:
    prefix_counts = Counter()
    for trace in distinct_traces:
        for i in range(1, len(trace) + 1):
            prefix_counts[trace[:i]] += 1
    return entropy_from_counts(prefix_counts.values())


def prefix_entropy_k(traces: list[tuple[str, ...]], k: int) -> float | None:
    prefix_counts = Counter(trace[:k] for trace in traces if len(trace) >= k)
    return entropy_from_counts(prefix_counts.values())


def graph_metrics_from_pairs(pair_counts: pd.DataFrame, activities: set[str]) -> dict:
    n_nodes = len(activities)
    if pair_counts.empty or n_nodes == 0:
        return {
            "number_of_nodes": n_nodes,
            "number_of_arcs": 0,
            "average_node_degree": 0,
            "maximum_node_degree": 0,
            "density": 0,
            "structure": 1 if n_nodes == 0 else 1,
            "dfg_entropy_variable_degree": None,
            "number_of_graph_communities": None,
        }

    edges = [(row.source, row.target) for row in pair_counts.itertuples(index=False)]
    n_arcs = len(edges)
    degree = Counter({activity: 0 for activity in activities})
    in_degree = Counter({activity: 0 for activity in activities})
    out_degree = Counter({activity: 0 for activity in activities})
    for source, target in edges:
        out_degree[source] += 1
        in_degree[target] += 1
        degree[source] += 1
        degree[target] += 1

    density = n_arcs / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    result = {
        "number_of_nodes": n_nodes,
        "number_of_arcs": n_arcs,
        "average_node_degree": (2 * n_arcs / n_nodes) if n_nodes else 0,
        "maximum_node_degree": max(degree.values()) if degree else 0,
        "avg_in_degree_custom": sum(in_degree.values()) / n_nodes if n_nodes else 0,
        "avg_out_degree_custom": sum(out_degree.values()) / n_nodes if n_nodes else 0,
        "max_in_degree_custom": max(in_degree.values()) if in_degree else 0,
        "max_out_degree_custom": max(out_degree.values()) if out_degree else 0,
        "density": density,
        "structure": 1 - (n_arcs / (n_nodes ** 2)) if n_nodes else 1,
        "dfg_entropy_variable_degree": entropy_from_counts(degree.values()),
    }

    if nx is not None:
        graph = nx.DiGraph()
        graph.add_nodes_from(activities)
        graph.add_edges_from(edges)
        result["weakly_connected_components_custom"] = nx.number_weakly_connected_components(graph)
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(graph.to_undirected())
            result["number_of_graph_communities"] = len(list(communities))
        except Exception:
            result["number_of_graph_communities"] = None
    else:
        result["weakly_connected_components_custom"] = None
        result["number_of_graph_communities"] = None

    return result


def compute_extended_features_for_log(log_name: str, frames: dict[str, pd.DataFrame]) -> dict:
    events = frames["events"]
    case_features = frames["case_features"]
    row = {"log": log_name}
    if case_features.empty:
        return row

    trace_lengths = case_features.get("event_count_derived", pd.Series(dtype=float)).dropna()
    variant_series = case_features.get("variant", pd.Series(dtype=str)).dropna()
    trace_tuples = [split_variant(value) for value in case_features.get("variant_tuple", variant_series).dropna()]
    variant_counts = variant_series.value_counts(dropna=True)
    distinct_trace_tuples = []
    seen = set()
    for trace in trace_tuples:
        if trace not in seen:
            seen.add(trace)
            distinct_trace_tuples.append(trace)

    start_counts = case_features["first_activity"].value_counts(dropna=True) if "first_activity" in case_features else pd.Series(dtype=float)
    end_counts = case_features["last_activity"].value_counts(dropna=True) if "last_activity" in case_features else pd.Series(dtype=float)
    activity_counts = events["concept:name"].value_counts(dropna=True) if "concept:name" in events else pd.Series(dtype=float)

    n_traces = len(case_features)
    n_events = len(events)
    n_activities = int(activity_counts.size)

    # RS4PD baseline features.
    row["rs4pd_distinct_traces"] = int(variant_counts.size)
    row["rs4pd_total_traces"] = n_traces
    row["rs4pd_trace_length_avg"] = float(trace_lengths.mean()) if not trace_lengths.empty else None
    row["rs4pd_repetitions_intra_trace_avg"] = sum(repeated_activity_count(t) for t in trace_tuples) / len(trace_tuples) if trace_tuples else None
    row["rs4pd_distinct_events"] = n_activities
    row["rs4pd_total_events"] = n_events
    row["rs4pd_start_events"] = int(start_counts.size)
    row["rs4pd_end_events"] = int(end_counts.size)

    pairs = direct_follow_pairs(events)
    pair_counts = pairs.value_counts(["source", "target"]).reset_index(name="count") if not pairs.empty else pd.DataFrame(columns=["source", "target", "count"])
    if pair_counts.empty or n_activities == 0:
        row["rs4pd_flow_entropy"] = None
        row["rs4pd_flow_concurrency_ratio"] = None
        row["rs4pd_flow_density"] = None
        row["rs4pd_length_one_loops_count"] = None
    else:
        row["rs4pd_flow_entropy"] = entropy_from_counts(pair_counts["count"])
        observed_pairs = {(item.source, item.target) for item in pair_counts.itertuples(index=False)}
        bidirectional_pairs = {tuple(sorted((source, target))) for source, target in observed_pairs if source != target and (target, source) in observed_pairs}
        unordered_possible_pairs = n_activities * (n_activities - 1) / 2
        row["rs4pd_flow_concurrency_ratio"] = len(bidirectional_pairs) / unordered_possible_pairs if unordered_possible_pairs else 0
        row["rs4pd_flow_density"] = len(observed_pairs) / (n_activities * n_activities) if n_activities else None
        row["rs4pd_length_one_loops_count"] = int((pair_counts["source"] == pair_counts["target"]).sum())

    # ProReco/Fig4PM and custom extensions.
    row["n_traces"] = n_traces
    row["n_unique_traces"] = int(variant_counts.size)
    row["ratio_unique_traces_per_trace"] = row["n_unique_traces"] / n_traces if n_traces else None
    row["n_events"] = n_events

    if not variant_counts.empty:
        occurrences = variant_counts.tolist()
        row["ratio_most_common_variant"] = occurrences[0] / n_traces if n_traces else None
        for pct in [1, 5, 10, 20, 50, 75]:
            cutoff = int(len(occurrences) * (pct / 100))
            row[f"ratio_top_{pct}_variants"] = sum(occurrences[:cutoff]) / n_traces if n_traces else None
        row["ratio_top_3_variants_custom"] = sum(occurrences[:3]) / n_traces if n_traces else None
        row["mean_variant_occurrence"] = float(pd.Series(occurrences).mean())
        row["std_variant_occurrence"] = population_std(occurrences)
        row["trace_entropy"] = entropy_from_counts(occurrences)
    else:
        row["trace_entropy"] = None

    if not trace_lengths.empty:
        row["trace_len_min"] = float(trace_lengths.min())
        row["trace_len_max"] = float(trace_lengths.max())
        row["trace_len_mean"] = float(trace_lengths.mean())
        row["trace_len_median"] = float(trace_lengths.median())
        row["trace_len_std"] = population_std(trace_lengths)
        row["trace_len_variance"] = float(trace_lengths.var(ddof=0))
        row["trace_len_q1"] = safe_percentile(trace_lengths, 0.25)
        row["trace_len_q3"] = safe_percentile(trace_lengths, 0.75)
        row["trace_len_iqr"] = row["trace_len_q3"] - row["trace_len_q1"]
        row["trace_len_p90_custom"] = safe_percentile(trace_lengths, 0.90)

    row["n_unique_activities"] = n_activities
    if not activity_counts.empty:
        row["activities_max"] = float(activity_counts.max())
        row["activities_mean"] = float(activity_counts.mean())
        row["activities_std"] = population_std(activity_counts)
        row["activities_iqr"] = safe_percentile(activity_counts, 0.75) - safe_percentile(activity_counts, 0.25)
        row["activity_entropy_custom"] = entropy_from_counts(activity_counts)
        row["most_common_activity_share_custom"] = float(activity_counts.max() / n_events) if n_events else None
        row["rare_activity_ratio_custom"] = float((activity_counts <= 1).sum() / len(activity_counts)) if len(activity_counts) else None

    row["n_unique_start_activities"] = int(start_counts.size)
    if not start_counts.empty:
        row["start_activities_max"] = float(start_counts.max())
        row["start_activities_mean"] = float(start_counts.mean())
        row["start_activities_std"] = population_std(start_counts)
        row["start_event_entropy_custom"] = entropy_from_counts(start_counts)
        row["most_common_start_share_custom"] = float(start_counts.max() / n_traces) if n_traces else None

    row["n_unique_end_activities"] = int(end_counts.size)
    if not end_counts.empty:
        row["end_activities_max"] = float(end_counts.max())
        row["end_activities_mean"] = float(end_counts.mean())
        row["end_activities_std"] = population_std(end_counts)
        row["end_event_entropy_custom"] = entropy_from_counts(end_counts)
        row["most_common_end_share_custom"] = float(end_counts.max() / n_traces) if n_traces else None

    activities = set(activity_counts.index.astype(str)) if not activity_counts.empty else set()
    row.update(graph_metrics_from_pairs(pair_counts, activities))
    if not pair_counts.empty:
        row["dfg_edge_entropy_custom"] = entropy_from_counts(pair_counts["count"])
        row["top_dfg_edge_share_custom"] = float(pair_counts["count"].max() / pair_counts["count"].sum())
        row["rare_dfg_edge_ratio_custom"] = float((pair_counts["count"] <= 1).sum() / len(pair_counts))
        self_loop_activities = set(pair_counts.loc[pair_counts["source"] == pair_counts["target"], "source"])
        row["length_one_loops"] = len(self_loop_activities) / n_activities if n_activities else 0
    else:
        row["length_one_loops"] = None

    row["relative_number_of_traces_with_repetition"] = sum(has_non_self_loop_repetition(t) for t in trace_tuples) / len(trace_tuples) if trace_tuples else None
    row["avg_event_repetition_intra_trace"] = sum(repeated_activity_count(t) for t in trace_tuples) / len(trace_tuples) if trace_tuples else None
    row["length_two_loops_custom"] = sum(count_length_two_loop_patterns(t) for t in trace_tuples)
    row["prefix_entropy"] = flattened_prefix_entropy(distinct_trace_tuples)
    row["prefix_entropy_2_custom"] = prefix_entropy_k(trace_tuples, 2)
    row["prefix_entropy_3_custom"] = prefix_entropy_k(trace_tuples, 3)

    return row


# ---------------------------------------------------------------------------
# Wrapper: run the notebook's pipeline on one log file.
# ---------------------------------------------------------------------------

def extract_log_features(xes_path, max_traces: int | None = None) -> dict:
    """Feature dict for a single XES (.xes/.xes.gz) file, notebook semantics."""
    path = Path(xes_path)
    events, cases = load_xes(path, max_traces=max_traces)
    case_features = build_case_features(events, cases)
    frames = {"events": events, "cases": cases, "case_features": case_features}
    row = compute_extended_features_for_log(path.stem, frames)
    row.pop("log", None)
    return row


# --- canonical feature list & pipeline helpers ---
import warnings
import numpy as np

FEATURE_NAMES = [
    # RS4PD baseline (12)
    "rs4pd_distinct_traces",
    "rs4pd_total_traces",
    "rs4pd_trace_length_avg",
    "rs4pd_repetitions_intra_trace_avg",
    "rs4pd_distinct_events",
    "rs4pd_total_events",
    "rs4pd_start_events",
    "rs4pd_end_events",
    "rs4pd_flow_entropy",
    "rs4pd_flow_concurrency_ratio",
    "rs4pd_flow_density",
    "rs4pd_length_one_loops_count",
    # variant diversity
    "ratio_unique_traces_per_trace",
    "ratio_most_common_variant",
    "ratio_top_1_variants",
    "ratio_top_5_variants",
    "ratio_top_10_variants",
    "ratio_top_3_variants_custom",
    "trace_entropy",
    # trace length
    "trace_len_min",
    "trace_len_max",
    "trace_len_median",
    "trace_len_std",
    "trace_len_iqr",
    "trace_len_p90_custom",
    # start/end distribution
    "start_activities_max",
    "end_activities_max",
    "start_event_entropy_custom",
    "end_event_entropy_custom",
    "most_common_start_share_custom",
    "most_common_end_share_custom",
    # activity distribution
    "activity_entropy_custom",
    "most_common_activity_share_custom",
    "rare_activity_ratio_custom",
    # DFG/graph
    "number_of_arcs",
    "average_node_degree",
    "maximum_node_degree",
    "density",
    "dfg_entropy_variable_degree",
    "dfg_edge_entropy_custom",
    "top_dfg_edge_share_custom",
    # loops/rework
    "length_one_loops",
    "relative_number_of_traces_with_repetition",
    "avg_event_repetition_intra_trace",
    "length_two_loops_custom",
    # entropy
    "prefix_entropy",
    "prefix_entropy_2_custom",
    "prefix_entropy_3_custom",
]


def extract_features_from_xes(xes_path: str) -> dict:
    """The canonical 48 features for a XES (.xes/.xes.gz) file.

    The notebook's extractor emits ~80 raw keys (including intermediates and
    duplicates across feature families); this keeps the documented 48. Missing
    keys are filled with NaN so the schema is always complete.
    """
    row = extract_log_features(xes_path)
    return {k: row.get(k, np.nan) for k in FEATURE_NAMES}


def features_to_array(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEATURE_NAMES], dtype=float)


def nan_safe_normalize(feat_matrix, query):
    """Median-impute NaNs column-wise, then normalise by column std.

    Returns ``(feat_norm, query_norm)`` ready for a Euclidean KNN. Some features
    can be NaN (e.g. when networkx is unavailable); imputing keeps the distance
    well-defined instead of poisoning every row with NaN.
    """
    feat_matrix = np.array(feat_matrix, dtype=float)
    query = np.array(query, dtype=float)
    if feat_matrix.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
            col_median = np.nanmedian(feat_matrix, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median)
        nan_idx = np.where(np.isnan(feat_matrix))
        feat_matrix[nan_idx] = np.take(col_median, nan_idx[1])
        q_nan = np.isnan(query)
        query[q_nan] = col_median[q_nan]
    col_std = feat_matrix.std(axis=0)
    col_std[col_std == 0] = 1.0
    return feat_matrix / col_std, query / col_std
