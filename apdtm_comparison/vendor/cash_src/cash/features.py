"""
Log feature extraction.

Feature *values* are computed by the extended log-feature implementation in
``cash.log_features``. This module only exposes the stable feature-name list and
the thin helpers the rest of the pipeline relies on.

``FEATURE_NAMES`` is the fixed 80-feature set used to build the CASH dataset.
Do not reorder or trim it -- it must match what
``compute_extended_features_for_log`` returns, so the dataset columns stay
identical to the trained data.
"""

from __future__ import annotations

import warnings

import numpy as np

from cash.log_features import extract_log_features

LOG_FEATURE_NAMES = [
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
    "n_traces",
    "n_unique_traces",
    "ratio_unique_traces_per_trace",
    "n_events",
    "ratio_most_common_variant",
    "ratio_top_1_variants",
    "ratio_top_5_variants",
    "ratio_top_10_variants",
    "ratio_top_20_variants",
    "ratio_top_50_variants",
    "ratio_top_75_variants",
    "ratio_top_3_variants_custom",
    "mean_variant_occurrence",
    "std_variant_occurrence",
    "trace_entropy",
    "trace_len_min",
    "trace_len_max",
    "trace_len_mean",
    "trace_len_median",
    "trace_len_std",
    "trace_len_variance",
    "trace_len_q1",
    "trace_len_q3",
    "trace_len_iqr",
    "trace_len_p90_custom",
    "n_unique_activities",
    "activities_max",
    "activities_mean",
    "activities_std",
    "activities_iqr",
    "activity_entropy_custom",
    "most_common_activity_share_custom",
    "rare_activity_ratio_custom",
    "n_unique_start_activities",
    "start_activities_max",
    "start_activities_mean",
    "start_activities_std",
    "start_event_entropy_custom",
    "most_common_start_share_custom",
    "n_unique_end_activities",
    "end_activities_max",
    "end_activities_mean",
    "end_activities_std",
    "end_event_entropy_custom",
    "most_common_end_share_custom",
    "number_of_nodes",
    "number_of_arcs",
    "average_node_degree",
    "maximum_node_degree",
    "avg_in_degree_custom",
    "avg_out_degree_custom",
    "max_in_degree_custom",
    "max_out_degree_custom",
    "density",
    "structure",
    "dfg_entropy_variable_degree",
    "weakly_connected_components_custom",
    "number_of_graph_communities",
    "dfg_edge_entropy_custom",
    "top_dfg_edge_share_custom",
    "rare_dfg_edge_ratio_custom",
    "length_one_loops",
    "relative_number_of_traces_with_repetition",
    "avg_event_repetition_intra_trace",
    "length_two_loops_custom",
    "prefix_entropy",
    "prefix_entropy_2_custom",
    "prefix_entropy_3_custom",
]

FEATURE_NAMES = LOG_FEATURE_NAMES


def extract_features_from_xes(xes_path: str) -> dict:
    """Extended log features for a XES (.xes/.xes.gz) file as a dict.

    Missing keys (rare edge cases in the extractor) are filled with NaN so the
    schema is always the full FEATURE_NAMES set.
    """
    row = extract_log_features(xes_path)
    return {k: row.get(k, np.nan) for k in FEATURE_NAMES}


def extract_features_from_split(split: dict) -> dict:
    """Deprecated. Extended features need the full log, not the split_features
    summary, so this returns NaN for every feature (kept only so the legacy
    aggregate path does not crash)."""
    return {k: np.nan for k in FEATURE_NAMES}


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
