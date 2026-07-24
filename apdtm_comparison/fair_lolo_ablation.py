"""
Fairness-aware LOLO ablation for APDTM vs CASH.

This script decomposes the comparison into model, feature, data, action-space,
and hyperparameter dimensions. It evaluates every setup with leave-one-real-log-
out on the 18 real logs that have both CASH and APDTM data, removes augmented
logs from the held-out family, and writes model artifacts plus result tables.

Outputs:
  apdtm_comparison/outputs/fair_lolo_ablation/
    setup_manifest.csv
    lolo_results.csv
    lolo_summary.csv
    model_manifest.csv
    models/*.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "fair_lolo_ablation"

warnings.filterwarnings("ignore", category=PerformanceWarning)

MEASURES = ["fitness", "precision", "generalization", "simplicity"]
APDTM_VARIANTS = ["AM", "HM", "IM", "IMf", "IMd"]
APDTM_RANK_METRICS = [
    "fitness_rank",
    "precision_rank",
    "generalization_rank",
    "simplicity_rank",
    "discovery_time_rank",
]

APDTM_TO_CASH_ALGORITHM = {
    "AM": "alpha_miner_classic",
    "HM": "heuristics_miner",
    "IM": "inductive_miner_im",
    "IMd": "inductive_miner_imd",
    "IMf": "inductive_miner_imf",
}
CASH_TO_APDTM_VARIANT = {v: k for k, v in APDTM_TO_CASH_ALGORITHM.items()}

APDTM_LOG_TO_CASH_FAMILY = {
    "bpi_2012": "bpi2012",
    "bpi_2013_closed_problems": "bpi2013_closed_problems",
    "bpi_2013_incidents": "bpi2013_incidents",
    "sepsis": "sepsis",
}

HYPERPARAM_COLS = [
    "and_threshold",
    "dependency_threshold",
    "dfg_pre_cleaning_noise_thresh",
    "loop_two_threshold",
    "min_act_count",
    "min_dfg_occurrences",
    "noise_threshold",
    "disable_fallthroughs",
    "generations",
    "population_size",
    "mutation_rate",
    "crossover_rate",
    "elitism_rate",
    "elitism_min_sample",
    "alpha",
    "epsilon",
    "eta",
    "parallelismFirst",
    "removeLoopActivityMarkers",
    "replaceIORs",
]

# Reference defaults used only when dataset_v8 has no recorded v6_baseline_*
# row for a log/algorithm pair. The preferred source for "default-only" rows is
# still the measured baseline experiment itself. These values are only a
# fallback for algorithms/logs where no measured baseline row exists.
DEFAULT_HYPERPARAMS = {
    "heuristics_miner": {
        "dependency_threshold": 0.5,
        "and_threshold": 0.65,
        "loop_two_threshold": 0.5,
        "dfg_pre_cleaning_noise_thresh": 0.05,
        "min_act_count": 1.0,
        "min_dfg_occurrences": 1.0,
    },
    "heuristics_miner_plusplus": {
        "dependency_threshold": 0.5,
        "and_threshold": 0.65,
        "loop_two_threshold": 0.5,
        "dfg_pre_cleaning_noise_thresh": 0.05,
        "min_act_count": 1.0,
        "min_dfg_occurrences": 1.0,
    },
    "inductive_miner_im": {"disable_fallthroughs": 0.0},
    "inductive_miner_imd": {"disable_fallthroughs": 0.0},
    "inductive_miner_imf": {"noise_threshold": 0.0, "disable_fallthroughs": 0.0},
    "ilp_miner": {"alpha": 1.0},
    "split_miner": {
        "epsilon": 0.5,
        "eta": 0.5,
        "parallelismFirst": 0.0,
        "removeLoopActivityMarkers": 0.0,
        "replaceIORs": 0.0,
    },
    "genetic_miner": {
        "generations": 100.0,
        "population_size": 500.0,
        "mutation_rate": 0.01,
        "crossover_rate": 1.0,
        "elitism_rate": 0.01,
        "elitism_min_sample": 5.0,
    },
}


@dataclass(frozen=True)
class Setup:
    setup_id: str
    description: str
    model_kind: str
    feature_set: str
    data_regime: str
    action_space: str
    n_estimators: int
    random_state: int


SETUPS = [
    Setup(
        "B0_apdtm_cls_apdtm_features_original",
        "APDTM faithful baseline: original APDTM meta-database only.",
        "classifier",
        "apdtm",
        "apdtm_original",
        "apdtm5_default",
        100,
        10,
    ),
    Setup(
        "B1_apdtm_cls_apdtm_features_original_plus_cash_real",
        "APDTM classifier with original APDTM plus CASH real APDTM-feature rows.",
        "classifier",
        "apdtm",
        "apdtm_original_plus_cash_real",
        "apdtm5_default",
        100,
        10,
    ),
    Setup(
        "B2_apdtm_cls_cash_features_cash_real_default",
        "APDTM-style classifier trained on CASH features, real logs only.",
        "classifier",
        "cash",
        "cash_real",
        "apdtm5_default",
        100,
        10,
    ),
    Setup(
        "B3_apdtm_cls_cash_features_cash_real_synthetic_default",
        "APDTM-style classifier trained on CASH features, real plus synthetic logs.",
        "classifier",
        "cash",
        "cash_real_synthetic",
        "apdtm5_default",
        100,
        10,
    ),
    Setup(
        "B4_apdtm_cls_cash_features_cash_full_clean_default",
        "APDTM-style classifier trained on CASH features, full CASH dataset.",
        "classifier",
        "cash",
        "cash_full",
        "apdtm5_default",
        100,
        10,
    ),
    Setup(
        "C0_cash_reg_apdtm_features_cash_real_default",
        "CASH-style regressor using APDTM features and APDTM default metrics on real logs.",
        "regressor",
        "apdtm",
        "cash_real",
        "apdtm5_default",
        200,
        42,
    ),
    Setup(
        "C1_cash_reg_cash_features_cash_real_default",
        "CASH regressor using CASH features, real logs only, APDTM-5 default representatives.",
        "regressor",
        "cash",
        "cash_real",
        "apdtm5_default",
        200,
        42,
    ),
    Setup(
        "C2_cash_reg_cash_features_cash_full_clean_apdtm5_default",
        "CASH regressor, full CASH dataset, APDTM-5 default representatives.",
        "regressor",
        "cash",
        "cash_full",
        "apdtm5_default",
        200,
        42,
    ),
    Setup(
        "C3_cash_reg_cash_features_cash_full_clean_apdtm5_hparams",
        "CASH regressor, full CASH dataset, APDTM-compatible algorithms with hyperparameters.",
        "regressor",
        "cash",
        "cash_full",
        "apdtm5_hparams",
        200,
        42,
    ),
    Setup(
        "C4_cash_reg_cash_features_cash_full_clean_all_algos_default",
        "CASH regressor, full CASH dataset, all CASH algorithms default representatives.",
        "regressor",
        "cash",
        "cash_full",
        "cash_all_default",
        200,
        42,
    ),
    Setup(
        "C5_cash_reg_cash_features_cash_full_clean_all_algos_hparams",
        "CASH regressor, full CASH dataset, all CASH algorithms and observed hyperparameters.",
        "regressor",
        "cash",
        "cash_full",
        "cash_all_hparams",
        200,
        42,
    ),
]


def weight_sets() -> dict[str, dict[str, float]]:
    sets = {"equal": {m: 0.25 for m in MEASURES}}
    for m in MEASURES:
        sets[m[:4]] = {x: (1.0 if x == m else 0.0) for x in MEASURES}
    for i in range(len(MEASURES)):
        for j in range(i + 1, len(MEASURES)):
            a, b = MEASURES[i], MEASURES[j]
            sets[f"{a[:3]}+{b[:3]}"] = {x: (1.0 if x in (a, b) else 0.0) for x in MEASURES}
    sets["mix-fit"] = {"fitness": 0.4, "precision": 0.3, "generalization": 0.1, "simplicity": 0.2}
    sets["mix-prec"] = {"fitness": 0.2, "precision": 0.4, "generalization": 0.3, "simplicity": 0.1}
    sets["mix-simp"] = {"fitness": 0.3, "precision": 0.1, "generalization": 0.2, "simplicity": 0.4}
    sets["mix-gen"] = {"fitness": 0.1, "precision": 0.2, "generalization": 0.4, "simplicity": 0.3}
    return sets


def is_augmented(log_id: str) -> bool:
    return str(log_id).startswith("aug_")


def is_synthetic(log_id: str) -> bool:
    return str(log_id).startswith("syn_")


def is_real_cash_log(log_id: str) -> bool:
    return not is_augmented(log_id) and not is_synthetic(log_id)


def cash_family(log_id: str) -> str:
    """Map augmented CASH logs back to the real log family they came from."""
    log_id = str(log_id)
    if log_id.startswith("aug_"):
        return log_id[len("aug_") :].split("__", 1)[0]
    return log_id


def apdtm_family(log_id: str) -> str:
    """Map overlapping APDTM original logs to the corresponding CASH family."""
    return APDTM_LOG_TO_CASH_FAMILY.get(str(log_id), f"apdtm::{log_id}")


def composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values()) or 1.0
    return sum(df[m].astype(float) * weights[m] for m in MEASURES) / total


def score_row(
    test_full: pd.DataFrame,
    action_candidates: pd.DataFrame,
    chosen_index,
    weights: dict[str, float],
) -> dict:
    full_scores = composite(test_full, weights)
    action_scores = composite(action_candidates, weights)
    chosen_score = float(full_scores.loc[chosen_index])
    full_best = float(full_scores.max())
    full_worst = float(full_scores.min())
    action_best = float(action_scores.max())
    action_worst = float(action_scores.min())
    full_denom = full_best - full_worst
    action_denom = action_best - action_worst
    return {
        "real_score": chosen_score,
        "full_best_score": full_best,
        "full_worst_score": full_worst,
        "action_best_score": action_best,
        "action_worst_score": action_worst,
        "regret_full": full_best - chosen_score,
        "regret_action": action_best - chosen_score,
        "accuracy_full": 1.0 if full_denom == 0 else (chosen_score - full_worst) / full_denom,
        "accuracy_action": 1.0 if action_denom == 0 else (chosen_score - action_worst) / action_denom,
        "rank_full": int((full_scores > chosen_score).sum()) + 1,
        "rank_action": int((action_scores > chosen_score).sum()) + 1,
        "n_full_configs": int(len(test_full)),
        "n_action_configs": int(len(action_candidates)),
    }


def add_apdtm_ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    """Recreate APDTM's target ranking from discovery metrics.

    Runtime is ranked ascending because lower time is better. The four quality
    metrics are ranked descending because higher values are better. APDTM then
    averages these five ranks and treats the smallest average rank as the best
    algorithm for that log.
    """
    out = metrics.copy()
    if "status" in out.columns:
        out = out[out["status"].fillna("ok").eq("ok")].copy()
    if "fitness" in out.columns and "log_fitness" not in out.columns:
        out = out.rename(columns={"fitness": "log_fitness"})
    out = out.dropna(subset=["discovery_time", "log_fitness", "precision", "generalization", "simplicity"])
    out["discovery_time_rank"] = out.groupby("log")["discovery_time"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    out["fitness_rank"] = out.groupby("log")["log_fitness"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    out["precision_rank"] = out.groupby("log")["precision"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    out["generalization_rank"] = out.groupby("log")["generalization"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    out["simplicity_rank"] = out.groupby("log")["simplicity"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    out["target_final_rank"] = out[APDTM_RANK_METRICS].mean(axis=1)
    return out


def build_apdtm_meta(features: pd.DataFrame, metrics: pd.DataFrame, source: str) -> pd.DataFrame:
    ranked = add_apdtm_ranks(metrics)
    best = ranked.groupby("log", as_index=False)["target_final_rank"].min()
    targets = ranked.merge(best, on=["log", "target_final_rank"])
    # Match APDTM's original code: if more than one algorithm shares the best
    # average rank for a log, remove that log from the labelled meta-dataset.
    targets = targets.drop_duplicates(subset="log", keep=False)
    meta = (
        targets[["log", "variant"]]
        .set_index("log")
        .join(features.set_index("log"))
        .reset_index()
    )
    meta["source"] = source
    meta["family"] = meta["log"].map(apdtm_family if source == "apdtm_original" else cash_family)
    return meta


def load_inputs(project_root: Path, package_dir: Path):
    bundled_dataset = package_dir / "data" / "dataset_v8.csv"
    root_dataset = project_root / "dataset_v8.csv"
    dataset_path = bundled_dataset if bundled_dataset.exists() else root_dataset
    cash_df = pd.read_csv(dataset_path)
    cash_df["family"] = cash_df["log_id"].map(cash_family)

    apdtm_features = pd.read_csv(package_dir / "vendor/process_discovery_meta_learning/log_meta_features.csv")
    apdtm_metrics = pd.read_csv(package_dir / "vendor/process_discovery_meta_learning/discovery_metrics.csv")
    apdtm_original_meta = build_apdtm_meta(apdtm_features, apdtm_metrics, "apdtm_original")

    cash_apdtm_features = pd.read_csv(package_dir / "outputs/apdtm_cash_real_log_meta_features.csv")
    cash_apdtm_metrics = pd.read_csv(package_dir / "outputs/apdtm_cash_real_discovery_metrics.csv")
    cash_apdtm_meta = build_apdtm_meta(cash_apdtm_features, cash_apdtm_metrics, "cash_raw")

    cash_real_logs = {log for log in cash_df["log_id"].astype(str).unique() if is_real_cash_log(log)}
    apdtm_cash_logs = set(cash_apdtm_features["log"].astype(str).unique())
    # The APDTM comparison is evaluated on the common real-log set for which
    # CASH measurements and APDTM features/metrics are both available. The full
    # CASH dataset can still be used for training in CASH setups.
    real_logs = sorted(cash_real_logs & apdtm_cash_logs)
    cash_feature_cols = [c for c in cash_df.columns[: cash_df.columns.get_loc("algorithm")]]
    apdtm_feature_cols = [
        c
        for c in apdtm_original_meta.columns
        if c not in {"log", "variant", "source", "family"}
        and pd.api.types.is_numeric_dtype(apdtm_original_meta[c])
    ]
    return cash_df, apdtm_original_meta, cash_apdtm_meta, real_logs, cash_feature_cols, apdtm_feature_cols


def row_default_distance(row: pd.Series) -> float:
    """Return how close an observed CASH row is to the documented default.

    This is used only as a fallback when no measured v6_baseline_* row exists.
    For parameterless algorithms such as Alpha Miner, "distance" is simply the
    number of filled hyperparameter columns; an all-empty row is therefore the
    best/default representative.

    For algorithms with documented defaults, the distance is a small Euclidean-
    style score over the default hyperparameter columns:
      - matching numeric values add 0;
      - numeric deviations add squared error;
      - missing expected default columns receive a large penalty;
      - unexpected filled hyperparameter columns receive a tiny penalty.
    """
    defaults = DEFAULT_HYPERPARAMS.get(str(row["algorithm"]), {})
    if not defaults:
        present = row[HYPERPARAM_COLS].notna().sum()
        return float(present)
    dist = 0.0
    for col, default in defaults.items():
        value = row.get(col, np.nan)
        if pd.isna(value):
            dist += 10.0
        else:
            dist += float(value - default) ** 2
    for col in HYPERPARAM_COLS:
        if col not in defaults and pd.notna(row.get(col, np.nan)):
            dist += 0.01
    return float(math.sqrt(dist))


def default_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """Select one default representative per log and algorithm.

    Selection priority:
      1. Prefer measured v6_baseline_* rows from dataset_v8.
      2. If no baseline row exists, prefer rows with all hyperparameter columns
         empty. This covers parameterless Alpha Miner rows.
      3. Otherwise choose the observed row nearest to DEFAULT_HYPERPARAMS.
      4. If multiple rows are equally default-like, keep the one with the best
         equal-weight realised quality so the tie break is deterministic and
         performance-aware.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_is_recorded_baseline"] = out["experiment_id"].astype(str).str.contains(
        "baseline", case=False, na=False
    )
    out["_all_hyperparams_empty"] = out[HYPERPARAM_COLS].isna().all(axis=1)
    out["_default_distance"] = out.apply(row_default_distance, axis=1)
    out["_equal_score"] = composite(out, {m: 0.25 for m in MEASURES})
    out = (
        out.sort_values(
            [
                "log_id",
                "algorithm",
                "_is_recorded_baseline",
                "_all_hyperparams_empty",
                "_default_distance",
                "_equal_score",
            ],
            ascending=[True, True, False, False, True, False],
        )
        .drop_duplicates(subset=["log_id", "algorithm"], keep="first")
        .drop(columns=["_default_distance", "_equal_score", "_is_recorded_baseline", "_all_hyperparams_empty"])
    )
    return out


def cash_action_rows(df: pd.DataFrame, action_space: str) -> pd.DataFrame:
    """Restrict the evaluated candidates to the setup's action space."""
    out = df.dropna(subset=MEASURES).copy()
    if action_space.startswith("apdtm5"):
        out = out[out["algorithm"].isin(CASH_TO_APDTM_VARIANT)].copy()
    if action_space.endswith("default"):
        out = default_representatives(out)
    return out


def cash_data_regime_rows(df: pd.DataFrame, data_regime: str) -> pd.DataFrame:
    """Select which CASH logs are allowed before LOLO family filtering."""
    if data_regime == "cash_real":
        return df[df["log_id"].astype(str).map(is_real_cash_log)].copy()
    if data_regime == "cash_real_synthetic":
        return df[~df["log_id"].astype(str).map(is_augmented)].copy()
    if data_regime == "cash_full":
        return df.copy()
    raise ValueError(f"Unsupported CASH data regime: {data_regime}")


def cash_classifier_meta(df: pd.DataFrame, feature_cols: list[str], action_space: str) -> pd.DataFrame:
    """Build APDTM-style classifier labels from CASH rows.

    The classifier needs one label per log. We first score default candidates
    with equal metric weights, then keep the unique best algorithm. Logs with
    ties are removed, mirroring APDTM's single-label target construction.
    """
    candidates = cash_action_rows(df, action_space)
    if candidates.empty:
        return pd.DataFrame()
    candidates["_score"] = composite(candidates, {m: 0.25 for m in MEASURES})
    best_score = candidates.groupby("log_id", as_index=False)["_score"].max()
    targets = candidates.merge(best_score, on=["log_id", "_score"])
    targets = targets.drop_duplicates(subset="log_id", keep=False)
    rows = []
    for _, row in targets.iterrows():
        algorithm = str(row["algorithm"])
        if algorithm not in CASH_TO_APDTM_VARIANT:
            continue
        record = {"log_id": row["log_id"], "family": row["family"], "variant": CASH_TO_APDTM_VARIANT[algorithm]}
        for col in feature_cols:
            record[col] = row[col]
        rows.append(record)
    return pd.DataFrame(rows)


def apdtm_training_meta(
    setup: Setup,
    test_log: str | None,
    apdtm_original_meta: pd.DataFrame,
    cash_apdtm_meta: pd.DataFrame,
) -> pd.DataFrame:
    frames = [apdtm_original_meta.copy()]
    if setup.data_regime == "apdtm_original_plus_cash_real":
        frames.append(cash_apdtm_meta.copy())
    meta = pd.concat(frames, ignore_index=True, sort=False)
    if test_log is not None:
        # LOLO leakage guard: remove the held-out CASH family from original
        # APDTM logs and from added CASH real logs before fitting the classifier.
        meta = meta[meta["family"].ne(test_log)].copy()
    return meta


def cash_training_rows(setup: Setup, test_log: str | None, cash_df: pd.DataFrame) -> pd.DataFrame:
    rows = cash_data_regime_rows(cash_df, setup.data_regime)
    if test_log is not None:
        # Remove the held-out real log and all augmented variants derived from
        # that same family. This is the key fair LOLO split for CASH data.
        rows = rows[rows["family"].ne(test_log)].copy()
    return cash_action_rows(rows, setup.action_space)


def train_classifier(meta: pd.DataFrame, feature_cols: list[str], setup: Setup):
    train = meta.dropna(subset=["variant"]).copy()
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=setup.n_estimators, random_state=setup.random_state, n_jobs=-1)),
    ])
    clf.fit(train[feature_cols], train["variant"])
    return clf


def train_regressor(rows: pd.DataFrame, feature_cols: list[str], setup: Setup):
    train = rows.dropna(subset=MEASURES, how="all").copy()
    le = LabelEncoder()
    train["_algorithm_code"] = le.fit_transform(train["algorithm"].astype(str))
    numeric_cols = feature_cols + HYPERPARAM_COLS + ["_algorithm_code"]
    models = {}
    for measure in MEASURES:
        # CASH predicts each quality metric separately. Recommendation happens
        # later by combining the predicted metrics with the requested weights.
        sub = train.dropna(subset=[measure]).copy()
        preprocessor = ColumnTransformer(
            [("num", SimpleImputer(strategy="median"), numeric_cols)],
            remainder="drop",
        )
        pipe = Pipeline([
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=setup.n_estimators, random_state=setup.random_state, n_jobs=-1)),
        ])
        pipe.fit(sub[numeric_cols], sub[measure])
        models[measure] = pipe
    return {"models": models, "label_encoder": le, "numeric_cols": numeric_cols}


def predict_regressor(regressor_artifact: dict, candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    le = regressor_artifact["label_encoder"]
    known = out["algorithm"].astype(str).isin(le.classes_)
    out["_algorithm_code"] = np.nan
    out.loc[known, "_algorithm_code"] = le.transform(out.loc[known, "algorithm"].astype(str))
    numeric_cols = regressor_artifact["numeric_cols"]
    for measure, model in regressor_artifact["models"].items():
        out[f"_pred_{measure}"] = np.nan
        if known.any():
            out.loc[known, f"_pred_{measure}"] = model.predict(out.loc[known, numeric_cols])
    return out


def available_variants_from_candidates(candidates: pd.DataFrame) -> list[str]:
    variants = []
    for algorithm in sorted(candidates["algorithm"].astype(str).unique()):
        variant = CASH_TO_APDTM_VARIANT.get(algorithm)
        if variant:
            variants.append(variant)
    return variants


def constrained_classifier_predict(clf, feature_row: pd.DataFrame, allowed_variants: list[str]) -> str:
    """Predict the most likely APDTM variant that is evaluable for this log."""
    probabilities = clf.predict_proba(feature_row)[0]
    class_probabilities = dict(zip(clf.classes_, probabilities))
    available = [variant for variant in allowed_variants if variant in class_probabilities]
    if not available:
        return str(clf.predict(feature_row)[0])
    return max(available, key=lambda variant: class_probabilities[variant])


def choose_default_candidate_for_variant(default_candidates: pd.DataFrame, variant: str):
    algorithm = APDTM_TO_CASH_ALGORITHM.get(variant)
    if algorithm is None:
        return None
    sub = default_candidates[default_candidates["algorithm"].astype(str).eq(algorithm)]
    if sub.empty:
        return None
    return sub.index[0]


def evaluate_classifier_setup(
    setup: Setup,
    cash_df: pd.DataFrame,
    apdtm_original_meta: pd.DataFrame,
    cash_apdtm_meta: pd.DataFrame,
    real_logs: list[str],
    cash_feature_cols: list[str],
    apdtm_feature_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    rows = []
    train_counts = []
    for fold_idx, test_log in enumerate(real_logs, start=1):
        print(f"[fold] {setup.setup_id} {fold_idx}/{len(real_logs)} test={test_log}", flush=True)
        test_full = cash_df[cash_df["log_id"].astype(str).eq(test_log)].dropna(subset=MEASURES).copy()
        # APDTM-style classifiers recommend an algorithm family only, not a
        # tuned configuration, so evaluation uses one default candidate per
        # APDTM-compatible algorithm.
        default_candidates = cash_action_rows(test_full, "apdtm5_default")
        if test_full.empty or default_candidates.empty:
            continue

        if setup.feature_set == "apdtm":
            train_meta = apdtm_training_meta(setup, test_log, apdtm_original_meta, cash_apdtm_meta)
            feature_cols = apdtm_feature_cols
            test_feature = cash_apdtm_meta[cash_apdtm_meta["log"].astype(str).eq(test_log)]
            if test_feature.empty:
                continue
            test_feature = test_feature.iloc[[0]][feature_cols]
        else:
            regime_rows = cash_data_regime_rows(cash_df, setup.data_regime)
            if setup.data_regime == "cash_full":
                regime_rows = regime_rows[regime_rows["family"].ne(test_log)].copy()
            elif setup.data_regime in {"cash_real", "cash_real_synthetic"}:
                regime_rows = regime_rows[regime_rows["family"].ne(test_log)].copy()
            train_meta = cash_classifier_meta(regime_rows, cash_feature_cols, setup.action_space)
            feature_cols = cash_feature_cols
            test_feature = test_full.iloc[[0]][feature_cols]

        if train_meta.empty or train_meta["variant"].nunique() < 2:
            continue
        clf = train_classifier(train_meta, feature_cols, setup)
        allowed = available_variants_from_candidates(default_candidates)
        pred_variant = constrained_classifier_predict(clf, test_feature, allowed)
        chosen_index = choose_default_candidate_for_variant(default_candidates, pred_variant)
        if chosen_index is None:
            continue
        train_counts.append(len(train_meta))
        for weight_name, weights in weight_sets().items():
            score = score_row(test_full, default_candidates, chosen_index, weights)
            rows.append({
                "setup_id": setup.setup_id,
                "cash_log_id": test_log,
                "weights": weight_name,
                "prediction_type": "variant",
                "predicted_variant": pred_variant,
                "recommended_algorithm": APDTM_TO_CASH_ALGORITHM[pred_variant],
                "chosen_config_hash": default_candidates.loc[chosen_index].get("config_hash", ""),
                "n_train_instances": len(train_meta),
                **score,
            })
    return pd.DataFrame(rows), {"mean_train_instances": float(np.mean(train_counts)) if train_counts else 0.0}


def evaluate_regressor_setup(
    setup: Setup,
    cash_df: pd.DataFrame,
    cash_apdtm_meta: pd.DataFrame,
    real_logs: list[str],
    cash_feature_cols: list[str],
    apdtm_feature_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    rows = []
    train_counts = []
    for fold_idx, test_log in enumerate(real_logs, start=1):
        print(f"[fold] {setup.setup_id} {fold_idx}/{len(real_logs)} test={test_log}", flush=True)
        test_full = cash_df[cash_df["log_id"].astype(str).eq(test_log)].dropna(subset=MEASURES).copy()
        if test_full.empty:
            continue

        if setup.feature_set == "apdtm":
            # Regressor trained on APDTM default metrics for real logs only.
            metrics = pd.read_csv(PACKAGE_DIR / "outputs/apdtm_cash_real_discovery_metrics.csv")
            metrics = metrics[metrics["status"].fillna("ok").eq("ok")].copy()
            metrics["algorithm"] = metrics["variant"].map(APDTM_TO_CASH_ALGORITHM)
            metrics = metrics.rename(columns={"log": "log_id", "log_fitness": "fitness"})
            features = cash_apdtm_meta[["log"] + apdtm_feature_cols].rename(columns={"log": "log_id"})
            reg_rows = metrics.merge(features, on="log_id", how="inner")
            reg_rows["family"] = reg_rows["log_id"].map(cash_family)
            reg_rows = reg_rows[reg_rows["family"].ne(test_log)].copy()
            for col in HYPERPARAM_COLS:
                reg_rows[col] = np.nan
            feature_cols = apdtm_feature_cols
            train_rows = reg_rows

            test_candidates = cash_action_rows(test_full, "apdtm5_default")
            test_feature_row = cash_apdtm_meta[cash_apdtm_meta["log"].astype(str).eq(test_log)]
            if test_feature_row.empty or test_candidates.empty:
                continue
            for col in feature_cols:
                test_candidates[col] = test_feature_row.iloc[0][col]
        else:
            train_rows = cash_training_rows(setup, test_log, cash_df)
            feature_cols = cash_feature_cols
            test_candidates = cash_action_rows(test_full, setup.action_space)

        if train_rows.empty or test_candidates.empty or train_rows["algorithm"].nunique() < 2:
            continue
        artifact = train_regressor(train_rows, feature_cols, setup)
        predicted = predict_regressor(artifact, test_candidates)
        train_counts.append(len(train_rows))
        for weight_name, weights in weight_sets().items():
            total = sum(weights.values()) or 1.0
            pred_cols = [f"_pred_{m}" for m in MEASURES]
            # Convert predicted metric values into the same composite utility
            # used for realised evaluation, then recommend the top candidate.
            predicted["_pred_composite"] = sum(predicted[f"_pred_{m}"] * weights[m] for m in MEASURES) / total
            ranked = predicted.dropna(subset=pred_cols + ["_pred_composite"]).sort_values("_pred_composite", ascending=False)
            if ranked.empty:
                continue
            chosen = ranked.iloc[0]
            score = score_row(test_full, test_candidates, chosen.name, weights)
            rows.append({
                "setup_id": setup.setup_id,
                "cash_log_id": test_log,
                "weights": weight_name,
                "prediction_type": "candidate",
                "predicted_variant": CASH_TO_APDTM_VARIANT.get(str(chosen["algorithm"]), ""),
                "recommended_algorithm": chosen["algorithm"],
                "chosen_config_hash": chosen.get("config_hash", ""),
                "predicted_composite": float(chosen["_pred_composite"]),
                "n_train_instances": len(train_rows),
                **score,
            })
    return pd.DataFrame(rows), {"mean_train_instances": float(np.mean(train_counts)) if train_counts else 0.0}


def fit_final_model_for_setup(
    setup: Setup,
    cash_df: pd.DataFrame,
    apdtm_original_meta: pd.DataFrame,
    cash_apdtm_meta: pd.DataFrame,
    cash_feature_cols: list[str],
    apdtm_feature_cols: list[str],
):
    if setup.model_kind == "classifier":
        if setup.feature_set == "apdtm":
            train_meta = apdtm_training_meta(setup, None, apdtm_original_meta, cash_apdtm_meta)
            artifact = train_classifier(train_meta, apdtm_feature_cols, setup)
            return {"model": artifact, "feature_cols": apdtm_feature_cols, "n_train_instances": len(train_meta)}
        regime_rows = cash_data_regime_rows(cash_df, setup.data_regime)
        train_meta = cash_classifier_meta(regime_rows, cash_feature_cols, setup.action_space)
        artifact = train_classifier(train_meta, cash_feature_cols, setup)
        return {"model": artifact, "feature_cols": cash_feature_cols, "n_train_instances": len(train_meta)}

    if setup.feature_set == "apdtm":
        metrics = pd.read_csv(PACKAGE_DIR / "outputs/apdtm_cash_real_discovery_metrics.csv")
        metrics = metrics[metrics["status"].fillna("ok").eq("ok")].copy()
        metrics["algorithm"] = metrics["variant"].map(APDTM_TO_CASH_ALGORITHM)
        metrics = metrics.rename(columns={"log": "log_id", "log_fitness": "fitness"})
        features = cash_apdtm_meta[["log"] + apdtm_feature_cols].rename(columns={"log": "log_id"})
        train_rows = metrics.merge(features, on="log_id", how="inner")
        for col in HYPERPARAM_COLS:
            train_rows[col] = np.nan
        artifact = train_regressor(train_rows, apdtm_feature_cols, setup)
        return {"model": artifact, "feature_cols": apdtm_feature_cols, "n_train_instances": len(train_rows)}
    train_rows = cash_action_rows(cash_data_regime_rows(cash_df, setup.data_regime), setup.action_space)
    artifact = train_regressor(train_rows, cash_feature_cols, setup)
    return {"model": artifact, "feature_cols": cash_feature_cols, "n_train_instances": len(train_rows)}


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    base = (
        results.groupby(["setup_id", "weights"], as_index=False)
        .agg(
            n_logs=("cash_log_id", "nunique"),
            mean_accuracy_full=("accuracy_full", "mean"),
            mean_accuracy_action=("accuracy_action", "mean"),
            mean_regret_full=("regret_full", "mean"),
            mean_regret_action=("regret_action", "mean"),
            median_rank_full=("rank_full", "median"),
            median_rank_action=("rank_action", "median"),
            mean_train_instances=("n_train_instances", "mean"),
        )
    )
    mean_rows = (
        results.groupby("setup_id", as_index=False)
        .agg(
            n_logs=("cash_log_id", "nunique"),
            mean_accuracy_full=("accuracy_full", "mean"),
            mean_accuracy_action=("accuracy_action", "mean"),
            mean_regret_full=("regret_full", "mean"),
            mean_regret_action=("regret_action", "mean"),
            median_rank_full=("rank_full", "median"),
            median_rank_action=("rank_action", "median"),
            mean_train_instances=("n_train_instances", "mean"),
        )
    )
    mean_rows["weights"] = "MEAN"
    return pd.concat([base, mean_rows[base.columns]], ignore_index=True)


def run(output_dir: Path, only: set[str] | None = None) -> None:
    warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    cash_df, apdtm_original_meta, cash_apdtm_meta, real_logs, cash_feature_cols, apdtm_feature_cols = load_inputs(
        PROJECT_ROOT, PACKAGE_DIR
    )

    setup_manifest = pd.DataFrame([setup.__dict__ for setup in SETUPS])
    setup_manifest["notes"] = ""
    setup_manifest.loc[
        setup_manifest["action_space"].str.endswith("default"),
        "notes",
    ] = "Default-only CASH rows prefer recorded v6_baseline_* rows; fallback is all-empty hyperparameters or nearest documented default when no baseline row exists."
    setup_manifest.to_csv(output_dir / "setup_manifest.csv", index=False)

    all_results = []
    model_rows = []
    selected_setups = [setup for setup in SETUPS if only is None or setup.setup_id in only]
    all_results = []
    existing_results_path = output_dir / "lolo_results.csv"
    if existing_results_path.exists():
        existing = pd.read_csv(existing_results_path)
        if only is not None:
            existing = existing[~existing["setup_id"].isin(only)].copy()
        if not existing.empty:
            all_results.append(existing)

    model_rows = []
    existing_model_manifest = output_dir / "model_manifest.csv"
    if existing_model_manifest.exists():
        existing_models = pd.read_csv(existing_model_manifest)
        if only is not None:
            existing_models = existing_models[~existing_models["setup_id"].isin(only)].copy()
        model_rows.extend(existing_models.to_dict("records"))

    for setup in selected_setups:
        print(f"[setup] Training/evaluating {setup.setup_id}", flush=True)
        if setup.model_kind == "classifier":
            results, stats = evaluate_classifier_setup(
                setup,
                cash_df,
                apdtm_original_meta,
                cash_apdtm_meta,
                real_logs,
                cash_feature_cols,
                apdtm_feature_cols,
            )
        else:
            results, stats = evaluate_regressor_setup(
                setup,
                cash_df,
                cash_apdtm_meta,
                real_logs,
                cash_feature_cols,
                apdtm_feature_cols,
            )
        if results.empty:
            print(f"[setup] WARNING no LOLO results for {setup.setup_id}", flush=True)
        else:
            all_results.append(results)
            print(
                f"[setup] OK {setup.setup_id}: {results['cash_log_id'].nunique()} logs, "
                f"mean equal accuracy_full="
                f"{results[results['weights'].eq('equal')]['accuracy_full'].mean():.3f}",
                flush=True,
            )

        artifact = fit_final_model_for_setup(
            setup,
            cash_df,
            apdtm_original_meta,
            cash_apdtm_meta,
            cash_feature_cols,
            apdtm_feature_cols,
        )
        model_path = model_dir / f"{setup.setup_id}.joblib"
        joblib.dump({"setup": setup.__dict__, **artifact}, model_path, compress=3)
        model_rows.append({
            "setup_id": setup.setup_id,
            "model_path": str(model_path),
            "n_final_train_instances": artifact["n_train_instances"],
            **stats,
        })
        print(f"[model] saved {model_path}", flush=True)

        partial_results = pd.concat(all_results, ignore_index=True, sort=False) if all_results else pd.DataFrame()
        partial_results.to_csv(output_dir / "lolo_results.csv", index=False)
        partial_summary = summarize_results(partial_results) if not partial_results.empty else pd.DataFrame()
        partial_summary = partial_summary.merge(setup_manifest, on="setup_id", how="left") if not partial_summary.empty else partial_summary
        partial_summary.to_csv(output_dir / "lolo_summary.csv", index=False)
        pd.DataFrame(model_rows).to_csv(output_dir / "model_manifest.csv", index=False)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True, sort=False)
    else:
        results_df = pd.DataFrame()
    results_df.to_csv(output_dir / "lolo_results.csv", index=False)
    summary = summarize_results(results_df) if not results_df.empty else pd.DataFrame()
    summary = summary.merge(setup_manifest, on="setup_id", how="left") if not summary.empty else summary
    summary.to_csv(output_dir / "lolo_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(output_dir / "model_manifest.csv", index=False)

    run_meta = {
        "n_real_test_logs": len(real_logs),
        "real_test_logs": real_logs,
        "n_cash_rows": len(cash_df),
        "n_cash_logs": int(cash_df["log_id"].nunique()),
        "n_apdtm_original_meta_instances": int(apdtm_original_meta["log"].nunique()),
        "n_cash_apdtm_meta_instances": int(cash_apdtm_meta["log"].nunique()),
        "cash_feature_count": len(cash_feature_cols),
        "apdtm_feature_count": len(apdtm_feature_cols),
        "outputs": {
            "setup_manifest": str(output_dir / "setup_manifest.csv"),
            "lolo_results": str(output_dir / "lolo_results.csv"),
            "lolo_summary": str(output_dir / "lolo_summary.csv"),
            "model_manifest": str(output_dir / "model_manifest.csv"),
        },
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[done] wrote outputs to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional setup_id list to run. Existing rows for these setup_ids are replaced.",
    )
    args = parser.parse_args()
    run(args.output_dir, set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
