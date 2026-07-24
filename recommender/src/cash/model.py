"""
Random Forest surrogate model for CASH.

One independent RF regressor is trained per quality measure (fitness,
precision, generalization, simplicity). Each predicts the value of its measure
for the model that a given configuration (algorithm + hyperparameters) would
produce on a log described by its feature vector. The composite score is the
*weighted sum* of the four predicted measures, with weights supplied at
inference time -- so the user can change priorities without retraining.

Missing hyperparameters (irrelevant for a given algorithm) are filled with NaN
before median imputation.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

from cash.features import FEATURE_NAMES

# All hyperparameters occurring in the portfolio. Those not defined for a
# given algorithm stay NaN and are median-imputed in the pipeline.
ALL_HYPER_NAMES = [
    # heuristic miner
    "and_threshold",
    "dependency_threshold",
    "dfg_pre_cleaning_noise_thresh",
    "loop_two_threshold",
    "min_act_count",
    "min_dfg_occurrences",
    # inductive miner
    "noise_threshold",
    "disable_fallthroughs",
    # genetic miner
    "generations",
    "population_size",
    "mutation_rate",
    "crossover_rate",
    "elitism_rate",
    "elitism_min_sample",
    # ilp miner
    "alpha",
    # split miner (v1/v2)
    "epsilon",
    "eta",
    "parallelismFirst",
    "removeLoopActivityMarkers",
    "replaceIORs",
]

# No hardcoded algorithm list: the model knows exactly the algorithms seen in
# training (the label encoder's classes).

MEASURES = ["fitness", "precision", "generalization", "simplicity"]
TARGET = "composite_score"  # derived column name (weighted sum of MEASURES)
DEFAULT_WEIGHTS = {m: 0.25 for m in MEASURES}
NUMERIC_COLS = FEATURE_NAMES + ALL_HYPER_NAMES + ["algorithm"]


def parse_weights(spec: str | None) -> dict:
    """Parse a 'fitness,precision,generalization,simplicity' weight string.

    Returns DEFAULT_WEIGHTS when spec is None/empty. Order matches MEASURES.
    """
    if not spec:
        return dict(DEFAULT_WEIGHTS)
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != len(MEASURES):
        raise ValueError(
            f"--weights expects {len(MEASURES)} comma-separated values "
            f"({','.join(MEASURES)}), got {len(parts)}: {spec!r}"
        )
    return {meas: float(val) for meas, val in zip(MEASURES, parts)}


def composite_score(measure_values: dict, weights: dict | None = None) -> float:
    """Weighted sum of the four measures, normalised by the total weight.

    Normalising by the total weight means weights need not sum to 1 (e.g. the
    user can pass values in [0, 100] like ProReco).
    """
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights[m] for m in MEASURES) or 1.0
    return sum(float(measure_values[m]) * weights[m] for m in MEASURES) / total_w


def composite_from_df(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Vectorised composite score over a DataFrame holding the MEASURES columns."""
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights[m] for m in MEASURES) or 1.0
    return sum(df[m] * weights[m] for m in MEASURES) / total_w


def build_row(log_features: dict, algorithm: str, hyperparams: dict, metrics: dict) -> dict:
    """Combine all components into a flat dict for one training row.

    Stores the four raw measures as separate columns and a *recomputed*
    composite_score (default equal weights). The stored ``metrics["composite_score"]``
    is deliberately ignored because historical weights have drifted.
    """
    row = {f: log_features[f] for f in FEATURE_NAMES}
    row["algorithm"] = algorithm
    for h in ALL_HYPER_NAMES:
        v = hyperparams.get(h, np.nan)
        row[h] = float(v) if isinstance(v, bool) else v  # booleans -> 0/1 for the numeric pipeline

    for meas in MEASURES:
        v = metrics.get(meas)
        row[meas] = float(v) if v is not None else np.nan

    if all(not pd.isna(row[meas]) for meas in MEASURES):
        row[TARGET] = composite_score({m: row[m] for m in MEASURES})
    else:
        row[TARGET] = np.nan
    return row


def _make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
    ], remainder="drop")
    return Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def train(df: pd.DataFrame) -> tuple[dict, LabelEncoder]:
    """Train one RF per measure. Returns ({measure: Pipeline}, LabelEncoder).

    Each measure model is fit on the rows where that measure is available, so a
    log/config with one failed metric still contributes to the other three.
    """
    le = LabelEncoder()
    df = df.copy()
    df["algorithm"] = le.fit_transform(df["algorithm"])

    models: dict = {}
    for meas in MEASURES:
        sub = df.dropna(subset=[meas])
        pipe = _make_pipeline()
        pipe.fit(sub[NUMERIC_COLS], sub[meas])
        models[meas] = pipe
    return models, le


def _build_X(le: LabelEncoder, log_features: dict, algorithm: str, hyperparams: dict) -> pd.DataFrame:
    algo_enc = int(le.transform([algorithm])[0])
    row = {f: log_features[f] for f in FEATURE_NAMES}
    row["algorithm"] = algo_enc
    for h in ALL_HYPER_NAMES:
        row[h] = hyperparams.get(h, np.nan)
    return pd.DataFrame([row])


def predict_measures(
    models: dict,
    le: LabelEncoder,
    log_features: dict,
    algorithm: str,
    hyperparams: dict,
) -> dict:
    """Return {measure: (mean, std_across_trees)} for a single configuration."""
    X = _build_X(le, log_features, algorithm, hyperparams)
    out: dict = {}
    for meas, pipe in models.items():
        prep_out = pipe["prep"].transform(X)
        tree_preds = np.array([t.predict(prep_out)[0] for t in pipe["rf"].estimators_])
        out[meas] = (float(tree_preds.mean()), float(tree_preds.std()))
    return out


def predict(
    models: dict,
    le: LabelEncoder,
    log_features: dict,
    algorithm: str,
    hyperparams: dict,
    weights: dict | None = None,
) -> tuple[float, float]:
    """Return (composite_score, std). Composite is the weighted sum of the four
    predicted measures; std is the weight-blended per-measure tree std (a
    confidence proxy)."""
    pm = predict_measures(models, le, log_features, algorithm, hyperparams)
    means = {m: pm[m][0] for m in MEASURES}
    comp = composite_score(means, weights)

    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights[m] for m in MEASURES) or 1.0
    std = sum(pm[m][1] * weights[m] for m in MEASURES) / total_w
    return comp, std


def save(models: dict, le: LabelEncoder, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"models": models, "le": le}, f)


def load(path: str | Path) -> tuple[dict, LabelEncoder]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["models"], obj["le"]
