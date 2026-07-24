"""
Intrinsic leave-one-family-out evaluation of the CASH recommender.

Each log is held out together with its augmented siblings (see ``log_family``),
so near-duplicate logs never leak into training. A single pass produces:

  * composite min-max accuracy of cash_rf / cash@3 / knn_transfer /
    best_train_config across the 15 weightings, split by log group;
  * per-measure prediction error (MAE / RMSE / R^2) and recommendation
    accuracy (Acc@1 / Acc@3);
  * Spearman rho between predicted and real config ordering;
  * a per-(log, weighting, method) detail table.

The surrogate is trained once per fold and reused for every weighting: the
per-measure predictions do not depend on the weights.

Usage:
    python evaluation/intrinsic_eval.py \
        --dataset output/datasets/dataset_v8.csv \
        --output-prefix output/eval/intrinsic_v8_famholdout
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from cash import model as m
from cash.features import FEATURE_NAMES, nan_safe_normalize
from weightings import weight_sets

def log_family(lid: str) -> str:
    """Family a log belongs to: an augmented log shares its parent's family
    (aug_<parent>__<augmentation>__<seed> -> <parent>); every other log is its
    own family. Held-out families prevent leakage from near-duplicate logs."""
    return lid[4:].split("__")[0] if lid.startswith("aug_") else lid


def log_group(lid: str) -> str:
    """Population of a log, by naming convention (works for v7 and v8 datasets):
    syn_gedi_* -> gedi, aug_* -> augmented, syn_* -> synth (aspect logs), else real."""
    if lid.startswith("syn_gedi"):
        return "gedi"
    if lid.startswith("aug_"):
        return "augmented"
    if lid.startswith("syn_"):
        return "synth"
    return "real"

METHODS = ["cash", "cash3", "knn", "best_train"]
LABELS = {"cash": "cash_rf", "cash3": "cash@3", "knn": "knn_transfer",
          "best_train": "best_train_config"}


# ---------------------------------------------------------------------------
# Hold-out machinery and baselines
# ---------------------------------------------------------------------------

def train_model(train_df: pd.DataFrame):
    train_df = train_df.dropna(subset=m.MEASURES, how="all")
    if train_df.empty:
        return None, None
    return m.train(train_df)


def predict_measures_df(models: dict, le, df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Add _pred_<measure> and _pred_composite columns for every row in df."""
    df = df.copy()
    known = df["algorithm"].isin(le.classes_)
    X = df[known].copy()
    X["algorithm"] = le.transform(X["algorithm"])
    for meas, pipe in models.items():
        col = f"_pred_{meas}"
        df[col] = np.nan
        if not X.empty:
            df.loc[known, col] = pipe.predict(X[m.NUMERIC_COLS])
    total_w = sum(weights[meas] for meas in m.MEASURES) or 1.0
    df["_pred_composite"] = (
        sum(df[f"_pred_{meas}"] * weights[meas] for meas in m.MEASURES) / total_w
    )
    return df


def _config_key(df: pd.DataFrame) -> pd.Series:
    """Per-row config identity = algorithm + all hyperparameter values."""
    def fmt(v) -> str:
        return "_" if pd.isna(v) else repr(round(float(v), 6))
    key = df["algorithm"].astype(str)
    for h in m.ALL_HYPER_NAMES:
        col = df[h] if h in df.columns else pd.Series(np.nan, index=df.index)
        key = key + "|" + col.map(fmt)
    return key


def _nearest_training_log(train_df: pd.DataFrame, log_features: dict):
    feat_matrix = train_df.groupby("log_id")[FEATURE_NAMES].first()
    query = np.array([log_features[f] for f in FEATURE_NAMES], dtype=float)
    feat_norm, query_norm = nan_safe_normalize(feat_matrix.values, query)
    distances = np.linalg.norm(feat_norm - query_norm, axis=1)
    return feat_matrix.index[np.argmin(distances)]


def _spearman(pred, real) -> float:
    """Spearman rho, NaN-safe; NaN when there is nothing meaningful to rank
    (fewer than 3 pairs, or one side constant)."""
    pred, real = np.asarray(pred, float), np.asarray(real, float)
    ok = ~np.isnan(pred) & ~np.isnan(real)
    if ok.sum() < 3 or np.unique(pred[ok]).size < 2 or np.unique(real[ok]).size < 2:
        return float("nan")
    return float(spearmanr(pred[ok], real[ok]).statistic)


def _graded_score(test_df: pd.DataFrame, ranked_keys, test_by_key: dict) -> float:
    for k in ranked_keys:
        if k in test_by_key:
            return float(test_by_key[k])
    return float(test_df["_real_composite"].mean())


def baseline_best_train_config(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    ranked = (train_df.assign(_k=_config_key(train_df))
              .groupby("_k")["_real_composite"].mean()
              .sort_values(ascending=False).index)
    test_by_key = (test_df.assign(_k=_config_key(test_df))
                   .groupby("_k")["_real_composite"].max().to_dict())
    return _graded_score(test_df, ranked, test_by_key)


def baseline_knn_transfer(train_df: pd.DataFrame, test_df: pd.DataFrame, log_features: dict) -> float:
    nearest_log = _nearest_training_log(train_df, log_features)
    near = train_df[train_df["log_id"] == nearest_log].assign(_k=lambda d: _config_key(d))
    ranked = near.sort_values("_real_composite", ascending=False)["_k"]
    test_by_key = (test_df.assign(_k=_config_key(test_df))
                   .groupby("_k")["_real_composite"].max().to_dict())
    return _graded_score(test_df, ranked, test_by_key)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, test_group: str = "all"):
    wsets = weight_sets()
    records = {(w, me): [] for w in wsets for me in METHODS}  # -> list of (log_id, acc)
    perlog_rows = []
    rho_rows = []
    reg_true = {meas: [] for meas in m.MEASURES}
    reg_pred = {meas: [] for meas in m.MEASURES}
    meas_acc1 = {meas: [] for meas in m.MEASURES}
    meas_acc3 = {meas: [] for meas in m.MEASURES}
    reg_true_algo = defaultdict(list)  # (algorithm, measure) -> true values
    reg_pred_algo = defaultdict(list)  # (algorithm, measure) -> predicted values

    fam = df.log_id.map(log_family)
    for lid in df.log_id.unique():
        if test_group != "all" and log_group(lid) != test_group:
            continue
        # leave-one-FAMILY-out: a real parent and its aug_* variants are near
        # duplicates, so none of them may train the model tested on any of them
        test = df[df.log_id == lid].dropna(subset=m.MEASURES).copy()
        fam_train = df[fam != log_family(lid)]
        # training keeps partial rows (each per-measure RF fits where its
        # measure exists); baselines and grading need all four measures
        train_full = fam_train.copy()
        train = fam_train.dropna(subset=m.MEASURES).copy()
        if test.empty or train.empty:
            continue
        models, le = train_model(train_full)
        if models is None:
            continue
        log_features = {f: test.iloc[0][f] for f in FEATURE_NAMES}

        # Per-measure predictions: weight-independent -> compute once.
        test = predict_measures_df(models, le, test, {x: 0.25 for x in m.MEASURES})

        # (once) per-measure regression error (global + per algorithm).
        for meas in m.MEASURES:
            pred_col = test[f"_pred_{meas}"]
            valid = pred_col.notna()
            reg_true[meas].extend(test.loc[valid, meas].tolist())
            reg_pred[meas].extend(pred_col[valid].tolist())
            for algo, sub in test.loc[valid].groupby("algorithm"):
                reg_true_algo[(algo, meas)].extend(sub[meas].tolist())
                reg_pred_algo[(algo, meas)].extend(sub[f"_pred_{meas}"].tolist())

        # (once) per-measure recommendation accuracy (Acc@1 / Acc@3).
        for meas in m.MEASURES:
            best_m, worst_m = test[meas].max(), test[meas].min()
            denom_m = best_m - worst_m
            ranked_m = test.dropna(subset=[f"_pred_{meas}"]).sort_values(
                f"_pred_{meas}", ascending=False)
            if ranked_m.empty:
                continue
            top1 = float(ranked_m.iloc[0][meas])
            top3 = float(ranked_m.iloc[:3][meas].max())
            meas_acc1[meas].append(1.0 if denom_m == 0 else (top1 - worst_m) / denom_m)
            meas_acc3[meas].append(1.0 if denom_m == 0 else (top3 - worst_m) / denom_m)

        # Composite min-max accuracy across the 15 weightings.
        for wname, w in wsets.items():
            total_w = sum(w[x] for x in m.MEASURES) or 1.0
            test["_pred_composite"] = sum(test[f"_pred_{x}"] * w[x] for x in m.MEASURES) / total_w
            test["_real_composite"] = m.composite_from_df(test, w)
            train["_real_composite"] = m.composite_from_df(train, w)
            best, worst = test._real_composite.max(), test._real_composite.min()
            denom = best - worst

            def acc(s):
                return 1.0 if denom == 0 else (s - worst) / denom

            # rank correlation of predicted vs real ordering; working-only
            # excludes the all-zero discovery failures
            working = test[~(test[m.MEASURES] == 0).all(axis=1)]
            rho_rows.append({
                "log_id": lid, "group": log_group(lid), "weights": wname,
                "n_configs": len(test), "n_working": len(working),
                "rho_all": _spearman(test["_pred_composite"], test["_real_composite"]),
                "rho_working": _spearman(working["_pred_composite"], working["_real_composite"]),
            })

            ranked = test.sort_values("_pred_composite", ascending=False)
            picks = {
                "cash": float(test.loc[ranked.index[0], "_real_composite"]),
                "cash3": float(test.loc[ranked.index[:3], "_real_composite"].max()),
                "knn": baseline_knn_transfer(train, test, log_features),
                "best_train": baseline_best_train_config(train, test),
            }
            for me, sc in picks.items():
                records[(wname, me)].append((lid, acc(sc)))
                perlog_rows.append({
                    "log_id": lid,
                    "group": log_group(lid),
                    "weights": wname,
                    "method": LABELS[me],
                    "real_score": round(sc, 4),
                    "best_score": round(float(best), 4),
                    "worst_score": round(float(worst), 4),
                    "accuracy": round(acc(sc), 4),
                })

    return (records, perlog_rows, reg_true, reg_pred, meas_acc1, meas_acc3,
            reg_true_algo, reg_pred_algo, rho_rows)


def _agg(vals, grp):
    if grp == "combined":
        vals = [v for _, v in vals]
    else:
        vals = [v for l, v in vals if log_group(l) == grp]
    return float(np.mean(vals)) if vals else float("nan")


def report(records, reg_true, reg_pred, meas_acc1, meas_acc3, reg_true_algo, reg_pred_algo,
           rho_rows, out_prefix):
    wsets = weight_sets()

    # --- composite table (per group) ---
    comp_rows = []
    groups = sorted({log_group(l) for vals in records.values() for l, _ in vals})
    for grp in ["combined"] + groups:
        print(f"\n=== {grp.upper()} (min-max accuracy of the composite) ===")
        print(f"{'weights':<10} " + " ".join(f"{LABELS[me]:>17}" for me in METHODS))
        print("-" * 80)
        for wname in wsets:
            line = {me: _agg(records[(wname, me)], grp) for me in METHODS}
            print(f"{wname:<10} " + " ".join(f"{line[me]:>17.3f}" for me in METHODS))
            for me in METHODS:
                comp_rows.append((grp, wname, LABELS[me], round(line[me], 4)))
        means = {me: float(np.mean([_agg(records[(wn, me)], grp) for wn in wsets])) for me in METHODS}
        print("-" * 80)
        print(f"{'MEAN':<10} " + " ".join(f"{means[me]:>17.3f}" for me in METHODS))
    pd.DataFrame(comp_rows, columns=["group", "weights", "method", "accuracy"]).to_csv(
        f"{out_prefix}_composite.csv", index=False)

    # --- per-measure table (once, weight-independent) ---
    print("\n=== PER-MEASURE (prediction error + recommendation accuracy) ===")
    print(f"  {'measure':<16} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Acc@1':>8} {'Acc@3':>8}")
    print("  " + "-" * 60)
    pm_rows = []
    for meas in m.MEASURES:
        if not reg_true[meas]:
            continue
        yt, yp = np.array(reg_true[meas]), np.array(reg_pred[meas])
        mae = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = r2_score(yt, yp) if len(yt) > 1 else float("nan")
        a1 = float(np.mean(meas_acc1[meas])) if meas_acc1[meas] else float("nan")
        a3 = float(np.mean(meas_acc3[meas])) if meas_acc3[meas] else float("nan")
        print(f"  {meas:<16} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {a1:>8.4f} {a3:>8.4f}")
        pm_rows.append((meas, round(mae, 4), round(rmse, 4), round(r2, 4),
                        round(a1, 4), round(a3, 4)))
    if pm_rows:
        oa1 = float(np.mean([r[4] for r in pm_rows]))
        oa3 = float(np.mean([r[5] for r in pm_rows]))
        print("  " + "-" * 60)
        print(f"  {'overall':<16} {'':>8} {'':>8} {'':>8} {oa1:>8.4f} {oa3:>8.4f}")
    pd.DataFrame(pm_rows, columns=["measure", "MAE", "RMSE", "R2", "Acc@1", "Acc@3"]).to_csv(
        f"{out_prefix}_permeasure.csv", index=False)

    # --- ranking quality (Spearman rho, predicted vs real composite order) ---
    if rho_rows:
        rho_df = pd.DataFrame(rho_rows)
        print("\n=== RANKING QUALITY (Spearman rho, predicted vs real composite) ===")
        print(f"  {'group':<12} {'rho(all configs)':>18} {'rho(working only)':>19}")
        print("  " + "-" * 51)
        for grp in ["combined"] + groups:
            sub = rho_df if grp == "combined" else rho_df[rho_df.group == grp]
            print(f"  {grp:<12} {sub.rho_all.mean():>18.3f} {sub.rho_working.mean():>19.3f}")
        rho_df.to_csv(f"{out_prefix}_ranking.csv", index=False)

    # --- per-algorithm prediction error (MAE per measure) ---
    # Reveals whether rare algorithms (few configs) are predicted as reliably as
    # the dominant one; the global per-measure MAE above is dominated by heuristic.
    algos = sorted({a for (a, _) in reg_true_algo})
    print("\n=== PER-ALGORITHM PREDICTION ERROR (MAE per measure; lower is better) ===")
    print(f"  {'algorithm':<26} {'n':>6} " + " ".join(f"{meas[:8]:>9}" for meas in m.MEASURES)
          + f" {'overall':>9}")
    print("  " + "-" * 82)
    pa_rows = []
    for algo in algos:
        maes = {}
        n = 0
        for meas in m.MEASURES:
            yt = np.array(reg_true_algo.get((algo, meas), []))
            yp = np.array(reg_pred_algo.get((algo, meas), []))
            maes[meas] = mean_absolute_error(yt, yp) if len(yt) else float("nan")
            n = max(n, len(yt))
            pa_rows.append((algo, meas, round(float(maes[meas]), 4), len(yt)))
        overall = float(np.nanmean([maes[meas] for meas in m.MEASURES]))
        print(f"  {algo:<26} {n:>6} "
              + " ".join(f"{maes[meas]:>9.4f}" for meas in m.MEASURES) + f" {overall:>9.4f}")
    pd.DataFrame(pa_rows, columns=["algorithm", "measure", "MAE", "n"]).to_csv(
        f"{out_prefix}_permeasure_by_algo.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description="Intrinsic leave-one-family-out evaluation of the CASH recommender.")
    ap.add_argument("--dataset", required=True, help="CSV produced by aggregate.py")
    ap.add_argument("--output-prefix", required=True,
                    help="Prefix for the output CSVs")
    ap.add_argument("--test-group", choices=["all", "real"], default="all",
                    help="grade only held-out logs of this group; training always "
                         "uses all remaining families")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    (records, perlog_rows, reg_true, reg_pred, meas_acc1, meas_acc3,
     reg_true_algo, reg_pred_algo, rho_rows) = run(df, args.test_group)

    pd.DataFrame(perlog_rows).to_csv(f"{args.output_prefix}_perlog.csv", index=False)
    report(records, reg_true, reg_pred, meas_acc1, meas_acc3,
           reg_true_algo, reg_pred_algo, rho_rows, args.output_prefix)
    print(f"\nsaved -> {args.output_prefix}_composite.csv, {args.output_prefix}_permeasure.csv, "
          f"{args.output_prefix}_permeasure_by_algo.csv, {args.output_prefix}_perlog.csv, "
          f"{args.output_prefix}_ranking.csv")


if __name__ == "__main__":
    main()
