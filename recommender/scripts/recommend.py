"""
Recommend the best process discovery algorithm + hyperparameters for a new XES log.

Steps:
  1. Load the XES log and extract its features.
  2. Build the candidate set: every distinct (algorithm, hyperparameters) config
     present in the training dataset.
  3. Predict the four quality measures for every candidate with the RF surrogate
     and rank by the weighted composite (the same procedure the evaluation grades).

Usage:
    python scripts/recommend.py --xes path/to/log.xes \
                                 --model model.pkl \
                                 --dataset dataset.csv \
                                 [--weights '0.4,0.3,0.2,0.1'] \
                                 [--top 5] \
                                 [--output result.json]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cash import model as m
from cash.features import extract_features_from_xes


def rank_candidates(models, le, log_features: dict, dataset: pd.DataFrame,
                    weights: dict) -> pd.DataFrame:
    """Score every distinct config of the training dataset for this log.

    Returns the candidates sorted by predicted composite (best first), with one
    ``_pred_<measure>`` column per measure and ``_pred_composite``.
    """
    cand = dataset[["algorithm"] + m.ALL_HYPER_NAMES].drop_duplicates().reset_index(drop=True)
    cand = cand[cand["algorithm"].isin(le.classes_)].copy()

    X = pd.DataFrame([log_features] * len(cand))
    X["algorithm"] = le.transform(cand["algorithm"])
    for h in m.ALL_HYPER_NAMES:
        X[h] = cand[h].values

    total_w = sum(weights[meas] for meas in m.MEASURES) or 1.0
    for meas in m.MEASURES:
        cand[f"_pred_{meas}"] = models[meas].predict(X)
    cand["_pred_composite"] = sum(
        cand[f"_pred_{meas}"] * weights[meas] for meas in m.MEASURES
    ) / total_w
    return cand.sort_values("_pred_composite", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Recommend algorithm + hyperparameters for a new XES log.")
    parser.add_argument("--xes", required=True, help="Path to the input XES file")
    parser.add_argument("--model", required=True, help="Trained model .pkl (from train.py)")
    parser.add_argument("--dataset", required=True, help="Training dataset CSV (from aggregate.py)")
    parser.add_argument("--weights", default=None,
                        help="Comma-separated measure weights as fitness,precision,generalization,"
                             "simplicity (e.g. '0.4,0.3,0.2,0.1'). Default: equal weights.")
    parser.add_argument("--top", type=int, default=5, help="Number of top configs to print (default: 5)")
    parser.add_argument("--output", default=None, help="Optional: save result as JSON to this path")
    args = parser.parse_args()

    weights = m.parse_weights(args.weights)

    print(f"[1/3] Loading {args.xes} ...")
    log_features = extract_features_from_xes(args.xes)
    print(f"      Measure weights: {weights}")

    print(f"\n[2/3] Loading model from {args.model} ...")
    models, le = m.load(args.model)

    print(f"\n[3/3] Ranking every known config with the RF surrogate ...")
    dataset = pd.read_csv(args.dataset)
    ranked = rank_candidates(models, le, log_features, dataset, weights)
    print(f"      Candidates scored: {len(ranked)}")

    best = ranked.iloc[0]
    best_hyperparams = {h: float(best[h]) for h in m.ALL_HYPER_NAMES if pd.notna(best[h])}
    score, std = m.predict(models, le, log_features, best["algorithm"], best_hyperparams, weights)
    confidence = "high" if std < 0.05 else "medium" if std < 0.15 else "low"

    print("\n" + "=" * 50)
    print("RF SURROGATE RECOMMENDATION")
    print("=" * 50)
    print(f"  Algorithm       : {best['algorithm']}")
    print(f"  Hyperparameters : {best_hyperparams}")
    print(f"  Predicted score : {round(score, 4)}")
    print(f"  Std (±)         : {round(std, 4)}")
    print(f"  Confidence      : {confidence}")
    print("=" * 50)

    print(f"\nTop {min(args.top, len(ranked))} candidates:")
    for i, row in ranked.head(args.top).iterrows():
        hp = {h: round(float(row[h]), 4) for h in m.ALL_HYPER_NAMES if pd.notna(row[h])}
        print(f"  {i + 1}. {row['algorithm']:<28} pred={row['_pred_composite']:.4f}  {hp}")

    result = {
        "algorithm": str(best["algorithm"]),
        "hyperparameters": best_hyperparams,
        "predicted_score": round(score, 4),
        "prediction_std": round(std, 4),
        "confidence": confidence,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
