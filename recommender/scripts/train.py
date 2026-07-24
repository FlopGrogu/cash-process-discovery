"""
Train the Random Forest surrogate model from the aggregated dataset.

Usage:
    python scripts/train.py --dataset dataset.csv --model-output model.pkl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cash import model as m


def main():
    parser = argparse.ArgumentParser(description="Train RF surrogate model.")
    parser.add_argument("--dataset", required=True, help="CSV produced by aggregate.py")
    parser.add_argument("--model-output", required=True, help="Output .pkl path for the trained model")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} rows, {df['algorithm'].nunique()} algorithms, {df['log_id'].nunique()} logs")

    missing_measures = [meas for meas in m.MEASURES if meas not in df.columns]
    if missing_measures:
        print(f"ERROR: dataset is missing measure columns {missing_measures}.")
        print("Re-run scripts/aggregate.py to regenerate the dataset with per-measure columns.")
        sys.exit(1)

    # Need at least one measure to train; rows with all measures NaN are useless.
    before = len(df)
    df = df.dropna(subset=m.MEASURES, how="all")
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with no measure values")

    models, le = m.train(df)
    m.save(models, le, args.model_output)

    print(f"Model saved to {args.model_output}  ({len(models)} per-measure regressors)")
    print(f"Algorithms known to model: {list(le.classes_)}")

    # Quick sanity check: in-sample prediction on a fully-measured row.
    from cash.features import FEATURE_NAMES
    from cash.model import ALL_HYPER_NAMES
    complete = df.dropna(subset=m.MEASURES)
    if not complete.empty:
        sample = complete.iloc[0]
        log_features = {f: sample[f] for f in FEATURE_NAMES}
        hyperparams = {h: sample[h] for h in ALL_HYPER_NAMES}
        pm = m.predict_measures(models, le, log_features, sample["algorithm"], hyperparams)
        print("Sanity check (per measure)  actual -> predicted:")
        for meas in m.MEASURES:
            print(f"  {meas:<15} {float(sample[meas]):.4f} -> {pm[meas][0]:.4f}")
        score, std = m.predict(models, le, log_features, sample["algorithm"], hyperparams)
        print(f"  {'composite':<15} {float(sample[m.TARGET]):.4f} -> {score:.4f}  (std {std:.4f})")


if __name__ == "__main__":
    main()
