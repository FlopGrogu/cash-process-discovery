"""Per-log inventory: row counts by outcome, gradability, evaluation membership.

Source of the log-inventory table in the paper (which logs are gradable and
which enter the intrinsic evaluation / head-to-head).

Usage:
    python evaluation/log_inventory.py [--dataset ...] [--proreco-features ...]
"""

import argparse

import pandas as pd

MEASURES = ["fitness", "precision", "generalization", "simplicity"]


def group(lid: str) -> str:
    if lid.startswith("syn_gedi_"):
        return "gedi"
    if lid.startswith("aug_"):
        return "augmented"
    return "real"


def main():
    ap = argparse.ArgumentParser(description="Per-log inventory table.")
    ap.add_argument("--dataset", default="output/datasets/dataset_v8.csv")
    ap.add_argument("--proreco-features", default="output/data/proreco/proreco162_v8.csv")
    ap.add_argument("--output", default="output/eval/log_inventory_v8.csv")
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset, usecols=["log_id"] + MEASURES)
    featured = set(pd.read_csv(args.proreco_features).log_id.unique())

    rows = []
    for lid, g in ds.groupby("log_id"):
        complete = g.dropna(subset=MEASURES)
        zeros = complete[(complete[MEASURES] == 0).all(axis=1)]
        rows.append({
            "log_id": lid, "group": group(lid), "rows_total": len(g),
            "complete": len(complete) - len(zeros), "timeout_zeros": len(zeros),
            "partial": len(g) - len(complete),
            "proreco162": lid in featured,
            "gradable": len(complete) > 0,
        })
    inv = pd.DataFrame(rows)
    inv["in_intrinsic"] = inv.gradable
    inv["in_headtohead"] = inv.gradable & inv.proreco162
    inv = inv.sort_values(["group", "log_id"])
    inv.to_csv(args.output, index=False)

    s = inv.groupby("group").agg(logs=("log_id", "count"), gradable=("gradable", "sum"),
                                 proreco162=("proreco162", "sum"),
                                 headtohead=("in_headtohead", "sum"))
    s.loc["TOTAL"] = s.sum()
    print(s.to_string())
    print(f"\nsaved -> {args.output}")


if __name__ == "__main__":
    main()
