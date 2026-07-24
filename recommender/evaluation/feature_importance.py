"""Feature importance of the trained surrogate (impurity-based, per measure).

Reports importances per input column, grouped (log features vs hyperparameters
vs algorithm), and a top-N figure colored by group.

Usage:
    python evaluation/feature_importance.py [--model ...] [--out-prefix ...]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import cash.model as m
from cash.features import FEATURE_NAMES

GROUP_COLORS = {"log feature": "#0072B2", "hyperparameter": "#D55E00",
                "algorithm": "#9A9A9A"}


def column_group(col: str) -> str:
    if col in FEATURE_NAMES:
        return "log feature"
    if col == "algorithm":
        return "algorithm"
    return "hyperparameter"


def main():
    ap = argparse.ArgumentParser(description="RF feature importance of the surrogate.")
    ap.add_argument("--model", default="output/models/model_v8.pkl")
    ap.add_argument("--out-prefix", default="output/eval/feature_importance_v8")
    ap.add_argument("--top", type=int, default=20, help="bars in the figure (default 20)")
    args = ap.parse_args()

    models, _le = m.load(args.model)
    cols = m.NUMERIC_COLS

    imp = pd.DataFrame({meas: pipe.named_steps["rf"].feature_importances_
                        for meas, pipe in models.items()}, index=cols)
    imp["composite"] = imp[m.MEASURES].mean(axis=1)
    imp["group"] = [column_group(c) for c in imp.index]

    long = (imp.reset_index(names="column")
            .melt(id_vars=["column", "group"], var_name="measure", value_name="importance"))
    long.to_csv(f"{args.out_prefix}_all.csv", index=False)

    groups = imp.groupby("group")[m.MEASURES + ["composite"]].sum().round(4)
    groups.to_csv(f"{args.out_prefix}_groups.csv")
    print("=== importance mass per input group (sums to 1 per measure) ===")
    print(groups.to_string())

    top = imp.sort_values("composite", ascending=False).head(args.top)
    print(f"\n=== top {args.top} columns (equal-weight composite) ===")
    for c, r in top.iterrows():
        print(f"  {c:<36} {r['composite']:.4f}  ({r['group']})")

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * args.top + 1.2))
    t = top.iloc[::-1]
    ax.barh(range(len(t)), t["composite"],
            color=[GROUP_COLORS[g] for g in t["group"]], zorder=3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("mean RF importance over the four measures")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}.{ext}", dpi=180)

    print(f"\nsaved -> {args.out_prefix}_all.csv, _groups.csv, .png/.pdf")


if __name__ == "__main__":
    main()
