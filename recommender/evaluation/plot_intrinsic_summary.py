"""Method-comparison dot plot of the intrinsic evaluation, for the paper.

Reads the per-group composite accuracies produced by intrinsic_eval.py and
draws one row per method: filled dots = per-group means (real / augmented /
synthetic), open ring = mean over all graded logs. Saves PNG + PDF.

Usage:
    python evaluation/plot_intrinsic_summary.py \
        [--composite output/eval/intrinsic_v8_famholdout_composite.csv] \
        [--out output/eval/intrinsic_v8_famholdout_summary]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = [  # bottom-to-top row order
    ("cash@3", "cash@3"),
    ("cash_rf", "cash_rf"),
    ("best_train_config", "best_train_config"),
    ("knn_transfer", "knn_transfer"),
]
GROUPS = [  # CSV group 
    ("real", "real", "#D55E00"),
    ("augmented", "augmented", "#0072B2"),
    ("gedi", "synthetic", "#009E73"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", default="output/eval/intrinsic_v8_famholdout_composite.csv")
    ap.add_argument("--out", default="output/eval/intrinsic_v8_famholdout_summary")
    args = ap.parse_args()

    df = pd.read_csv(args.composite)
    mean = df.groupby(["group", "method"]).accuracy.mean()
    groups = [(g, l, c) for g, l, c in GROUPS if g in df.group.unique()]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for y, (meth, label) in enumerate(METHODS):
        for grp, _, color in groups:
            ax.plot(mean[(grp, meth)], y, "o", ms=9, color=color, zorder=3)
        overall = mean[("combined", meth)]
        ax.plot(overall, y, "o", ms=13, mfc="none", mec="black", mew=1.8, zorder=4)
        ax.annotate(f"{overall:.3f}", (overall, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=11)

    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([label for _, label in METHODS], fontsize=11)
    ax.set_xlabel("mean min-max accuracy (15 weightings, leave-one-family-out)", fontsize=11)
    ax.set_xlim(0.80, 1.00)
    ax.set_ylim(-0.6, len(METHODS) - 0.25)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, label=l)
               for _, l, c in groups]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=11, mfc="none",
                              mec="black", mew=1.8, label="all graded logs"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=180)
    print(f"saved -> {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
