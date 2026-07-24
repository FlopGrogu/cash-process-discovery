"""
Dataset characterization: what we train on (not how well the recommender does).

Reports, split by log group, as CSVs + one figure: feature dispersion/coverage
of the 48 log features; winning-algorithm distribution per log; headroom per
log (best - worst composite over the log's configs).

Usage:
    python evaluation/dataset_analysis.py \
        --dataset output/datasets/dataset_v8.csv \
        --output-prefix output/eval/dataset_v8
"""

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from cash import model as m
from cash.features import FEATURE_NAMES
from weightings import weight_sets  # same 15-weighting grid as the evals

def _group(lid):
    """Population by naming convention (v7 and v8): syn_gedi_* -> gedi,
    aug_* -> augmented, syn_* -> synth (aspect logs), else real."""
    if lid.startswith("syn_gedi"):
        return "gedi"
    if lid.startswith("aug_"):
        return "augmented"
    if lid.startswith("syn_"):
        return "synth"
    return "real"


# --- 1. feature dispersion / coverage ---------------------------------------

def feature_dispersion(df):
    X = df.groupby("log_id")[FEATURE_NAMES].first()
    grp = pd.Series([_group(l) for l in X.index], index=X.index)
    Xn = X.loc[:, X.nunique(dropna=True) > 1]          # drop zero-variance columns
    Xn = Xn.fillna(Xn.mean())
    Z = (Xn - Xn.mean()) / Xn.std(ddof=0)

    def mean_pairwise(sub):
        idx = list(sub.index)
        if len(idx) < 2:
            return float("nan")
        return float(np.mean([np.linalg.norm(sub.loc[a].values - sub.loc[b].values)
                              for a, b in combinations(idx, 2)]))

    def mean_nn(sub):
        idx = list(sub.index)
        if len(idx) < 2:
            return float("nan")
        return float(np.mean([min(np.linalg.norm(sub.loc[a].values - sub.loc[b].values)
                                  for b in idx if b != a) for a in idx]))

    rows = []
    for g in ["combined"] + sorted(grp.unique()):
        sub = Z if g == "combined" else Z[grp == g]
        rows.append((g, len(sub), round(mean_pairwise(sub), 3),
                     round(mean_nn(sub), 3), round(float(sub.var(ddof=0).sum()), 1)))

    # coverage: how far the generated logs extend beyond the real per-feature range
    cov = None
    if (grp == "real").any() and (grp != "real").any():
        r, s = Xn[grp == "real"], Xn[grp != "real"]
        ext = sum(1 for c in Xn.columns if s[c].min() < r[c].min() or s[c].max() > r[c].max())
        ratios = [(s[c].max() - s[c].min()) / (r[c].max() - r[c].min())
                  for c in Xn.columns if (r[c].max() - r[c].min()) > 0]
        cov = (Xn.shape[1], ext, round(float(np.mean(ratios)), 2))
    return rows, cov


# --- 2. winning algorithm per log -------------------------------------------

def winning_algo(df, wsets):
    """Winner per (log, weighting) cell, aggregated per group; plus per-log
    winner stability (how many distinct algorithms win across the 15 weightings)."""
    d = df.dropna(subset=m.MEASURES).copy()
    dist = {}
    stability = {}
    for wname, w in wsets.items():
        d["_c"] = m.composite_from_df(d, w)
        for lid, g in d.groupby("log_id"):
            algo = g.groupby("algorithm")["_c"].max().idxmax()
            dist.setdefault(_group(lid), Counter())[algo] += 1
            stability.setdefault(lid, set()).add(algo)
    n_distinct = pd.Series({lid: len(s) for lid, s in stability.items()})
    return dist, n_distinct


# --- 3. headroom per log ----------------------------------------------------

def headroom(df, wsets):
    """Per log: best/worst/headroom of the composite, averaged over the 15
    weightings (complete rows only -- partial rows have no composite)."""
    d = df.dropna(subset=m.MEASURES).copy()
    per_w = []
    for wname, w in wsets.items():
        d["_c"] = m.composite_from_df(d, w)
        g = d.groupby("log_id")["_c"].agg(["max", "min"])
        g["weights"] = wname
        per_w.append(g.reset_index())
    allw = pd.concat(per_w)
    agg = allw.groupby("log_id")[["max", "min"]].mean()
    rows = [(lid, _group(lid), round(r["max"], 4), round(r["min"], 4),
             round(r["max"] - r["min"], 4)) for lid, r in agg.iterrows()]
    return rows


def main():
    ap = argparse.ArgumentParser(description="Characterize the training dataset (features, winners, headroom).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)

    # 1. feature dispersion
    disp_rows, cov = feature_dispersion(df)
    print("=== FEATURE DISPERSION (48 features, z-scored on the pooled logs) ===")
    print(f"  {'group':<10} {'n_logs':>7} {'pairwise':>9} {'nn_dist':>8} {'total_var':>10}")
    for g, n, mp, nn, tv in disp_rows:
        print(f"  {g:<10} {n:>7} {mp:>9} {nn:>8} {tv:>10}")
    if cov:
        print(f"  coverage: synthetic extends beyond the real range on {cov[1]}/{cov[0]} "
              f"features; mean (synth range / real range) = {cov[2]}x")
    pd.DataFrame(disp_rows, columns=["group", "n_logs", "mean_pairwise", "mean_nn", "total_var"]).to_csv(
        f"{args.output_prefix}_feature_dispersion.csv", index=False)
    if cov:
        pd.DataFrame([cov], columns=["n_features", "synth_extends_beyond_real", "mean_range_ratio"]).to_csv(
            f"{args.output_prefix}_feature_coverage.csv", index=False)

    # 2. winning algorithm
    wsets = weight_sets()
    n_graded = df.dropna(subset=m.MEASURES).log_id.nunique()
    if n_graded < df.log_id.nunique():
        print(f"\n[note] winner/headroom computed on the {n_graded} logs with complete "
              f"configs ({df.log_id.nunique() - n_graded} partial-only logs excluded)")
    dist, n_distinct = winning_algo(df, wsets)
    groups = sorted(dist)
    print(f"\n=== WINNING ALGORITHM PER (LOG, WEIGHTING) ({len(wsets)} weightings) ===")
    wa_rows = []
    for g in groups:
        tot = sum(dist[g].values()) or 1
        print(f"  {g} ({tot} log-weighting cells): {len(dist[g])} distinct winners")
        for algo, c in dist[g].most_common():
            print(f"     {algo:<28} {c:>4}  ({100*c/tot:.0f}%)")
            wa_rows.append((g, algo, c, round(100 * c / tot, 1)))
    pd.DataFrame(wa_rows, columns=["group", "algorithm", "count", "pct"]).to_csv(
        f"{args.output_prefix}_winning_algo.csv", index=False)
    print(f"  winner stability: {float(n_distinct.mean()):.2f} distinct winners per log "
          f"across the {len(wsets)} weightings (1 = weight-insensitive); "
          f"{int((n_distinct == 1).sum())}/{len(n_distinct)} logs have a single winner")

    # 3. headroom
    hr_rows = headroom(df, wsets)
    hr = pd.DataFrame(hr_rows, columns=["log_id", "group", "best", "worst", "headroom"])
    hr.to_csv(f"{args.output_prefix}_headroom.csv", index=False)
    print("\n=== HEADROOM PER LOG (best - worst composite, mean over 15 weightings) ===")
    for g in groups + ["combined"]:
        sub = hr if g == "combined" else hr[hr.group == g]
        if len(sub):
            print(f"  {g:<10} mean {sub.headroom.mean():.3f}  median {sub.headroom.median():.3f}  "
                  f"[{sub.headroom.min():.3f}, {sub.headroom.max():.3f}]")

    # --- figure: winning-algo distribution + headroom ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    algos = sorted(set().union(*dist.values()))
    palette = {"real": "#d1495b", "synth": "#2e86ab", "gedi": "#2e86ab", "augmented": "#edae49"}
    x = np.arange(len(algos))
    w = 0.8 / len(groups)
    for i, g in enumerate(groups):
        tot = sum(dist[g].values()) or 1
        ax1.bar(x + (i - (len(groups) - 1) / 2) * w,
                [100 * dist[g].get(a, 0) / tot for a in algos], w,
                label=g, color=palette.get(g))
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("_miner", "").replace("_", " ") for a in algos], rotation=45, ha="right")
    ax1.set_ylabel("% of (log, weighting) cells where best")
    ax1.set_title("Winning algorithm (15 weightings)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    box = [hr[hr.group == g].headroom.values for g in groups]
    ax2.boxplot([b for b in box if len(b)], labels=[g for g, b in zip(groups, box) if len(b)])
    ax2.set_ylabel("best - worst composite")
    ax2.set_title("Headroom per log")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}_dataset_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved -> {args.output_prefix}_feature_dispersion.csv, _winning_algo.csv, "
          f"_headroom.csv, _dataset_analysis.png")


if __name__ == "__main__":
    main()
