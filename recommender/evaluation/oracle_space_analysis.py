"""Oracle (ground-truth) analyses of the configuration space, no models.

On the head-to-head log set, all on the full-grid min-max scale:
  1. ladder: best-of-8-defaults vs shared-8-tuned vs full grid;
  2. per-algorithm tuning headroom (best tuned vs default, same log);
  3. per-log menu gap (what the grid offers beyond the best default).

Usage:
    python evaluation/oracle_space_analysis.py [--dataset ...] [--out-prefix ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import compare_proreco as C
import cash.model as m


def search_space_from_dataset(df, hyper_names):
    """{algorithm: {param: (kind, lo, hi)}} for hyperparameters that vary."""
    space = {}
    for algo, g in df.groupby("algorithm"):
        algo_space = {}
        for h in hyper_names:
            col = g[h].dropna() if h in g.columns else None
            if col is None or col.empty or col.min() == col.max():
                continue
            kind = "int" if (col == col.round()).all() else "float"
            algo_space[h] = (kind, float(col.min()), float(col.max()))
        space[str(algo)] = algo_space
    return space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/datasets/dataset_v8.csv")
    ap.add_argument("--proreco-features", default="output/data/proreco/proreco162_v8.csv")
    ap.add_argument("--out-prefix", default="output/eval/oracle_space_v8")
    args = ap.parse_args()

    v5 = C.normalize_labels(pd.read_csv(args.dataset)).dropna(subset=C.MEASURES)
    pro = set(pd.read_csv(args.proreco_features).log_id.unique())
    v5 = v5[v5.log_id.isin(pro)]
    dg = C.build_default_gt(v5)
    shared = set(C.PRORECO_ALGOS)
    wsets = C.weight_sets()

    o_def, o_shared = [], []
    per_algo: dict = {}
    gap_rows = []
    for lid, g in v5.groupby("log_id"):
        d = dg[dg.log_id == lid]
        if d.empty:
            continue
        gs = g[g.algorithm.isin(shared)]
        for wname, w in wsets.items():
            tot = sum(w.values())
            full = sum(g[x] * w[x] for x in C.MEASURES) / tot
            deft = sum(d[x] * w[x] for x in C.MEASURES) / tot
            shared_full = sum(gs[x] * w[x] for x in C.MEASURES) / tot
            lo, hi = full.min(), full.max()
            if hi - lo < 1e-9:
                continue
            acc_def = (deft.max() - lo) / (hi - lo)
            o_def.append(acc_def)
            o_shared.append((shared_full.max() - lo) / (hi - lo))
            best_algo = g.loc[full.idxmax(), "algorithm"]
            gap_rows.append({"log_id": lid, "weights": wname,
                             "menu_gap": round(1.0 - acc_def, 4),
                             "best_algo": best_algo})
            for algo, ga in g.groupby("algorithm"):
                da = d[d.algorithm == algo]
                if da.empty:
                    continue
                best_a = (sum(ga[x] * w[x] for x in C.MEASURES) / tot).max()
                def_a = float(sum(da.iloc[0][x] * w[x] for x in C.MEASURES) / tot)
                per_algo.setdefault(algo, []).append((best_a - def_a) / (hi - lo))

    ladder = pd.DataFrame([
        ("oracle_best_of_defaults", float(np.mean(o_def))),
        ("oracle_best_shared_tuned", float(np.mean(o_shared))),
        ("oracle_full_grid", 1.0),
    ], columns=["rung", "minmax_accuracy"]).round(4)
    ladder.to_csv(f"{args.out_prefix}_ladder.csv", index=False)

    space = search_space_from_dataset(v5, m.ALL_HYPER_NAMES)
    algo_df = pd.DataFrame(
        [(a, len(space.get(a, {})), float(np.mean(v)), len(v))
         for a, v in per_algo.items()],
        columns=["algorithm", "tunable_hyperparameters", "tuning_headroom", "n_cells"],
    ).sort_values("tuning_headroom", ascending=False).round(4)
    algo_df.to_csv(f"{args.out_prefix}_tuning_by_algo.csv", index=False)

    pd.DataFrame(gap_rows).to_csv(f"{args.out_prefix}_menu_gap_per_log.csv", index=False)

    n_logs = len({r["log_id"] for r in gap_rows})
    print(f"cells: {len(o_def)} ({n_logs} logs x {len(wsets)} weightings)")
    print(ladder.to_string(index=False))
    print()
    print(algo_df.to_string(index=False))
    print(f"\nsaved -> {args.out_prefix}_ladder.csv, _tuning_by_algo.csv, _menu_gap_per_log.csv")


if __name__ == "__main__":
    main()
