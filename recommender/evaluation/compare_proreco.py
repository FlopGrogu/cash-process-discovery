"""
Head-to-head: our CASH recommender vs a faithful ProReco reproduction, on the
logs shared by both inputs (the measured dataset and ProReco's 162-feature
matrix), graded with the same min-max accuracy.

Two analyses:
  DECOMPOSITION -- both systems pick one of the 8 ProReco algorithms at its
    default config; one design choice changes at a time from their system to
    ours (RFE, pooling, model class, features), win/tie/loss vs PR-real.
  CASH DELTA -- ProReco picks among the 8 defaults, CASH picks from the full
    configuration grid; both graded against best/worst of the full grid.

Protocol: leave-one-family-out by default; --folds N switches to grouped
k-fold (families stay together) with per-fold fit caching, ~40x faster and
within +-0.015 of LOFO on every rung.

ProReco reproduction: xgboost per (algorithm, measure) pair, RFE feature
subsets, prediction clamping, their 162 features (thesis p.46). Ours:
RandomForest per measure with the algorithm as an input (src/cash/model.py).
"""

import sys
import os
import argparse
import pickle
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from cash.features import FEATURE_NAMES
from weightings import weight_sets

MEASURES = ["fitness", "precision", "generalization", "simplicity"]

# The 8 ProReco algorithms, expressed as our v8 'algorithm' labels.
PRORECO_ALGOS = [
    "alpha_miner_classic", "alpha_miner_plus", "heuristics_miner",
    "ilp_miner", "inductive_miner_im", "inductive_miner_imf",
    "inductive_miner_imd", "split_miner",
]
# Older datasets (v5-v7) used different algorithm labels; normalise on load.
LEGACY_LABELS = {
    "heuristic_miner_classic": "heuristics_miner",
    "heuristic_miner_plusplus": "heuristics_miner_plusplus",
    "ilp_miner_classic": "ilp_miner",
    "split_miner_v1": "split_miner",
}
# Default hyperparameters per algorithm, as run in the v8 'baseline' block
# (nearest measured config is used when the exact default is absent).
DEFAULT_HP = {
    "alpha_miner_classic": {},
    "alpha_miner_plus": {},
    "heuristics_miner": {"dependency_threshold": 0.5, "and_threshold": 0.65,
                         "loop_two_threshold": 0.5, "dfg_pre_cleaning_noise_thresh": 0.05,
                         "min_act_count": 1.0, "min_dfg_occurrences": 1.0},
    "ilp_miner": {"alpha": 1.0},
    "inductive_miner_im": {"disable_fallthroughs": 0.0},
    "inductive_miner_imf": {"noise_threshold": 0.0, "disable_fallthroughs": 0.0},
    "inductive_miner_imd": {"disable_fallthroughs": 0.0},
    "split_miner": {"epsilon": 0.5, "eta": 0.5, "parallelismFirst": 0.0,
                    "removeLoopActivityMarkers": 0.0, "replaceIORs": 0.0},
}


def normalize_labels(df):
    """Map legacy (v5-v7) algorithm labels onto the v8 canonical ones."""
    df = df.copy()
    df["algorithm"] = df["algorithm"].replace(LEGACY_LABELS)
    return df


def is_real(lid: str) -> bool:
    return not str(lid).startswith(("aug_", "syn_"))


def log_family(lid: str) -> str:
    """Family a log belongs to: an augmented log shares its parent's family
    (aug_<parent>__... -> <parent>); every other log is its own family. Holding
    out whole families prevents leakage from near-duplicate logs."""
    return lid[4:].split("__")[0] if lid.startswith("aug_") else lid


def make_folds(logs, n_folds, seed=42):
    """{log_id: fold}: grouped K-fold by family (whole families share a fold,
    preserving the no-leakage guarantee), greedily balanced by log count."""
    import random as _r
    fams: dict = {}
    for l in logs:
        fams.setdefault(log_family(l), []).append(l)
    order = sorted(fams)
    _r.Random(seed).shuffle(order)
    order.sort(key=lambda f: -len(fams[f]))     # big families first, ties shuffled
    counts = [0] * n_folds
    fold_of = {}
    for f in order:
        k = counts.index(min(counts))
        for l in fams[f]:
            fold_of[l] = k
        counts[k] += len(fams[f])
    return fold_of


def _train_logs_for(logs, test_log, fold_of):
    """LOFO by family (fold_of=None) or grouped K-fold train set."""
    if fold_of is None:
        return [l for l in logs if log_family(l) != log_family(test_log)]
    return [l for l in logs if fold_of[l] != fold_of[test_log]]


# Fit cache: in K-fold mode every test log of a fold shares its train set, so
# each system's models are fit once per fold instead of once per test log
# (this is where the ~40x speedup comes from). Disabled under LOFO -- caching
# 200+ distinct train sets would only burn memory.
_FIT_CACHE: dict = {}
_CACHE_FITS = False


def _cached(key, builder):
    if not _CACHE_FITS:
        return builder()
    if key not in _FIT_CACHE:
        _FIT_CACHE[key] = builder()
    return _FIT_CACHE[key]

XGB_KW = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
              booster="gbtree", tree_method="hist", verbosity=0)

# ProReco applies RFE per (algorithm, measure): each regressor uses its own
# precomputed optimal feature subset (shipped in their repo). Map our labels to
# theirs and load those subsets so the faithful baseline matches exactly.
PRORECO_FA = Path(os.path.expanduser("~/Downloads/ProReco-main/backend/flask_app"))
ALGO_MAP = {
    "alpha_miner_classic": "alpha", "alpha_miner_plus": "alpha_plus",
    "heuristics_miner": "heuristic", "ilp_miner": "ILP",
    "inductive_miner_im": "inductive", "inductive_miner_imf": "inductive_infrequent",
    "inductive_miner_imd": "inductive_direct", "split_miner": "split",
}
MEAS_MAP = {"fitness": "token_fitness", "precision": "token_precision",
            "generalization": "generalization", "simplicity": "pm4py_simplicity"}
FEAT_NAMES = []          # set in main() to the 162 feature-column names
_rfe_idx_cache = {}


def _rfe_indices(algo, measure):
    """Column indices (into FEAT_NAMES) of ProReco's RFE subset for this regressor."""
    key = (algo, measure)
    if key not in _rfe_idx_cache:
        p = (PRORECO_FA / "constants/optimal_features_list/regression/xgboost"
             / f"optimal_features_{ALGO_MAP[algo]}_{MEAS_MAP[measure]}.pk")
        sel = set(pickle.load(open(p, "rb")))
        _rfe_idx_cache[key] = [i for i, f in enumerate(FEAT_NAMES) if f in sel]
    return _rfe_idx_cache[key]


def wscore(meas: dict, w: dict) -> float:
    tot = sum(w.values()) or 1.0
    return sum(float(meas[m]) * w[m] for m in MEASURES) / tot


def build_default_gt(v5: pd.DataFrame) -> pd.DataFrame:
    """One row per (log, ProReco-algo) = the default (or nearest) config's measures."""
    v5 = v5.dropna(subset=MEASURES)  # a partial row must never be the default row
    rows = []
    for (lid, algo), g in v5.groupby(["log_id", "algorithm"]):
        if algo not in DEFAULT_HP:
            continue
        tgt = DEFAULT_HP[algo]
        if tgt:
            dist = np.zeros(len(g))
            for hp, val in tgt.items():
                if hp in g.columns:
                    col = g[hp].astype(float).fillna(val)  # NaN HP -> no penalty
                    dist = dist + (col.values - val) ** 2
            idx = g.index[int(np.argmin(dist))]
        else:
            idx = g.index[0]
        r = v5.loc[idx]
        rows.append({"log_id": lid, "algorithm": algo,
                     **{m: float(r[m]) for m in MEASURES}})
    return pd.DataFrame(rows)


# --- recommenders: return {algo: {measure: predicted}} for the test log -------

def proreco_predict(default_gt, feats, train_logs, test_log, cand_algos, rfe=True):
    """Faithful ProReco baseline. With rfe=True (default) it matches their
    deployed system exactly: per (algorithm, measure) it uses their precomputed
    RFE feature subset and clamps each prediction to [0, 1]. rfe=False uses all
    passed features (for the controlled 2x2 probe)."""
    tk = (hash(tuple(train_logs)), len(feats[test_log]))
    out = {}
    for algo in cand_algos:
        tr = default_gt[(default_gt.algorithm == algo) &
                        (default_gt.log_id.isin(train_logs))]
        if tr.empty:
            continue
        out[algo] = {}
        for m in MEASURES:
            if rfe:
                idx = _rfe_indices(algo, m)
                xte = np.array([feats[test_log][idx]])
            else:
                xte = np.array([feats[test_log]])

            def build(tr=tr, m=m, algo=algo):
                if rfe:
                    Xtr = np.array([feats[l][_rfe_indices(algo, m)] for l in tr.log_id])
                else:
                    Xtr = np.array([feats[l] for l in tr.log_id])
                mdl = xgb.XGBRegressor(**XGB_KW)
                mdl.fit(Xtr, tr[m].values)
                return mdl
            mdl = _cached(("pro", rfe, tk, algo, m), build)
            pred = float(mdl.predict(xte)[0])
            out[algo][m] = max(min(pred, 1.0), 0.0)  # ProReco clamps to [0, 1]
    return out


def ours_predict(default_gt, feats, train_logs, test_log, cand_algos):
    def build():
        tr = default_gt[default_gt.log_id.isin(train_logs)].copy()
        le = LabelEncoder().fit(tr.algorithm)
        Xtr = np.array([list(feats[l]) + [enc]
                        for l, enc in zip(tr.log_id, le.transform(tr.algorithm))])
        models = {}
        for m in MEASURES:
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
            pipe.fit(Xtr, tr[m].values)
            models[m] = pipe
        return models, le
    models, le = _cached(("ours", len(feats[test_log]), hash(tuple(train_logs))), build)
    out = {}
    for algo in cand_algos:
        if algo not in le.classes_:
            continue
        x = np.array([list(feats[test_log]) + [int(le.transform([algo])[0])]])
        out[algo] = {m: float(models[m].predict(x)[0]) for m in MEASURES}
    return out


def proreco_pooled_predict(default_gt, feats, train_logs, test_log, cand_algos):
    """ProReco's MODEL (xgboost) but trained the POOLED way: one regressor per
    measure with the algorithm as an input feature (same data regime as ours).
    Control to separate the per-algo vs pooled ARCHITECTURE from the model choice.
    """
    def build():
        tr = default_gt[default_gt.log_id.isin(train_logs)].copy()
        le = LabelEncoder().fit(tr.algorithm)
        Xtr = np.array([list(feats[l]) + [enc]
                        for l, enc in zip(tr.log_id, le.transform(tr.algorithm))])
        models = {}
        for m in MEASURES:
            mdl = xgb.XGBRegressor(**XGB_KW)
            mdl.fit(Xtr, tr[m].values)
            models[m] = mdl
        return models, le
    models, le = _cached(("pool", len(feats[test_log]), hash(tuple(train_logs))), build)
    out = {}
    for algo in cand_algos:
        if algo not in le.classes_:
            continue
        x = np.array([list(feats[test_log]) + [int(le.transform([algo])[0])]])
        out[algo] = {m: float(models[m].predict(x)[0]) for m in MEASURES}
    return out


def ours_per_algo_predict(default_gt, feats, train_logs, test_log, cand_algos, rfe=False):
    """Our MODEL (RandomForest) trained PER (algorithm, measure). With rfe=True it
    also uses ProReco's per-(algorithm, measure) RFE feature subset, so it isolates
    the pure MODEL effect vs ProReco-faithful (same per-algo architecture, same
    RFE-selected features, only RandomForest instead of xgboost)."""
    tk = (hash(tuple(train_logs)), len(feats[test_log]))
    out = {}
    for algo in cand_algos:
        tr = default_gt[(default_gt.algorithm == algo) &
                        (default_gt.log_id.isin(train_logs))]
        if tr.empty:
            continue
        out[algo] = {}
        for m in MEASURES:
            if rfe:
                xte = np.array([feats[test_log][_rfe_indices(algo, m)]])
            else:
                xte = np.array([feats[test_log]])

            def build(tr=tr, m=m, algo=algo):
                if rfe:
                    Xtr = np.array([feats[l][_rfe_indices(algo, m)] for l in tr.log_id])
                else:
                    Xtr = np.array([feats[l] for l in tr.log_id])
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
                pipe.fit(Xtr, tr[m].values)
                return pipe
            pipe = _cached(("rf_per_algo", rfe, tk, algo, m), build)
            out[algo][m] = float(pipe.predict(xte)[0])
    return out


def run_fair_fight(default_gt, systems, label, block, fold_of=None, test_group="all"):
    """systems = list of (name, predict_fn, feats_dict). The FIRST system is the
    baseline (put ProReco-real first); we tally win/tie/loss of every other system
    vs that baseline. Returns (rows, wtl_rows) for CSV export while printing.
    fold_of: optional {log_id: fold} for grouped K-fold instead of LOFO."""
    logs = sorted(default_gt.log_id.unique())
    wsets = weight_sets()
    names = [s[0] for s in systems]
    baseline = names[0]
    acc = {n: {w: [] for w in wsets} for n in names}
    wins = {n: [0, 0, 0] for n in names[1:]}  # each system's win/tie/loss vs baseline

    # In fold mode, process fold by fold and drop cached fits at each boundary
    # (bounds memory to one fold's models).
    logs_iter = logs if fold_of is None else sorted(logs, key=lambda l: (fold_of[l], l))
    prev_fold = None
    for test_log in logs_iter:
        if test_group != "all" and not is_real(test_log):
            continue
        if fold_of is not None and fold_of[test_log] != prev_fold:
            _FIT_CACHE.clear()
            prev_fold = fold_of[test_log]
        train_logs = _train_logs_for(logs, test_log, fold_of)
        test_rows = default_gt[default_gt.log_id == test_log]
        trainable = set(default_gt[default_gt.log_id.isin(train_logs)].algorithm)
        cand = [a for a in test_rows.algorithm if a in trainable]
        if len(cand) < 2:
            continue
        actual = {a: {m: float(test_rows[test_rows.algorithm == a].iloc[0][m]) for m in MEASURES}
                  for a in cand}
        preds = {n: fn(default_gt, feats, train_logs, test_log, cand)
                 for n, fn, feats in systems}

        for wname, w in wsets.items():
            real = {a: wscore(actual[a], w) for a in cand}
            best, worst = max(real.values()), min(real.values())
            denom = best - worst
            picks = {}
            for n in names:
                cs = [a for a in cand if a in preds[n]]
                if not cs:
                    continue
                rec = max(cs, key=lambda a: wscore(preds[n][a], w))
                acc[n][wname].append(1.0 if denom == 0 else (real[rec] - worst) / denom)
                picks[n] = real[rec]
            for n in names[1:]:
                if n in picks and baseline in picks:
                    rn, rb = picks[n], picks[baseline]
                    wins[n][0 if rn > rb + 1e-9 else 2 if rb > rn + 1e-9 else 1] += 1

    width = 12 + 18 * len(names)
    print(f"\n{'='*max(72, width)}\n{label}\n{'='*max(72, width)}")
    print(f"  {'weights':<12}" + "".join(f"{n:>18}" for n in names)
          + "   (mean min-max accuracy)")
    print(f"  {'-'*width}")
    rows, means = [], {n: [] for n in names}
    for wname in wsets:
        line = f"  {wname:<12}"
        for n in names:
            v = np.mean(acc[n][wname]) if acc[n][wname] else float("nan")
            means[n].append(v)
            rows.append((block, wname, n, round(float(v), 4)))
            line += f"{v:>18.3f}"
        print(line)
    print(f"  {'-'*width}")
    print(f"  {'MEAN':<12}" + "".join(f"{np.nanmean(means[n]):>18.3f}" for n in names))
    for n in names:
        rows.append((block, "MEAN", n, round(float(np.nanmean(means[n])), 4)))
    wtl = []
    for n in names[1:]:
        t = wins[n]
        print(f"  {n} vs {baseline}: win {t[0]} / tie {t[1]} / loss {t[2]}  (over log x weight)")
        wtl.append((block, n, baseline, t[0], t[1], t[2]))
    return rows, wtl


def run_cash_delta(v5, default_gt, feats_pro, fold_of=None, test_group="all"):
    """Ours (full action space: all configs, incl. HP and the extra algorithms
    genetic & heuristic++) vs ProReco @ 8 defaults. Realised weighted score,
    normalised by best/worst over the FULL action space."""
    import cash.model as m
    logs = sorted(v5.log_id.unique())
    wsets = weight_sets()
    VARIANTS = ("full", "grid")  # menu + scale: all measured configs vs grid only
    acc = {v: {s: {w: [] for w in wsets} for s in ("ProReco@default", "Ours-CASH")}
           for v in VARIANTS}
    raw = {v: {"ProReco@default": [], "Ours-CASH": []} for v in VARIANTS}

    logs_iter = logs if fold_of is None else sorted(logs, key=lambda l: (fold_of[l], l))
    prev_fold = None
    for test_log in logs_iter:
        if test_group != "all" and not is_real(test_log):
            continue
        if fold_of is not None and fold_of[test_log] != prev_fold:
            _FIT_CACHE.clear()
            prev_fold = fold_of[test_log]
        train_logs = _train_logs_for(logs, test_log, fold_of)
        test_v5 = v5[v5.log_id == test_log].dropna(subset=MEASURES).copy()
        train_v5 = v5[v5.log_id.isin(train_logs)].dropna(subset=MEASURES).copy()
        if test_v5.empty or train_v5.empty:
            continue
        models, le = _cached(("cashfull", hash(tuple(train_logs))),
                             lambda: m.train(train_v5))
        test_pred = predict_measures_full(models, le, test_v5)
        # grid rows = the shared designed grid (HPO runs carry 'hpo' experiment ids)
        grid_mask = ~test_v5["experiment_id"].astype(str).str.contains("hpo").values
        masks = {"full": np.ones(len(test_v5), dtype=bool), "grid": grid_mask}

        # ProReco @ 8 defaults on this log
        dg = default_gt[default_gt.log_id == test_log]
        trainable = set(default_gt[default_gt.log_id.isin(train_logs)].algorithm)
        cand = [a for a in dg.algorithm if a in trainable]
        pred_pro = proreco_predict(default_gt, feats_pro, train_logs, test_log, cand)
        actual_def = {a: {me: float(dg[dg.algorithm == a].iloc[0][me]) for me in MEASURES} for a in cand}

        for wname, w in wsets.items():
            real_full = (sum(test_v5[me] * w[me] for me in MEASURES) / (sum(w.values()) or 1.0)).values
            pscore = (sum(test_pred[f"_p_{me}"] * w[me] for me in MEASURES) / (sum(w.values()) or 1.0)).values

            # proreco pick (realised default score); same pick under both variants
            cands = [a for a in cand if a in pred_pro]
            rec = max(cands, key=lambda a: wscore(pred_pro[a], w))
            pro_real = wscore(actual_def[rec], w)

            for var, mask in masks.items():
                if not mask.any():
                    continue
                best, worst = real_full[mask].max(), real_full[mask].min()
                denom = best - worst
                # CASH picks the best predicted config within this menu
                idx = np.flatnonzero(mask)[int(np.argmax(pscore[mask]))]
                our_real = float(real_full[idx])
                acc[var]["Ours-CASH"][wname].append(1.0 if denom == 0 else (our_real - worst) / denom)
                acc[var]["ProReco@default"][wname].append(1.0 if denom == 0 else (pro_real - worst) / denom)
                raw[var]["Ours-CASH"].append(our_real); raw[var]["ProReco@default"].append(pro_real)

    rows, realised = [], []
    for var in VARIANTS:
        if not any(acc[var]["Ours-CASH"][w] for w in wsets):
            continue
        print(f"\n{'='*64}\nCASH DELTA [{var} menu]  (vs ProReco @ 8 defaults)\n{'='*64}")
        print(f"  {'weights':<12} {'ProReco':>9} {'Ours-CASH':>10}")
        print(f"  {'-'*46}")
        mp_all, mo_all = [], []
        for wname in wsets:
            mp = np.mean(acc[var]["ProReco@default"][wname]); mo = np.mean(acc[var]["Ours-CASH"][wname])
            mp_all.append(mp); mo_all.append(mo)
            rows.append((var, wname, round(float(mp), 4), round(float(mo), 4)))
            print(f"  {wname:<12} {mp:>9.3f} {mo:>10.3f}")
        print(f"  {'-'*46}")
        print(f"  {'MEAN':<12} {np.mean(mp_all):>9.3f} {np.mean(mo_all):>10.3f}")
        rows.append((var, "MEAN", round(float(np.mean(mp_all)), 4), round(float(np.mean(mo_all)), 4)))
        rp, ro = float(np.mean(raw[var]["ProReco@default"])), float(np.mean(raw[var]["Ours-CASH"]))
        realised.append((var, "ProReco", round(rp, 4)))
        realised.append((var, "CASH", round(ro, 4)))
        print(f"  mean realised score: ProReco {rp:.3f}  Ours-CASH {ro:.3f}  (delta {ro-rp:+.3f})")
    return rows, realised


def predict_measures_full(models, le, df):
    import cash.model as m
    df = df.copy()
    known = df["algorithm"].isin(le.classes_)
    X = df[known].copy()
    X["algorithm"] = le.transform(X["algorithm"])
    for me, pipe in models.items():
        df[f"_p_{me}"] = np.nan
        if not X.empty:
            df.loc[known, f"_p_{me}"] = pipe.predict(X[m.NUMERIC_COLS])
    return df


def main():
    global FEAT_NAMES
    root = Path(__file__).parent.parent
    ap = argparse.ArgumentParser(description="Head-to-head: CASH vs a faithful ProReco baseline.")
    ap.add_argument("--dataset", default=str(root / "output/datasets/dataset_v6.csv"),
                    help="CSV produced by aggregate.py")
    ap.add_argument("--proreco-features",
                    default=str(root / "output/data/proreco/proreco162_43logs.csv"),
                    help="ProReco 162-feature matrix (log_id + 162 columns)")
    ap.add_argument("--output-prefix", default=str(root / "output/eval/compare_proreco"),
                    help="Prefix for the output CSVs")
    ap.add_argument("--test-group", choices=["all", "real"], default="all",
                    help="grade only held-out logs of this group; training still "
                         "uses all remaining families")
    ap.add_argument("--folds", type=int, default=0,
                    help="0 (default) = leave-one-family-out; N>0 = grouped N-fold CV "
                         "(families stay together). Folds share models via a fit "
                         "cache, so e.g. --folds 5 is ~40x faster than LOFO.")
    args = ap.parse_args()

    v5 = normalize_labels(pd.read_csv(args.dataset))
    pro = pd.read_csv(args.proreco_features)
    # Keep only logs present in both (ProReco features AND measured ground truth).
    common = set(v5.log_id) & set(pro.log_id)
    v5 = v5[v5.log_id.isin(common)].copy()
    pro = pro[pro.log_id.isin(common)].copy()

    feat162_cols = [c for c in pro.columns if c != "log_id"]
    FEAT_NAMES = feat162_cols
    feats_pro = {r.log_id: r[feat162_cols].to_numpy(dtype=float) for _, r in pro.iterrows()}
    feats48 = {lid: g.iloc[0][FEATURE_NAMES].to_numpy(dtype=float)
               for lid, g in v5.groupby("log_id")}

    default_gt = build_default_gt(v5)
    print(f"logs={v5.log_id.nunique()}  default-config rows={len(default_gt)} "
          f"(of {v5.log_id.nunique()*len(PRORECO_ALGOS)} possible)")

    fold_of = None
    if args.folds > 0:
        global _CACHE_FITS
        _CACHE_FITS = True
        fold_of = make_folds(sorted(v5.log_id.unique()), args.folds)
        print(f"protocol: grouped {args.folds}-fold CV (families stay together, "
              f"fits cached per fold)", flush=True)
    else:
        print("protocol: leave-one-family-out", flush=True)

    proreco_exact = functools.partial(proreco_predict, rfe=True)
    all_rows, all_wtl = [], []

    # Decomposition ladder: one change at a time from ProReco-real to our system.
    # Its endpoints ARE the head-to-head: PR-real vs Ours-RF-162 (same 162 features)
    # and PR-real vs CASH-48 (our 48 features); win/tie/loss of each config is taken
    # vs PR-real (the first/baseline system).
    #   PR-real -> PR-162 (RFE effect) -> PR-pooled (pooling, xgb) ; RF-RFE vs
    #   PR-real (model) ; Ours-RF-162 -> CASH-48 (features).
    r, w = run_fair_fight(default_gt, [
        ("PR-real", proreco_exact, feats_pro),
        ("PR-162", functools.partial(proreco_predict, rfe=False), feats_pro),
        ("PR-pooled", proreco_pooled_predict, feats_pro),
        ("RF-RFE", functools.partial(ours_per_algo_predict, rfe=True), feats_pro),
        ("Ours-RF-162", ours_predict, feats_pro),
        ("CASH-48", ours_predict, feats48),
    ], "DECOMPOSITION -- one change at a time (RFE / pooling / model / features)", "decomposition",
        fold_of=fold_of, test_group=args.test_group)
    all_rows += r; all_wtl += w

    cd_rows, cd_realised = run_cash_delta(v5, default_gt, feats_pro, fold_of=fold_of,
                                          test_group=args.test_group)

    # --- save results to CSV ---
    pd.DataFrame(all_rows, columns=["block", "weights", "system", "accuracy"]).to_csv(
        f"{args.output_prefix}_headtohead.csv", index=False)
    pd.DataFrame(all_wtl, columns=["block", "system", "baseline", "win", "tie", "loss"]).to_csv(
        f"{args.output_prefix}_wintieloss.csv", index=False)
    pd.DataFrame(cd_rows, columns=["variant", "weights", "ProReco", "CASH"]).to_csv(
        f"{args.output_prefix}_cashdelta.csv", index=False)
    pd.DataFrame(cd_realised, columns=["variant", "system", "mean_realised"]).to_csv(
        f"{args.output_prefix}_cashdelta_realised.csv", index=False)
    print(f"\nsaved -> {args.output_prefix}_headtohead.csv, _wintieloss.csv, "
          f"_cashdelta.csv, _cashdelta_realised.csv")


if __name__ == "__main__":
    main()
