"""Aggregate experiment result JSONs into a flat CSV dataset for training.

Each result JSON is self-contained: log_id, algorithm_name, hyperparameters,
status and the four measures. Row policy (see paper, Sec. 6.1): partial rows
(some measure missing) are kept for training only; discovery timeouts become
all-zero rows on logs that have at least one complete configuration.

Usage:
    python scripts/aggregate.py --results-dir <runs> --xes-dir <logs> \
        --output dataset.csv [--feature-cache cache.csv] \
        [--save-feature-cache cache.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cash.features import extract_features_from_xes, FEATURE_NAMES
from cash.model import MEASURES, build_row


def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def resolve_xes(xes_dirs, ref_path: str):
    """Locate a log in the xes dir(s) from a (possibly cluster-side) path,
    tolerant to .xes/.xes.gz and case differences."""
    if not ref_path:
        return None
    base = Path(ref_path).name
    core = base[:-3] if base.endswith(".gz") else base
    stem = core[:-4] if core.endswith(".xes") else core
    candidates = {c.lower() for c in
                  {base, core, core + ".gz", stem + ".xes", stem + ".xes.gz"}}
    for d in xes_dirs:
        for p in Path(d).rglob("*"):
            if p.is_file() and p.name.lower() in candidates:
                return p
    return None


def _metrics_complete(metrics: dict) -> bool:
    return bool(metrics) and all(metrics.get(k) is not None for k in MEASURES)


def aggregate(results_dirs, output, xes_dir, excludes=None,
              feature_cache_csv=None, save_feature_cache=None) -> None:
    roots = [Path(d) for d in results_dirs]
    xes_dirs = [Path(d) for d in xes_dir]
    excludes = excludes or []

    feature_cache: dict = {}
    if feature_cache_csv:
        fc = pd.read_csv(feature_cache_csv)
        for lid, g in fc.groupby("log_id"):
            feature_cache[str(lid)] = {f: g.iloc[0][f] for f in FEATURE_NAMES}
        print(f"Pre-loaded cached features for {len(feature_cache)} logs")

    def dump_cache():
        if save_feature_cache:
            rows = [{"log_id": lid, **feats}
                    for lid, feats in feature_cache.items() if feats]
            pd.DataFrame(rows).to_csv(save_feature_cache, index=False)

    def features_for(ref_path: str, log_id: str):
        if log_id in feature_cache:
            return feature_cache[log_id]
        xes_path = resolve_xes(xes_dirs, ref_path)
        if xes_path is None:
            print(f"[WARN] No XES for log_id '{log_id}' (ref '{ref_path}')")
            feature_cache[log_id] = None
            return None
        print(f"  Extracting features from {xes_path.name} ...", flush=True)
        feature_cache[log_id] = extract_features_from_xes(str(xes_path))
        dump_cache()  # persist progressively so an interrupted run resumes
        return feature_cache[log_id]

    json_files = [p for root in roots for p in root.rglob("*.json")
                  if not any(pat in str(p) for pat in excludes)]
    print(f"Found {len(json_files)} JSON files in {[str(r) for r in roots]}")

    rows = []
    skipped = 0
    seen: set = set()  # (log_id, config_hash); baseline and grid may overlap
    status_skipped: Counter = Counter()
    failures = []  # discovery timeouts, materialised as all-zero rows below

    for path in json_files:
        data = load_json(path)
        if data is None or data.get("status") != "success":
            st = (data or {}).get("status", "unreadable")
            if st in ("timeout", "success_missing", "failed"):
                # timeout = ran out of time, failed = crashed (memory, miner
                # error); either way discovery produced no model
                failures.append({
                    "log_id": data.get("log_id", "unknown"),
                    "config_hash": data.get("config_hash")
                                   or data.get("metadata", {}).get("config_hash"),
                    "algorithm": data.get("algorithm_name"),
                    "hyperparams": data.get("hyperparameters") or {},
                    "ref": data.get("test_log_path") or data.get("log_path"),
                    "experiment_id": data.get("experiment_id", ""),
                })
            status_skipped[st] += 1
            skipped += 1
            continue

        metrics = data.get("metrics", {})
        # partial rows (>=1 measure present) are kept; the per-measure
        # regressors train on whatever is available
        if not any(metrics.get(k) is not None for k in MEASURES):
            status_skipped["metrics_all_missing"] += 1
            skipped += 1
            continue
        if not _metrics_complete(metrics):
            status_skipped["kept_partial"] += 1

        log_id = data.get("log_id", "unknown")
        ch = data.get("config_hash") or data.get("metadata", {}).get("config_hash")
        if ch and (log_id, ch) in seen:
            status_skipped["duplicate_config"] += 1
            skipped += 1
            continue

        ref = data.get("test_log_path") or data.get("log_path")
        log_features = features_for(ref, log_id)
        if log_features is None:
            skipped += 1
            continue

        try:
            row = build_row(log_features=log_features,
                            algorithm=data["algorithm_name"],
                            hyperparams=data.get("hyperparameters", {}),
                            metrics=metrics)
            row["log_id"] = log_id
            row["experiment_id"] = data.get("experiment_id", "")
            row["config_hash"] = ch
            rows.append(row)
            if ch:
                seen.add((log_id, ch))
        except Exception as e:
            print(f"[WARN] Error processing {path}: {e}")
            skipped += 1

    # Discovery timeouts -> all-zero rows, but only on logs that also have a
    # complete configuration: on all-failure logs the zeros would be the whole
    # log and min-max grading would degenerate (best = worst = 0).
    complete_logs = {r["log_id"] for r in rows
                     if all(not pd.isna(r[meas]) for meas in MEASURES)}
    zero_rows = 0
    for fl in failures:
        if fl["log_id"] not in complete_logs or not fl["algorithm"]:
            continue
        if fl["config_hash"] and (fl["log_id"], fl["config_hash"]) in seen:
            continue
        log_features = features_for(fl["ref"], fl["log_id"])
        if log_features is None:
            continue
        row = build_row(log_features=log_features, algorithm=fl["algorithm"],
                        hyperparams=fl["hyperparams"],
                        metrics={meas: 0.0 for meas in MEASURES})
        row["log_id"] = fl["log_id"]
        row["experiment_id"] = fl["experiment_id"]
        row["config_hash"] = fl["config_hash"]
        rows.append(row)
        if fl["config_hash"]:
            seen.add((fl["log_id"], fl["config_hash"]))
        zero_rows += 1

    print(f"Materialised {zero_rows} discovery timeouts as all-zero rows "
          f"(on {len(complete_logs)} logs with a complete config)")
    if status_skipped:
        print(f"Skipped by reason: {dict(status_skipped)}")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} rows to {output}  ({skipped} files skipped)")
    if not df.empty:
        print(f"Logs: {df['log_id'].nunique()} | "
              f"Algorithms: {sorted(df['algorithm'].unique())}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate experiment JSONs into a CSV dataset.")
    ap.add_argument("--results-dir", required=True, action="append",
                    help="Directory with result JSONs (repeatable)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--xes-dir", required=True, action="append",
                    help="Directory containing the XES logs (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Path substring to exclude (repeatable)")
    ap.add_argument("--feature-cache", default=None,
                    help="CSV with per-log features to reuse (skips extraction)")
    ap.add_argument("--save-feature-cache", default=None,
                    help="CSV path to persist newly extracted features to")
    args = ap.parse_args()
    aggregate(args.results_dir, args.output, args.xes_dir, args.exclude,
              args.feature_cache, args.save_feature_cache)


if __name__ == "__main__":
    main()
