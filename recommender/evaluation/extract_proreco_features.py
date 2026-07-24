"""Extract ProReco's 162 log features for every log of a dataset.

Runs ProReco's own feature code from a local checkout of their repository
(--proreco-backend). The feature names and their order come from their
feature_portfolio.pk.

Important: do not "fix" ProReco's read_log. It returns a pm4py DataFrame, but
part of their feature code was written for the older EventLog object, so some
features compute something else than their name says (for example 'n_traces'
actually holds the number of events). ProReco was trained and shipped with
exactly these values, so a corrected read_log would produce features that no
longer match their published model and feature subsets. We reproduce their
behavior as is.

A feature that fails or exceeds --per-feature-timeout becomes NaN. Rows are
appended to the output CSV one log at a time and already-present logs are
skipped, so an interrupted run can simply be restarted.

Usage:
    python evaluation/extract_proreco_features.py \
        --dataset output/datasets/dataset_v8.csv \
        --output output/data/proreco/proreco162_v8.csv
"""

import argparse
import csv
import os
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).parent.parent

# real log_id -> file name in the raw 4TU folder (--raw-dir)
RAW_DIR = os.path.expanduser("~/Downloads/test_pascal/data/raw")
REAL_FILES = {
    "bpi2012": "BPI_Challenge_2012.xes.gz",
    "bpi2013_closed_problems": "BPI_Challenge_2013_closed_problems.xes.gz",
    "bpi2013_incidents": "BPI_Challenge_2013_incidents.xes.gz",
    "bpi2013_open_problems": "BPI_Challenge_2013_open_problems.xes.gz",
    "bpi2017": "BPI Challenge 2017.xes.gz",
    "bpi2018": "BPI Challenge 2018.xes.gz",
    "bpi2019": "BPI_Challenge_2019.xes",
    "bpi2020_domestic": "DomesticDeclarations.xes.gz",
    "bpi2020_international": "InternationalDeclarations.xes.gz",
    "bpi2020_payment": "RequestForPayment.xes.gz",
    "bpi2020_permit": "PermitLog.xes.gz",
    "bpi2020_prepaid": "PrepaidTravelCost.xes.gz",
    "bpic15_1": "BPIC15_1.xes", "bpic15_2": "BPIC15_2.xes", "bpic15_3": "BPIC15_3.xes",
    "bpic15_4": "BPIC15_4.xes", "bpic15_5": "BPIC15_5.xes",
    "hospital": "Hospital_log.xes.gz",
    "hospital_billing": "Hospital Billing - Event Log.xes.gz",
    "road_traffic_fines": "Road_Traffic_Fine_Management_Process.xes.gz",
    "sepsis": "Sepsis Cases - Event Log.xes.gz",
}

_worker = {}  # per-process state: feature functions, portfolio, timeout


def _init_worker(backend, timeout_s):
    """Set up ProReco's code in this worker process (imported unmodified)."""
    fa = str(Path(backend) / "flask_app")
    os.chdir(fa)                       # their code loads ./constants/*.pk
    os.environ["PRORECO_FLASK"] = fa
    sys.path.insert(0, str(backend))   # flask_app.*
    sys.path.insert(0, fa)             # import globals

    # Their read_log parses the log from disk again for every feature call
    # (162 times per log). Cache the parse in memory instead; the values are
    # unchanged. Their modules import utils under two names, so patch both.
    import pm4py
    import utils as U1
    import flask_app.utils as U2

    _log_cache = {}

    def read_log_cached(log_path):
        if log_path not in _log_cache:
            _log_cache.clear()         # one log at a time per worker
            _log_cache[log_path] = pm4py.read_xes(str(log_path))
        return _log_cache[log_path]

    U1.read_log = read_log_cached
    U2.read_log = read_log_cached

    import globals as G
    from flask_app.feature_controller import get_total_feature_functions_dict
    _worker["portfolio"] = list(G.feature_portfolio)
    _worker["funcs"] = get_total_feature_functions_dict()  # binds the patched read_log
    _worker["timeout"] = timeout_s


def _alarm(signum, frame):
    raise TimeoutError()


def extract_one(item):
    lid, path = item
    t0 = time.time()
    funcs, portfolio = _worker["funcs"], _worker["portfolio"]
    row, failed = {"log_id": lid}, []
    signal.signal(signal.SIGALRM, _alarm)
    for name in portfolio:
        try:
            signal.alarm(_worker["timeout"])
            row[name] = float(funcs[name](path))
        except Exception:
            row[name] = float("nan")
            failed.append(name)
        finally:
            signal.alarm(0)
    print(f"  ok  {lid}  ({time.time()-t0:.0f}s"
          + (f", {len(failed)} failed: {failed[:5]}" if failed else "") + ")",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description="Extract ProReco's 162 features for a dataset's logs.")
    ap.add_argument("--dataset", required=True, help="dataset CSV; its log_ids define the log set")
    ap.add_argument("--output", required=True, help="output CSV (appended progressively, resumable)")
    ap.add_argument("--proreco-backend", default=os.path.expanduser("~/Downloads/ProReco-main/backend"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help="folder with the raw 4TU logs (the real logs)")
    ap.add_argument("--logs-dir", default=str(REPO / "event-logs"),
                    help="root containing augmented/logs and synthetic/gedi/logs")
    ap.add_argument("--per-feature-timeout", type=int, default=180,
                    help="seconds before a single feature is abandoned as NaN (default 180)")
    args = ap.parse_args()

    import pandas as pd
    log_ids = sorted(pd.read_csv(args.dataset, usecols=["log_id"]).log_id.unique())

    el = Path(args.logs_dir)
    paths = {}
    for lid in log_ids:
        if lid in REAL_FILES:
            paths[lid] = os.path.join(args.raw_dir, REAL_FILES[lid])
        elif lid.startswith("aug_"):
            paths[lid] = str(el / "augmented/logs" / f"{lid}.xes.gz")
        elif lid.startswith("syn_gedi"):
            paths[lid] = str(el / "synthetic/gedi/logs" / f"{lid}.xes.gz")
        else:
            print(f"[WARN] no XES found for {lid}, skipping")

    done = set()
    out = Path(args.output)
    if out.exists():
        done = set(pd.read_csv(out, usecols=["log_id"]).log_id)
    # only the logs still to extract need their files present
    missing = [l for l, p in paths.items() if l not in done and not os.path.exists(p)]
    if missing:
        sys.exit(f"[ERROR] missing files for: {missing}")
    todo = sorted(((l, p) for l, p in paths.items() if l not in done),
                  key=lambda t: os.path.getsize(t[1]))
    print(f"{len(paths)} logs, {len(done)} already done, {len(todo)} to extract "
          f"({args.workers} workers, {args.per_feature_timeout}s/feature cap)", flush=True)

    # feature names for the CSV header
    import pickle
    portfolio = pickle.load(open(Path(args.proreco_backend) / "flask_app/constants/feature_portfolio.pk", "rb"))

    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["log_id"] + list(portfolio))
        if write_header:
            w.writeheader()
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(args.proreco_backend, args.per_feature_timeout)) as pool:
            for row in pool.imap_unordered(extract_one, todo):
                w.writerow(row)
                f.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
