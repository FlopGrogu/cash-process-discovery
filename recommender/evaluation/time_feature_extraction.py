"""Wall-clock extraction time: our 48 features vs ProReco's 162.

Five real logs across the size range, one end-to-end run per (log, pipeline).
ProReco runs charitably: log parsed once and memoized, no per-feature timeout.
The giant logs are excluded; their extraction is known not to complete. In the
recorded run (feature_timing_v8.csv) ProReco's extraction on bpi2017 was
stopped after 3 h and is reported as censored.

Usage:  python evaluation/time_feature_extraction.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))

LOGS = [  # small -> large
    "bpi2013_open_problems",
    "sepsis",
    "bpi2012",
    "bpi2017",
    "road_traffic_fines",
]


def main():
    import pandas as pd
    import extract_proreco_features as X

    paths = {lid: Path(X.RAW_DIR) / X.REAL_FILES[lid] for lid in LOGS}
    for lid, p in paths.items():
        assert p.exists(), f"missing raw log: {p}"

    sizes = pd.read_csv(REPO / "output/data/feature_cache_v8.csv",
                        usecols=["log_id", "rs4pd_total_traces", "rs4pd_total_events"]
                        ).set_index("log_id")

    # ours first (the ProReco init chdir's away and patches modules)
    from cash.features import extract_features_from_xes
    ours = {}
    for lid in LOGS:
        t0 = time.perf_counter()
        extract_features_from_xes(str(paths[lid]))
        ours[lid] = time.perf_counter() - t0
        print(f"ours     {lid:<24} {ours[lid]:8.1f}s", flush=True)

    # ProReco's stack, unmodified semantics, cached read, no per-feature timeout
    X._init_worker(str(Path.home() / "Downloads/ProReco-main/backend"),
                   timeout_s=0)  # 0 disables signal.alarm
    theirs = {}
    for lid in LOGS:
        t0 = time.perf_counter()
        X.extract_one((lid, str(paths[lid])))
        theirs[lid] = time.perf_counter() - t0
        print(f"proreco  {lid:<24} {theirs[lid]:8.1f}s", flush=True)

    rows = [{
        "log_id": lid,
        "cases": int(sizes.loc[lid, "rs4pd_total_traces"]),
        "events": int(sizes.loc[lid, "rs4pd_total_events"]),
        "ours_48_s": round(ours[lid], 1),
        "proreco_162_s": round(theirs[lid], 1),
    } for lid in LOGS]
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "output/eval/feature_timing_v8.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nsaved -> output/eval/feature_timing_v8.csv")


if __name__ == "__main__":
    main()
