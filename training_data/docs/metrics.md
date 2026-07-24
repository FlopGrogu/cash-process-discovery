# Metric manifests

Metric manifests are a pure transformation of a source discovery manifest.
Generation never scans result state and never writes model paths. Every source
row produces one metric row in the same order, even if discovery later fails,
times out, or produces no model.

```bash
.venv/bin/python scripts/generate_metric_manifest.py \
  --source-manifest build/manifests/v6/model/baseline/alpha_plus/v1.csv \
  --output build/manifests/v6/metrics/baseline/alpha_plus/v1/token_metrics.csv \
  --output-root results/cluster/v6/metrics/baseline/alpha_plus/v1/token
```

At runtime the metric worker reads `source_result_path`, resolves the model
artifact from that result, and evaluates it against the source-XES
`test_log_path`. A current local Parquet cache is used transparently. Missing or
failed source results remain visible as structured zero/failure metric rows
rather than disappearing from the analysis denominator.

Supported profiles are `pm4py_default`, `token`, and `alignment`. Metric
generation is a separate opt-in workflow; the default-run runtime and failure
survey does not generate metric manifests.
