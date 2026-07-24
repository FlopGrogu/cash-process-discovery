# External release validation

These checks require non-redistributed inputs and are mandatory before creating
a public archive:

1. `.venv/bin/python scripts/verify_inputs.py --all` succeeds for all 21 real
   logs and the Split Miner 1.7.1 JAR.
2. `.venv/bin/python -m pytest -q tests/test_data_pipeline_integration.py`
   succeeds in the pip-managed core environment.
3. One real GEDI target completes with `.venv-gedi/bin/python`.
4. One Split Miner discovery and metric row completes with the external JAR
   and the configured Java 8 runtime.
5. The full data pipeline reproduces 21/77/200/117/215 and
   `.venv/bin/python scripts/verify_generated_data.py --json` validates every
   generated checksum and emits the release receipt hash.
6. All 30 primary baseline and explore manifests match
   `release/v6-manifest-receipts.json`.
7. One baseline discovery-plus-metric row and one row from each explore family
   complete from their source XES inputs.
8. `make check` passes and `git diff --exit-code` remains clean.
9. `make submission-archive` produces a source-only archive and SHA-256.

These checks are excluded from ordinary CI with the `external` pytest marker.
Run the data-generation stages in fresh cluster storage roots as described in
[cluster.md](cluster.md#fresh-root-release-validation). Preserve:

- input and generated-data JSON receipts;
- all-30 primary manifest receipt confirmation;
- one real GEDI target and one Split Miner discovery-plus-metric smoke outcome;
- Slurm job IDs and `sacct` resource summaries for external smoke runs;
- the source archive SHA-256 and successful archive-verifier output.

The default-run survey is an optional extension and can be verified with
`make manifests-survey` or `make manifests-all`. Deprecated HPO artifacts are
not part of release validation or submission.
