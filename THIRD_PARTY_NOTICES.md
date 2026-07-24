# Third-party notices

## GEDI

The optional synthetic-log environment installs GEDI 1.0.8 from its published
Python distribution. GEDI is not vendored in this repository. Its transitive
versions are recorded by the fully pinned
`environments/gedi/requirements.txt`; users must review their licenses before
redistribution.

## Split Miner

Split Miner 1.7.1 is an external research artifact and is not redistributed.
Users provide `split-miner-1.7.1-all.jar` themselves. The repository accepts
only SHA-256
`472c006623d99a6e440aa93a58e29b867cc331cec2b12b3d7fb61fb2a5de8328`.

## 48-feature extractor

`src/process_discovery_cash/data/features.py` is derived from the
`origin/test-anitan` branch at commit `04d91d52`, specifically the feature
pipeline in `Feature_Extension_Exploration.ipynb`. It is retained to preserve
the exact 48-feature semantics used by the v6 GEDI design. The surrounding
wrapper, tests, and provenance notice were added for this artifact.
