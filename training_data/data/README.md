# Data

Only tiny example logs under `data/example/` are committed.

Use these local directories for real data:

- `data/raw/`: immutable source event logs
- `data/interim/`: temporary converted logs
- `data/processed/`: processed train/test splits or feature tables
- `data/external/`: externally supplied files that are not produced by this repo

Real logs are ignored by Git. On the cluster, prefer setting `DATA_ROOT` to a
shared filesystem location and point experiment configs at those paths.
