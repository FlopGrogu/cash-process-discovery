# Reproducibility contract

The artifact pins Python 3.11.15 and two pip-installable dependency graphs:
`requirements.txt` for the application and
`environments/gedi/requirements.txt` for GEDI 1.0.8. The two environments are
kept separate because their NumPy, SciPy, and PM4Py pins are incompatible.

XES and XES.GZ files are the authoritative event-log inputs. Discovery
manifests identify runs using the portable source and test XES paths, not
machine-local preprocessing artifacts. Optional Parquet caches are validated
against the source path, byte size, modification time, and SHA-256 before use.
Consequently, a manifest runs with no cache and remains unchanged when a cache
is created or removed.

Deterministic artifacts include:

- augmentation child IDs, records, and canonical XES gzip bytes;
- the 48-feature vectors and feature anchor;
- the 200 GEDI targets and all seed derivations;
- accepted/rejected generated-log records and checksums;
- primary and metric manifest bytes, plus optional survey manifests;
- source-XES-based run configuration identities.

Volatile execution metadata—timestamps, elapsed time, host/job IDs, resource
measurements, and temporary paths—may appear in scientific results but is
excluded from deterministic data and manifests.

All CSV writers use a fixed column order and LF line endings. JSON identities
use sorted canonical serialization. XES gzip output uses a blank filename and
zero modification time.

Run:

```bash
make check
```

This runs Ruff, supported non-external tests, exact regeneration of the 30
primary manifest receipts, and the submission audit with `.venv/bin/python`.
The default-run survey is not generated; use `make manifests-survey` or
`make manifests-all` when it is needed. Deprecated HPO tooling and its tests
are retained only for compatibility and are not part of this reproducibility
contract. External checks are listed in
[external-validation.md](external-validation.md).
