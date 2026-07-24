# Feature-space generation

The v6 synthetic stage uses the fixed 48-feature anchor, a deterministic
200-target design, and GEDI 1.0.8 in its separate frozen environment. See
[data.md](data.md#3-feature-anchor-and-gedi-design) for the executable workflow.

The target design, target/child identifiers, generated XES serialization,
acceptance decisions, and checksums are deterministic. Runtime diagnostics are
kept separate. Generated logs feed the primary synthetic-explore runs. The
other supported real-log workflows use only the 21 real logs.
