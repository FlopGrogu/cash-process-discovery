# Augmentation

Augmentation is a deterministic upstream v6 data stage, not a separate
experiment family. See [data.md](data.md#2-deterministic-augmentation) for the
complete procedure, seed policy, validation, paths, and expected count.

Augmented children inherit their real parent’s data provenance and must never
be treated as independent held-out real test logs.
