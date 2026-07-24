# Algorithms

The v6 benchmark covers alpha classic, alpha plus, genetic, heuristic classic,
heuristic plus-plus, ILP, inductive IM, inductive IMd, inductive IMF, and Split
Miner.

PM4Py-backed parameters are defined in `configs/algorithms/*.yaml`. Split
Miner accepts only the upstream 1.7.1 CLI parameter names recorded in
`configs/algorithms/split.yaml`; aliases are rejected. It consumes the
canonical `splitminer-v1.xes` preprocessing artifact and requires the pinned
external JAR and Java 8.

Discovery runs default to a 24-hour timeout. The supplied Slurm entry points
default to a 24-hour walltime with 16 GB of memory and one CPU per row or
worker. Individual experiment configs may declare shorter limits.
