# Algorithms

The v6 benchmark covers alpha classic, alpha plus, genetic, heuristic classic,
heuristic plus-plus, ILP, inductive IM, inductive IMd, inductive IMF, and Split
Miner.

PM4Py-backed parameters are defined in `configs/algorithms/*.yaml`. Split
Miner accepts only the upstream 1.7.1 CLI parameter names recorded in
`configs/algorithms/split.yaml`; aliases are rejected. It consumes the
canonical `splitminer-v1.xes` preprocessing artifact and requires the pinned
external JAR and Java 8.

The 10 default-run survey configs each use one declared default configuration
and the same 21 real logs. Their runtime and failure records are descriptive,
not evidence for comparative algorithm performance.

Every algorithm has the same default runtime controls: a 24-hour discovery
timeout and, when submitted through the supplied Slurm entry points, a 24-hour
walltime with 16G of memory and one CPU per row or worker. Experiment configs
may state shorter limits explicitly for exploration or HPO.
