from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.hpo.study import (
    journal_path_for,
    open_study,
    study_name_for,
)
from process_discovery_cash.hpo.study_manifest import load_study_manifest_rows
from process_discovery_cash.hpo.summary import write_study_summary
from process_discovery_cash.hpo.trial_runner import StudyContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "(Re)generate study summary JSONs from HPO journal files, e.g. after a "
            "crashed or walltime-killed run."
        )
    )
    parser.add_argument("--study-manifest", help="Summarize every study in this manifest.")
    parser.add_argument("--config", help="Experiment config YAML (direct selection).")
    parser.add_argument("--log-id", help="Log id (direct selection).")
    parser.add_argument("--algorithm", help="Algorithm name (direct selection).")
    parser.add_argument("--output", help="Summary path override (direct selection only).")
    return parser


def _summarize(
    config: str,
    log_id: str,
    algorithm: str,
    output: str | None = None,
) -> Path | None:
    ctx = StudyContext.from_experiment(config, log_id, algorithm)
    study_name = study_name_for(
        ctx.experiment.experiment_id, ctx.log_ref.log_id, ctx.algorithm_ref.name
    )
    journal_path = journal_path_for(ctx.hpo.storage_root, ctx.experiment.experiment_id, study_name)
    if not journal_path.exists():
        print(f"study={study_name} has no journal at {journal_path}; skipping", flush=True)
        return None
    study = open_study(
        study_name=study_name,
        journal_path=journal_path,
        sampler_seed=ctx.hpo.sampler_seed,
        n_startup_trials=ctx.hpo.n_startup_trials,
        multivariate=ctx.hpo.multivariate,
        group=ctx.hpo.group,
        constant_liar=ctx.hpo.constant_liar,
    )
    summary_path = write_study_summary(study, ctx, output)
    print(f"Wrote study summary {summary_path}", flush=True)
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.study_manifest:
        if args.output:
            parser.error("--output only applies to direct --config selection")
        for row in load_study_manifest_rows(args.study_manifest):
            _summarize(row["experiment_config_path"], row["log_id"], row["algorithm_name"])
        return
    if args.config and args.log_id and args.algorithm:
        _summarize(args.config, args.log_id, args.algorithm, args.output)
        return
    parser.error("Select studies with --study-manifest or with --config --log-id --algorithm")


if __name__ == "__main__":
    main()
