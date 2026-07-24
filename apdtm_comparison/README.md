# APDTM Fair LOLO Comparison

This folder contains the cleaned APDTM comparison used for the current
submission. The relevant analysis is:

- `notebooks/fair_lolo_ablation.ipynb`
- `fair_lolo_ablation.py`

The notebook is output-driven and should run quickly when the prepared outputs
are present. It does not recompute expensive APDTM feature extraction or process
discovery metrics.

## How To View Results

Open and run:

```text
notebooks/fair_lolo_ablation.ipynb
```

The notebook shows the main leave-one-real-log-out result in the section:

```text
Main LOLO Accuracy Table
```

The headline column is:

```text
mean_accuracy_full
```

## Reproducibility

Fast reproduction:

1. Open `notebooks/fair_lolo_ablation.ipynb`.
2. Run all cells.
3. The notebook loads the existing files in `outputs/fair_lolo_ablation/`.

If all expected outputs already exist, the training cell prints:

```text
Existing complete outputs found in apdtm_comparison/outputs/fair_lolo_ablation; skipping retraining.
```

To rerun the fair LOLO training/evaluation explicitly from the project root:

```bash
python apdtm_comparison/fair_lolo_ablation.py
```

This recreates:

- `outputs/fair_lolo_ablation/lolo_results.csv`
- `outputs/fair_lolo_ablation/lolo_summary.csv`
- `outputs/fair_lolo_ablation/model_manifest.csv`
- `outputs/fair_lolo_ablation/setup_manifest.csv`
- `outputs/fair_lolo_ablation/run_metadata.json`
- `outputs/fair_lolo_ablation/models/*.joblib`

## Required Inputs

The fair LOLO comparison uses:

- `data/dataset_v8.csv`
- `outputs/apdtm_cash_real_log_meta_features.csv`
- `outputs/apdtm_cash_real_discovery_metrics.csv`
- `vendor/process_discovery_meta_learning/log_meta_features.csv`
- `vendor/process_discovery_meta_learning/discovery_metrics.csv`

The `cash_real_xes/` files are kept as intermediate APDTM data artifacts, but
the current fair LOLO notebook does not need to regenerate them.

## Current Setup Matrix

The notebook compares 11 setups:

- `B0` APDTM classifier, APDTM features, original APDTM data
- `B1` APDTM classifier, APDTM features, original APDTM + CASH real logs
- `B2` APDTM-style classifier, CASH features, real logs only
- `B3` APDTM-style classifier, CASH features, real + synthetic logs
- `B4` APDTM-style classifier, CASH features, full CASH dataset
- `C0` CASH-style regressor, APDTM features, real logs only
- `C1` CASH regressor, CASH features, real logs only
- `C2` CASH regressor, CASH features, full data, APDTM-5 default candidates
- `C3` CASH regressor, CASH features, full data, APDTM-5 with hyperparameters
- `C4` CASH regressor, CASH features, full data, all CASH default candidates
- `C5` CASH regressor, CASH features, full data, all CASH algorithms and
  observed hyperparameters

All setups are evaluated with leave-one-real-log-out. Augmented logs from the
held-out real log family are removed from training in each fold.

## Default Hyperparameter Handling

Default-only CASH rows prefer recorded `v6_baseline_*` rows in `dataset_v8.csv`.
If a log/algorithm pair has no recorded baseline row, the script falls back to
parameterless Alpha rows or the nearest documented/project default.

## Cleaned Scope

Older APDTM-only scripts, old plotting scripts, old APDTM-only notebooks, and
old common/full comparison outputs were removed from this submission folder.
The retained material is the current fair LOLO comparison and its required
inputs/outputs.
