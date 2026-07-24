# CASH for Process Discovery

Project for LMU Practical Process Mining. Recommends a discovery algorithm
*and* its hyperparameters for an event log, tailored to user-weighted quality
measures (fitness, precision, generalization, simplicity), without running
discovery on the candidates. A Random Forest surrogate (one regressor per
measure) predicts the quality of every configuration in a measured grid;
candidates are ranked by the weighted composite of the predictions.

## Layout

    src/cash/          library: features (48 log features), model (surrogate)
    scripts/           aggregate.py (build dataset), train.py, recommend.py
    evaluation/        all evaluation scripts (see map below)
    output/eval/       result CSVs backing the paper's tables and figures
    output/datasets/   dataset_v8.csv (40,388 rows, 215 logs)

## Setup

    python 3.9+, pip install -r requirements.txt

The trained model (`output/models/model_v8.pkl`, ~1.7 GB) is not shipped;
rebuild it with `train.py` (deterministic, seeded).

## Pipeline

    # 1. aggregate experiment outputs into the training dataset
    python scripts/aggregate.py --results-dir <runs> --xes-dir <logs> \
        --output output/datasets/dataset_v8.csv \
        --feature-cache output/data/feature_cache_v8.csv

    # 2. train the surrogate (4 RFs + label encoder, one pickle)
    python scripts/train.py --dataset output/datasets/dataset_v8.csv \
        --model-output output/models/model_v8.pkl

    # 3. recommend for a new log
    python scripts/recommend.py --xes mylog.xes \
        --model output/models/model_v8.pkl \
        --dataset output/datasets/dataset_v8.csv --weights 0.4,0.3,0.2,0.1

The experiment runs themselves (215 logs x 196 configurations) come from an
external cluster pipeline; each run is a self-contained JSON.

## Reproducing the paper's results

| Paper artifact | Command |
|---|---|
| Log inventory table | `python evaluation/log_inventory.py` |
| Surrogate quality table (MAE/RMSE/R2, Acc@1/@3, Spearman rho) and intrinsic accuracies | `python evaluation/intrinsic_eval.py --dataset output/datasets/dataset_v8.csv --output-prefix output/eval/intrinsic_v8_famholdout` |
| Intrinsic comparison figure | `python evaluation/plot_intrinsic_summary.py` |
| ProReco 162-feature matrix (input to the comparison) | `python evaluation/extract_proreco_features.py --dataset output/datasets/dataset_v8.csv --output output/data/proreco/proreco162_v8.csv` |
| ProReco head-to-head + CASH delta | `python evaluation/compare_proreco.py --dataset output/datasets/dataset_v8.csv --proreco-features output/data/proreco/proreco162_v8.csv --output-prefix output/eval/compare_proreco_v8` (`--folds 5` for the k-fold cross-check) |
| Oracle ladder, per-algorithm tuning headroom, menu gap | `python evaluation/oracle_space_analysis.py` |
| Feature-extraction timing | `python evaluation/time_feature_extraction.py` |
| Feature importance | `python evaluation/feature_importance.py` |
| Dataset characterization (dispersion, winners, headroom) | `python evaluation/dataset_analysis.py --dataset output/datasets/dataset_v8.csv --output-prefix output/eval/dataset_v8` |

All evaluations share the 15 measure weightings (`evaluation/weightings.py`)
and the leave-one-family-out protocol (a real log and its augmented variants
are held out together).

External inputs some scripts need: the raw 4TU event logs (cited individually
in the paper's appendix; `--raw-dir`), the generated logs (`--logs-dir`),
and, for the ProReco comparison, the ProReco repository checkout
(`--proreco-backend`).

## Event logs

The 215 logs are 21 public real-life logs (4TU.ResearchData), 77 augmented
variants, and 117 GEDI-generated synthetic logs. The augmented and synthetic
logs are produced by the external generation pipeline and are not part of this
repository.
