# CASH for Process Discovery Algorithms

This repository contains the implementation and evaluation material for a CASH-based recommender for process discovery algorithms. Instead of recommending only a discovery algorithm at its default settings, the project treats recommendation as a Combined Algorithm Selection and Hyperparameter Optimization (CASH) problem: given an event log and user-specified weights over process-model quality measures, it recommends a complete configuration consisting of an algorithm and its hyperparameters.

The recommender uses event-log features and surrogate models to estimate process-model quality across the standard dimensions fitness, precision, generalization, and simplicity. The repository also includes the training-data generation pipeline and comparison material against existing process discovery recommenders such as ProReco and APDTM.

## Repository Structure

- `training_data/` contains the data generation, experiment, and evaluation pipeline used to create the measured training data for the CASH recommender. See `training_data/README.md` for setup and usage details.

- `recommender/` contains the recommendation code for selecting process discovery algorithms and hyperparameter configurations from event-log features and quality preferences. See `recommender/README.md` for usage details.

- `apdtm_comparison/` contains the APDTM comparison and fair leave-one-real-log-out ablation used in the project evaluation. See `apdtm_comparison/README.md` for reproduction notes and expected outputs.

## Project Context

Process discovery algorithms produce process models from event logs, but no single algorithm works best for every log. Model quality is evaluated along multiple competing dimensions, so the best choice depends on both the input log and the user’s priorities.

Existing recommenders mostly focus on selecting an algorithm under fixed or default settings. This project extends that idea by also recommending hyperparameters.
