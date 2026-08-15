# Model and parameter documentation

## Distributed inference models

This release provides exactly two complete TensorFlow SavedModels:

- `models/latest_models/classification/saved_model/`: classifier copied
  byte-for-byte from `pTyr_antibody-analog/model/CNN_classification`;
- `models/latest_models/regression/saved_model/`: regressor copied
  byte-for-byte from `pTyr_antibody-analog/model/CNN_regression_model`.

The model manifests record the encoding order, sequence length, architecture
source, and prediction scale. Historical SavedModels, redundant model copies,
and intermediate seed weights are intentionally excluded.

These two artifacts are supplied for downstream sequence screening and
ranking. They are not presented as the models from which the supplied
repeated-seed or Hamming-separated evaluation summaries were calculated. The
corresponding predictions, histories, seed-level metrics, calibration outputs,
and aggregate statistics remain under `results/robustness_analysis/`.

## Architecture and retraining

The 80:20 holdout plus 10-fold cross-validation training programs are:

- `src/model_training/03_CNN_classification.py`;
- `src/model_training/07_CNN_regression.py`.

They freeze the independent 20% test set before cross-validation, perform the
outer 10-fold analysis only within the 80% development set, and use inner
validation subsets for early stopping. The final SavedModel is refitted on the
complete development set and the test set is evaluated once.

The more stringent repeated-seed and Hamming-separated analysis is implemented
by `src/robustness_analysis/10_robustness_analysis.py`. Fixed architecture and
training settings are recorded in `config/model_hyperparameters.json`. No
formal grid, random, or Bayesian hyperparameter optimization was performed.

## Deep ensemble used in the robustness analysis

The robustness-analysis deep ensemble is a prediction-level combination, not
a weight average. For classification, each seed's probabilities are calibrated
using its validation data and then averaged. For regression, predictions in
original target units are averaged. The sample standard deviation across the
ten predictions is reported as model-disagreement uncertainty.

This standard deviation is an epistemic disagreement measure; it is not a
complete predictive interval and does not include all experimental noise.

## Prediction scales

- The distributed classification SavedModel returns a raw sigmoid probability.
  No post-hoc calibration file is distributed for this inference model.
- The distributed regression SavedModel returns a tanh-scaled target.
  `regression/target_scaler.json` contains the transformation needed to recover
  original target units; it was reconstructed from the 4,506 values in the
  distributed processed regression dataset, matching the original inference
  workflow.

Thresholds for new applications should be selected on validation data from the
intended application domain.
