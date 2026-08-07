# Model and parameter documentation

## Distributed models

This release provides exactly two complete TensorFlow SavedModels:

- `models/latest_models/classification/saved_model/`: random-split seed 7,
  selected by minimum validation loss at epoch 125.
- `models/latest_models/regression/saved_model/`: random-split seed 5,
  selected by minimum validation loss at epoch 158.

Test metrics were not used during seed selection. Model manifests record the
selection rule, encoding order, sequence length, and prediction scale.

Historical SavedModels, redundant model copies, and the 40 intermediate
`best.weights.h5` files are intentionally excluded. The repeated-run training
histories, predictions, calibration parameters, seed-level metrics, and
aggregate results are retained under `results/robustness_analysis/`.

## Architecture and retraining

The classification and regression architectures are constructed by
`src/robustness_analysis/10_robustness_analysis.py`. Fixed architecture and
training settings are recorded in `config/model_hyperparameters.json`. No
formal grid, random, or Bayesian hyperparameter optimization was performed.

Running `bash run_robustness_analysis.sh` retrains seeds 1-10 for both tasks and
split designs. Intermediate weights are generated locally in the rerun result
tree, but are ignored by Git so that future public releases continue to carry
only the latest two SavedModels.

## Deep ensemble

The deep ensemble is a prediction-level combination, not a weight average.
For classification, each seed's sigmoid probabilities are calibrated using a
validation-fitted Platt model and then averaged. For regression, predictions
in original target units are averaged. The sample standard deviation across
the ten predictions is reported as model-disagreement uncertainty.

This standard deviation is an epistemic disagreement measure; it is not a
complete predictive interval and does not include all experimental noise.

## Prediction scales

- The classification SavedModel returns a raw sigmoid probability.
  `classification/calibration.json` contains the Platt coefficient and
  intercept for calibrated probabilities.
- The regression SavedModel returns a tanh-scaled target.
  `regression/target_scaler.json` contains the training-only transformation
  needed to recover original target units.

Thresholds must be defined on validation data for the intended raw,
calibrated, or ensemble prediction scale.
