# Trained models

`latest_models/` contains the only two trained inference artifacts distributed
in this release:

- `classification/saved_model/`: classifier returning a raw sigmoid
  probability. No post-hoc calibration file is distributed for this model.
- `regression/saved_model/`: regressor returning a tanh-scaled target, together
  with `target_scaler.json` for conversion to the original target units.

Both are complete TensorFlow SavedModels copied byte-for-byte from the
`pTyr_antibody-analog/model` archive (`CNN_classification` and
`CNN_regression_model`). Historical SavedModels and intermediate seed weights
are not included. The models used for the repeated-seed evaluation remain
represented by their predictions, histories, and metrics in
`../results/holdout_10fold_analysis/`; those evaluation statistics must not be
attributed to the two downstream inference exports in this directory.

See [`../docs/MODELS.md`](../docs/MODELS.md) for architecture, prediction-scale,
and evaluation details.
