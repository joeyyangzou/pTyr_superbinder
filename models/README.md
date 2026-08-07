# Trained models

`latest_models/` contains the only trained model artifacts distributed in this
release:

- `classification/`: validation-selected classifier (random-split seed 7),
  with Platt-calibration metadata.
- `regression/`: validation-selected regressor (random-split seed 5), with
  target-scaling metadata.

Both are complete TensorFlow SavedModels. Historical SavedModels and
intermediate seed weight files are not included. Per-seed training histories,
predictions, and metrics remain in `../results/robustness_analysis/`.

See [`../docs/MODELS.md`](../docs/MODELS.md) for architecture, selection,
prediction-scale, and ensemble details.
