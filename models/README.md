# Trained models

`latest_models/` contains the two current TensorFlow SavedModels used by the
prediction programs:

```text
latest_models/classification/saved_model/
latest_models/regression/saved_model/
```

The classification model returns a sigmoid probability. The regression model
is accompanied by `target_scaler.json`, which is loaded automatically to
convert predictions back to the original target scale.

Model architecture and training parameters are recorded in
`../config/model_hyperparameters.json`. Model-specific file manifests are kept
next to each SavedModel.
