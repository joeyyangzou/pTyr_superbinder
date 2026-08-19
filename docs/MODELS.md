# Models and parameters

## Current trained models

The repository contains one classifier and one regressor:

- `models/latest_models/classification/saved_model/`;
- `models/latest_models/regression/saved_model/`.

These are the selected primary models used to obtain the reported independent-
test results: classification seed 5 trained for 132 epochs and regression seed
8 trained for 199 epochs. Model selection used inner-development validation
loss and did not use the independent test sets.

No alternative or historical model copies are distributed. File manifests are
stored beside each model.

## Architecture and inputs

Both models receive one-hot encoded eight-residue sequences with amino-acid
order `ILVFMCAGPTSYWQNHEDKR`. The convolutional stack uses 128 filters with
kernel sizes 1, 3, 9 and 10 and `same` padding, followed by dense layers with
64, 32 and 8 units. Complete task-specific dropout, output, optimizer,
early-stopping and target-scaling settings are recorded in
`config/model_hyperparameters.json`.

## Training and evaluation

The training programs are:

- `src/model_training/03_CNN_classification.py`;
- `src/model_training/07_CNN_regression.py`.

They freeze a 20% independent test set before model development. Ten-fold
cross-validation is performed within the 80% development set, and separate
inner validation subsets determine the training duration. Ten independent
training seeds, calibration, bootstrap confidence intervals and
model-disagreement summaries are produced by the same programs.

## Prediction scales

- The classifier returns a raw sigmoid probability. Development-only Platt
  parameters are supplied in
  `models/latest_models/classification/platt_calibration.json`. The prediction
  program applies them when `--apply_platt` is specified. The accompanying
  development-selected calibrated cutoff is 0.510.
- The regressor returns a scaled output. The prediction program automatically
  loads `models/latest_models/regression/target_scaler.json` and reports values
  in the original regression-target units.

For a new application domain, select any classification threshold using an
appropriate validation dataset rather than the independent test set.
