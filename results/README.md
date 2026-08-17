# Results

`holdout_10fold_analysis/` contains the supplied manuscript-aligned baseline:

- a fixed 80% development and 20% independent-test split;
- outer 10-fold cross-validation performed only within development data;
- inner-validation histories used for early stopping;
- fold assignments, split audits, out-of-fold and test predictions;
- classification calibration, ROC, precision-recall, and reliability files;
- regression observed-versus-predicted and residual analyses;
- 2,000-replicate bootstrap 95% confidence intervals; and
- ten-seed mean/SD and prediction-disagreement uncertainty outputs.

The SavedModels produced during this evaluation are excluded. The only
distributed inference models remain under `../models/latest_models/` and are
byte-identical to the models in the `pTyr_antibody-analog` model archive.
