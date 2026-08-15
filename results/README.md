# Results

`robustness_analysis/` contains the immutable supplied outputs from the formal
run:

- random and Hamming split manifests;
- per-seed histories, predictions, calibration parameters, and metrics;
- ensemble predictions and metrics;
- calibration, ROC, precision-recall, residual, and uncertainty plots;
- 2,000-replicate bootstrap confidence intervals; and
- combined manuscript tables.

Intermediate seed weight files and duplicate model exports are not included.
Run `bash run_robustness_analysis.sh` to write an independent rerun to
`robustness_analysis_rerun/` without overwriting these reference outputs.

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
