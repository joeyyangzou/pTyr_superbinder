# Model-analysis results

`holdout_10fold_analysis/` contains the supplied classification and regression
analysis outputs:

- fixed 80% development and 20% independent-test partitions;
- development-set ten-fold assignments and fold metrics;
- inner-validation training histories used for early stopping;
- out-of-fold and independent-test predictions;
- classification ROC, precision-recall and reliability plots;
- regression observed-versus-predicted and residual plots;
- 2,000-replicate bootstrap 95% confidence intervals;
- ten-seed metrics and model-disagreement summaries; and
- run configurations and target-scaling parameters.

The current trained models used by the prediction programs are stored once,
under `../models/latest_models/`.

`hamming_buffer_sensitivity/` contains the fixed-test sequence-similarity
sensitivity analysis. It retains the primary independent test set, removes all
distance-0/1 development candidates, and supplies the resulting partitions,
fold and seed outputs, ensemble predictions, bootstrap metrics, and
Supplementary Figure S6/Table S6 source files. Sensitivity-model weights are
not distributed because they are not downstream inference artifacts.
