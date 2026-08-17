# Evaluation response draft

## Expanded metrics, calibration, confidence intervals, and repeated evaluation

**Response:** Thank you for this suggestion. We revised the random-split
evaluation to prevent test-set leakage. A fixed 20% test set was removed before
model development. Ten-fold outer cross-validation was performed exclusively
within the remaining 80% development set. For every outer fold, early stopping
used a separate inner validation subset drawn only from the outer training
partition. The outer validation fold was not used for early stopping. For the
final analysis, the epoch was selected within the development set, a newly
initialized CNN was refitted using all development observations, and the fixed
20% test set was evaluated once. Regression target scalers were also fitted
using training data only.

For regression, the fixed 4:1 split followed the original systematic procedure:
sequences were ordered by descending log(ratio), consecutive blocks of five
were formed, one sequence per block was assigned to the independent test set,
and the remaining sequences formed the development set. Ten-fold evaluation
and inner early stopping were subsequently restricted to that development set.

For classification, the mean AUROC across the ten development-set folds was
0.971 +/- 0.004. On the untouched 20% test set, AUROC was
0.974 (95% CI, 0.968-0.978), AUPRC was 0.979 (95% CI, 0.975-0.982), F1 was 0.925 (95% CI, 0.915-0.934), and MCC was
0.853 (95% CI, 0.835-0.870). The Brier score was 0.057 (95% CI, 0.051-0.063), and the expected
calibration error was 0.020 (95% CI, 0.016-0.030). We added ROC and precision-recall curves,
a reliability diagram, and stratified-bootstrap 95% confidence intervals. The
classifier produces raw sigmoid probabilities. Platt scaling was fitted using
pooled out-of-fold predictions from the development set only; no test data were
used to fit the calibrator or select the classification threshold.

Ten independently initialized final-training runs gave a classification AUROC
of 0.973 +/- 0.001. The calibrated deep-ensemble AUROC
was 0.976; the mean standard deviation across seed-model
probabilities was 0.035 and was reported as epistemic model
disagreement.

For regression, the mean Pearson correlation across the ten development-set
folds was 0.868 +/- 0.017. On the untouched test set,
Pearson r was 0.870 (95% CI, 0.849-0.889), Spearman rho was 0.825 (95% CI, 0.808-0.842), MAE was
0.206 (95% CI, 0.194-0.219), and RMSE was 0.303 (95% CI, 0.277-0.332). We also added observed-versus-predicted
and residual plots and stratified-bootstrap 95% confidence intervals.

Across ten independently initialized regression runs, Pearson r was
0.871 +/- 0.003. The deep-ensemble Pearson r was
0.879; mean model-disagreement SD was 0.066.

The complete classification and regression evaluation workflows required
10254.5 and 4026.9 seconds, respectively, in the recorded
software environment. Per-fold timing information and the complete run
configurations are supplied with the output.

## Related variants and optimistic random splits

**Response:** We agree that a random split among related library variants tests
within-library interpolation and may overestimate performance on dissimilar
sequence regions. We therefore describe the 80:20/10-fold results explicitly
as the primary within-library evaluation. Exact duplicate sequences are
rejected before splitting, and the generated split manifests confirm zero
development/test row overlap. We do not claim that this random partition is a
homology-separated extrapolation test.
