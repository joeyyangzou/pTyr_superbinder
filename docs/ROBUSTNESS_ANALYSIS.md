# Robustness analysis

## Analysis levels

The repository contains three connected levels:

1. NGS preprocessing converts paired-end reads into sequence/count tables.
2. Dataset construction produces balanced classification and regression model
   inputs.
3. Robustness analysis evaluates the task-specific CNN architectures with
   isolated partitions, repeated training, calibration, confidence intervals,
   and prediction-level model disagreement.

The complete path beginning with raw FASTQ data is documented in
`END_TO_END_WORKFLOW.md`. The model evaluation begins with the public
processed sequence-level tables.

## Processed inputs

Classification:

```text
data/processed/classification/positive.tsv
data/processed/classification/negative.tsv
```

Regression:

```text
data/processed/regression/regression_dataset.tsv
```

All model inputs are eight-residue sequences encoded in the amino-acid order
`ILVFMCAGPTSYWQNHEDKR`.

## Frozen partitions

### Random baseline

The combined data are stratified into 70% training, 10% validation, and 20%
test partitions with split seed 2026. Classification is stratified by class;
regression is stratified by observed-target quantile bins.

### Hamming-distance sensitivity design

1. Freeze a stratified 20% test set.
2. Calculate each remaining sequence's minimum Hamming distance to the test set.
3. Assign sequences with distance <2 to an exclusion buffer.
4. Select validation sequences from the remaining development data.
5. Assign training candidates within distance <2 of validation to the buffer.
6. Audit train-validation, train-test, and validation-test distances.

The minimum distance between every pair of retained partitions is 2. Buffer
sequences are excluded from fitting, calibration, and testing but remain
available in the public TSV files with their exclusion reason.

Frozen partitions are under `data/frozen_splits/` and
`results/robustness_analysis/*/*/splits/`.

## Training and overfitting controls

For every task and split design, models are independently initialized and
trained with seeds 1-10. Python, NumPy, and TensorFlow random seeds are set.
Early stopping monitors validation loss and restores the best validation-loss
weights. The test partition is evaluated only after fitting is complete.

No formal grid, random, or Bayesian hyperparameter optimization was performed.
All fixed settings are recorded in `config/model_hyperparameters.json`.

## Calibration and metrics

Each classification seed has a Platt calibrator fitted only on validation
predictions. The untouched test set is evaluated with AUROC, AUPRC, accuracy,
precision, recall, F1, MCC, Brier score, and 10-bin expected calibration error.
Raw and calibrated probabilities are retained, together with ROC,
precision-recall, and reliability plots.

Regression metrics include Pearson's r, Spearman's rho, R2, MAE, and RMSE.
Observed-versus-predicted and residual plots are supplied.

## Confidence intervals and uncertainty

Percentile 95% confidence intervals use 2,000 test-sequence bootstrap
replicates. Classification bootstrap samples are stratified by outcome class;
regression samples are stratified by observed-target quantile strata.

Classification ensemble probabilities are the mean of ten calibrated seed
probabilities. Regression ensemble scores are the mean of ten predictions in
original target units. Sample SD across the ten predictions is reported as
epistemic model-disagreement uncertainty. It is not a complete predictive
interval and does not include all experimental noise.

## Run the formal analysis

```bash
bash run_robustness_analysis.sh
```

Short installation check:

```bash
python src/robustness_analysis/10_robustness_analysis.py \
  --task classification \
  --positive-file data/processed/classification/positive.tsv \
  --negative-file data/processed/classification/negative.tsv \
  --output-dir results/smoke_test \
  --split-modes random \
  --seeds 1 \
  --classification-epochs 2 \
  --bootstrap-replicates 20 \
  --calibration platt
```

A smoke test checks installation only and must not replace the formal results.

## Export one classifier and one regressor after a full run

```bash
bash export_single_models.sh
```

This command selects one seed per task using validation loss only, loads the
locally generated weights, and writes TensorFlow SavedModels under
`results/robustness_analysis_rerun/single_models/`. The public release already
writes validation-selected exports to the rerun output tree. The models under
`models/latest_models/` are the separate downstream inference artifacts and
are not the source of the supplied repeated-run summaries.
