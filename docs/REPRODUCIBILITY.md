# Reproducibility workflow

## 1. Workflow levels

The release separates three related but distinct workflows:

1. **NGS preprocessing** converts sequencing reads into sequence-count tables.
2. **Original model workflow** constructs balanced classification/regression
   datasets and trains the task-specific CNNs used in the original discovery
   analysis.
3. **Robustness workflow** re-evaluates the same task-specific CNN
   architectures with leakage-free partitions, ten seeds, calibration,
   confidence intervals, and uncertainty estimates.

The robustness workflow starts from the public processed sequence-level tables,
so raw FASTQ processing is not required to reproduce the reported ML metrics.

## 2. Processed inputs

Classification uses:

```text
data/processed/classification/positive.tsv
data/processed/classification/negative.tsv
```

Regression uses:

```text
data/processed/regression/regression_dataset.tsv
```

All model inputs contain fixed-length eight-residue sequences. The amino-acid
encoding order is `ILVFMCAGPTSYWQNHEDKR`.

## 3. Frozen partition construction

### Random baseline

The combined data are stratified into 70% training, 10% validation, and 20%
test partitions with split seed 2026. Classification is stratified by class;
regression is stratified by observed-target quantile bins.

### Hamming-distance sensitivity analysis

1. Freeze a stratified 20% test set.
2. Calculate the minimum Hamming distance from every remaining sequence to the
   test set.
3. Move all sequences with distance <2 into an exclusion buffer.
4. Select validation sequences from the remaining development data.
5. Move all training candidates with distance <2 from validation into the
   exclusion buffer.
6. Audit train–validation, train–test, and validation–test distances.

This procedure guarantees a minimum Hamming distance of 2 between all three
partitions. Buffer sequences are excluded from fitting, calibration, and
testing; they are retained in public TSV files with the exclusion reason.

Frozen partitions are available in both `data/frozen_splits/` and the supplied
result tree under `results/robustness_analysis/*/*/splits/`.

## 4. Training

For every task and split design, models are independently initialized and
trained with seeds 1–10. Python, NumPy, and TensorFlow random seeds are set.
Early stopping monitors validation loss and restores the best validation-loss
weights. Test observations are evaluated only after fitting is complete.

No formal grid, random, or Bayesian hyperparameter optimization was performed.
All fixed parameters are listed in `config/model_hyperparameters.json`.

## 5. Calibration and evaluation

For classification, a separate Platt calibrator is fitted for each seed using
validation predictions only. The untouched test set is evaluated using AUROC,
AUPRC, accuracy, precision, recall, F1, MCC, Brier score, and 10-bin expected
calibration error. Both raw and calibrated probabilities are retained.

Regression is evaluated using Pearson's r, Spearman's rho, R², MAE, RMSE,
observed-versus-predicted plots, and residual diagnostics.

## 6. Confidence intervals and uncertainty

Percentile 95% confidence intervals use 2,000 test-sequence bootstrap
replicates. Classification replicates are stratified by outcome class;
regression replicates are stratified by up to ten observed-target quantile
strata.

The ten models are combined by averaging calibrated classification
probabilities or regression predictions. The sample standard deviation among
the ten predictions is reported as deep-ensemble epistemic model disagreement.
It is not a complete predictive interval and does not quantify all experimental
noise.

## 7. Exact execution

```bash
bash run_robustness_analysis.sh
```

For a short installation test:

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

Smoke-test results must not be substituted for the supplied formal analysis.

## 8. Single-model downstream export

The repeated-run statistics require all ten seeds. A convenient single model
may nevertheless be selected by minimum validation loss, without consulting
test performance:

```bash
bash export_single_models.sh
```

Run this command after a complete reproduction. It reconstructs the
architecture, loads the validation-selected local weights, and writes
TensorFlow SavedModels under
`results/robustness_analysis_rerun/single_models/`.
