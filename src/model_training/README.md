# CNN model training

The two training programs implement the manuscript's random-split baseline:

- `03_CNN_classification.py`: binary classification;
- `07_CNN_regression.py`: regression.

Each program first freezes an independent 20% test set. Outer 10-fold
cross-validation is performed only within the remaining 80% development set.
Each outer training partition is divided again to create an inner validation
subset for early stopping. Neither the outer validation fold nor the final
test set is used for epoch selection. After epoch selection, a newly
initialized final model is fitted on all development observations and tested
once on the fixed 20% holdout.

These programs also run ten independently seeded final-model fits, calibration,
bootstrap confidence intervals, and prediction-disagreement analyses on the
same frozen development/test partition.

## Commands

```bash
python src/model_training/03_CNN_classification.py \
  --positive-file data/processed/classification/positive.tsv \
  --negative-file data/processed/classification/negative.tsv \
  --output-dir results/classification_80_20_10fold \
  --seed 2026 --folds 10

python src/model_training/07_CNN_regression.py \
  --development-file results/split_check_seed2026/train_set.tsv \
  --test-file results/split_check_seed2026/test_set.tsv \
  --split-manifest results/split_check_seed2026/regression_80_20_split_manifest.json \
  --output-dir results/regression_80_20_10fold \
  --seed 2026 --folds 10
```

Use `--split-only` to write and audit assignments without importing
TensorFlow. Use a new output directory for every full run.

## Main outputs

Classification outputs include per-fold metrics, mean and standard deviation,
pooled out-of-fold predictions, ROC and precision-recall curves, independent
test metrics with stratified-bootstrap 95% confidence intervals, a reliability
plot, and one final SavedModel.

Regression outputs include per-fold Pearson, Spearman, R2, MAE, and RMSE;
pooled out-of-fold predictions; independent-test confidence intervals; scatter
and residual plots; target scalers; and one final SavedModel.

Numerical values in reports must be taken from the generated output rather
than assumed from an earlier run.

From the repository root, `bash check_80_20_splits.sh` checks both task splits
without loading TensorFlow. `bash run_80_20_10fold_analysis.sh` runs the full
classification and regression analysis and then creates a combined summary.
