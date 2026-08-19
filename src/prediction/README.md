# Full-sequence prediction

The two programs in this directory load the latest public SavedModels by
default. Input files contain one eight-residue sequence per line. If additional
tab-separated columns are present, only the first column is used.

## Classification screening

```bash
python src/prediction/classification_Multi-thread_new.py \
  output_T.txt output_T_predict.txt \
  --threshold 0.99 --batch_size 10240
```

The classifier returns the raw sigmoid probability used by the historical
screening workflow. To return probabilities calibrated using pooled out-of-fold
predictions from the 80% development set, add `--apply_platt`; the program then
loads `models/latest_models/classification/platt_calibration.json` by default.
The threshold is applied to whichever probability scale is selected. Use
`--model` or `--calibration` only when overriding the public defaults.
The calibrated F1-selected development-set cutoff reported in the model
analysis is 0.510; the historical full-library screening command above retains
the original raw-probability cutoff of 0.99.

## Regression ranking

```bash
cut -f 1 output_T_predict.txt > pass_classification99_seq
python src/prediction/regression_multi_thread.py \
  pass_classification99_seq pass_classification99_seq_regression_score \
  --batch_size 10240
```

The program automatically loads `target_scaler.json` and reports values in the
original regression-target units. Use `--model` and `--scaler` to override
the public defaults.
