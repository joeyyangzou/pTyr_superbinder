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

The default is the raw sigmoid probability used by the historical screening
workflow. Add `--calibrated` to use the validation-fitted Platt transformation.
Use `--model` and `--calibration` only when overriding the public defaults.

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

