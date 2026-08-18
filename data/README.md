# Processed data

`processed/` contains the sequence-level inputs needed by the classification
and regression programs:

- `classification/positive.tsv` and `classification/negative.tsv`;
- `classification/classification_source_counts.tsv`;
- `regression/regression_dataset.tsv` and `regression/regression_source_values.tsv`;
- `merged_sequences_all.txt`, the combined sequence/count input used during
  dataset construction.

Fixed development/test partitions and fold assignments are stored with the
analysis outputs under `../results/holdout_10fold_analysis/`.

See [`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md) for table schemas
and row counts.
