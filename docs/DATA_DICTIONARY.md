# Data dictionary

## Processed inputs

| File | Rows | Columns | Purpose |
|---|---:|---|---|
| `data/processed/classification/positive.tsv` | 8,460 | `sequence`, `label` | Positive classification examples (`label=1`) |
| `data/processed/classification/negative.tsv` | 8,460 | `sequence`, `label` | Balanced negative examples (`label=0`) |
| `data/processed/classification/classification_source_counts.tsv` | 102,746 | `Sequence`, `Ratio_R2`, `Ratio_R4`, `Diff_R4_R2`, `Flag` | R2/R4 copy-frequency table used to assign classification labels |
| `data/processed/regression/regression_dataset.tsv` | 4,506 | `sequence`, `value` | Balanced regression model input |
| `data/processed/regression/regression_source_values.tsv` | 102,746 | `sequence`, `value` | Regression target values before balancing |
| `data/processed/merged_sequences_all.txt` | 132,037 | sequence and sample copy counts | Combined denoised sequence/count table used by dataset preparation |

`sequence` is an eight-residue amino-acid string. Classification `label` is 0
or 1. Regression `value` is the processed log-enrichment target.

## Model-analysis partitions

| Task | Total | Development | Independent test |
|---|---:|---:|---:|
| Classification | 16,920 | 13,536 | 3,384 |
| Regression | 4,506 | 3,604 | 902 |

The 20% independent test set is fixed before model development. Ten-fold
cross-validation and inner validation for early stopping are confined to the
80% development set. Exact split manifests, fold assignments and predictions
are under `results/holdout_10fold_analysis/`.

## Prediction files

The prediction programs accept a text file with one eight-residue sequence per
line. If a row contains additional tab-separated columns, only the first column
is used.

- classification output: sequence and classifier probability;
- regression output: sequence and predicted enrichment score in the original
  target units.

Large exhaustive sequence-space prediction tables are derived outputs and are
not stored in the repository; they can be regenerated with the supplied models
and prediction programs.

## Raw sequencing reads

Raw FASTQ files are available from the NCBI Sequence Read Archive under
accession PRJNA664254. The processing programs required to create the public
sequence-level tables are under `src/ngs_preprocessing/`.
