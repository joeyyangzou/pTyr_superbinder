# Data dictionary

## Public processed model inputs

| File | Rows excluding header | Columns | Purpose |
|---|---:|---|---|
| `data/processed/classification/positive.tsv` | 8,460 | `sequence`, `label` | Positive classification examples (`label=1`) |
| `data/processed/classification/negative.tsv` | 8,460 | `sequence`, `label` | Balanced negative examples (`label=0`) |
| `data/processed/classification/classification_source_counts.tsv` | 102,746 | `Sequence`, `Ratio_R2`, `Ratio_R4`, `Diff_R4_R2`, `Flag` | Processed R2/R4 frequency-derived classification source table |
| `data/processed/regression/regression_dataset.tsv` | 4,506 | `sequence`, `value` | Balanced regression dataset used by the robustness workflow |
| `data/processed/regression/regression_source_values.tsv` | 102,746 | `sequence`, `value` | Processed sequence-level regression values before balancing |
| `data/processed/regression/legacy_train.tsv` | 3,604 | no header; sequence and value | Original regression training partition |
| `data/processed/regression/legacy_test.tsv` | 902 | no header; sequence and value | Original regression held-out partition |

`sequence` is an eight-character amino-acid string. Classification `label` is
0 or 1. Regression `value` is the processed enrichment target used by the
regression model.

## Frozen robustness-analysis partitions

Each split directory contains:

- `train.tsv`: samples used for gradient-based fitting.
- `validation.tsv`: samples used for early stopping and classification
  calibration.
- `test.tsv`: frozen samples used for final evaluation only.
- `excluded_hamming_buffer.tsv`: samples excluded because they were too close
  to test or validation sequences; empty for random splitting.
- `split_metadata.json`: sizes, split seed, class counts, and pairwise Hamming
  audits.

### Partition sizes

| Task | Design | Train | Validation | Test | Excluded buffer |
|---|---|---:|---:|---:|---:|
| Classification | Random | 11,844 | 1,692 | 3,384 | 0 |
| Classification | Hamming | 2,734 | 432 | 3,384 | 10,370 |
| Regression | Random | 3,153 | 451 | 902 | 0 |
| Regression | Hamming | 1,568 | 237 | 902 | 1,799 |

The Hamming partitions have minimum pairwise inter-partition distance 2 and no
test sequence with a distance-0 or distance-1 training neighbor.

## Supplied predictions and derived results

Per-seed test predictions are stored below each `seed_*` directory. Ensemble
files contain the test label/value, prediction mean, model-disagreement SD, and
for regression, residual and absolute error.

Large exhaustive full-sequence-space prediction tables from the original local
analysis are not included in this GitHub package:

| Local derived file | Approximate size | Reason omitted |
|---|---:|---|
| `final_predict.txt` | 477 MB | Fully derived and reproducible; unsuitable for standard GitHub storage |
| `predict.txt` | 247 MB | Fully derived and reproducible |
| `reduced_sequence_prediction.txt` | 67 MB | Fully derived intermediate |

These tables are not training inputs and are not required to reproduce any
reported test metric. They can be regenerated using the published prediction
scripts and trained models. If the journal requires direct hosting of these
large derived tables, deposit them in Zenodo/Figshare or use Git LFS and add the
permanent DOI to the release description.

## Raw reads

Raw FASTQ files are not duplicated in this repository. The scripts needed to
produce sequence-level processed tables are supplied in
`src/ngs_preprocessing/`. The final manuscript should cite the permanent raw
read archive accession if one is available.
