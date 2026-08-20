# ANCHOR

ANCHOR (AI-NGS Consensus Hierarchy for Optimized Refinement) is an end-to-end
workflow for processing display-library NGS data, training classification and
regression CNNs, and using the trained models to screen new SH2 variants.

## Workflow

```text
paired-end FASTQ reads
        |
        v
read merging, template filtering and translation
        |
        v
peptide sequences and copy counts
        |
        v
denoising and classification/regression dataset construction
        |
        v
80% development + 20% independent-test CNN analysis
        |
        v
optional fixed-test Hamming-buffer sensitivity audit
        |
        v
classification screening of new sequences
        |
        v
regression ranking of sequences passing the classifier
```

The complete commands from FASTQ input to final prediction are documented in
[`docs/END_TO_END_WORKFLOW.md`](docs/END_TO_END_WORKFLOW.md).

## Repository contents

| Path | Contents |
|---|---|
| `src/ngs_preprocessing/` | FASTQ/FASTA processing, translation, copy counting and denoising |
| `src/dataset_preparation/` | Classification labels and regression target construction |
| `src/model_training/` | Classification and regression CNN training and evaluation |
| `src/prediction/` | Classification screening and regression scoring of new sequences |
| `data/processed/` | Sequence-level inputs used by the model programs |
| `models/latest_models/` | Current classifier, regressor and regression target scaler |
| `results/holdout_10fold_analysis/` | Fixed splits, predictions, metrics, confidence intervals and plots |
| `results/hamming_buffer_sensitivity/` | Fixed-test Hamming-buffer partitions, predictions, metrics and Supplementary Figure S6/Table S6 |
| `config/` | Model architecture and analysis parameters |
| `environment/` | Conda and Python dependency specifications |

## 1. Install the software environment

The model analysis used Python 3.8.20, TensorFlow 2.4.0, CUDA 11.0 and cuDNN 8.

```bash
conda env create -f environment/environment.yml
conda activate anchor-tf24
python -m pip install -r environment/requirements_tensorflow24_runtime.txt
python -m pip install -r environment/requirements_analysis.txt
```

TensorFlow 2.4 requires the versions pinned in the environment files,
including NumPy 1.19.2 and six 1.15.0.

FLASH, Perl, `awk`, `grep`, `sed` and standard Unix command-line tools are also
required for the raw NGS stage.

## 2. Process raw NGS reads

Raw paired-end reads are merged with FLASH and then processed by the programs
under `src/ngs_preprocessing/`:

```text
FLASH
 -> 2.rmadpator.pl
 -> 3.extract_forward_revserse.pl
 -> 4.reverse_fastq.pl
 -> 5.fastq2fasta.sh
 -> 6.dna2pep.pl
 -> stat_uniq_pep_num.pl / stat_uniq_seq_num.pl
 -> join.pl
 -> sequence_denoise.py
 -> stat_average_copy.py
```

Exact motifs, commands, file naming rules and filtering conditions are given in
the [end-to-end workflow](docs/END_TO_END_WORKFLOW.md). Raw sequencing data are
available from the NCBI Sequence Read Archive under accession PRJNA664254.

## 3. Build the model datasets

Create a working directory because the dataset scripts use the stage filenames
shown in the end-to-end workflow:

```bash
REPO_ROOT="$(pwd)"
mkdir -p work/dataset_preparation
cp data/processed/merged_sequences_all.txt work/dataset_preparation/
cd work/dataset_preparation
```

Classification data:

```bash
python "${REPO_ROOT}/src/dataset_preparation/01_calculate_classification.py"
perl "${REPO_ROOT}/src/dataset_preparation/02_classification_positive_negative.pl"
```

Regression data:

```bash
python "${REPO_ROOT}/src/dataset_preparation/04_regression_preprocess.py"
python "${REPO_ROOT}/src/dataset_preparation/05_split_pos_neg_regression.py"
python "${REPO_ROOT}/src/dataset_preparation/06_train_test_split.py" \
  --input regression_dataset.txt
cd "${REPO_ROOT}"
```

Ready-to-use inputs are included at:

```text
data/processed/classification/positive.tsv
data/processed/classification/negative.tsv
data/processed/regression/regression_dataset.tsv
```

See [`src/dataset_preparation/README.md`](src/dataset_preparation/README.md)
and [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) for schemas and row
counts.

## 4. Train and evaluate the CNNs

First check the fixed partitions without loading TensorFlow:

```bash
bash check_80_20_splits.sh
```

Run the complete classification and regression analysis:

```bash
bash run_80_20_10fold_analysis.sh
```

The analysis freezes a 20% independent test set before model development.
Ten-fold cross-validation is performed only within the remaining 80%
development set. Inner validation subsets control early stopping; the outer
folds and independent test set do not participate in epoch selection.

The programs report classification AUROC, AUPRC, F1, MCC, Brier score and ECE,
and regression Pearson correlation, Spearman correlation, MAE and RMSE. They
also generate calibration/residual plots, 2,000-replicate bootstrap confidence
intervals and summaries from ten independent training seeds.

Detailed options are described in
[`src/model_training/README.md`](src/model_training/README.md).

### Fixed-test Hamming-buffer sensitivity analysis

To assess extrapolation beyond one-residue development-test neighbors while
retaining the exact primary independent test set, run:

```bash
bash run_fixed_test_hamming_buffer.sh
```

This excludes every non-test sequence at Hamming distance 0 or 1 from any test
sequence, enforces a minimum development-test distance of 2, and then repeats
the development-only 10-fold/ten-seed protocol. Published outputs are under
`results/hamming_buffer_sensitivity/`; full design and run instructions are in
[`docs/FIXED_TEST_HAMMING_BUFFER.md`](docs/FIXED_TEST_HAMMING_BUFFER.md).

## 5. Predict new sequences

Prepare a text file containing one eight-residue amino-acid sequence per line.
Additional tab-separated columns are ignored.

Classification screening:

```bash
python src/prediction/classification_Multi-thread_new.py \
  output_T.txt output_T_predict.txt \
  --threshold 0.99 --batch_size 10240
```

Regression ranking of sequences passing classification:

```bash
cut -f 1 output_T_predict.txt > pass_classification99_seq
python src/prediction/regression_multi_thread.py \
  pass_classification99_seq pass_classification99_seq_regression_score \
  --batch_size 10240
```

Both prediction programs load the models under `models/latest_models/` by
default. The classifier is the selected seed-5/132-epoch primary model and the
regressor is the selected seed-8/199-epoch primary model. Development-only
Platt parameters are supplied for optional calibrated classification
probabilities (`--apply_platt`). See
[`src/prediction/README.md`](src/prediction/README.md) for input and output
details.

## Supplied model-analysis results

| Task | Metric | Independent-test estimate (95% bootstrap CI) |
|---|---|---|
| Classification | AUROC | 0.974 (0.968-0.978) |
| Classification | AUPRC | 0.979 (0.975-0.982) |
| Regression | Pearson's r | 0.870 (0.849-0.889) |
| Regression | Spearman's rho | 0.825 (0.808-0.842) |
| Hamming-buffer classification ensemble | AUROC | 0.950 (0.942-0.957) |
| Hamming-buffer classification ensemble | AUPRC | 0.963 (0.957-0.968) |

Primary metrics, predictions, split assignments, training histories and plots
are under `results/holdout_10fold_analysis/`. The fixed-test Hamming-buffer
sensitivity outputs are under `results/hamming_buffer_sensitivity/`.

## Version and file integrity

This workflow is version `v1.2.1`.

```bash
python scripts/maintenance/validate_release.py
```

`MANIFEST.tsv` lists every distributed file and `CHECKSUMS.sha256` provides its
SHA-256 checksum.
