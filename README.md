# ANCHOR

ANCHOR (AI-NGS Consensus Hierarchy for Optimized Refinement) is an end-to-end
workflow for affinity maturation of the Fyn SH2 domain:

```text
paired-end NGS reads
  -> merged and template-filtered reads
  -> translated variable-region peptides and copy counts
  -> denoised classification/regression datasets
  -> CNN training and robustness analysis
  -> classification screening and regression ranking
```

This release contains the preprocessing programs, dataset-construction code,
model-training and evaluation code, full-sequence prediction programs,
processed inputs, frozen partitions, the latest trained models, environment
files, and supplied result tables and figures.

Release: **v1.2.0 (2026-08-14)**

## Start here

- [Complete raw-read-to-prediction workflow](docs/END_TO_END_WORKFLOW.md)
- [Robustness analysis](docs/ROBUSTNESS_ANALYSIS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Models and parameters](docs/MODELS.md)

## Repository structure

```text
ANCHOR/
|-- src/
|   |-- ngs_preprocessing/       FASTQ/FASTA processing and copy counting
|   |-- dataset_preparation/     Classification/regression dataset construction
|   |-- model_training/          80:20 holdout and 10-fold CNN evaluation
|   |-- robustness_analysis/     Isolated splits and repeated-run evaluation
|   `-- prediction/              Classification screening and regression scoring
|-- data/
|   |-- processed/               Public sequence-level model inputs
|   `-- frozen_splits/           Exact random and Hamming-separated partitions
|-- models/latest_models/        Latest classifier and regressor SavedModels
|-- results/robustness_analysis/ Supplied metrics, predictions, and figures
|-- results/holdout_10fold_analysis/ 80:20/10-fold outputs and process records
|-- config/                      Frozen parameters and run configuration
|-- environment/                 Dependency specifications
|-- docs/                        Workflow, data, model, and release documentation
`-- run_robustness_analysis.sh
```

Historical source files from the earlier repository layout are retained under
`legacy/` for provenance. Maintained programs are under `src/`.

## NGS preprocessing

The NGS workflow merges paired reads with FLASH, trims fixed DNA motifs,
separates orientations, reverse-complements reverse reads, filters the DNA and
protein templates, translates reads, extracts the variable regions, counts
unique peptide copies, and applies denoising rules.

The full commands, input/output filenames, exact motifs, and sample-dependent
checks are documented in
[docs/END_TO_END_WORKFLOW.md](docs/END_TO_END_WORKFLOW.md).

## Dataset construction

Classification labels compare normalized round-2 and round-4 copy frequencies.
Positive and negative examples are balanced before fitting. Regression targets
are log10 enrichment values, followed by balanced sampling across positive and
negative values. Public processed tables are under `data/processed/`.

## Environment

The formal model analysis used Python 3.8.20 and TensorFlow 2.4.0. The original
GPU server used CUDA 11.0 and cuDNN 8.

```bash
conda env create -f environment/environment.yml
conda activate anchor-tf24
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pip install -r environment/requirements_tensorflow24_runtime.txt
python -m pip install -r environment/requirements_analysis.txt
```

For TensorFlow 2.4, keep NumPy at 1.19.2 and six at 1.15.0 as specified in the
environment files.

## Model analysis

From the repository root:

```bash
bash run_robustness_analysis.sh
```

This runs classification and regression with random and Hamming-separated
partitions, independent seeds 1-10, validation-only early stopping,
validation-only Platt calibration, expanded metrics, and 2,000 bootstrap
replicates. Output is written to `results/robustness_analysis_rerun/`; the
supplied reference results are not overwritten.

All train, validation, and test sets are mutually exclusive. In the stringent
design, the minimum inter-partition Hamming distance is 2. Test data are never
used for early stopping, calibration, threshold selection, seed selection, or
model selection.

The manuscript-aligned 80:20 holdout analysis with 10-fold cross-validation
inside the 80% development set can be rerun with:

```bash
bash check_80_20_splits.sh
bash run_80_20_10fold_analysis.sh
```

The supplied output, including split manifests, fold assignments, training
histories, out-of-fold and independent-test predictions, calibration files,
bootstrap intervals, repeated-seed summaries, uncertainty estimates, and
plots, is archived under `results/holdout_10fold_analysis/`. SavedModels
created during that evaluation are intentionally omitted so that the public
repository contains only the two downstream inference models described below.

## Latest models

Only two downstream inference models are distributed: one classifier and one
regressor copied byte-for-byte from the `pTyr_antibody-analog/model` archive. They are
under `models/latest_models/`; the classifier returns a raw sigmoid probability
and the regressor is accompanied by target-scaling metadata. These inference
exports are separate from the seed-level models used for the supplied
repeated-run and Hamming-separated evaluation summaries. Historical
SavedModels, redundant copies, and intermediate seed weights are not included.

## Full-sequence prediction

Classification screening:

```bash
python src/prediction/classification_Multi-thread_new.py \
  output_T.txt output_T_predict.txt \
  --threshold 0.99 --batch_size 10240
```

Regression ranking of the sequences passing classification:

```bash
cut -f 1 output_T_predict.txt > pass_classification99_seq
python src/prediction/regression_multi_thread.py \
  pass_classification99_seq pass_classification99_seq_regression_score \
  --batch_size 10240
```

Both programs default to the latest SavedModels in this repository.

## Reference results

Key Hamming-separated ensemble estimates are:

| Task | Metric | Estimate (95% bootstrap CI) |
|---|---|---|
| Classification | AUROC | 0.945 (0.937-0.954) |
| Classification | AUPRC | 0.961 (0.955-0.967) |
| Regression | Pearson's r | 0.780 (0.755-0.804) |
| Regression | Spearman's rho | 0.771 (0.749-0.794) |

Complete seed-level and ensemble metrics, confidence intervals, predictions,
calibration outputs, residuals, and uncertainty values are under
`results/robustness_analysis/`.

The separate manuscript-aligned 80:20/10-fold results and their generated
process records are under `results/holdout_10fold_analysis/`.

## Documentation

- [Complete workflow](docs/END_TO_END_WORKFLOW.md)
- [Robustness analysis](docs/ROBUSTNESS_ANALYSIS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Models and parameters](docs/MODELS.md)
- [Versioned release procedure](docs/VERSIONED_RELEASE.md)
- [Upload guide](docs/UPLOAD_GUIDE.md)

## Integrity check

```bash
python scripts/maintenance/validate_release.py
```

`MANIFEST.tsv` and `CHECKSUMS.sha256` enumerate the published files and their
SHA-256 checksums.

Raw FASTQ files and very large exhaustive sequence-space prediction tables are
not duplicated in this repository. Raw-read archive accession and final
manuscript citation should be added when available.
