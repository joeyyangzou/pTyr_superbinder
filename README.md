# ANCHOR

ANCHOR (AI-NGS Consensus Hierarchy for Optimized Refinement) is the analysis
workflow used to train and evaluate sequence-based CNN models for affinity
maturation of the Fyn SH2 domain. This public release contains source code,
processed model inputs, frozen data partitions, the latest trained models,
software environments, and complete robustness and reproducibility outputs.

Release: **v1.2.0 (2026-08-07)**

## Included resources

- NGS preprocessing and sequence-count processing scripts.
- Processed classification and regression datasets.
- Mutually exclusive training, validation, and test partitions.
- Random and Hamming-distance-separated validation designs.
- Ten independent training seeds for each task and split design.
- Classification calibration, bootstrap confidence intervals, residual
  analyses, and deep-ensemble model-disagreement uncertainty.
- Per-seed histories, predictions, metrics, and aggregate figures.
- One latest validation-selected classification SavedModel and one latest
  validation-selected regression SavedModel.

Only the two latest inference models are distributed. Historical SavedModels
and the 40 intermediate seed weight files are intentionally not included.
The code, frozen inputs, seeds, histories, and predictions needed to audit or
repeat the analyses remain available.

## Repository structure

```text
ANCHOR/
|-- src/
|   |-- ngs_preprocessing/       Raw-read preprocessing utilities
|   |-- dataset_preparation/     Classification/regression table construction
|   |-- model_training/          Original classifier and regressor scripts
|   |-- prediction/              Full-sequence prediction utilities
|   `-- robustness_analysis/     Leakage-free repeated-run analysis
|-- data/
|   |-- processed/               Public processed model inputs
|   `-- frozen_splits/           Exact random and Hamming partitions
|-- models/latest_models/        Latest classifier and regressor SavedModels
|-- results/robustness_analysis/ Supplied metrics, predictions, and plots
|-- config/                      Frozen parameters and run configuration
|-- environment/                 Dependency specifications
|-- docs/                        Workflow and data/model documentation
`-- run_robustness_analysis.sh
```

Historical source scripts from the earlier repository layout are retained
under `legacy/` for provenance. New analyses should use `src/`.

## Environment

The formal analysis used Python 3.8.20 and TensorFlow 2.4.0. TensorFlow must
match the installed CUDA/cuDNN stack; the original server used CUDA 11.0 and
cuDNN 8.

```bash
conda env create -f environment/environment.yml
conda activate anchor-tf24
python -c "import tensorflow as tf; print(tf.__version__)"
python -m pip install -r environment/requirements_tensorflow24_runtime.txt
python -m pip install -r environment/requirements_analysis.txt
```

For TensorFlow 2.4, keep NumPy at 1.19.2 and six at 1.15.0 as specified in the
environment files.

## Reproduce the robustness analysis

From the repository root:

```bash
bash run_robustness_analysis.sh
```

This runs classification and regression with random and Hamming-separated
partitions, seeds 1-10, validation-only early stopping, validation-only Platt
calibration, and 2,000 bootstrap replicates. New output is written to
`results/robustness_analysis_rerun/`; supplied reference results under
`results/robustness_analysis/` are not overwritten.

Parameters are frozen in `config/robustness_run_configuration.json` and
`config/model_hyperparameters.json`.

For the stringent design, all training and validation sequences within Hamming
distance <2 of the held-out partition are placed in an exclusion buffer. The
minimum inter-partition Hamming distance is therefore 2. The test set is never
used for early stopping, calibration, threshold selection, seed selection, or
model selection.

## Latest models

The two current inference artifacts are under `models/latest_models/`:

- classification: random-split seed 7, selected by validation loss;
- regression: random-split seed 5, selected by validation loss.

Test performance was not used for model selection. Classification calibration
and regression target-scaling metadata are supplied beside each SavedModel.

The ten-seed ensemble is used only for repeated-training statistics and
model-disagreement uncertainty; it is a prediction-level ensemble, not a
single merged weight file. To reconstruct fresh single-model exports after a
complete rerun, run `bash export_single_models.sh`.

## Reference results

Key Hamming-separated ensemble estimates are:

| Task | Metric | Estimate (95% bootstrap CI) |
|---|---|---|
| Classification | AUROC | 0.945 (0.937-0.954) |
| Classification | AUPRC | 0.961 (0.955-0.967) |
| Regression | Pearson's r | 0.780 (0.755-0.804) |
| Regression | Spearman's rho | 0.771 (0.749-0.794) |

Complete seed-level and ensemble metrics, confidence intervals, predictions,
calibration outputs, residuals, and uncertainty values are in
`results/robustness_analysis/`.

## Documentation

- [Reproducibility workflow](docs/REPRODUCIBILITY.md)
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

Raw FASTQ files and very large derived exhaustive sequence-space tables are
not duplicated in this release. Their generation scripts and provenance are
documented in `src/ngs_preprocessing/` and `docs/DATA_DICTIONARY.md`.

## Citation

Please cite the associated ANCHOR manuscript and this versioned repository
release. Add the final manuscript citation to the GitHub release description
when the bibliographic record becomes available.
