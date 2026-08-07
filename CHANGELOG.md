# Changelog

## v1.2.0 - 2026-08-07

- Documented the complete raw-read-to-prediction ANCHOR workflow.
- Updated classification and regression prediction programs to load the
  latest public models and scaling metadata by default.
- Added FASTQ-to-FASTA, unique-peptide counting, and deterministic split
  helpers.
- Renamed the public robustness workflow and removed internal response-oriented
  naming.
- Retained only the latest validation-selected classification and regression
  SavedModels.
- Removed historical SavedModels, redundant model copies, and intermediate
  seed weight files.
- Retained ten-seed histories, predictions, metrics, calibration results,
  confidence intervals, uncertainty estimates, and residual analyses.
- Updated paths, documentation, environment files, manifests, checksums, and
  automated release validation.

## v1.1.0 - 2026-08-07

- Added mutually exclusive training, validation, and test partitions.
- Added frozen random and Hamming-distance-separated partitions.
- Added ten independent seeds, expanded metrics, Platt calibration,
  stratified bootstrap confidence intervals, residual analyses, and
  prediction-level ensemble uncertainty.

## v1.0.0

- Original ANCHOR preprocessing, CNN training, prediction scripts, processed
  datasets, and trained TensorFlow SavedModels.
