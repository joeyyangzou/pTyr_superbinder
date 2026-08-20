# Changelog

## v1.2.1 - 2026-08-20

- Added the fixed-test Hamming-buffer sequence-similarity sensitivity analysis.
- Published the retained and excluded partitions, fold assignments,
  cross-validation outputs, ten-seed ensemble predictions, calibration files,
  and 2,000-replicate bootstrap confidence intervals.
- Added Supplementary Figure S6 and Supplementary Table S6 source files.
- Added an end-to-end sensitivity-analysis runner, summary generator, run
  instructions, parameter metadata, and release validation checks.
- Retained the v1.2.0 classification and regression inference models without
  adding sensitivity-model weights.

## v1.2.0 - 2026-08-18

- Organized the repository as one continuous workflow from paired-end NGS
  reads to classification screening and regression ranking of new sequences.
- Included the maintained NGS preprocessing, dataset-construction, CNN
  training and prediction programs.
- Added processed sequence-level model inputs, fixed 80:20 partitions,
  development-set ten-fold assignments and complete evaluation outputs.
- Added calibration, bootstrap confidence intervals, repeated-training
  summaries, uncertainty outputs and regression residual analysis.
- Included only the current classification and regression SavedModels and the
  required regression target scaler.
- Added pinned software environments, model parameters, file manifests and
  SHA-256 checksums.
