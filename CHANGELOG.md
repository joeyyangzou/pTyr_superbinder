# Changelog

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
