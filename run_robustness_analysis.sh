#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" src/robustness_analysis/10_robustness_analysis.py \
  --task all \
  --positive-file data/processed/classification/positive.tsv \
  --negative-file data/processed/classification/negative.tsv \
  --regression-file data/processed/regression/regression_dataset.tsv \
  --output-dir results/robustness_analysis_rerun \
  --split-modes random hamming \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 2026 \
  --test-size 0.20 \
  --validation-size 0.10 \
  --minimum-test-train-hamming 2 \
  --classification-epochs 200 \
  --regression-epochs 1000 \
  --classification-patience 20 \
  --regression-patience 50 \
  --classification-batch-size 64 \
  --regression-batch-size 128 \
  --bootstrap-replicates 2000 \
  --mc-samples 0 \
  --calibration platt \
  "$@"
