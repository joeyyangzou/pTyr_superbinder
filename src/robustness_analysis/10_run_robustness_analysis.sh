#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" 10_robustness_analysis.py \
  --task all \
  --split-modes random hamming \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --split-seed 2026 \
  --test-size 0.20 \
  --validation-size 0.10 \
  --minimum-test-train-hamming 2 \
  --bootstrap-replicates 2000 \
  --mc-samples 0 \
  --calibration platt \
  "$@"
