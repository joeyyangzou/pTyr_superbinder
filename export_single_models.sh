#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" src/robustness_analysis/10_robustness_analysis.py \
  --task all \
  --output-dir results/robustness_analysis_rerun \
  --single-model-split random \
  --export-single-models-only
