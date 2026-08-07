#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" 10_robustness_analysis.py \
  --task all \
  --output-dir robustness_results \
  --single-model-split random \
  --export-single-models-only
