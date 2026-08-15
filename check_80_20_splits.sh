#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
OUTPUT_ROOT="${1:-${REPO_ROOT}/results/split_check_seed${SEED}}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  printf 'ERROR: output directory already exists: %s\n' "${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/03_CNN_classification.py" \
  --positive-file "${REPO_ROOT}/data/processed/classification/positive.tsv" \
  --negative-file "${REPO_ROOT}/data/processed/classification/negative.tsv" \
  --output-dir "${OUTPUT_ROOT}/classification" \
  --seed "${SEED}" --folds 10 --split-only

"${PYTHON_BIN}" "${REPO_ROOT}/src/dataset_preparation/06_train_test_split.py" \
  --input "${REPO_ROOT}/data/processed/regression/regression_dataset.tsv" \
  --development-output "${OUTPUT_ROOT}/train_set.tsv" \
  --test-output "${OUTPUT_ROOT}/test_set.tsv" \
  --assignments "${OUTPUT_ROOT}/regression_systematic_split_assignments.tsv" \
  --manifest "${OUTPUT_ROOT}/regression_80_20_split_manifest.json" \
  --within-block-seed 1

"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/07_CNN_regression.py" \
  --development-file "${OUTPUT_ROOT}/train_set.tsv" \
  --test-file "${OUTPUT_ROOT}/test_set.tsv" \
  --split-manifest "${OUTPUT_ROOT}/regression_80_20_split_manifest.json" \
  --output-dir "${OUTPUT_ROOT}/regression" \
  --seed "${SEED}" --folds 10 --split-only

printf 'Split audits completed: %s\n' "${OUTPUT_ROOT}"
