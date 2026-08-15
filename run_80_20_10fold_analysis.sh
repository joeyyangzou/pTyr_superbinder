#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2026}"
OUTPUT_ROOT="${1:-${REPO_ROOT}/results/holdout_10fold_analysis_rerun_seed${SEED}}"

CLASSIFICATION_OUTPUT="${OUTPUT_ROOT}/classification_80_20_10fold_results"
REGRESSION_OUTPUT="${OUTPUT_ROOT}/regression_80_20_10fold_results"
SUMMARY_OUTPUT="${OUTPUT_ROOT}/summary"
SPLIT_WORK="${OUTPUT_ROOT}/regression_split"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  printf 'ERROR: output directory already exists: %s\n' "${OUTPUT_ROOT}" >&2
  printf 'Choose a new output path to avoid mixing independent runs.\n' >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${SPLIT_WORK}"

printf 'Step 1/4: classification 80:20 holdout, 10-fold CV, and repeated seeds\n'
"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/03_CNN_classification.py" \
  --positive-file "${REPO_ROOT}/data/processed/classification/positive.tsv" \
  --negative-file "${REPO_ROOT}/data/processed/classification/negative.tsv" \
  --output-dir "${CLASSIFICATION_OUTPUT}" \
  --seed "${SEED}" --folds 10 \
  --training-seeds 1 2 3 4 5 6 7 8 9 10 \
  --bootstrap-replicates 2000 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/classification.log"

printf 'Step 2/4: manuscript-defined systematic regression 80:20 split\n'
"${PYTHON_BIN}" "${REPO_ROOT}/src/dataset_preparation/06_train_test_split.py" \
  --input "${REPO_ROOT}/data/processed/regression/regression_dataset.tsv" \
  --development-output "${SPLIT_WORK}/train_set.tsv" \
  --test-output "${SPLIT_WORK}/test_set.tsv" \
  --assignments "${SPLIT_WORK}/regression_systematic_split_assignments.tsv" \
  --manifest "${SPLIT_WORK}/regression_80_20_split_manifest.json" \
  --within-block-seed 1

printf 'Step 3/4: regression 10-fold CV and repeated seeds\n'
"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/07_CNN_regression.py" \
  --development-file "${SPLIT_WORK}/train_set.tsv" \
  --test-file "${SPLIT_WORK}/test_set.tsv" \
  --split-manifest "${SPLIT_WORK}/regression_80_20_split_manifest.json" \
  --output-dir "${REGRESSION_OUTPUT}" \
  --seed "${SEED}" --folds 10 \
  --training-seeds 1 2 3 4 5 6 7 8 9 10 \
  --bootstrap-replicates 2000 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/regression.log"

printf 'Step 4/4: generate evaluation summary\n'
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/maintenance/summarize_80_20_results.py" \
  --classification-dir "${CLASSIFICATION_OUTPUT}" \
  --regression-dir "${REGRESSION_OUTPUT}" \
  --output-dir "${SUMMARY_OUTPUT}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/summary.log"

printf 'All analyses completed. Results: %s\n' "${OUTPUT_ROOT}"
