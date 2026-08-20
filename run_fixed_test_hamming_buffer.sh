#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${1:-${REPO_ROOT}/results/hamming_buffer_sensitivity_rerun}"
PRIMARY_RESULTS_DIR="${REPO_ROOT}/results/holdout_10fold_analysis/classification_80_20_10fold_results"
FIXED_TEST_FILE="${PRIMARY_RESULTS_DIR}/splits/independent_test_20.tsv"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VERBOSE="${VERBOSE:-2}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  printf 'ERROR: output directory already exists: %s\n' "${OUTPUT_ROOT}" >&2
  printf 'Choose a new output path to avoid mixing independent runs.\n' >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/14_make_fixed_test_hamming_buffer.py" \
  --positive-file "${REPO_ROOT}/data/processed/classification/positive.tsv" \
  --negative-file "${REPO_ROOT}/data/processed/classification/negative.tsv" \
  --fixed-test-file "${FIXED_TEST_FILE}" \
  --output-dir "${OUTPUT_ROOT}/buffer_partitions" \
  --minimum-development-test-hamming 2

"${PYTHON_BIN}" "${REPO_ROOT}/src/model_training/03_CNN_classification.py" \
  --development-file "${OUTPUT_ROOT}/buffer_partitions/development_hamming_buffer.tsv" \
  --test-file "${OUTPUT_ROOT}/buffer_partitions/independent_test_fixed.tsv" \
  --split-manifest "${OUTPUT_ROOT}/buffer_partitions/split_manifest.json" \
  --output-dir "${OUTPUT_ROOT}/classification_results" \
  --seed 2026 \
  --folds 10 \
  --inner-validation-size 0.10 \
  --epochs 200 \
  --patience 20 \
  --batch-size "${BATCH_SIZE}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --training-seeds 1 2 3 4 5 6 7 8 9 10 \
  --verbose "${VERBOSE}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/maintenance/summarize_fixed_test_hamming_buffer.py" \
  --primary-results-dir "${PRIMARY_RESULTS_DIR}" \
  --hamming-results-dir "${OUTPUT_ROOT}/classification_results" \
  --hamming-split-manifest "${OUTPUT_ROOT}/buffer_partitions/split_manifest.json" \
  --output-dir "${OUTPUT_ROOT}/summary"

printf 'Hamming-buffer sensitivity analysis completed: %s\n' "${OUTPUT_ROOT}"
