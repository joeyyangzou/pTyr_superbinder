# Fixed-test Hamming-buffer sensitivity analysis

This analysis tests local sequence extrapolation without changing the primary
random 80:20 evaluation. It retains the exact 3,384-sequence independent test
set from the primary classification analysis and removes every non-test
sequence at Hamming distance 0 or 1 from any test sequence. The resulting
development pool has a minimum development-test Hamming distance of 2.

The fixed test set is not used for cross-validation, early stopping, Platt
calibration, threshold selection, seed selection, or epoch selection.

## Published design and results

- Source classification library: 16,920 unique eight-residue sequences
- Fixed independent test set: 3,384 sequences (1,692 positive, 1,692 negative)
- Excluded development candidates at Hamming distance <= 1: 10,084
- Retained Hamming-buffered development set: 3,452 sequences
  (1,535 positive, 1,917 negative)
- Minimum development-test Hamming distance: 2
- Test sequences with a development neighbor at Hamming distance <= 1: 0
- Training protocol: development-only 10-fold cross-validation and ten final
  fits using seeds 1-10
- Ensemble AUROC: 0.950 (95% bootstrap CI, 0.942-0.957)
- Ensemble AUPRC: 0.963 (95% bootstrap CI, 0.957-0.968)

The corresponding primary random-split ensemble AUROC was 0.976. Because the
Hamming buffer removed 74.5% of the original development pool, the performance
difference reflects both sequence separation and substantially reduced
training-set coverage.

Published partitions, predictions, per-fold and per-seed outputs, bootstrap
summaries, and Supplementary Figure S6/Table S6 are under
`results/hamming_buffer_sensitivity/`. Sensitivity-analysis model weights are
not distributed because the two models under `models/latest_models/` are the
maintained inference artifacts.

## Re-run the full analysis

Create the pinned TensorFlow 2.4 environment described in the root README,
then run from the repository root:

```bash
conda activate anchor-tf24
bash run_fixed_test_hamming_buffer.sh
```

The default output directory is
`results/hamming_buffer_sensitivity_rerun/`. The script refuses to overwrite an
existing output directory. To select another destination:

```bash
bash run_fixed_test_hamming_buffer.sh /path/to/new/output
```

The complete analysis performs ten-fold cross-validation, ten independently
seeded final fits, Platt calibration, 2,000 class-stratified bootstrap
replicates, and figure/table generation. It may take several hours. A larger
batch size can be requested without changing the scripts:

```bash
BATCH_SIZE=256 bash run_fixed_test_hamming_buffer.sh /path/to/new/output
```

Do not increase the batch size if it causes GPU out-of-memory errors.

## Split-only audit

The split can be regenerated and checked without training the CNN:

```bash
python src/model_training/14_make_fixed_test_hamming_buffer.py \
  --output-dir results/hamming_buffer_split_check

python src/model_training/03_CNN_classification.py \
  --development-file results/hamming_buffer_split_check/development_hamming_buffer.tsv \
  --test-file results/hamming_buffer_split_check/independent_test_fixed.tsv \
  --split-manifest results/hamming_buffer_split_check/split_manifest.json \
  --output-dir results/hamming_buffer_split_audit \
  --split-only
```

The generated manifest should report 3,384 fixed test sequences, zero exact
overlap, a minimum development-test Hamming distance of 2, and no test sequence
with a development neighbor at distance 0 or 1.
