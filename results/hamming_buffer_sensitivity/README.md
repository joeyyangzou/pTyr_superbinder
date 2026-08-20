# Published fixed-test Hamming-buffer outputs

These files support Supplementary Figure S6 and Supplementary Table S6. The
primary random-split test set was retained unchanged, and all non-test
sequences at Hamming distance 0 or 1 from any test sequence were excluded
before model development.

## Directory map

- `buffer_partitions/`: retained development set, frozen test set, excluded
  candidates, and the distance audit
- `splits/`: model-ready development/test tables and ten-fold assignments
- `cross_validation/`: fold predictions, histories, metrics, and ROC/PR plots
- `final_model/`: single-model predictions, calibration, metrics, and plots;
  sensitivity-model weights are intentionally not distributed
- `repeated_training/`: ten-seed metrics, ensemble predictions, calibration,
  bootstrap confidence intervals, and uncertainty summaries
- `summary/`: Supplementary Figure S6, Supplementary Table S6, and a concise
  interpretation
- `run_configuration.json`: software versions, seeds, epochs, batch size,
  bootstrap count, and elapsed runtime from the reported run

Use `bash run_fixed_test_hamming_buffer.sh` from the repository root to
regenerate these outputs in a new directory. Full instructions are in
`docs/FIXED_TEST_HAMMING_BUFFER.md`.
