# ANCHOR v1.2.1

This release adds the fixed-test Hamming-buffer sequence-similarity sensitivity
analysis reported in Supplementary Figure S6 and Supplementary Table S6.

## Added

- Script for retaining the frozen primary classification test set while
  excluding all non-test sequences at Hamming distance 0 or 1
- End-to-end runner for the development-only ten-fold/ten-seed Hamming-buffer
  analysis
- Summary script for ROC/precision-recall comparison and tabular bootstrap
  confidence intervals
- Published Hamming-buffer partitions, fold assignments, predictions, training
  histories, per-seed metrics, calibration outputs, and uncertainty summaries
- Supplementary Figure S6 and Supplementary Table S6 source files
- Full run instructions and release validation checks

## Reported sensitivity result

The exact primary independent test set was retained (n = 3,384). The Hamming
buffer removed 10,084 development candidates and retained 3,452 sequences with
a minimum development-test Hamming distance of 2. The ten-seed ensemble
achieved AUROC = 0.950 (95% bootstrap CI, 0.942-0.957) and AUPRC = 0.963
(0.957-0.968).

The maintained downstream inference models are unchanged: classification seed
5/epoch 132 and regression seed 8/epoch 199.
