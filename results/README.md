# Results

`robustness_analysis/` contains the immutable supplied outputs from the formal
run:

- random and Hamming split manifests;
- per-seed histories, predictions, calibration parameters, and metrics;
- ensemble predictions and metrics;
- calibration, ROC, precision-recall, residual, and uncertainty plots;
- 2,000-replicate bootstrap confidence intervals; and
- combined manuscript tables.

Intermediate seed weight files and duplicate model exports are not included.
Run `bash run_robustness_analysis.sh` to write an independent rerun to
`robustness_analysis_rerun/` without overwriting these reference outputs.
