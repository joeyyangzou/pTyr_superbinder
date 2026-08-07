# Source-code map

| Directory | Role | Primary entry point |
|---|---|---|
| `ngs_preprocessing/` | Read processing, translation, denoising, and count aggregation | Numbered Perl/Python utilities |
| `dataset_preparation/` | Construct classification labels and regression inputs | Numbered scripts 01, 02, 04, 05, 06 |
| `model_training/` | Original single-model classifier and regressor training | `03_CNN_classification.py`, `07_CNN_regression.py` |
| `prediction/` | Batched/full-sequence inference | Classification and regression prediction scripts |
| `robustness_analysis/` | Leakage-free splits, ten-seed training, calibration, bootstrap CIs, ensemble uncertainty, and model export | `10_robustness_analysis.py` |

The maintained reproducibility workflow is under `robustness_analysis/`.
Historical source files are retained under `legacy/` for provenance and are
not used to generate the supplied robustness results.
