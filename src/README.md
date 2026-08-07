# Source-code map

| Directory | Role | Primary entry point |
|---|---|---|
| `ngs_preprocessing/` | Read processing, translation, denoising, and count aggregation | See `ngs_preprocessing/README.md` |
| `dataset_preparation/` | Construct classification labels and regression inputs | See `dataset_preparation/README.md` |
| `model_training/` | Original single-model classifier and regressor training | `03_CNN_classification.py`, `07_CNN_regression.py` |
| `prediction/` | Batched/full-sequence inference with the latest models | See `prediction/README.md` |
| `robustness_analysis/` | Leakage-free splits, ten-seed training, calibration, bootstrap CIs, ensemble uncertainty, and model export | `10_robustness_analysis.py` |

The maintained robustness workflow is under `robustness_analysis/`.
Historical source files are retained under `legacy/` for provenance and are
not used to generate the supplied robustness results.

The complete connection from raw reads through model prediction is documented
in [`../docs/END_TO_END_WORKFLOW.md`](../docs/END_TO_END_WORKFLOW.md).
