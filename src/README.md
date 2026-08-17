# Source-code map

| Directory | Role | Primary entry point |
|---|---|---|
| `ngs_preprocessing/` | Read processing, translation, denoising, and count aggregation | See `ngs_preprocessing/README.md` |
| `dataset_preparation/` | Construct classification labels and regression inputs | See `dataset_preparation/README.md` |
| `model_training/` | Original single-model classifier and regressor training | `03_CNN_classification.py`, `07_CNN_regression.py` |
| `prediction/` | Batched/full-sequence inference with the latest models | See `prediction/README.md` |
The maintained evaluation workflow is implemented by the two programs under
`model_training/` and the root-level `run_80_20_10fold_analysis.sh` wrapper.
Historical source files are retained under `legacy/` for provenance and are
not used to generate the supplied current results.

The complete connection from raw reads through model prediction is documented
in [`../docs/END_TO_END_WORKFLOW.md`](../docs/END_TO_END_WORKFLOW.md).
