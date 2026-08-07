# Dataset preparation

These programs convert denoised sequence/count tables into classification and
regression model inputs.

| Program | Purpose |
|---|---|
| `01_calculate_classification.py` | Compare normalized round-2 and round-4 frequencies and assign labels |
| `02_classification_positive_negative.pl` | Create balanced positive and negative classification files |
| `04_regression_preprocess.py` | Calculate log10 enrichment targets |
| `05_split_pos_neg_regression.py` | Balance positive and negative regression values |
| `split_train_test.py` | Create a deterministic original-workflow 80/20 split |
| `06_train_test_split.py` | Historical regression split implementation |

The maintained robustness analysis creates isolated train, validation, and test
partitions internally. The processed public inputs are under `data/processed/`.
See [the complete workflow](../../docs/END_TO_END_WORKFLOW.md).

