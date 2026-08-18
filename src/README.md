# Source-code map

The source code follows the order of the ANCHOR workflow.

| Step | Directory | Purpose |
|---|---|---|
| 1 | `ngs_preprocessing/` | Convert paired-end reads into denoised peptide copy counts |
| 2 | `dataset_preparation/` | Construct classification labels and regression targets |
| 3 | `model_training/` | Train and evaluate the classification and regression CNNs |
| 4 | `prediction/` | Apply the trained models to new sequence files |

Start with [`../docs/END_TO_END_WORKFLOW.md`](../docs/END_TO_END_WORKFLOW.md)
for the commands linking these four stages.
