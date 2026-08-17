# Data

`processed/` contains the public sequence-level model inputs. The fixed
80:20 split manifests and fold assignments generated for model evaluation are
archived under `../results/holdout_10fold_analysis/`. Schemas, row counts,
split construction, and omitted large derived files are documented in
[`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).

The processed datasets are sufficient to repeat all reported machine-learning
metrics without access to raw FASTQ files.
