# Data

`processed/` contains the public sequence-level model inputs.
`frozen_splits/` contains the exact random and Hamming-distance-separated
partitions used in the robustness analysis. Schemas, row counts, split
construction, and omitted large derived files are documented in
[`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).

The processed datasets are sufficient to repeat all reported machine-learning
metrics without access to raw FASTQ files.
