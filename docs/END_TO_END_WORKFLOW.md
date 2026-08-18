# Complete ANCHOR workflow

This document connects raw paired-end NGS reads to sequence/count tables,
classification and regression datasets, CNN analysis, and final prediction.
Commands are examples and should be run in a sample-specific working directory,
not inside the source tree.

```bash
REPO_ROOT=/absolute/path/to/ANCHOR
WORK_DIR=/absolute/path/to/anchor_run
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
```

File prefixes such as `pYS2`, `PYS2_CN`, and `PYS4_CN` are sample-specific.
Keep a sample sheet that records the biological round represented by each
column. Commands below use the public filenames under `src/`.

## A. Paired-end read processing and peptide counting

### 1. Merge paired-end reads

Run FLASH with its default overlap settings and eight threads. The `-o`
prefix determines the output filename.

```bash
flash -o pYS2 -t 8 SH2_R1.fastq SH2_R2.fastq
```

Primary output: `pYS2.extendedFrags.fastq`.

### 2. Trim reads to the outer DNA motifs

Retain the matched motifs and everything between them:

- forward: `CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA`
- reverse: `TGTTTCAAACTGGGC.*GTACCACTCTTCTGCCTG`

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/2.rmadpator.pl" \
  pYS2.extendedFrags.fastq \
  pYS2.extendedFrags.rmadaptor.fastq
```

### 3. Separate forward and reverse reads

Only reads matching one of the two complete templates are retained. Reads with
one or more mismatches are discarded by the supplied strict script.

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/3.extract_forward_revserse.pl" \
  pYS2.extendedFrags.rmadaptor.fastq \
  pYS2.extendedFrags.rmadaptor.forward.fastq \
  pYS2.extendedFrags.rmadaptor.reverse.fastq
```

The current analysis uses exact matching. A mismatch-tolerant rule permitting
up to two mismatches would define a different preprocessing setting and must be
recorded explicitly if used.

### 4. Reverse-complement reverse reads and merge orientations

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/4.reverse_fastq.pl" \
  pYS2.extendedFrags.rmadaptor.reverse.fastq \
  pYS2.extendedFrags.rmadaptor.reverse.re.fastq

cat pYS2.extendedFrags.rmadaptor.forward.fastq \
    pYS2.extendedFrags.rmadaptor.reverse.re.fastq \
    > pYS2.final.fq
```

### 5. Convert FASTQ to FASTA and translate DNA

The helper retains DNA reads beginning with `CAAAACAAGAA` and ending with
`TGCCCTGAC`.

```bash
bash "${REPO_ROOT}/src/ngs_preprocessing/5.fastq2fasta.sh" \
  pYS2.final.fq pYS2.final.fa

perl "${REPO_ROOT}/src/ngs_preprocessing/6.dna2pep.pl" \
  pYS2.final.fa pYS2.final.fa.pep
```

Translation uses `TAA -> _`, `TAG -> q`, `TGA -> w`, and codons containing
an undetermined base `N -> X`.

### 6. Filter translated reads against the protein template

```bash
grep "^NKKKVEEVLEEEE.*EKVLDRRVVKGKVEYLLK.*PEENLDCP" \
  pYS2.final.fa.pep > pYS2.final.fa.pep.filter
```

If a construct version contains an additional leading `Q`, use
`^QNKKKVEEVLEEEE` consistently and record that template version.

### 7. Remove fixed protein regions

The two variable regions are temporarily joined by a hyphen.

```bash
perl -pe 's/^NKKKVEEVLEEEE|-EKVLDRRVVKGKVEYLLK|PEENLDCP$//g; \
s/EKVLDRRVVKGKVEYLLK/-/g' \
  pYS2.final.fa.pep.filter > pYS2.final.fa.pep.filter.short
```

### 8. Count unique peptides in each sample

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/stat_uniq_pep_num.pl" \
  pYS2.final.fa.pep.filter.short \
  pYS2.final.fa.pep.filter.short.stat.tsv
```

The output columns are sequence and copy count.

### 9. Build a combined multi-sample count table

List the `*.short` files in the intended biological-round order. The script
outputs the sequence, one copy-count column per sample, sequence length, and
total copies.

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/stat_uniq_seq_num.pl" \
  R0.short R1.short R2.short R3.short R4.short > stat.txt
```

### 10. Join the count table to the reference table

Place `stat.txt` and the sequence reference `ref-SH2.txt` in the working
directory, then run:

```bash
perl "${REPO_ROOT}/src/ngs_preprocessing/join.pl"
```

Output: `stat.ref.txt`.

### 11. Apply sequence denoising rules

```bash
python "${REPO_ROOT}/src/ngs_preprocessing/sequence_denoise.py" \
  > stat.ref.denoise.txt
```

The supplied rule removes peptides containing stop-code symbols and removes
low-support rows with zeros unless at least one round has more than three
copies.

### 12. Apply the copy-number threshold

```bash
awk '$4 > 10 || $6 > 10' stat.ref.denoise.txt \
  > stat.ref.denoise.10.txt
```

Column positions are sample-sheet dependent. Confirm that columns 4 and 6 are
the intended rounds before running this command.

### 13. Summarize average copy support

```bash
python "${REPO_ROOT}/src/ngs_preprocessing/stat_average_copy.py"
```

## B. Classification and regression dataset construction

The dataset scripts expect a tabular file named `merged_sequences_all.txt`
with a header containing `PYS2_CN` and `PYS4_CN`. Copy or rename the filtered
count table only after confirming those column names.

```bash
cp stat.ref.denoise.10.txt merged_sequences_all.txt
```

### 14. Compare round 2 and round 4 for classification labels

```bash
python "${REPO_ROOT}/src/dataset_preparation/01_calculate_classification.py"
```

Output: `stat.ref.denoise.classification1.txt`.

### 15. Create balanced positive and negative sets

```bash
perl "${REPO_ROOT}/src/dataset_preparation/02_classification_positive_negative.pl"
```

The script retains all positive rows and selects an equal number of negative
rows. Inspect the printed counts before continuing.

### 16. Verify classification headers

Both files must use:

```text
sequence<TAB>label
```

The supplied balancing script writes this header automatically.

### 17. Calculate regression targets

```bash
python "${REPO_ROOT}/src/dataset_preparation/04_regression_preprocess.py"
```

Output: `regression_input.txt`, with `sequence` and log10 enrichment
`value`.

### 18. Balance positive and negative regression values

```bash
python "${REPO_ROOT}/src/dataset_preparation/05_split_pos_neg_regression.py"
```

Output: `regression_dataset.txt`.

### 19. Combine the classification files without duplicate headers

```bash
{
  head -n 1 positive
  tail -n +2 positive
  tail -n +2 negative
} > classification_dataset.tsv
```

### 20. Create original-workflow train/test files

```bash
python "${REPO_ROOT}/src/dataset_preparation/split_train_test.py" \
  classification_dataset.tsv classification_train.tsv classification_test.tsv \
  --stratify-column label --seed 1

python "${REPO_ROOT}/src/dataset_preparation/split_train_test.py" \
  regression_dataset.txt regression_train.tsv regression_test.tsv --seed 1
```

These files support the original single-model workflow. The current robustness
analysis constructs separate train, validation, and untouched test partitions
internally and does not use this helper split.

### 21. Optional direct extraction of a regression table

When column 4 of the filtered table is already the intended continuous target:

```bash
cut -f 1,4 stat.ref.denoise.10.regression.txt > regression_direct.tsv
sed -i '1c sequence\tvalue' regression_direct.tsv
```

Confirm the column identity and header before model fitting.

## C. Model analysis

Copy the finalized model inputs to the public data layout:

```bash
cp positive "${REPO_ROOT}/data/processed/classification/positive.tsv"
cp negative "${REPO_ROOT}/data/processed/classification/negative.tsv"
cp regression_dataset.txt \
  "${REPO_ROOT}/data/processed/regression/regression_dataset.tsv"
```

The maintained 80:20/10-fold programs are
`src/model_training/03_CNN_classification.py` and
`src/model_training/07_CNN_regression.py`. They freeze the independent 20%
test set first, run 10-fold cross-validation only within the 80% development
set, and use inner validation data for early stopping. Run them together with:

```bash
cd "${REPO_ROOT}"
bash run_80_20_10fold_analysis.sh
```

The public inference artifacts remain one classifier and one regressor under
`models/latest_models/`. Intermediate seed weights are not distributed.

## D. Full-sequence prediction

The prediction programs default to the two SavedModels under
`models/latest_models/`. The regression model additionally uses the supplied
target-scaling metadata.

### 22. Classification screening

`output_T.txt` must contain one eight-residue sequence per line.

```bash
cd "${REPO_ROOT}"
conda activate anchor-tf24
nohup python src/prediction/classification_Multi-thread_new.py \
  output_T.txt output_T_predict.txt \
  --threshold 0.99 --batch_size 10240 \
  > classification_prediction.log 2>&1 &
```

The threshold is applied to the raw sigmoid probability, matching the
historical prediction scale. No post-hoc calibration file is distributed for
this inference model.

### 23. Extract classification-passing sequences

```bash
cut -f 1 output_T_predict.txt > pass_classification99_seq
```

### 24. Regression scoring

```bash
python src/prediction/regression_multi_thread.py \
  pass_classification99_seq \
  pass_classification99_seq_regression_score \
  --batch_size 10240
```

The regression program uses
`models/latest_models/regression/target_scaler.json` to convert the model's
tanh-scaled output back to the original enrichment scale.

## Output traceability

For every run, retain the sample sheet, raw filenames, FLASH version, commands,
template motifs, mismatch policy, threshold values, environment specification,
Git commit identifier, and output checksums. Do not overwrite the supplied
reference metrics under `results/holdout_10fold_analysis/`.
