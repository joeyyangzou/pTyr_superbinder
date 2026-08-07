# NGS preprocessing

These programs convert paired-end sequencing reads into denoised peptide
copy-count tables.

| Program | Purpose |
|---|---|
| External FLASH | Merge paired-end FASTQ reads |
| `2.rmadpator.pl` | Trim reads to the outer DNA motifs |
| `3.extract_forward_revserse.pl` | Separate exact forward/reverse templates and discard nonmatching reads |
| `4.reverse_fastq.pl` | Reverse-complement reverse-orientation FASTQ reads |
| `5.fastq2fasta.sh` | Convert FASTQ to template-filtered FASTA |
| `6.dna2pep.pl` | Translate DNA sequences to protein |
| `stat_uniq_pep_num.pl` | Count unique peptides within one sample |
| `stat_uniq_seq_num.pl` | Build a combined multi-sample copy-count table |
| `join.pl` | Join sequence counts to the reference table |
| `sequence_denoise.py` | Apply stop-code and low-support filters |
| `stat_average_copy.py` | Summarize sequence types and copy totals |

See [the complete workflow](../../docs/END_TO_END_WORKFLOW.md) for exact
commands, motifs, expected filenames, and output interpretation.

