#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 input.final.fq output.final.fa" >&2
  exit 1
fi

input_file="$1"
output_file="$2"

# Convert FASTQ records to FASTA and retain reads matching both fixed DNA
# template boundaries. Adjust these two motifs only if the library design has
# changed.
awk '
  NR % 4 == 1 {
    header = $0
    sub(/^@/, ">", header)
  }
  NR % 4 == 2 && /^CAAAACAAGAA/ && /TGCCCTGAC$/ {
    print header
    print $0
  }
' "${input_file}" > "${output_file}"
