#!/usr/bin/env python3
"""Create a fixed-test Hamming-buffer classification sensitivity split.

The independent test set is read from the frozen primary 80:20 analysis and
is never resampled.  Every non-test sequence within Hamming distance 0 or 1
of any test sequence is excluded from the development pool.  The retained
development set can then be supplied to 03_CNN_classification.py, which keeps
the test set out of cross-validation, calibration, epoch selection and early
stopping.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-file",
        default="data/processed/classification/positive.tsv",
    )
    parser.add_argument(
        "--negative-file",
        default="data/processed/classification/negative.tsv",
    )
    parser.add_argument(
        "--fixed-test-file",
        default=(
            "results/holdout_10fold_analysis/"
            "classification_80_20_10fold_results/splits/independent_test_20.tsv"
        ),
        help="Frozen test TSV from the primary 80:20 analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hamming_buffer_sensitivity_rerun/buffer_partitions",
    )
    parser.add_argument(
        "--minimum-development-test-hamming",
        type=int,
        default=2,
        help="Retain development candidates at least this far from every test sequence.",
    )
    parser.add_argument("--expected-test-rows", type=int, default=3384)
    parser.add_argument("--expected-test-positive", type=int, default=1692)
    parser.add_argument("--expected-test-negative", type=int, default=1692)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def normalise_partition(frame, require_source_row=False):
    required = {"sequence", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Input classification tables must contain sequence and label columns")
    columns = ["sequence", "label"]
    if "source_row" in frame.columns:
        columns.insert(0, "source_row")
    elif require_source_row:
        raise ValueError("The full dataset must contain source_row")
    result = frame.loc[:, columns].copy()
    result["sequence"] = result["sequence"].astype(str).str.strip().str.upper()
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    if "source_row" in result.columns:
        result["source_row"] = pd.to_numeric(result["source_row"], errors="raise").astype(int)
    return result


def read_full_dataset(positive_file, negative_file):
    positive = normalise_partition(pd.read_csv(positive_file, sep="\t"))
    negative = normalise_partition(pd.read_csv(negative_file, sep="\t"))
    data = pd.concat([positive, negative], ignore_index=True)
    data.insert(0, "source_row", np.arange(len(data), dtype=int))
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("The full classification dataset must contain labels 0 and 1")
    duplicated = data["sequence"].duplicated(False)
    if duplicated.any():
        examples = data.loc[duplicated, "sequence"].drop_duplicates().head(5).tolist()
        raise ValueError("Duplicate sequences must be resolved first: " + ", ".join(examples))
    lengths = data["sequence"].str.len().unique()
    if len(lengths) != 1:
        raise ValueError("All sequences must have the same length for Hamming distance")
    return data


def encode_sequences(sequences, alphabet):
    return np.asarray(
        [[alphabet[amino_acid] for amino_acid in sequence] for sequence in sequences],
        dtype=np.int16,
    )


def minimum_hamming_distances(reference_sequences, query_sequences, chunk_size):
    reference = [str(value) for value in reference_sequences]
    query = [str(value) for value in query_sequences]
    if not reference or not query:
        raise ValueError("Both reference and query sequence collections are required")
    observed_lengths = {len(sequence) for sequence in reference + query}
    if len(observed_lengths) != 1:
        raise ValueError("All sequences must have equal length")
    alphabet = {
        amino_acid: index
        for index, amino_acid in enumerate(sorted(set("".join(reference + query))))
    }
    reference_array = encode_sequences(reference, alphabet)
    minimum = np.empty(len(query), dtype=np.int16)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        query_array = encode_sequences(query[start:stop], alphabet)
        distances = np.count_nonzero(
            query_array[:, None, :] != reference_array[None, :, :], axis=2
        )
        minimum[start:stop] = distances.min(axis=1)
    return minimum.astype(int)


def partition_sha256(frame):
    canonical = frame.loc[:, ["sequence", "label"]].copy()
    canonical["label"] = canonical["label"].astype(int)
    canonical = canonical.sort_values(["sequence", "label"], kind="mergesort")
    payload = "".join(
        "%s\t%d\n" % (row.sequence, int(row.label))
        for row in canonical.itertuples(index=False)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def class_counts(frame):
    counts = frame["label"].value_counts().to_dict()
    return {"negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0))}


def main():
    args = parse_args()
    if args.minimum_development_test_hamming < 1:
        raise ValueError("The minimum Hamming distance must be at least 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full = read_full_dataset(args.positive_file, args.negative_file)
    supplied_test = normalise_partition(pd.read_csv(args.fixed_test_file, sep="\t"))
    if supplied_test["sequence"].duplicated().any():
        raise ValueError("The frozen test file contains duplicate sequences")

    full_by_sequence = full.set_index("sequence", drop=False)
    missing = sorted(set(supplied_test["sequence"]) - set(full["sequence"]))
    if missing:
        raise ValueError("Frozen test sequences are absent from the full dataset: " + ", ".join(missing[:5]))
    expected_labels = full_by_sequence.loc[supplied_test["sequence"], "label"].to_numpy(dtype=int)
    if not np.array_equal(expected_labels, supplied_test["label"].to_numpy(dtype=int)):
        raise ValueError("One or more frozen test labels do not match the full dataset")

    # Recover the canonical source_row while preserving the exact frozen-test order.
    test = full_by_sequence.loc[supplied_test["sequence"]].reset_index(drop=True)
    if len(test) != args.expected_test_rows:
        raise ValueError(
            "Frozen test row count is %d; expected %d"
            % (len(test), args.expected_test_rows)
        )
    observed_test_counts = class_counts(test)
    if observed_test_counts["positive"] != args.expected_test_positive:
        raise ValueError("Unexpected positive count in the frozen test set")
    if observed_test_counts["negative"] != args.expected_test_negative:
        raise ValueError("Unexpected negative count in the frozen test set")

    test_sequences = set(test["sequence"])
    candidates = full.loc[~full["sequence"].isin(test_sequences)].copy().reset_index(drop=True)
    distance_to_test = minimum_hamming_distances(
        test["sequence"], candidates["sequence"], args.chunk_size
    )
    candidates["nearest_test_hamming_distance"] = distance_to_test
    keep = distance_to_test >= args.minimum_development_test_hamming
    development = candidates.loc[keep, ["source_row", "sequence", "label"]].reset_index(drop=True)
    excluded = candidates.loc[~keep].copy().reset_index(drop=True)
    excluded["exclusion_reason"] = "within_test_hamming_buffer"

    if development.empty:
        raise ValueError("Hamming buffering removed every development candidate")
    if set(development["label"].unique()) != {0, 1}:
        raise ValueError("The retained development set must contain both classes")

    test_to_development = minimum_hamming_distances(
        development["sequence"], test["sequence"], args.chunk_size
    )
    minimum_observed = int(test_to_development.min())
    if minimum_observed < args.minimum_development_test_hamming:
        raise AssertionError("Development-test Hamming separation failed")
    if set(development["source_row"]) & set(test["source_row"]):
        raise AssertionError("Development and test source rows overlap")

    nearest_counts = Counter(int(value) for value in test_to_development)
    candidate_distance_counts = Counter(int(value) for value in distance_to_test)
    manifest = {
        "design": (
            "Frozen primary 20% independent test plus a Hamming-buffered development pool; "
            "all non-test sequences within the requested Hamming distance were excluded"
        ),
        "split_mode": "fixed_test_hamming_buffer",
        "fixed_test_source": Path(args.fixed_test_file).as_posix(),
        "fixed_test_sha256_sequence_label": partition_sha256(test),
        "source_total_rows": int(len(full)),
        "development_rows": int(len(development)),
        "independent_test_rows": int(len(test)),
        "excluded_rows": int(len(excluded)),
        "minimum_requested_development_test_hamming": int(
            args.minimum_development_test_hamming
        ),
        "class_counts": {
            "full": class_counts(full),
            "development": class_counts(development),
            "independent_test": class_counts(test),
            "excluded": class_counts(excluded),
        },
        "hamming_audit": {
            "minimum_development_test_hamming": minimum_observed,
            "test_sequences_with_development_neighbor_hamming_le_1": int(
                np.sum(test_to_development <= 1)
            ),
            "nearest_development_hamming_counts_for_test": {
                str(key): int(value) for key, value in sorted(nearest_counts.items())
            },
            "candidate_nearest_test_hamming_counts_before_filtering": {
                str(key): int(value)
                for key, value in sorted(candidate_distance_counts.items())
            },
            "exact_sequence_overlap": 0,
        },
        "test_used_for_early_stopping": False,
        "test_used_for_calibration": False,
        "test_resampled": False,
    }

    development.to_csv(output_dir / "development_hamming_buffer.tsv", sep="\t", index=False)
    test.to_csv(output_dir / "independent_test_fixed.tsv", sep="\t", index=False)
    excluded.to_csv(output_dir / "excluded_within_hd1_of_test.tsv", sep="\t", index=False)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    summary = pd.DataFrame(
        [
            {
                "partition": "development_hamming_buffer",
                "n": len(development),
                "positive": class_counts(development)["positive"],
                "negative": class_counts(development)["negative"],
            },
            {
                "partition": "independent_test_fixed",
                "n": len(test),
                "positive": observed_test_counts["positive"],
                "negative": observed_test_counts["negative"],
            },
            {
                "partition": "excluded_within_buffer",
                "n": len(excluded),
                "positive": class_counts(excluded)["positive"],
                "negative": class_counts(excluded)["negative"],
            },
        ]
    )
    summary.to_csv(output_dir / "split_summary.tsv", sep="\t", index=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
