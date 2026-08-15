#!/usr/bin/env python3
"""Create the manuscript-described systematic 4:1 regression split.

Rows are first sorted by the regression target (log10 ratio) in descending
order. Consecutive blocks of five are then formed. One row from every block is
assigned to the independent test set using the original fixed within-block
seed (1), and the remaining rows form the development/training set. A final
incomplete block contributes one test row, matching the historical workflow.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="regression_dataset.txt")
    parser.add_argument("--development-output", default="train_set.txt")
    parser.add_argument("--test-output", default="test_set.txt")
    parser.add_argument(
        "--alias-development-output",
        default="",
        help="Optional additional copy, e.g. regression_training",
    )
    parser.add_argument(
        "--alias-test-output",
        default="",
        help="Optional additional copy, e.g. regression_indep",
    )
    parser.add_argument("--assignments", default="regression_systematic_split_assignments.tsv")
    parser.add_argument("--manifest", default="regression_80_20_split_manifest.json")
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--within-block-seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_size != 5:
        raise ValueError("The manuscript-described 4:1 split requires --block-size 5")

    data = pd.read_csv(args.input, sep="\t")
    if not {"sequence", "value"}.issubset(data.columns):
        raise ValueError("Input must contain sequence and value columns")
    data = data[["sequence", "value"]].copy()
    data["sequence"] = data["sequence"].astype(str).str.strip().str.upper()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if data.empty:
        raise ValueError("Regression dataset is empty")
    if data["sequence"].duplicated().any():
        raise ValueError("Exact duplicate sequences must be resolved before splitting")

    # Stable sorting makes ties deterministic while preserving their input order.
    ordered = data.sort_values("value", ascending=False, kind="mergesort").reset_index(drop=True)
    ordered["sorted_rank"] = range(1, len(ordered) + 1)
    ordered["block_id"] = (ordered.index // args.block_size) + 1

    development_parts = []
    test_parts = []
    assignment_parts = []
    for _, block in ordered.groupby("block_id", sort=True):
        if len(block) == 1:
            block_development = block.iloc[0:0].copy()
            block_test = block.copy()
        else:
            # This matches train_test_split(..., test_size=1,
            # random_state=1) from the historical script. Resetting the fixed
            # RNG in every full five-row block selects within-block index 2.
            selected_position = int(np.random.RandomState(args.within_block_seed).permutation(len(block))[0])
            selected_index = block.index[selected_position]
            block_test = block.loc[[selected_index]].copy()
            block_development = block.drop(index=selected_index).copy()
        block_development = block_development.copy()
        block_test = block_test.copy()
        block_development["partition"] = "train_set"
        block_test["partition"] = "test_set"
        development_parts.append(block_development)
        test_parts.append(block_test)
        assignment_parts.extend([block_development, block_test])

    development_audit = pd.concat(development_parts, ignore_index=True)
    test_audit = pd.concat(test_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True).sort_values("sorted_rank")
    development = development_audit[["sequence", "value"]].reset_index(drop=True)
    test = test_audit[["sequence", "value"]].reset_index(drop=True)

    overlap = set(development["sequence"]).intersection(test["sequence"])
    if overlap:
        raise AssertionError("Regression training/independent-test sequence overlap detected")
    if len(development) + len(test) != len(data):
        raise AssertionError("Systematic split did not assign every input row exactly once")

    development.to_csv(args.development_output, sep="\t", index=False)
    test.to_csv(args.test_output, sep="\t", index=False)
    assignments.to_csv(args.assignments, sep="\t", index=False)

    if args.alias_development_output:
        development.to_csv(args.alias_development_output, sep="\t", index=False)
    if args.alias_test_output:
        test.to_csv(args.alias_test_output, sep="\t", index=False)

    block_counts = assignments.groupby("block_id")["partition"].value_counts().unstack(fill_value=0)
    full_blocks = int((block_counts.sum(axis=1) == args.block_size).sum())
    manifest = {
        "design": "log(ratio)-ordered systematic 4:1 split",
        "ordering": "value (log10 ratio) descending; stable ordering for ties",
        "block_size": args.block_size,
        "sampling": "one independent-test sequence per consecutive block",
        "within_block_selection": "one row selected by the historical fixed-seed within-block permutation",
        "within_block_seed": args.within_block_seed,
        "full_blocks": full_blocks,
        "final_incomplete_block_size": int(len(data) % args.block_size),
        "total_rows": int(len(data)),
        "development_rows": int(len(development)),
        "independent_test_rows": int(len(test)),
        "development_fraction": float(len(development) / len(data)),
        "test_fraction": float(len(test) / len(data)),
        "exact_sequence_overlap": 0,
        "test_used_for_training_validation_scaling_calibration_or_model_selection": False,
        "development_output": str(Path(args.development_output)),
        "independent_test_output": str(Path(args.test_output)),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
