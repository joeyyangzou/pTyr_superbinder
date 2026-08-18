#!/usr/bin/env python3
"""Create a deterministic random 80/20 train/test split."""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stratify-column",
        default=None,
        help="Column used for stratification, for example label; omit for regression",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.input_file, sep="\t")
    strata = frame[args.stratify_column] if args.stratify_column else None
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=strata,
    )
    train.to_csv(args.train_file, sep="\t", index=False)
    test.to_csv(args.test_file, sep="\t", index=False)
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
