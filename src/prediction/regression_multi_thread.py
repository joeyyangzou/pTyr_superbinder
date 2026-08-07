#!/usr/bin/env python3
"""Batch regression scoring of eight-residue SH2 variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


AMINO_ACIDS = "ILVFMCAGPTSYWQNHEDKR"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPOSITORY_ROOT / "models/latest_models/regression/saved_model"
DEFAULT_SCALER = REPOSITORY_ROOT / "models/latest_models/regression/target_scaler.json"


def configure_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="One eight-residue sequence per line; extra tab-separated columns are ignored")
    parser.add_argument("output_file", help="Output TSV containing sequence and regression score")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="TensorFlow SavedModel directory")
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER), help="Target-scaling JSON")
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


def encode_sequences(sequences):
    encoded = np.zeros((len(sequences), 8, len(AMINO_ACIDS)), dtype=np.float32)
    amino_acid_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    for row, sequence in enumerate(sequences):
        if len(sequence) != 8:
            raise ValueError(f"Sequence must contain exactly 8 residues: {sequence!r}")
        for position, amino_acid in enumerate(sequence):
            if amino_acid not in amino_acid_index:
                raise ValueError(f"Unsupported amino acid {amino_acid!r} in {sequence!r}")
            encoded[row, position, amino_acid_index[amino_acid]] = 1.0
    return encoded


def inverse_target_scale(normalized_values, scaler_path):
    with Path(scaler_path).open("r", encoding="utf-8") as handle:
        scaler = json.load(handle)
    data_min = scaler["minmax_data_min"]
    data_max = scaler["minmax_data_max"]
    standardized = ((normalized_values + 1.0) / 2.0) * (data_max - data_min) + data_min
    return standardized * scaler["standard_scaler_scale"] + scaler["standard_scaler_mean"]


def read_sequences(handle):
    for line in handle:
        sequence = line.strip().split("\t", 1)[0].strip()
        if sequence and sequence.lower() != "sequence":
            yield sequence


def write_batch(model, sequences, output_handle, args):
    normalized = np.asarray(
        model.predict(encode_sequences(sequences), batch_size=args.batch_size, verbose=0)
    ).reshape(-1)
    scores = inverse_target_scale(normalized, args.scaler)
    for sequence, score in zip(sequences, scores):
        output_handle.write(f"{sequence}\t{score:.6f}\n")


def main():
    args = parse_args()
    configure_gpu()
    model = tf.keras.models.load_model(args.model)
    with open(args.input_file, "r", encoding="utf-8") as input_handle, open(
        args.output_file, "w", encoding="utf-8"
    ) as output_handle:
        batch = []
        for sequence in read_sequences(input_handle):
            batch.append(sequence)
            if len(batch) >= args.batch_size:
                write_batch(model, batch, output_handle, args)
                batch = []
        if batch:
            write_batch(model, batch, output_handle, args)


if __name__ == "__main__":
    main()
