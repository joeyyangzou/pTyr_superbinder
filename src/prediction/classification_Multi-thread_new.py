#!/usr/bin/env python3
"""Batch classification of eight-residue SH2 variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


AMINO_ACIDS = "ILVFMCAGPTSYWQNHEDKR"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPOSITORY_ROOT / "models/latest_models/classification/saved_model"
DEFAULT_CALIBRATION = REPOSITORY_ROOT / "models/latest_models/classification/calibration.json"


def configure_gpu():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(exc)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="One eight-residue sequence per line; extra tab-separated columns are ignored")
    parser.add_argument("output_file", help="Output TSV containing sequence and classification probability")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="TensorFlow SavedModel directory")
    parser.add_argument("--threshold", type=float, default=0.99, help="Minimum probability written to the output")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Apply the validation-fitted Platt calibration before thresholding",
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="Platt calibration JSON")
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


def platt_calibrate(probabilities, calibration_path):
    with Path(calibration_path).open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated_logits = parameters["coefficient"] * logits + parameters["intercept"]
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def read_sequences(handle):
    for line in handle:
        sequence = line.strip().split("\t", 1)[0].strip()
        if sequence and sequence.lower() != "sequence":
            yield sequence


def write_batch(model, sequences, output_handle, args):
    probabilities = np.asarray(
        model.predict(encode_sequences(sequences), batch_size=args.batch_size, verbose=0)
    ).reshape(-1)
    if args.calibrated:
        probabilities = platt_calibrate(probabilities, args.calibration)
    for sequence, probability in zip(sequences, probabilities):
        if probability >= args.threshold:
            output_handle.write(f"{sequence}\t{probability:.6f}\n")


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
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
