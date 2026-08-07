#!/usr/bin/env python3
"""Robustness and repeated-run analyses for the SH2 peptide CNNs.

This script intentionally leaves the original training scripts unchanged. It
uses the same CNN architectures while correcting test-set leakage and adding:

* disjoint train/validation/test sets;
* random and Hamming-buffered evaluation;
* ten independent training seeds by default;
* AUROC/AUPRC/F1/MCC and regression metrics;
* validation-only Platt calibration, reliability plots, Brier score and ECE;
* stratified/non-parametric bootstrap 95% confidence intervals; and
* optional Monte Carlo dropout uncertainty estimates.
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from robustness_utils import (
    bootstrap_confidence_intervals,
    classification_metrics,
    make_robustness_split,
    one_hot_encode,
    regression_metrics,
    save_json,
    set_global_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "classification", "regression"], default="all")
    parser.add_argument("--positive-file", default="positive")
    parser.add_argument("--negative-file", default="negative")
    parser.add_argument("--regression-file", default="regression_dataset.txt")
    parser.add_argument("--output-dir", default="robustness_results")
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["random", "hamming", "cluster"],
        default=["random", "hamming"],
        help="Run conventional and stringent evaluations. Cluster is optional because the library contains a giant connected component.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--minimum-test-train-hamming", type=int, default=2)
    parser.add_argument("--classification-epochs", type=int, default=200)
    parser.add_argument("--regression-epochs", type=int, default=1000)
    parser.add_argument("--classification-patience", type=int, default=20)
    parser.add_argument("--regression-patience", type=int, default=50)
    parser.add_argument("--classification-batch-size", type=int, default=64)
    parser.add_argument("--regression-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="Optional supplementary MC-dropout draws per seed; deep-ensemble SD is the primary uncertainty.",
    )
    parser.add_argument("--calibration", choices=["platt", "none"], default="platt")
    parser.add_argument(
        "--single-model-split",
        choices=["random", "hamming", "cluster"],
        default="random",
        help="Split whose validation-selected seed is exported as one deployable model per task.",
    )
    parser.add_argument(
        "--skip-single-model-export",
        action="store_true",
        help="Do not export a validation-selected single SavedModel after the repeated-run analysis.",
    )
    parser.add_argument(
        "--export-single-models-only",
        action="store_true",
        help="Export single SavedModels from existing seed results without retraining.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)
    return parser.parse_args()


def read_classification(positive_file, negative_file):
    positive = pd.read_csv(positive_file, sep="\t")
    negative = pd.read_csv(negative_file, sep="\t")
    data = pd.concat([positive, negative], ignore_index=True)
    if not {"sequence", "label"}.issubset(data.columns):
        raise ValueError("Classification files must contain sequence and label columns")
    data = data[["sequence", "label"]].copy()
    data["label"] = data["label"].astype(int)
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Classification labels must contain both 0 and 1")
    return data


def read_regression(regression_file):
    data = pd.read_csv(regression_file, sep="\t")
    if not {"sequence", "value"}.issubset(data.columns):
        raise ValueError("Regression file must contain sequence and value columns")
    data = data[["sequence", "value"]].copy()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    return data


def configure_tensorflow():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def build_classification_model(tf, sequence_length):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(sequence_length, 20)),
            layers.Conv1D(128, 1, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 9, padding="same", activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dropout(0.5),
            layers.Conv1D(128, 10, padding="same", activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dropout(0.7),
            layers.Dense(64, activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dense(32, activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dense(8, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auroc"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
        ],
    )
    return model


def build_regression_model(tf, sequence_length):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(sequence_length, 20)),
            layers.Conv1D(128, 1, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 9, padding="same", activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dropout(0.5),
            layers.Conv1D(128, 10, padding="same", activation="relu"),
            layers.MaxPooling1D(2, 1),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(8, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(1, activation="tanh"),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def callbacks_for(tf, output_dir, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def select_seed_by_validation_loss(split_dir, seeds):
    """Select one already-trained seed without consulting the test set."""
    candidates = []
    for seed in seeds:
        history_path = split_dir / f"seed_{seed}" / "training_history.csv"
        weights_path = split_dir / f"seed_{seed}" / "best.weights.h5"
        if not history_path.exists() or not weights_path.exists():
            continue
        history = pd.read_csv(history_path)
        if "val_loss" not in history.columns or history.empty:
            continue
        best_row = int(history["val_loss"].astype(float).idxmin())
        candidates.append(
            {
                "seed": int(seed),
                "best_epoch": int(history.loc[best_row, "epoch"]),
                "minimum_validation_loss": float(history.loc[best_row, "val_loss"]),
                "weights_path": weights_path,
            }
        )
    if not candidates:
        raise FileNotFoundError(
            f"No complete seed results were found under {split_dir}. "
            "Run the repeated training analysis first."
        )
    return min(candidates, key=lambda item: (item["minimum_validation_loss"], item["seed"]))


def export_single_model(tf, task, split_dir, output_root, seeds):
    """Export one validation-selected SavedModel for old-style downstream inference.

    The ten models remain the basis of repeated-run and ensemble reporting. This
    export merely provides one convenient model artifact and never uses test-set
    performance to select the seed.
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "splits" / "train.tsv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing frozen training split: {train_path}")
    train = pd.read_csv(train_path, sep="\t")
    if "sequence" not in train.columns or train.empty:
        raise ValueError(f"Invalid frozen training split: {train_path}")
    sequence_lengths = train["sequence"].astype(str).str.len().unique()
    if len(sequence_lengths) != 1:
        raise ValueError(f"Expected one fixed sequence length; observed {sequence_lengths.tolist()}")
    sequence_length = int(sequence_lengths[0])

    selected = select_seed_by_validation_loss(split_dir, seeds)
    tf.keras.backend.clear_session()
    if task == "classification":
        model = build_classification_model(tf, sequence_length)
    elif task == "regression":
        model = build_regression_model(tf, sequence_length)
    else:
        raise ValueError("task must be classification or regression")
    model.load_weights(str(selected["weights_path"]))

    export_dir = Path(output_root) / "single_models" / task
    export_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = export_dir / "saved_model"
    model.save(str(saved_model_dir), save_format="tf", include_optimizer=False)

    manifest = {
        "task": task,
        "source_split": split_dir.name,
        "selection_rule": "minimum validation loss across requested training seeds; test metrics were not used",
        "selected_seed": selected["seed"],
        "best_epoch": selected["best_epoch"],
        "minimum_validation_loss": selected["minimum_validation_loss"],
        "sequence_length": sequence_length,
        "amino_acid_order": "ILVFMCAGPTSYWQNHEDKR",
        "saved_model": str(saved_model_dir),
        "prediction_scope": (
            "raw sigmoid probability; apply calibration.json only when calibrated probabilities are required"
            if task == "classification"
            else "tanh-scaled target; use target_scaler.json to convert predictions back to original target units"
        ),
        "reporting_note": "Repeated-run mean +/- SD and uncertainty use all ten independently trained models, not this single export.",
    }

    if task == "classification":
        calibration_path = split_dir / f"seed_{selected['seed']}" / "platt_calibration.json"
        if calibration_path.exists():
            with calibration_path.open("r", encoding="utf-8") as handle:
                calibration = json.load(handle)
            save_json(calibration, export_dir / "calibration.json")
            manifest["calibration"] = str(export_dir / "calibration.json")
    else:
        scaler_path = split_dir / "training_only_target_scaler.json"
        if scaler_path.exists():
            with scaler_path.open("r", encoding="utf-8") as handle:
                scaler = json.load(handle)
            save_json(scaler, export_dir / "target_scaler.json")
            manifest["target_scaler"] = str(export_dir / "target_scaler.json")

    save_json(manifest, export_dir / "single_model_manifest.json")
    print(
        f"Exported {task} single model from seed {selected['seed']} "
        f"(validation loss {selected['minimum_validation_loss']:.6g}) to {saved_model_dir}",
        flush=True,
    )
    del model
    return manifest


def export_requested_single_models(tf, args, output_root):
    tasks = [args.task] if args.task != "all" else ["classification", "regression"]
    manifests = {}
    for task in tasks:
        split_dir = output_root / task / args.single_model_split
        manifests[task] = export_single_model(tf, task, split_dir, output_root, args.seeds)
    save_json(
        {
            "source_split": args.single_model_split,
            "tasks": manifests,
            "purpose": "Convenient single-model artifacts for downstream inference; repeated-run statistics remain ensemble based.",
        },
        output_root / "single_models" / "export_summary.json",
    )
    return manifests


def mc_dropout_predict(model, features, samples):
    if samples <= 0:
        return None
    draws = []
    for _ in range(samples):
        draws.append(np.asarray(model(features, training=True)).reshape(-1))
    return np.asarray(draws, dtype=float)


def fit_platt(y_validation, probability_validation):
    probability_validation = np.clip(np.asarray(probability_validation), 1e-6, 1 - 1e-6)
    logits = np.log(probability_validation / (1 - probability_validation)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, np.asarray(y_validation, dtype=int))
    return calibrator


def apply_platt(calibrator, probability):
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def save_split_files(base_dir, train, validation, test, excluded, metadata):
    split_dir = base_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.tsv", sep="\t", index=False)
    validation.to_csv(split_dir / "validation.tsv", sep="\t", index=False)
    test.to_csv(split_dir / "test.tsv", sep="\t", index=False)
    excluded.to_csv(split_dir / "excluded_hamming_buffer.tsv", sep="\t", index=False)
    save_json(metadata, split_dir / "split_metadata.json")


def save_history(history, output_dir):
    frame = pd.DataFrame(history.history)
    frame.index = np.arange(1, len(frame) + 1)
    frame.index.name = "epoch"
    frame.to_csv(output_dir / "training_history.csv")
    best_epoch = int(frame["val_loss"].idxmin())
    return len(frame), best_epoch


def summarize_seed_metrics(metrics_frame, metric_prefix, output_dir):
    columns = [column for column in metrics_frame.columns if column.startswith(metric_prefix)]
    summary = pd.DataFrame(
        {
            "mean": metrics_frame[columns].mean(axis=0),
            "standard_deviation": metrics_frame[columns].std(axis=0, ddof=1),
            "minimum": metrics_frame[columns].min(axis=0),
            "maximum": metrics_frame[columns].max(axis=0),
            "n_independent_runs": len(metrics_frame),
        }
    )
    summary.index.name = "metric"
    summary.to_csv(output_dir / "metrics_mean_sd_across_seeds.csv")


def write_combined_tables(output_root, tasks, split_modes):
    metric_rows = []
    split_rows = []
    for task in tasks:
        prefix = "calibrated_" if task == "classification" else "test_"
        for split_mode in split_modes:
            directory = output_root / task / split_mode
            metrics_file = directory / "ensemble_metrics.json"
            if not metrics_file.exists():
                continue
            point = json.loads(metrics_file.read_text(encoding="utf-8"))
            confidence = pd.read_csv(directory / "bootstrap_95CI.csv").set_index("metric")
            repeated = pd.read_csv(directory / "metrics_mean_sd_across_seeds.csv").set_index("metric")
            for metric, estimate in point.items():
                repeated_key = prefix + metric
                metric_rows.append(
                    {
                        "task": task,
                        "split_mode": split_mode,
                        "metric": metric,
                        "ensemble_estimate": estimate,
                        "bootstrap_ci_2.5%": confidence.loc[metric, "ci_2.5%"],
                        "bootstrap_ci_97.5%": confidence.loc[metric, "ci_97.5%"],
                        "seed_mean": repeated.loc[repeated_key, "mean"],
                        "seed_standard_deviation": repeated.loc[repeated_key, "standard_deviation"],
                        "n_training_seeds": repeated.loc[repeated_key, "n_independent_runs"],
                    }
                )
            metadata = json.loads(
                (directory / "splits" / "split_metadata.json").read_text(encoding="utf-8")
            )
            audit = metadata["homology_audit"]
            pairwise = metadata["pairwise_homology_audits"]
            split_rows.append(
                {
                    "task": task,
                    "split_mode": split_mode,
                    "n_train": metadata["n_train"],
                    "n_validation": metadata["n_validation"],
                    "n_test": metadata["n_test"],
                    "n_excluded_hamming_buffer": metadata["n_excluded_hamming_buffer"],
                    "minimum_train_test_hamming": audit["minimum_nearest_hamming_distance"],
                    "minimum_train_validation_hamming": pairwise["train_vs_validation"][
                        "minimum_nearest_hamming_distance"
                    ],
                    "minimum_validation_test_hamming": pairwise["validation_vs_test"][
                        "minimum_nearest_hamming_distance"
                    ],
                    "test_fraction_with_train_neighbor_hamming_le_1": audit[
                        "fraction_with_train_neighbor_hamming_le_1"
                    ],
                }
            )
    pd.DataFrame(metric_rows).to_csv(output_root / "combined_metrics_for_manuscript.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_root / "combined_split_summary.csv", index=False)


def plot_classification(y_true, raw_probability, calibrated_probability, uncertainty, output_dir):
    fpr, tpr, _ = roc_curve(y_true, calibrated_probability)
    precision, recall, _ = precision_recall_curve(y_true, calibrated_probability)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.axhline(np.mean(y_true), linestyle="--", color="gray")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, probability in [("Raw", raw_probability), ("Platt calibrated", calibrated_probability)]:
        observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_plot.png", dpi=300)
    plt.close(fig)

    if uncertainty is not None:
        error = np.abs(np.asarray(y_true) - calibrated_probability)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(uncertainty, error, s=10, alpha=0.35)
        ax.set(xlabel="Deep-ensemble predictive SD", ylabel="Absolute probability error", title="Uncertainty vs. error")
        fig.tight_layout()
        fig.savefig(output_dir / "uncertainty_vs_error.png", dpi=300)
        plt.close(fig)


def plot_regression(y_true, prediction, uncertainty, output_dir):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, prediction, s=12, alpha=0.45)
    lower = min(np.min(y_true), np.min(prediction))
    upper = max(np.max(y_true), np.max(prediction))
    ax.plot([lower, upper], [lower, upper], "--", color="black")
    ax.set(xlabel="Observed value", ylabel="Predicted value", title="Regression performance")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_predicted.png", dpi=300)
    plt.close(fig)

    residual = np.asarray(y_true) - np.asarray(prediction)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(prediction, residual, s=12, alpha=0.45)
    axes[0].axhline(0.0, linestyle="--", color="black")
    axes[0].set(xlabel="Predicted value", ylabel="Residual (observed - predicted)", title="Residuals vs. predictions")
    axes[1].hist(residual, bins=30, edgecolor="white")
    axes[1].axvline(0.0, linestyle="--", color="black")
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_analysis.png", dpi=300)
    plt.close(fig)

    if uncertainty is not None:
        error = np.abs(np.asarray(y_true) - prediction)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(uncertainty, error, s=10, alpha=0.35)
        ax.set(xlabel="Deep-ensemble predictive SD", ylabel="Absolute error", title="Uncertainty vs. error")
        fig.tight_layout()
        fig.savefig(output_dir / "uncertainty_vs_error.png", dpi=300)
        plt.close(fig)


def train_classification_split(tf, data, split_mode, args, task_root):
    output_dir = task_root / split_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, excluded, metadata = make_robustness_split(
        data,
        target_col="label",
        task="classification",
        split_mode=split_mode,
        test_size=args.test_size,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
        minimum_test_train_hamming=args.minimum_test_train_hamming,
    )
    save_split_files(output_dir, train, validation, test, excluded, metadata)

    x_train = one_hot_encode(train["sequence"])
    x_validation = one_hot_encode(validation["sequence"])
    x_test = one_hot_encode(test["sequence"])
    y_train = train["label"].to_numpy(dtype=np.float32)
    y_validation = validation["label"].to_numpy(dtype=np.float32)
    y_test = test["label"].to_numpy(dtype=int)

    rows = []
    raw_predictions = []
    calibrated_predictions = []
    stochastic_predictions = []
    for seed in args.seeds:
        tf.keras.backend.clear_session()
        set_global_seed(seed, tf)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model = build_classification_model(tf, x_train.shape[1])
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=args.classification_batch_size,
            epochs=args.classification_epochs,
            callbacks=callbacks_for(tf, seed_dir, args.classification_patience),
            shuffle=True,
            verbose=args.verbose,
        )
        epochs_trained, best_epoch = save_history(history, seed_dir)
        probability_validation = model.predict(x_validation, batch_size=args.classification_batch_size, verbose=0).reshape(-1)
        probability_test_raw = model.predict(x_test, batch_size=args.classification_batch_size, verbose=0).reshape(-1)
        calibrator = None
        if args.calibration == "platt":
            calibrator = fit_platt(y_validation, probability_validation)
            probability_test = apply_platt(calibrator, probability_test_raw)
            save_json(
                {"coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
                seed_dir / "platt_calibration.json",
            )
        else:
            probability_test = probability_test_raw

        mc_raw = mc_dropout_predict(model, x_test, args.mc_samples)
        mc_used = None
        if mc_raw is not None:
            mc_used = np.asarray([apply_platt(calibrator, draw) for draw in mc_raw]) if calibrator is not None else mc_raw
            stochastic_predictions.append(mc_used)

        raw_metric = classification_metrics(y_test, probability_test_raw)
        calibrated_metric = classification_metrics(y_test, probability_test)
        row = {"seed": seed, "epochs_trained": epochs_trained, "best_epoch": best_epoch}
        row.update({f"raw_{key}": value for key, value in raw_metric.items()})
        row.update({f"calibrated_{key}": value for key, value in calibrated_metric.items()})
        rows.append(row)

        predictions = test[["source_row", "sequence", "label"]].copy()
        predictions["probability_raw"] = probability_test_raw
        predictions["probability_calibrated"] = probability_test
        if mc_used is not None:
            predictions["mc_probability_mean"] = mc_used.mean(axis=0)
            predictions["mc_probability_sd"] = mc_used.std(axis=0, ddof=1)
        predictions.to_csv(seed_dir / "test_predictions.tsv", sep="\t", index=False)
        raw_predictions.append(probability_test_raw)
        calibrated_predictions.append(probability_test)
        del model

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output_dir / "metrics_per_seed.csv", index=False)
    summarize_seed_metrics(metrics_frame, "calibrated_", output_dir)
    raw_ensemble = np.mean(raw_predictions, axis=0)
    calibrated_ensemble = np.mean(calibrated_predictions, axis=0)
    deep_ensemble_sd = np.std(calibrated_predictions, axis=0, ddof=1)
    mc_dropout_and_seed_sd = (
        np.concatenate(stochastic_predictions, axis=0).std(axis=0, ddof=1)
        if stochastic_predictions
        else None
    )
    ensemble = test[["source_row", "sequence", "label"]].copy()
    ensemble["probability_raw_mean"] = raw_ensemble
    ensemble["probability_calibrated_mean"] = calibrated_ensemble
    ensemble["deep_ensemble_probability_sd"] = deep_ensemble_sd
    if mc_dropout_and_seed_sd is not None:
        ensemble["supplementary_mc_dropout_and_seed_sd"] = mc_dropout_and_seed_sd
    ensemble.to_csv(output_dir / "ensemble_test_predictions.tsv", sep="\t", index=False)
    save_json(classification_metrics(y_test, raw_ensemble), output_dir / "ensemble_metrics_raw.json")
    save_json(classification_metrics(y_test, calibrated_ensemble), output_dir / "ensemble_metrics.json")
    absolute_probability_error = np.abs(y_test - calibrated_ensemble)
    save_json(
        {
            "method": "standard deviation across independently trained seed models",
            "n_training_seeds": int(len(args.seeds)),
            "mean_predictive_sd": float(np.mean(deep_ensemble_sd)),
            "median_predictive_sd": float(np.median(deep_ensemble_sd)),
            "predictive_sd_95th_percentile": float(np.percentile(deep_ensemble_sd, 95)),
            "spearman_uncertainty_vs_absolute_probability_error": float(
                spearmanr(deep_ensemble_sd, absolute_probability_error)[0]
            ),
            "scope": "deep-ensemble epistemic uncertainty; not a full prediction interval",
        },
        output_dir / "uncertainty_summary.json",
    )
    if mc_dropout_and_seed_sd is not None:
        save_json(
            {
                "method": "supplementary standard deviation over MC-dropout draws and training seeds",
                "mc_draws_per_seed": int(args.mc_samples),
                "mean_predictive_sd": float(np.mean(mc_dropout_and_seed_sd)),
                "median_predictive_sd": float(np.median(mc_dropout_and_seed_sd)),
            },
            output_dir / "supplementary_mc_dropout_uncertainty.json",
        )
    bootstrap_confidence_intervals(
        y_test,
        calibrated_ensemble,
        classification_metrics,
        args.bootstrap_replicates,
        args.split_seed,
    ).to_csv(output_dir / "bootstrap_95CI.csv", index=False)
    save_json(
        {
            "replicates": int(args.bootstrap_replicates),
            "interval": "2.5th and 97.5th percentiles",
            "resampling_unit": "test sequence",
            "stratification": "binary outcome class",
        },
        output_dir / "bootstrap_design.json",
    )
    plot_classification(y_test, raw_ensemble, calibrated_ensemble, deep_ensemble_sd, output_dir)


def inverse_regression(values, standardizer, minmax):
    values = np.asarray(values).reshape(-1, 1)
    return standardizer.inverse_transform(minmax.inverse_transform(values)).reshape(-1)


def train_regression_split(tf, data, split_mode, args, task_root):
    output_dir = task_root / split_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, excluded, metadata = make_robustness_split(
        data,
        target_col="value",
        task="regression",
        split_mode=split_mode,
        test_size=args.test_size,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
        minimum_test_train_hamming=args.minimum_test_train_hamming,
    )
    save_split_files(output_dir, train, validation, test, excluded, metadata)

    x_train = one_hot_encode(train["sequence"])
    x_validation = one_hot_encode(validation["sequence"])
    x_test = one_hot_encode(test["sequence"])
    y_train_raw = train["value"].to_numpy(dtype=float)
    y_validation_raw = validation["value"].to_numpy(dtype=float)
    y_test = test["value"].to_numpy(dtype=float)
    standardizer = StandardScaler().fit(y_train_raw.reshape(-1, 1))
    y_train_standard = standardizer.transform(y_train_raw.reshape(-1, 1))
    minmax = MinMaxScaler(feature_range=(-1, 1)).fit(y_train_standard)
    y_train = minmax.transform(y_train_standard).reshape(-1)
    y_validation = minmax.transform(standardizer.transform(y_validation_raw.reshape(-1, 1))).reshape(-1)
    save_json(
        {
            "standard_scaler_mean": float(standardizer.mean_[0]),
            "standard_scaler_scale": float(standardizer.scale_[0]),
            "minmax_data_min": float(minmax.data_min_[0]),
            "minmax_data_max": float(minmax.data_max_[0]),
        },
        output_dir / "training_only_target_scaler.json",
    )

    rows = []
    seed_predictions = []
    stochastic_predictions = []
    for seed in args.seeds:
        tf.keras.backend.clear_session()
        set_global_seed(seed, tf)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model = build_regression_model(tf, x_train.shape[1])
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            batch_size=args.regression_batch_size,
            epochs=args.regression_epochs,
            callbacks=callbacks_for(tf, seed_dir, args.regression_patience),
            shuffle=True,
            verbose=args.verbose,
        )
        epochs_trained, best_epoch = save_history(history, seed_dir)
        scaled_prediction = model.predict(x_test, batch_size=args.regression_batch_size, verbose=0).reshape(-1)
        prediction = inverse_regression(scaled_prediction, standardizer, minmax)
        mc_scaled = mc_dropout_predict(model, x_test, args.mc_samples)
        mc_values = None
        if mc_scaled is not None:
            mc_values = np.asarray([inverse_regression(draw, standardizer, minmax) for draw in mc_scaled])
            stochastic_predictions.append(mc_values)
        metric = regression_metrics(y_test, prediction)
        row = {"seed": seed, "epochs_trained": epochs_trained, "best_epoch": best_epoch}
        row.update({f"test_{key}": value for key, value in metric.items()})
        rows.append(row)
        predictions = test[["source_row", "sequence", "value"]].copy()
        predictions["prediction"] = prediction
        if mc_values is not None:
            predictions["mc_prediction_mean"] = mc_values.mean(axis=0)
            predictions["mc_prediction_sd"] = mc_values.std(axis=0, ddof=1)
        predictions.to_csv(seed_dir / "test_predictions.tsv", sep="\t", index=False)
        seed_predictions.append(prediction)
        del model

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(output_dir / "metrics_per_seed.csv", index=False)
    summarize_seed_metrics(metrics_frame, "test_", output_dir)
    ensemble_prediction = np.mean(seed_predictions, axis=0)
    deep_ensemble_sd = np.std(seed_predictions, axis=0, ddof=1)
    mc_dropout_and_seed_sd = (
        np.concatenate(stochastic_predictions, axis=0).std(axis=0, ddof=1)
        if stochastic_predictions
        else None
    )
    ensemble = test[["source_row", "sequence", "value"]].copy()
    ensemble["prediction_mean"] = ensemble_prediction
    ensemble["deep_ensemble_prediction_sd"] = deep_ensemble_sd
    ensemble["residual"] = y_test - ensemble_prediction
    ensemble["absolute_error"] = np.abs(y_test - ensemble_prediction)
    if mc_dropout_and_seed_sd is not None:
        ensemble["supplementary_mc_dropout_and_seed_sd"] = mc_dropout_and_seed_sd
    ensemble.to_csv(output_dir / "ensemble_test_predictions.tsv", sep="\t", index=False)
    save_json(regression_metrics(y_test, ensemble_prediction), output_dir / "ensemble_metrics.json")
    absolute_error = np.abs(y_test - ensemble_prediction)
    save_json(
        {
            "method": "standard deviation across independently trained seed models",
            "n_training_seeds": int(len(args.seeds)),
            "mean_predictive_sd": float(np.mean(deep_ensemble_sd)),
            "median_predictive_sd": float(np.median(deep_ensemble_sd)),
            "predictive_sd_95th_percentile": float(np.percentile(deep_ensemble_sd, 95)),
            "spearman_uncertainty_vs_absolute_error": float(
                spearmanr(deep_ensemble_sd, absolute_error)[0]
            ),
            "scope": "deep-ensemble epistemic uncertainty; not a full prediction interval",
        },
        output_dir / "uncertainty_summary.json",
    )
    save_json(
        {
            "residual_definition": "observed minus predicted",
            "mean_residual": float(np.mean(y_test - ensemble_prediction)),
            "median_residual": float(np.median(y_test - ensemble_prediction)),
            "residual_standard_deviation": float(np.std(y_test - ensemble_prediction, ddof=1)),
            "mean_absolute_residual": float(np.mean(absolute_error)),
            "spearman_prediction_vs_residual": float(
                spearmanr(ensemble_prediction, y_test - ensemble_prediction)[0]
            ),
        },
        output_dir / "residual_summary.json",
    )
    if mc_dropout_and_seed_sd is not None:
        save_json(
            {
                "method": "supplementary standard deviation over MC-dropout draws and training seeds",
                "mc_draws_per_seed": int(args.mc_samples),
                "mean_predictive_sd": float(np.mean(mc_dropout_and_seed_sd)),
                "median_predictive_sd": float(np.median(mc_dropout_and_seed_sd)),
            },
            output_dir / "supplementary_mc_dropout_uncertainty.json",
        )
    bootstrap_confidence_intervals(
        y_test,
        ensemble_prediction,
        regression_metrics,
        args.bootstrap_replicates,
        args.split_seed,
    ).to_csv(output_dir / "bootstrap_95CI.csv", index=False)
    save_json(
        {
            "replicates": int(args.bootstrap_replicates),
            "interval": "2.5th and 97.5th percentiles",
            "resampling_unit": "test sequence",
            "stratification": "observed regression target quantile strata (up to 10)",
        },
        output_dir / "bootstrap_design.json",
    )
    plot_regression(y_test, ensemble_prediction, deep_ensemble_sd, output_dir)


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tf = configure_tensorflow()

    if args.export_single_models_only:
        export_requested_single_models(tf, args, output_root)
        print(f"\nSingle-model export completed. Results: {output_root / 'single_models'}")
        return

    configuration = vars(args).copy()
    configuration.update(
        {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "tensorflow_version": tf.__version__,
        }
    )
    save_json(configuration, output_root / "run_configuration.json")

    completed_tasks = []
    if args.task in {"all", "classification"}:
        classification = read_classification(args.positive_file, args.negative_file)
        for split_mode in args.split_modes:
            print(f"\n=== Classification / {split_mode} ===", flush=True)
            train_classification_split(tf, classification, split_mode, args, output_root / "classification")
        completed_tasks.append("classification")

    if args.task in {"all", "regression"}:
        regression = read_regression(args.regression_file)
        for split_mode in args.split_modes:
            print(f"\n=== Regression / {split_mode} ===", flush=True)
            train_regression_split(tf, regression, split_mode, args, output_root / "regression")
        completed_tasks.append("regression")

    write_combined_tables(output_root, completed_tasks, args.split_modes)
    if not args.skip_single_model_export:
        if args.single_model_split not in args.split_modes:
            print(
                f"\nSingle-model export skipped because {args.single_model_split!r} "
                f"was not among the completed split modes {args.split_modes}.",
                flush=True,
            )
        else:
            export_requested_single_models(tf, args, output_root)
    print(f"\nAll requested analyses completed. Results: {output_root}")


if __name__ == "__main__":
    main()
