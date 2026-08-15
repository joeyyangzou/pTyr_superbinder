#!/usr/bin/env python3
"""Binary CNN evaluation with an 80:20 independent test and outer 10-fold CV.

The independent 20% test set is frozen before cross-validation. Ten-fold
cross-validation is performed only within the remaining 80% development set.
For each outer fold, an inner validation subset is used for early stopping;
the untouched outer fold is evaluated only after the best epoch is selected.
A final model is selected in the same way, refitted on all development data,
and evaluated once on the independent test set.
"""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split


AMINO_ACIDS = "ILVFMCAGPTSYWQNHEDKR"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-file", default="positive")
    parser.add_argument("--negative-file", default="negative")
    parser.add_argument("--output-dir", default="classification_80_20_10fold_results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--inner-validation-size", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Independent final-training seeds used for mean +/- SD and deep-ensemble uncertainty.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Write and audit the 80:20 and outer-fold assignments without importing TensorFlow.",
    )
    return parser.parse_args()


def save_json(value, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def read_data(positive_file, negative_file):
    positive = pd.read_csv(positive_file, sep="\t")
    negative = pd.read_csv(negative_file, sep="\t")
    required = {"sequence", "label"}
    if not required.issubset(positive.columns) or not required.issubset(negative.columns):
        raise ValueError("Both input files must contain sequence and label columns")
    data = pd.concat(
        [positive.loc[:, ["sequence", "label"]], negative.loc[:, ["sequence", "label"]]],
        ignore_index=True,
    )
    data["sequence"] = data["sequence"].astype(str).str.strip().str.upper()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Classification labels must contain both 0 and 1")
    if data["sequence"].duplicated().any():
        duplicates = data.loc[data["sequence"].duplicated(False), "sequence"].unique()[:5]
        raise ValueError("Duplicate sequences must be resolved before splitting: " + ", ".join(duplicates))
    invalid = [
        sequence
        for sequence in data["sequence"]
        if len(sequence) != 8 or any(amino_acid not in AA_TO_INDEX for amino_acid in sequence)
    ]
    if invalid:
        raise ValueError("All sequences must contain exactly eight standard amino acids; examples: " + ", ".join(invalid[:5]))
    data.insert(0, "source_row", np.arange(len(data), dtype=int))
    return data


def one_hot_encode(sequences):
    sequences = [str(sequence).strip().upper() for sequence in sequences]
    encoded = np.zeros((len(sequences), 8, 20), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, amino_acid in enumerate(sequence):
            encoded[row, position, AA_TO_INDEX[amino_acid]] = 1.0
    return encoded


def set_global_seed(seed, tf=None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def configure_tensorflow():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return tf


def build_network(tf):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(8, 20)),
            layers.Conv1D(128, 1, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 9, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
            layers.Dropout(0.5),
            layers.Conv1D(128, 10, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
            layers.Dropout(0.7),
            layers.Dense(64, activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
            layers.Dense(32, activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
            layers.Dense(8, activation="relu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def classification_metrics(y_true, probability, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, probability)),
        "AUPRC": float(average_precision_score(y_true, probability)),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predicted)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "ECE": float(expected_calibration_error(y_true, probability)),
    }


def expected_calibration_error(y_true, probability, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        if selected.any():
            error += selected.mean() * abs(y_true[selected].mean() - probability[selected].mean())
    return error


def fit_platt_calibrator(y_validation, raw_probability):
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, np.asarray(y_validation, dtype=int))
    return calibrator


def apply_platt_calibrator(calibrator, raw_probability):
    clipped = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def stratified_bootstrap(y_true, probability, replicates, seed, threshold=0.5):
    if replicates <= 0:
        return {}
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    rng = np.random.RandomState(seed)
    class_indices = [np.flatnonzero(y_true == label) for label in sorted(np.unique(y_true))]
    values = {}
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        metrics = classification_metrics(y_true[sampled], probability[sampled], threshold)
        for name, value in metrics.items():
            values.setdefault(name, []).append(value)
    return {
        name: {
            "estimate": classification_metrics(y_true, probability, threshold)[name],
            "ci_2.5": float(np.percentile(samples, 2.5)),
            "ci_97.5": float(np.percentile(samples, 97.5)),
            "replicates": int(replicates),
        }
        for name, samples in values.items()
    }


def select_f1_threshold(y_true, probability):
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(values))])


def fit_for_epoch_selection(tf, x_train, y_train, x_validation, y_validation, args, seed):
    tf.keras.backend.clear_session()
    set_global_seed(seed, tf)
    model = build_network(tf)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
        verbose=0,
    )
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[callback],
        shuffle=True,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - start
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    return model, history, best_epoch, elapsed


def refit_and_predict(tf, x_train, y_train, x_evaluation, best_epoch, args, seed):
    tf.keras.backend.clear_session()
    set_global_seed(seed, tf)
    model = build_network(tf)
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=max(1, best_epoch),
        shuffle=True,
        verbose=args.verbose,
    )
    training_seconds = time.perf_counter() - start
    prediction_start = time.perf_counter()
    probability = model.predict(x_evaluation, batch_size=args.batch_size, verbose=0).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_start
    return model, history, probability, training_seconds, prediction_seconds


def history_frame(history):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def plot_cv_roc(y_true, probability, fold_curves, output_path):
    figure, axis = plt.subplots(figsize=(6, 6))
    for fold, fpr, tpr, fold_auc in fold_curves:
        axis.plot(fpr, tpr, alpha=0.25, linewidth=1, label="Fold %d (%.3f)" % (fold, fold_auc))
    fpr, tpr, _ = roc_curve(y_true, probability)
    pooled_auc = roc_auc_score(y_true, probability)
    axis.plot(fpr, tpr, color="black", linewidth=2, label="Pooled OOF (%.3f)" % pooled_auc)
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="10-fold cross-validation ROC")
    axis.legend(loc="lower right", fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_independent_roc(y_true, probability, output_path):
    fpr, tpr, _ = roc_curve(y_true, probability)
    auc_value = roc_auc_score(y_true, probability)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(fpr, tpr, linewidth=2, label="Independent test (AUC = %.3f)" % auc_value)
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Independent 20% test ROC")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_precision_recall(y_true, probability, output_path, title):
    precision, recall, _ = precision_recall_curve(y_true, probability)
    auprc = average_precision_score(y_true, probability)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(recall, precision, linewidth=2, label="AUPRC = %.3f" % auprc)
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_reliability(y_true, probability, output_path, n_bins=10, raw_probability=None):
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probability, edges[1:-1]), 0, n_bins - 1)
    observed, predicted, counts = [], [], []
    for index in range(n_bins):
        selected = bin_index == index
        if selected.any():
            observed.append(float(y_true[selected].mean()))
            predicted.append(float(probability[selected].mean()))
            counts.append(int(selected.sum()))
    calibration = pd.DataFrame(
        {"mean_predicted_probability": predicted, "observed_positive_fraction": observed, "n": counts}
    )
    calibration.to_csv(Path(output_path).with_suffix(".tsv"), sep="\t", index=False)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    if raw_probability is not None:
        raw_probability = np.asarray(raw_probability, dtype=float)
        raw_bins = np.clip(np.digitize(raw_probability, edges[1:-1]), 0, n_bins - 1)
        raw_observed, raw_predicted = [], []
        for index in range(n_bins):
            selected = raw_bins == index
            if selected.any():
                raw_observed.append(float(y_true[selected].mean()))
                raw_predicted.append(float(raw_probability[selected].mean()))
        axis.plot(raw_predicted, raw_observed, marker="o", linewidth=1.5, label="Raw")
    axis.plot(predicted, observed, marker="o", linewidth=2, label="Platt calibrated")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Independent-test reliability plot",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main():
    args = parse_args()
    if args.folds != 10:
        print("Warning: the manuscript statement specifically requires --folds 10.")
    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    cv_dir = output_dir / "cross_validation"
    final_dir = output_dir / "final_model"
    split_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    data = read_data(args.positive_file, args.negative_file)
    all_indices = np.arange(len(data), dtype=int)
    development_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
        stratify=data["label"].to_numpy(),
    )
    development = data.iloc[development_indices].copy()
    independent_test = data.iloc[test_indices].copy()
    development.to_csv(split_dir / "development_80.tsv", sep="\t", index=False)
    independent_test.to_csv(split_dir / "independent_test_20.tsv", sep="\t", index=False)

    if set(development["source_row"]) & set(independent_test["source_row"]):
        raise AssertionError("Development and independent test sets overlap")

    cv_splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_assignments = development.loc[:, ["source_row", "sequence", "label"]].copy()
    fold_assignments["outer_fold"] = 0
    fold_splits = []
    for fold, (outer_train_relative, outer_validation_relative) in enumerate(
        cv_splitter.split(development["sequence"], development["label"]), start=1
    ):
        fold_assignments.iloc[outer_validation_relative, fold_assignments.columns.get_loc("outer_fold")] = fold
        fold_splits.append((fold, outer_train_relative, outer_validation_relative))
    fold_assignments.to_csv(split_dir / "development_10fold_assignments.tsv", sep="\t", index=False)

    split_manifest = {
        "design": "80% development plus 20% untouched independent test; outer 10-fold CV only within development",
        "split_seed": int(args.seed),
        "n_total": int(len(data)),
        "n_development": int(len(development)),
        "n_independent_test": int(len(independent_test)),
        "test_fraction": float(args.test_size),
        "outer_folds": int(args.folds),
        "inner_validation_fraction_of_outer_training": float(args.inner_validation_size),
        "test_used_for_early_stopping": False,
        "outer_fold_used_for_early_stopping": False,
    }
    save_json(split_manifest, split_dir / "split_manifest.json")

    if args.split_only:
        print("Split-only audit completed:", split_manifest)
        return

    analysis_started = time.perf_counter()
    tf = configure_tensorflow()
    x_development = one_hot_encode(development["sequence"])
    y_development = development["label"].to_numpy(dtype=int)
    x_test = one_hot_encode(independent_test["sequence"])
    y_test = independent_test["label"].to_numpy(dtype=int)

    oof_probability = np.full(len(development), np.nan, dtype=float)
    fold_rows = []
    fold_curves = []
    for fold, outer_train_relative, outer_validation_relative in fold_splits:
        fold_output = cv_dir / ("fold_%02d" % fold)
        fold_output.mkdir(parents=True, exist_ok=True)
        inner_train_relative, inner_validation_relative = train_test_split(
            outer_train_relative,
            test_size=args.inner_validation_size,
            random_state=args.seed + fold,
            shuffle=True,
            stratify=y_development[outer_train_relative],
        )
        if set(inner_train_relative) & set(inner_validation_relative):
            raise AssertionError("Inner training and validation sets overlap")
        if set(outer_train_relative) & set(outer_validation_relative):
            raise AssertionError("Outer training and validation sets overlap")

        selection_model, selection_history, best_epoch, selection_seconds = fit_for_epoch_selection(
            tf,
            x_development[inner_train_relative],
            y_development[inner_train_relative],
            x_development[inner_validation_relative],
            y_development[inner_validation_relative],
            args,
            args.seed + fold,
        )
        del selection_model
        model, refit_history, probability, refit_seconds, prediction_seconds = refit_and_predict(
            tf,
            x_development[outer_train_relative],
            y_development[outer_train_relative],
            x_development[outer_validation_relative],
            best_epoch,
            args,
            args.seed + fold,
        )
        del model
        oof_probability[outer_validation_relative] = probability
        metrics = classification_metrics(y_development[outer_validation_relative], probability)
        row = {
            "fold": fold,
            "n_outer_train": int(len(outer_train_relative)),
            "n_inner_train": int(len(inner_train_relative)),
            "n_inner_validation": int(len(inner_validation_relative)),
            "n_outer_validation": int(len(outer_validation_relative)),
            "best_epoch": int(best_epoch),
            "selection_training_seconds": float(selection_seconds),
            "refit_training_seconds": float(refit_seconds),
            "prediction_seconds": float(prediction_seconds),
        }
        row.update(metrics)
        fold_rows.append(row)
        fpr, tpr, _ = roc_curve(y_development[outer_validation_relative], probability)
        fold_curves.append((fold, fpr, tpr, metrics["AUROC"]))

        history_frame(selection_history).to_csv(fold_output / "epoch_selection_history.csv", index=False)
        history_frame(refit_history).to_csv(fold_output / "refit_history.csv", index=False)
        predictions = development.iloc[outer_validation_relative][["source_row", "sequence", "label"]].copy()
        predictions["probability"] = probability
        predictions.to_csv(fold_output / "outer_fold_predictions.tsv", sep="\t", index=False)

    if np.isnan(oof_probability).any():
        raise AssertionError("Each development observation must receive exactly one out-of-fold prediction")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(cv_dir / "fold_metrics.csv", index=False)
    metric_names = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1", "MCC", "Brier", "ECE"]
    cv_summary = {
        "n_folds": int(args.folds),
        "mean_fold_metrics": {name: float(fold_metrics[name].mean()) for name in metric_names},
        "sd_fold_metrics": {name: float(fold_metrics[name].std(ddof=1)) for name in metric_names},
        "pooled_out_of_fold_metrics": classification_metrics(y_development, oof_probability),
    }
    save_json(cv_summary, cv_dir / "cv_summary.json")
    oof = development[["source_row", "sequence", "label"]].copy()
    oof["outer_fold"] = fold_assignments["outer_fold"].to_numpy()
    oof["probability"] = oof_probability
    oof.to_csv(cv_dir / "out_of_fold_predictions.tsv", sep="\t", index=False)
    plot_cv_roc(y_development, oof_probability, fold_curves, cv_dir / "ten_fold_cv_roc.png")
    plot_precision_recall(
        y_development,
        oof_probability,
        cv_dir / "ten_fold_cv_precision_recall.png",
        "10-fold cross-validation precision-recall curve",
    )
    development_calibrator = fit_platt_calibrator(y_development, oof_probability)
    calibrated_oof_probability = apply_platt_calibrator(development_calibrator, oof_probability)
    classification_threshold = select_f1_threshold(y_development, calibrated_oof_probability)
    save_json(
        {
            "coefficient": float(development_calibrator.coef_[0, 0]),
            "intercept": float(development_calibrator.intercept_[0]),
            "fitted_on": "pooled out-of-fold predictions from the 80% development set",
            "classification_threshold": classification_threshold,
            "threshold_selection": "maximum F1 on calibrated development OOF predictions",
            "test_data_used": False,
        },
        cv_dir / "development_oof_platt_calibration.json",
    )

    final_inner_train, final_inner_validation = train_test_split(
        np.arange(len(development), dtype=int),
        test_size=args.inner_validation_size,
        random_state=args.seed + 1000,
        shuffle=True,
        stratify=y_development,
    )
    repeated_dir = output_dir / "repeated_training"
    repeated_dir.mkdir(parents=True, exist_ok=True)
    repeated_rows = []
    raw_test_predictions = []
    calibrated_test_predictions = []
    selected = None
    for training_seed in args.training_seeds:
        selection_model, selection_history, best_epoch, selection_seconds = fit_for_epoch_selection(
            tf,
            x_development[final_inner_train],
            y_development[final_inner_train],
            x_development[final_inner_validation],
            y_development[final_inner_validation],
            args,
            training_seed,
        )
        minimum_validation_loss = float(np.min(selection_history.history["val_loss"]))
        del selection_model
        model, final_history, raw_probability, refit_seconds, prediction_seconds = refit_and_predict(
            tf, x_development, y_development, x_test, best_epoch, args, training_seed
        )
        calibrated_probability = apply_platt_calibrator(development_calibrator, raw_probability)
        repeated_rows.append(
            {
                "training_seed": training_seed,
                "best_epoch": best_epoch,
                "minimum_validation_loss": minimum_validation_loss,
                "selection_training_seconds": selection_seconds,
                "refit_training_seconds": refit_seconds,
                "test_prediction_seconds": prediction_seconds,
            }
        )
        history_frame(selection_history).to_csv(
            repeated_dir / ("seed_%d_selection_history.csv" % training_seed), index=False
        )
        raw_test_predictions.append(raw_probability)
        calibrated_test_predictions.append(calibrated_probability)
        candidate = {
                "seed": training_seed,
                "validation_loss": minimum_validation_loss,
                "best_epoch": best_epoch,
                "history": final_history,
                "selection_seconds": selection_seconds,
                "refit_seconds": refit_seconds,
                "prediction_seconds": prediction_seconds,
            }
        if selected is None or (candidate["validation_loss"], candidate["seed"]) < (
            selected["validation_loss"], selected["seed"]
        ):
            selected = candidate
        del model

    # Test labels are used here only after all independently seeded models have
    # completed training; no test result is fed back into model selection.
    for metadata, calibrated_probability in zip(repeated_rows, calibrated_test_predictions):
        metadata.update(
            classification_metrics(y_test, calibrated_probability, classification_threshold)
        )
    repeated_metrics = pd.DataFrame(repeated_rows)
    repeated_metrics.to_csv(repeated_dir / "metrics_per_seed.csv", index=False)
    repeated_summary = {
        metric: {
            "mean": float(repeated_metrics[metric].mean()),
            "sd": float(repeated_metrics[metric].std(ddof=1)),
            "n_independent_runs": int(len(repeated_metrics)),
        }
        for metric in ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1", "MCC", "Brier", "ECE"]
    }
    save_json(repeated_summary, repeated_dir / "metrics_mean_sd.json")

    raw_ensemble_probability = np.mean(np.asarray(raw_test_predictions), axis=0)
    ensemble_probability = np.mean(np.asarray(calibrated_test_predictions), axis=0)
    ensemble_uncertainty = np.std(np.asarray(calibrated_test_predictions), axis=0, ddof=1)
    ensemble_metrics = classification_metrics(y_test, ensemble_probability, classification_threshold)
    ensemble_ci = stratified_bootstrap(
        y_test,
        ensemble_probability,
        args.bootstrap_replicates,
        args.seed + 2000,
        classification_threshold,
    )
    ensemble_frame = independent_test[["source_row", "sequence", "label"]].copy()
    ensemble_frame["raw_probability_mean"] = raw_ensemble_probability
    ensemble_frame["calibrated_probability_mean"] = ensemble_probability
    ensemble_frame["deep_ensemble_probability_sd"] = ensemble_uncertainty
    ensemble_frame.to_csv(repeated_dir / "ensemble_test_predictions.tsv", sep="\t", index=False)
    save_json(
        {
            "metrics": ensemble_metrics,
            "bootstrap_95_ci": ensemble_ci,
            "n_training_seeds": len(args.training_seeds),
        },
        repeated_dir / "ensemble_summary.json",
    )
    save_json(
        {
            "method": "standard deviation across independently initialized final-training models",
            "scope": "deep-ensemble epistemic disagreement; not a complete predictive interval",
            "n_training_seeds": len(args.training_seeds),
            "mean_predictive_sd": float(np.mean(ensemble_uncertainty)),
            "median_predictive_sd": float(np.median(ensemble_uncertainty)),
            "predictive_sd_95th_percentile": float(np.percentile(ensemble_uncertainty, 95)),
        },
        repeated_dir / "uncertainty_summary.json",
    )

    if selected is None:
        raise AssertionError("No final-training model was produced")
    best_epoch = selected["best_epoch"]
    selection_seconds = selected["selection_seconds"]
    final_model, final_history, test_probability_raw, refit_seconds, prediction_seconds = refit_and_predict(
        tf,
        x_development,
        y_development,
        x_test,
        best_epoch,
        args,
        selected["seed"],
    )
    test_probability = apply_platt_calibrator(development_calibrator, test_probability_raw)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(str(final_dir / "saved_model"), save_format="tf", include_optimizer=False)
    history_frame(final_history).to_csv(final_dir / "refit_history.csv", index=False)
    save_json(
        {
            "coefficient": float(development_calibrator.coef_[0, 0]),
            "intercept": float(development_calibrator.intercept_[0]),
            "fitted_on": "pooled OOF predictions from the 80% development set",
            "classification_threshold": classification_threshold,
        },
        final_dir / "platt_calibration.json",
    )

    test_predictions = independent_test[["source_row", "sequence", "label"]].copy()
    test_predictions["probability"] = test_probability
    test_predictions.to_csv(final_dir / "independent_test_predictions.tsv", sep="\t", index=False)
    test_metrics = classification_metrics(y_test, test_probability, classification_threshold)
    test_ci = stratified_bootstrap(
        y_test,
        test_probability,
        args.bootstrap_replicates,
        args.seed + 3000,
        classification_threshold,
    )
    final_summary = {
        "selected_training_seed": int(selected["seed"]),
        "selection_rule": "minimum inner-validation loss; test metrics were not used",
        "best_epoch_selected_without_test_data": int(best_epoch),
        "n_development_refit": int(len(development)),
        "n_independent_test": int(len(independent_test)),
        "selection_training_seconds": float(selection_seconds),
        "refit_training_seconds": float(refit_seconds),
        "independent_test_prediction_seconds": float(prediction_seconds),
        "independent_test_metrics": test_metrics,
        "bootstrap_95_ci": test_ci,
        "test_used_for_early_stopping_or_model_selection": False,
    }
    save_json(final_summary, final_dir / "independent_test_summary.json")
    plot_independent_roc(y_test, test_probability, final_dir / "independent_test_roc.png")
    plot_precision_recall(
        y_test,
        test_probability,
        final_dir / "independent_test_precision_recall.png",
        "Independent 20% test precision-recall curve",
    )
    plot_reliability(
        y_test,
        test_probability,
        final_dir / "independent_test_reliability.png",
        raw_probability=test_probability_raw,
    )
    plot_reliability(
        y_test,
        ensemble_probability,
        repeated_dir / "ensemble_reliability.png",
        raw_probability=raw_ensemble_probability,
    )

    run_configuration = vars(args).copy()
    run_configuration.update(
        {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "elapsed_seconds": float(time.perf_counter() - analysis_started),
            "test_used_for_model_selection": False,
        }
    )
    save_json(run_configuration, output_dir / "run_configuration.json")
    print("10-fold mean AUROC: %.6f +/- %.6f" % (
        cv_summary["mean_fold_metrics"]["AUROC"],
        cv_summary["sd_fold_metrics"]["AUROC"],
    ))
    print("Independent 20%% test AUROC: %.6f" % test_metrics["AUROC"])
    print("Results:", output_dir.resolve())


if __name__ == "__main__":
    main()
