#!/usr/bin/env python3
"""CNN regression with an 80:20 independent test and outer 10-fold CV.

The independent 20% test set is frozen before cross-validation. Ten-fold
cross-validation is performed only within the remaining 80% development set.
For each outer fold, an inner validation subset is used for early stopping;
target scaling is fitted only on the corresponding training data. The final
model is refitted on all development data and evaluated once on the test set.
"""

import argparse
import json
import os
import pickle
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


AMINO_ACIDS = "ILVFMCAGPTSYWQNHEDKR"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-file",
        default="train_set.txt",
        help="Fixed 80% development set produced by 06_train_test_split.py",
    )
    parser.add_argument(
        "--test-file",
        default="test_set.txt",
        help="Untouched 20% test set produced by 06_train_test_split.py",
    )
    parser.add_argument(
        "--split-manifest",
        default="regression_80_20_split_manifest.json",
        help="Split audit produced by 06_train_test_split.py",
    )
    parser.add_argument("--output-dir", default="regression_80_20_10fold_results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--inner-validation-size", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Independent final-training seeds for mean +/- SD and ensemble uncertainty.",
    )
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Write and audit split assignments without importing TensorFlow.",
    )
    return parser.parse_args()


def save_json(value, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def read_data(input_file):
    data = pd.read_csv(input_file, sep="\t")
    if not {"sequence", "value"}.issubset(data.columns):
        raise ValueError("Input file must contain sequence and value columns")
    data = data[["sequence", "value"]].copy()
    data["sequence"] = data["sequence"].astype(str).str.strip().str.upper()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["source_row"] = np.arange(len(data), dtype=int)
    if data.empty:
        raise ValueError("Input dataset is empty")
    if data["sequence"].duplicated().any():
        duplicated = int(data["sequence"].duplicated(keep=False).sum())
        raise ValueError(
            "%d rows have duplicated sequences. Aggregate or resolve exact duplicates "
            "before splitting so the same sequence cannot cross partitions." % duplicated
        )
    lengths = data["sequence"].str.len()
    if not (lengths == 8).all():
        raise ValueError("All sequences must have exactly 8 amino acids")
    invalid = sorted(set("".join(data["sequence"])) - set(AMINO_ACIDS + "X"))
    if invalid:
        raise ValueError("Unsupported amino-acid symbols: %s" % ", ".join(invalid))
    if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
        raise ValueError("Regression values must all be finite")
    return data


def encode_sequences(sequences):
    encoded = np.zeros((len(sequences), 8, 20), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for position, amino_acid in enumerate(sequence):
            if amino_acid == "X":
                encoded[row, position, :] = 0.05
            else:
                encoded[row, position, AA_TO_INDEX[amino_acid]] = 1.0
    return encoded


def quantile_strata(values, minimum_count, maximum_bins=10):
    """Return target-quantile labels when every bin is large enough."""
    values = np.asarray(values, dtype=float)
    for bin_count in range(min(maximum_bins, len(values)), 1, -1):
        try:
            labels = pd.qcut(values, q=bin_count, labels=False, duplicates="drop")
        except ValueError:
            continue
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            continue
        counts = pd.Series(labels).value_counts()
        if int(counts.min()) >= int(minimum_count):
            return labels.astype(int)
    return None


def make_outer_split(data, test_size, seed):
    strata = quantile_strata(data["value"].to_numpy(), minimum_count=2)
    development, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return development.reset_index(drop=True), test.reset_index(drop=True), strata is not None


def make_fold_indices(development, folds, seed):
    values = development["value"].to_numpy(dtype=float)
    strata = quantile_strata(values, minimum_count=folds)
    if strata is not None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(values)), strata)), "quantile-stratified"
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(values)))), "random-kfold"


def inner_split(relative_indices, values, validation_size, seed):
    values = np.asarray(values, dtype=float)
    strata = quantile_strata(values, minimum_count=2)
    train_relative, validation_relative = train_test_split(
        np.asarray(relative_indices),
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    return train_relative, validation_relative


def write_split_audit(development, test, fold_indices, output_dir, split_method):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    development_output = development.copy()
    development_output["partition"] = "development"
    test_output = test.copy()
    test_output["partition"] = "independent_test"
    pd.concat([development_output, test_output], ignore_index=True).to_csv(
        output_dir / "independent_80_20_split.tsv", sep="\t", index=False
    )

    assignments = development[["source_row", "sequence", "value"]].copy()
    assignments["outer_fold"] = 0
    for fold, (_, validation_relative) in enumerate(fold_indices, start=1):
        assignments.iloc[
            validation_relative, assignments.columns.get_loc("outer_fold")
        ] = fold
    assignments.to_csv(output_dir / "development_10fold_assignments.tsv", sep="\t", index=False)

    development_sequences = set(development["sequence"])
    test_sequences = set(test["sequence"])
    overlap = development_sequences.intersection(test_sequences)
    if overlap:
        raise AssertionError("Development/test sequence overlap detected")
    if sorted(assignments["outer_fold"].unique().tolist()) != list(
        range(1, len(fold_indices) + 1)
    ):
        raise AssertionError("Some development rows were not assigned to exactly one fold")

    audit = {
        "total_rows": int(len(development) + len(test)),
        "development_rows": int(len(development)),
        "independent_test_rows": int(len(test)),
        "development_test_exact_sequence_overlap": 0,
        "outer_folds": int(len(fold_indices)),
        "outer_fold_method": split_method,
        "test_used_in_cross_validation_or_early_stopping": False,
    }
    save_json(audit, output_dir / "split_audit.json")
    return assignments, audit


def import_tensorflow(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, TypeError):
        pass
    for device in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    return tf


def reset_seed(tf, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(tf):
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Conv1D(128, 1, padding="same", activation="relu", input_shape=(8, 20)),
            layers.Dropout(0.5),
            layers.Conv1D(128, 3, padding="same", activation="relu"),
            layers.Dropout(0.5),
            layers.Conv1D(128, 9, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
            layers.Dropout(0.5),
            layers.Conv1D(128, 10, padding="same", activation="relu"),
            layers.MaxPooling1D(pool_size=2, strides=1),
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
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error", metrics=["mae"])
    return model


def fit_scalers(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    standard = StandardScaler().fit(values)
    minmax = MinMaxScaler(feature_range=(-1, 1)).fit(standard.transform(values))
    return standard, minmax


def transform_values(values, standard, minmax):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return minmax.transform(standard.transform(values)).reshape(-1)


def inverse_values(values, standard, minmax):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return standard.inverse_transform(minmax.inverse_transform(values)).reshape(-1)


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    pearson = float(pearsonr(observed, predicted)[0]) if len(observed) > 1 else float("nan")
    spearman = float(spearmanr(observed, predicted)[0]) if len(observed) > 1 else float("nan")
    return {
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
    }


def select_best_epoch(
    tf, x_train, y_train, x_validation, y_validation, seed, epochs, patience, batch_size, verbose
):
    tf.keras.backend.clear_session()
    reset_seed(tf, seed)
    model = build_model(tf)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=verbose,
        shuffle=True,
    )
    validation_loss = history.history["val_loss"]
    best_epoch = int(np.argmin(validation_loss) + 1)
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    return best_epoch, history_frame


def refit_model(tf, x_train, y_train, seed, epochs, batch_size, verbose):
    tf.keras.backend.clear_session()
    reset_seed(tf, seed)
    model = build_model(tf)
    model.fit(
        x_train,
        y_train,
        epochs=max(1, int(epochs)),
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
    )
    return model


def bootstrap_confidence_intervals(observed, predicted, replicates, seed):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    strata = quantile_strata(observed, minimum_count=2)
    if strata is None:
        strata = np.zeros(len(observed), dtype=int)
    strata_indices = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    rng = np.random.RandomState(seed)
    samples = {name: [] for name in regression_metrics(observed, predicted)}
    for _ in range(replicates):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata_indices]
        )
        values = regression_metrics(observed[selected], predicted[selected])
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        if values:
            intervals[name] = {
                "lower_95": float(np.percentile(values, 2.5)),
                "upper_95": float(np.percentile(values, 97.5)),
                "valid_replicates": int(len(values)),
            }
        else:
            intervals[name] = {"lower_95": None, "upper_95": None, "valid_replicates": 0}
    return intervals


def save_scatter(observed, predicted, path, title):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(observed, predicted, s=15, alpha=0.55, edgecolors="none")
    plt.plot([low, high], [low, high], "--", color="black", linewidth=1)
    plt.xlabel("Observed value")
    plt.ylabel("Predicted value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_residual_plot(observed, predicted, path):
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    plt.figure(figsize=(6.0, 4.5))
    plt.scatter(predicted, residual, s=15, alpha=0.55, edgecolors="none")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Predicted value")
    plt.ylabel("Residual (observed - predicted)")
    plt.title("Independent-test residual analysis")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audit_dir = output_dir / "split_from_step06"
    split_audit_dir.mkdir(exist_ok=True)
    development = read_data(args.development_file).reset_index(drop=True)
    test = read_data(args.test_file).reset_index(drop=True)
    if set(development["sequence"]).intersection(test["sequence"]):
        raise ValueError("Development/test sequence overlap detected")
    split_manifest_path = Path(args.split_manifest)
    if split_manifest_path.is_file():
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8-sig"))
        if split_manifest.get("design") != "log(ratio)-ordered systematic 4:1 split":
            raise ValueError("The split manifest does not describe the expected regression split")
        if int(split_manifest.get("development_rows", -1)) != len(development):
            raise ValueError("Development row count does not match the split manifest")
        if int(split_manifest.get("independent_test_rows", -1)) != len(test):
            raise ValueError("Test row count does not match the split manifest")
        outer_stratified = bool(split_manifest.get("quantile_stratified", False))
    else:
        raise FileNotFoundError(
            "Split manifest not found. Run 06_train_test_split.py before 07_CNN_regression.py"
        )
    development.to_csv(split_audit_dir / "development_80.tsv", sep="\t", index=False)
    test.to_csv(split_audit_dir / "independent_test_20.tsv", sep="\t", index=False)
    save_json(split_manifest, split_audit_dir / "regression_80_20_split_manifest.json")
    fold_indices, fold_method = make_fold_indices(development, args.folds, args.seed)
    assignments, split_audit = write_split_audit(
        development, test, fold_indices, output_dir, fold_method
    )
    if args.split_only:
        print(json.dumps(split_audit, indent=2, sort_keys=True))
        return

    started = time.time()
    tf = import_tensorflow(args.seed)
    x_development = encode_sequences(development["sequence"].tolist())
    y_development = development["value"].to_numpy(dtype=float)
    x_test = encode_sequences(test["sequence"].tolist())
    y_test = test["value"].to_numpy(dtype=float)

    fold_rows = []
    oof_prediction = np.full(len(development), np.nan, dtype=float)
    fold_dir = output_dir / "cross_validation_folds"
    fold_dir.mkdir(exist_ok=True)

    for fold, (outer_train_relative, outer_validation_relative) in enumerate(
        fold_indices, start=1
    ):
        fold_seed = args.seed + fold
        local_train, local_validation = inner_split(
            outer_train_relative,
            y_development[outer_train_relative],
            args.inner_validation_size,
            fold_seed,
        )
        selection_standard, selection_minmax = fit_scalers(y_development[local_train])
        scaled_inner_train = transform_values(
            y_development[local_train], selection_standard, selection_minmax
        )
        scaled_inner_validation = transform_values(
            y_development[local_validation], selection_standard, selection_minmax
        )
        best_epoch, history = select_best_epoch(
            tf,
            x_development[local_train],
            scaled_inner_train,
            x_development[local_validation],
            scaled_inner_validation,
            fold_seed,
            args.epochs,
            args.patience,
            args.batch_size,
            args.verbose,
        )
        history.to_csv(fold_dir / ("fold_%02d_history.tsv" % fold), sep="\t", index=False)

        fold_standard, fold_minmax = fit_scalers(y_development[outer_train_relative])
        scaled_outer_train = transform_values(
            y_development[outer_train_relative], fold_standard, fold_minmax
        )
        model = refit_model(
            tf,
            x_development[outer_train_relative],
            scaled_outer_train,
            fold_seed,
            best_epoch,
            args.batch_size,
            args.verbose,
        )
        scaled_prediction = model.predict(
            x_development[outer_validation_relative], batch_size=args.batch_size, verbose=0
        ).reshape(-1)
        prediction = inverse_values(scaled_prediction, fold_standard, fold_minmax)
        oof_prediction[outer_validation_relative] = prediction
        metrics = regression_metrics(y_development[outer_validation_relative], prediction)
        fold_rows.append(
            {
                "fold": fold,
                "outer_train_n": int(len(outer_train_relative)),
                "inner_train_n": int(len(local_train)),
                "inner_validation_n": int(len(local_validation)),
                "outer_validation_n": int(len(outer_validation_relative)),
                "selected_epoch": best_epoch,
                **metrics,
            }
        )
        prediction_frame = development.iloc[outer_validation_relative][
            ["source_row", "sequence", "value"]
        ].copy()
        prediction_frame["prediction"] = prediction
        prediction_frame["residual"] = prediction_frame["value"] - prediction
        prediction_frame.to_csv(
            fold_dir / ("fold_%02d_predictions.tsv" % fold), sep="\t", index=False
        )

    if np.isnan(oof_prediction).any():
        raise AssertionError("Missing out-of-fold predictions")
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "cross_validation_fold_metrics.tsv", sep="\t", index=False)
    metric_names = ["pearson_r", "spearman_rho", "r2", "mae", "rmse"]
    cv_mean_sd = {
        name: {
            "mean": float(fold_metrics[name].mean()),
            "sd": float(fold_metrics[name].std(ddof=1)),
        }
        for name in metric_names
    }
    pooled_cv_metrics = regression_metrics(y_development, oof_prediction)
    cv_summary = {
        "scope": "10-fold outer cross-validation performed only within the 80% development set",
        "fold_method": fold_method,
        "fold_mean_and_sd": cv_mean_sd,
        "pooled_out_of_fold_metrics": pooled_cv_metrics,
    }
    save_json(cv_summary, output_dir / "cross_validation_summary.json")
    oof = assignments.copy()
    oof["prediction"] = oof_prediction
    oof["residual"] = oof["value"] - oof_prediction
    oof.to_csv(output_dir / "cross_validation_oof_predictions.tsv", sep="\t", index=False)
    save_scatter(
        y_development,
        oof_prediction,
        output_dir / "cross_validation_oof_scatter.png",
        "10-fold OOF predictions (80% development set)",
    )

    all_indices = np.arange(len(development))
    final_inner_train, final_inner_validation = inner_split(
        all_indices, y_development, args.inner_validation_size, args.seed + 1000
    )
    final_standard, final_minmax = fit_scalers(y_development)
    selection_standard, selection_minmax = fit_scalers(y_development[final_inner_train])
    repeated_dir = output_dir / "repeated_training"
    repeated_dir.mkdir(exist_ok=True)
    repeated_rows = []
    seed_predictions = []
    selected = None
    for training_seed in args.training_seeds:
        final_best_epoch, selection_history = select_best_epoch(
            tf,
            x_development[final_inner_train],
            transform_values(y_development[final_inner_train], selection_standard, selection_minmax),
            x_development[final_inner_validation],
            transform_values(
                y_development[final_inner_validation], selection_standard, selection_minmax
            ),
            training_seed,
            args.epochs,
            args.patience,
            args.batch_size,
            args.verbose,
        )
        minimum_validation_loss = float(selection_history["val_loss"].min())
        selection_history.to_csv(
            repeated_dir / ("seed_%d_selection_history.tsv" % training_seed),
            sep="\t",
            index=False,
        )
        model = refit_model(
            tf,
            x_development,
            transform_values(y_development, final_standard, final_minmax),
            training_seed,
            final_best_epoch,
            args.batch_size,
            args.verbose,
        )
        scaled_prediction = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)
        prediction = inverse_values(scaled_prediction, final_standard, final_minmax)
        repeated_rows.append(
            {
                "training_seed": training_seed,
                "best_epoch": final_best_epoch,
                "minimum_validation_loss": minimum_validation_loss,
            }
        )
        seed_predictions.append(prediction)
        candidate = {
            "seed": training_seed,
            "validation_loss": minimum_validation_loss,
            "best_epoch": final_best_epoch,
        }
        if selected is None or (candidate["validation_loss"], candidate["seed"]) < (
            selected["validation_loss"], selected["seed"]
        ):
            selected = candidate
        del model

    # Test labels are used in this single evaluation block after all seeded
    # training runs have completed; their metrics never affect model selection.
    for metadata, prediction in zip(repeated_rows, seed_predictions):
        metadata.update(regression_metrics(y_test, prediction))
    repeated_metrics = pd.DataFrame(repeated_rows)
    repeated_metrics.to_csv(repeated_dir / "metrics_per_seed.tsv", sep="\t", index=False)
    repeated_summary = {
        metric: {
            "mean": float(repeated_metrics[metric].mean()),
            "sd": float(repeated_metrics[metric].std(ddof=1)),
            "n_independent_runs": int(len(repeated_metrics)),
        }
        for metric in ["pearson_r", "spearman_rho", "r2", "mae", "rmse"]
    }
    save_json(repeated_summary, repeated_dir / "metrics_mean_sd.json")

    ensemble_prediction = np.mean(np.asarray(seed_predictions), axis=0)
    ensemble_uncertainty = np.std(np.asarray(seed_predictions), axis=0, ddof=1)
    ensemble_metrics = regression_metrics(y_test, ensemble_prediction)
    ensemble_intervals = bootstrap_confidence_intervals(
        y_test, ensemble_prediction, args.bootstrap_replicates, args.seed + 2000
    )
    ensemble_frame = test[["source_row", "sequence", "value"]].copy()
    ensemble_frame["prediction_mean"] = ensemble_prediction
    ensemble_frame["deep_ensemble_prediction_sd"] = ensemble_uncertainty
    ensemble_frame["residual"] = y_test - ensemble_prediction
    ensemble_frame.to_csv(repeated_dir / "ensemble_test_predictions.tsv", sep="\t", index=False)
    save_json(
        {
            "metrics": ensemble_metrics,
            "stratified_bootstrap_95_ci": ensemble_intervals,
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
            "spearman_uncertainty_vs_absolute_error": float(
                spearmanr(ensemble_uncertainty, np.abs(y_test - ensemble_prediction))[0]
            ),
        },
        repeated_dir / "uncertainty_summary.json",
    )
    save_scatter(
        y_test,
        ensemble_prediction,
        repeated_dir / "ensemble_independent_test_scatter.png",
        "Deep-ensemble independent 20% test set",
    )
    save_residual_plot(
        y_test, ensemble_prediction, repeated_dir / "ensemble_independent_test_residuals.png"
    )

    if selected is None:
        raise AssertionError("No final-training model was produced")
    final_best_epoch = selected["best_epoch"]
    final_model = refit_model(
        tf,
        x_development,
        transform_values(y_development, final_standard, final_minmax),
        selected["seed"],
        final_best_epoch,
        args.batch_size,
        args.verbose,
    )
    selected_scaled_prediction = final_model.predict(
        x_test, batch_size=args.batch_size, verbose=0
    ).reshape(-1)
    test_prediction = inverse_values(
        selected_scaled_prediction, final_standard, final_minmax
    )
    test_metrics = regression_metrics(y_test, test_prediction)
    test_intervals = bootstrap_confidence_intervals(
        y_test, test_prediction, args.bootstrap_replicates, args.seed + 3000
    )

    model_dir = output_dir / "final_model_savedmodel"
    final_model.save(str(model_dir))
    with (output_dir / "target_scalers.pkl").open("wb") as handle:
        pickle.dump({"standard_scaler": final_standard, "minmax_scaler": final_minmax}, handle)
    scaler_parameters = {
        "standard_mean": final_standard.mean_.tolist(),
        "standard_scale": final_standard.scale_.tolist(),
        "minmax_data_min": final_minmax.data_min_.tolist(),
        "minmax_data_max": final_minmax.data_max_.tolist(),
        "minmax_feature_range": [-1, 1],
        "fitted_on": "80% development set only",
    }
    save_json(scaler_parameters, output_dir / "target_scaler_parameters.json")

    test_predictions = test[["source_row", "sequence", "value"]].copy()
    test_predictions["prediction"] = test_prediction
    test_predictions["residual"] = y_test - test_prediction
    test_predictions.to_csv(
        output_dir / "independent_test_predictions.tsv", sep="\t", index=False
    )
    save_scatter(
        y_test,
        test_prediction,
        output_dir / "independent_test_scatter.png",
        "Independent 20% test set",
    )
    save_residual_plot(y_test, test_prediction, output_dir / "independent_test_residuals.png")

    elapsed = time.time() - started
    test_summary = {
        "scope": "untouched independent 20% test set",
        "selected_training_seed": int(selected["seed"]),
        "selection_rule": "minimum inner-validation loss; test metrics were not used",
        "selected_epoch": final_best_epoch,
        "metrics": test_metrics,
        "stratified_bootstrap_95_ci": test_intervals,
    }
    save_json(test_summary, output_dir / "independent_test_summary.json")
    run_configuration = {
        "development_file": str(Path(args.development_file).resolve()),
        "test_file": str(Path(args.test_file).resolve()),
        "split_manifest": str(split_manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "test_fraction": args.test_size,
        "development_fraction": 1.0 - args.test_size,
        "independent_split_quantile_stratified": outer_stratified,
        "outer_folds": args.folds,
        "outer_fold_method": fold_method,
        "inner_validation_fraction": args.inner_validation_size,
        "epochs_maximum": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "elapsed_seconds": elapsed,
        "test_used_for_model_selection": False,
        "target_scaling_fitted_on_test": False,
    }
    save_json(run_configuration, output_dir / "run_configuration.json")

    print("10-fold CV Pearson r: %.4f +/- %.4f" % (
        cv_mean_sd["pearson_r"]["mean"], cv_mean_sd["pearson_r"]["sd"]
    ))
    print("Independent-test Pearson r: %.4f" % test_metrics["pearson_r"])
    print("Results written to %s" % output_dir.resolve())


if __name__ == "__main__":
    main()
