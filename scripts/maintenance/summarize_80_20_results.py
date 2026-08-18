#!/usr/bin/env python3
"""Create a compact metrics table and run record from completed CNN runs."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-dir",
        default="classification_80_20_10fold_results",
        help="Completed CNN_classification.py output directory",
    )
    parser.add_argument(
        "--regression-dir",
        default="regression_80_20_10fold_results",
        help="Completed CNN_regression.py output directory",
    )
    parser.add_argument(
        "--output-dir",
        default="summary",
        help="Directory for the metrics table and run record",
    )
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Required result file was not found: %s" % path)
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sanitize_provenance(value, key=""):
    """Remove machine-specific absolute paths from archived run metadata."""
    if isinstance(value, dict):
        return {name: sanitize_provenance(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize_provenance(item, key) for item in value]
    if isinstance(value, str):
        path_key = any(
            token in key.lower()
            for token in ("file", "dir", "path", "manifest", "input", "output")
        )
        has_path_separator = "/" in value or "\\" in value
        if path_key and has_path_separator:
            basename = value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return "archived-run/" + basename
    return value


def add_row(rows, task, evaluation, metric, estimate, sd=None, lower=None, upper=None):
    rows.append(
        {
            "task": task,
            "evaluation": evaluation,
            "metric": metric,
            "estimate": estimate,
            "sd": sd,
            "ci_95_lower": lower,
            "ci_95_upper": upper,
        }
    )


def main():
    args = parse_args()
    classification_dir = Path(args.classification_dir).resolve()
    regression_dir = Path(args.regression_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_split = read_json(classification_dir / "splits" / "split_manifest.json")
    class_cv = read_json(classification_dir / "cross_validation" / "cv_summary.json")
    class_test = read_json(
        classification_dir / "final_model" / "independent_test_summary.json"
    )
    class_run = read_json(classification_dir / "run_configuration.json")
    class_repeated = read_json(classification_dir / "repeated_training" / "metrics_mean_sd.json")
    class_ensemble = read_json(classification_dir / "repeated_training" / "ensemble_summary.json")
    class_uncertainty = read_json(
        classification_dir / "repeated_training" / "uncertainty_summary.json"
    )

    reg_split = read_json(regression_dir / "split_audit.json")
    reg_cv = read_json(regression_dir / "cross_validation_summary.json")
    reg_test = read_json(regression_dir / "independent_test_summary.json")
    reg_run = read_json(regression_dir / "run_configuration.json")
    reg_repeated = read_json(regression_dir / "repeated_training" / "metrics_mean_sd.json")
    reg_ensemble = read_json(regression_dir / "repeated_training" / "ensemble_summary.json")
    reg_uncertainty = read_json(regression_dir / "repeated_training" / "uncertainty_summary.json")

    if class_split.get("test_used_for_early_stopping") is not False:
        raise ValueError("Classification split audit does not confirm test isolation")
    if class_split.get("outer_fold_used_for_early_stopping") is not False:
        raise ValueError("Classification outer folds were not isolated from early stopping")
    if reg_split.get("test_used_in_cross_validation_or_early_stopping") is not False:
        raise ValueError("Regression split audit does not confirm test isolation")
    if reg_run.get("target_scaling_fitted_on_test") is not False:
        raise ValueError("Regression run does not confirm training-only target scaling")

    rows = []
    for metric, estimate in class_cv["mean_fold_metrics"].items():
        add_row(
            rows,
            "classification",
            "10-fold mean within 80% development set",
            metric,
            estimate,
            class_cv["sd_fold_metrics"][metric],
        )
    for metric, estimate in class_cv["pooled_out_of_fold_metrics"].items():
        add_row(rows, "classification", "pooled out-of-fold", metric, estimate)
    for metric, estimate in class_test["independent_test_metrics"].items():
        interval = class_test["bootstrap_95_ci"].get(metric, {})
        add_row(
            rows,
            "classification",
            "independent 20% test",
            metric,
            estimate,
            lower=interval.get("ci_2.5"),
            upper=interval.get("ci_97.5"),
        )
    for metric, values in class_repeated.items():
        add_row(rows, "classification", "10 independent training seeds", metric, values["mean"], values["sd"])
    for metric, estimate in class_ensemble["metrics"].items():
        interval = class_ensemble["bootstrap_95_ci"].get(metric, {})
        add_row(
            rows, "classification", "deep ensemble on independent test", metric, estimate,
            lower=interval.get("ci_2.5"), upper=interval.get("ci_97.5")
        )

    for metric, values in reg_cv["fold_mean_and_sd"].items():
        add_row(
            rows,
            "regression",
            "10-fold mean within 80% development set",
            metric,
            values["mean"],
            values["sd"],
        )
    for metric, estimate in reg_cv["pooled_out_of_fold_metrics"].items():
        add_row(rows, "regression", "pooled out-of-fold", metric, estimate)
    for metric, estimate in reg_test["metrics"].items():
        interval = reg_test["stratified_bootstrap_95_ci"].get(metric, {})
        add_row(
            rows,
            "regression",
            "independent 20% test",
            metric,
            estimate,
            lower=interval.get("lower_95"),
            upper=interval.get("upper_95"),
        )
    for metric, values in reg_repeated.items():
        add_row(rows, "regression", "10 independent training seeds", metric, values["mean"], values["sd"])
    for metric, estimate in reg_ensemble["metrics"].items():
        interval = reg_ensemble["stratified_bootstrap_95_ci"].get(metric, {})
        add_row(
            rows, "regression", "deep ensemble on independent test", metric, estimate,
            lower=interval.get("lower_95"), upper=interval.get("upper_95")
        )

    table_path = output_dir / "evaluation_metrics_summary.tsv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "evaluation",
                "metric",
                "estimate",
                "sd",
                "ci_95_lower",
                "ci_95_upper",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "classification_result_directory": classification_dir.name,
        "regression_result_directory": regression_dir.name,
        "classification_split": sanitize_provenance(class_split),
        "regression_split": sanitize_provenance(reg_split),
        "classification_run_configuration": sanitize_provenance(class_run),
        "regression_run_configuration": sanitize_provenance(reg_run),
    }
    with (output_dir / "result_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    print("Evaluation summary written to %s" % output_dir)
    print("Metrics table: %s" % table_path)


if __name__ == "__main__":
    main()
