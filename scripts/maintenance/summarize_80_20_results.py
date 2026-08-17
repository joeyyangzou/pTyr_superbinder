#!/usr/bin/env python3
"""Create manuscript- and response-ready summaries from completed CNN runs."""

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
        help="Directory for tables and text drafts",
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


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return ("%%.%df" % digits) % float(value)


def ci_text(estimate, interval, lower_key, upper_key, digits=3):
    return "%s (95%% CI, %s-%s)" % (
        fmt(estimate, digits),
        fmt(interval.get(lower_key), digits),
        fmt(interval.get(upper_key), digits),
    )


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


def require_figures(base, relative_paths):
    return [(relative, (base / relative).is_file()) for relative in relative_paths]


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

    class_mean = class_cv["mean_fold_metrics"]
    class_sd = class_cv["sd_fold_metrics"]
    class_metrics = class_test["independent_test_metrics"]
    class_ci = class_test["bootstrap_95_ci"]
    reg_mean_sd = reg_cv["fold_mean_and_sd"]
    reg_metrics = reg_test["metrics"]
    reg_ci = reg_test["stratified_bootstrap_95_ci"]

    class_total_seconds = float(class_run.get("elapsed_seconds", 0.0))
    if class_total_seconds <= 0.0:
        class_total_seconds = sum(
            float(class_test.get(key, 0.0))
            for key in (
                "selection_training_seconds",
                "refit_training_seconds",
                "independent_test_prediction_seconds",
            )
        )
    regression_total_seconds = float(reg_run.get("elapsed_seconds", 0.0))

    response = """# Evaluation response draft

## Expanded metrics, calibration, confidence intervals, and repeated evaluation

**Response:** Thank you for this suggestion. We revised the random-split
evaluation to prevent test-set leakage. A fixed 20% test set was removed before
model development. Ten-fold outer cross-validation was performed exclusively
within the remaining 80% development set. For every outer fold, early stopping
used a separate inner validation subset drawn only from the outer training
partition. The outer validation fold was not used for early stopping. For the
final analysis, the epoch was selected within the development set, a newly
initialized CNN was refitted using all development observations, and the fixed
20% test set was evaluated once. Regression target scalers were also fitted
using training data only.

For regression, the fixed 4:1 split followed the original systematic procedure:
sequences were ordered by descending log(ratio), consecutive blocks of five
were formed, one sequence per block was assigned to the independent test set,
and the remaining sequences formed the development set. Ten-fold evaluation
and inner early stopping were subsequently restricted to that development set.

For classification, the mean AUROC across the ten development-set folds was
{class_cv_auc} +/- {class_cv_auc_sd}. On the untouched 20% test set, AUROC was
{class_auc_ci}, AUPRC was {class_auprc_ci}, F1 was {class_f1_ci}, and MCC was
{class_mcc_ci}. The Brier score was {class_brier_ci}, and the expected
calibration error was {class_ece_ci}. We added ROC and precision-recall curves,
a reliability diagram, and stratified-bootstrap 95% confidence intervals. The
classifier produces raw sigmoid probabilities. Platt scaling was fitted using
pooled out-of-fold predictions from the development set only; no test data were
used to fit the calibrator or select the classification threshold.

Ten independently initialized final-training runs gave a classification AUROC
of {class_seed_auc} +/- {class_seed_auc_sd}. The calibrated deep-ensemble AUROC
was {class_ensemble_auc}; the mean standard deviation across seed-model
probabilities was {class_uncertainty} and was reported as epistemic model
disagreement.

For regression, the mean Pearson correlation across the ten development-set
folds was {reg_cv_pearson} +/- {reg_cv_pearson_sd}. On the untouched test set,
Pearson r was {reg_pearson_ci}, Spearman rho was {reg_spearman_ci}, MAE was
{reg_mae_ci}, and RMSE was {reg_rmse_ci}. We also added observed-versus-predicted
and residual plots and stratified-bootstrap 95% confidence intervals.

Across ten independently initialized regression runs, Pearson r was
{reg_seed_pearson} +/- {reg_seed_pearson_sd}. The deep-ensemble Pearson r was
{reg_ensemble_pearson}; mean model-disagreement SD was {reg_uncertainty}.

The complete classification and regression evaluation workflows required
{class_runtime} and {reg_runtime} seconds, respectively, in the recorded
software environment. Per-fold timing information and the complete run
configurations are supplied with the output.

## Related variants and optimistic random splits

**Response:** We agree that a random split among related library variants tests
within-library interpolation and may overestimate performance on dissimilar
sequence regions. We therefore describe the 80:20/10-fold results explicitly
as the primary within-library evaluation. Exact duplicate sequences are
rejected before splitting, and the generated split manifests confirm zero
development/test row overlap. We do not claim that this random partition is a
homology-separated extrapolation test.
""".format(
        class_cv_auc=fmt(class_mean["AUROC"]),
        class_cv_auc_sd=fmt(class_sd["AUROC"]),
        class_auc_ci=ci_text(
            class_metrics["AUROC"], class_ci["AUROC"], "ci_2.5", "ci_97.5"
        ),
        class_auprc_ci=ci_text(
            class_metrics["AUPRC"], class_ci["AUPRC"], "ci_2.5", "ci_97.5"
        ),
        class_f1_ci=ci_text(class_metrics["F1"], class_ci["F1"], "ci_2.5", "ci_97.5"),
        class_mcc_ci=ci_text(
            class_metrics["MCC"], class_ci["MCC"], "ci_2.5", "ci_97.5"
        ),
        class_brier_ci=ci_text(
            class_metrics["Brier"], class_ci["Brier"], "ci_2.5", "ci_97.5"
        ),
        class_ece_ci=ci_text(
            class_metrics["ECE"], class_ci["ECE"], "ci_2.5", "ci_97.5"
        ),
        class_seed_auc=fmt(class_repeated["AUROC"]["mean"]),
        class_seed_auc_sd=fmt(class_repeated["AUROC"]["sd"]),
        class_ensemble_auc=fmt(class_ensemble["metrics"]["AUROC"]),
        class_uncertainty=fmt(class_uncertainty["mean_predictive_sd"]),
        reg_cv_pearson=fmt(reg_mean_sd["pearson_r"]["mean"]),
        reg_cv_pearson_sd=fmt(reg_mean_sd["pearson_r"]["sd"]),
        reg_pearson_ci=ci_text(
            reg_metrics["pearson_r"], reg_ci["pearson_r"], "lower_95", "upper_95"
        ),
        reg_spearman_ci=ci_text(
            reg_metrics["spearman_rho"],
            reg_ci["spearman_rho"],
            "lower_95",
            "upper_95",
        ),
        reg_mae_ci=ci_text(reg_metrics["mae"], reg_ci["mae"], "lower_95", "upper_95"),
        reg_rmse_ci=ci_text(
            reg_metrics["rmse"], reg_ci["rmse"], "lower_95", "upper_95"
        ),
        reg_seed_pearson=fmt(reg_repeated["pearson_r"]["mean"]),
        reg_seed_pearson_sd=fmt(reg_repeated["pearson_r"]["sd"]),
        reg_ensemble_pearson=fmt(reg_ensemble["metrics"]["pearson_r"]),
        reg_uncertainty=fmt(reg_uncertainty["mean_predictive_sd"]),
        class_runtime=fmt(class_total_seconds, 1),
        reg_runtime=fmt(regression_total_seconds, 1),
    )
    (output_dir / "evaluation_response_draft.md").write_text(response, encoding="utf-8")

    manuscript = """# Manuscript text draft

## Methods: model evaluation

The classification dataset was divided once into an 80% development set and
an untouched 20% test set using seed {seed}. The regression dataset was sorted
by descending log(ratio), partitioned into consecutive blocks of five, and one
sequence per block was assigned to the independent test set; the other four
were assigned to the development set. Ten-fold outer cross-validation was conducted only
within the development set. In each outer fold, an inner validation subset was
drawn from the outer training partition and used for early stopping. After the
optimal epoch was selected, a newly initialized CNN was refitted on the full
outer training partition and evaluated on the held-out outer fold. For final
testing, epoch selection was performed within the development set, after which
a newly initialized model was trained on the complete development set and
evaluated once on the independent test set. Regression target transformations
were fitted using training data only. Test-set confidence intervals were
estimated using 2,000 stratified bootstrap replicates.

## Results: classification

The classifier achieved a mean AUROC of {cv_auc} +/- {cv_auc_sd} across the ten
outer cross-validation folds conducted within the 80% development set. On the
untouched 20% test set, AUROC was {test_auc}, AUPRC was {test_auprc}, F1 was
{test_f1}, and MCC was {test_mcc}. The Brier score was {test_brier}, and ECE was
{test_ece}. Calibration was assessed using the Brier score, ECE, and a
reliability diagram. Final probabilities were Platt calibrated using only
development-set out-of-fold predictions. Ten independently initialized runs
and the corresponding deep-ensemble model-disagreement SD were also reported.

## Results: regression

Across the ten outer development-set folds, mean Pearson r was {cv_pearson} +/-
{cv_pearson_sd} and mean Spearman rho was {cv_spearman} +/- {cv_spearman_sd}.
On the untouched test set, Pearson r was {test_pearson}, Spearman rho was
{test_spearman}, MAE was {test_mae}, and RMSE was {test_rmse}. Residuals were
examined graphically as a function of the predicted value. Ten independently
initialized regression runs were summarized as mean +/- SD, and prediction SD
across the ten models was reported as deep-ensemble epistemic disagreement.

## Interpretation of split designs

The random 80:20 analysis evaluates interpolation within the mutational
library. Because related variants can occur across a random split, these
results should not be interpreted as direct evidence of extrapolation to
dissimilar sequence regions.
""".format(
        seed=class_split["split_seed"],
        cv_auc=fmt(class_mean["AUROC"]),
        cv_auc_sd=fmt(class_sd["AUROC"]),
        test_auc=ci_text(
            class_metrics["AUROC"], class_ci["AUROC"], "ci_2.5", "ci_97.5"
        ),
        test_auprc=ci_text(
            class_metrics["AUPRC"], class_ci["AUPRC"], "ci_2.5", "ci_97.5"
        ),
        test_f1=ci_text(class_metrics["F1"], class_ci["F1"], "ci_2.5", "ci_97.5"),
        test_mcc=ci_text(class_metrics["MCC"], class_ci["MCC"], "ci_2.5", "ci_97.5"),
        test_brier=ci_text(
            class_metrics["Brier"], class_ci["Brier"], "ci_2.5", "ci_97.5"
        ),
        test_ece=ci_text(
            class_metrics["ECE"], class_ci["ECE"], "ci_2.5", "ci_97.5"
        ),
        cv_pearson=fmt(reg_mean_sd["pearson_r"]["mean"]),
        cv_pearson_sd=fmt(reg_mean_sd["pearson_r"]["sd"]),
        cv_spearman=fmt(reg_mean_sd["spearman_rho"]["mean"]),
        cv_spearman_sd=fmt(reg_mean_sd["spearman_rho"]["sd"]),
        test_pearson=ci_text(
            reg_metrics["pearson_r"], reg_ci["pearson_r"], "lower_95", "upper_95"
        ),
        test_spearman=ci_text(
            reg_metrics["spearman_rho"],
            reg_ci["spearman_rho"],
            "lower_95",
            "upper_95",
        ),
        test_mae=ci_text(reg_metrics["mae"], reg_ci["mae"], "lower_95", "upper_95"),
        test_rmse=ci_text(
            reg_metrics["rmse"], reg_ci["rmse"], "lower_95", "upper_95"
        ),
    )
    (output_dir / "manuscript_text_draft.md").write_text(manuscript, encoding="utf-8")

    figure_entries = require_figures(
        classification_dir,
        [
            "cross_validation/ten_fold_cv_roc.png",
            "cross_validation/ten_fold_cv_precision_recall.png",
            "final_model/independent_test_roc.png",
            "final_model/independent_test_precision_recall.png",
            "final_model/independent_test_reliability.png",
            "final_model/independent_test_reliability.tsv",
            "repeated_training/metrics_per_seed.csv",
            "repeated_training/ensemble_test_predictions.tsv",
            "repeated_training/ensemble_reliability.png",
            "repeated_training/uncertainty_summary.json",
        ],
    ) + require_figures(
        regression_dir,
        [
            "cross_validation_oof_scatter.png",
            "independent_test_scatter.png",
            "independent_test_residuals.png",
            "repeated_training/metrics_per_seed.tsv",
            "repeated_training/ensemble_test_predictions.tsv",
            "repeated_training/ensemble_independent_test_scatter.png",
            "repeated_training/ensemble_independent_test_residuals.png",
            "repeated_training/uncertainty_summary.json",
        ],
    )
    inventory_lines = [
        "# Figure and file inventory",
        "",
        "Copy or cite the following generated files in the main text or supplement:",
        "",
    ]
    for relative, exists in figure_entries:
        inventory_lines.append("- [%s] `%s`" % ("present" if exists else "MISSING", relative))
    inventory_lines.extend(
        [
            "",
            "The classification reliability TSV contains the plotted bin-level values.",
            "The split manifests and run-configuration JSON files should be archived with the results.",
        ]
    )
    (output_dir / "supplementary_file_inventory.md").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )

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

    print("Evaluation summary files written to %s" % output_dir)
    print("Metrics table: %s" % table_path)


if __name__ == "__main__":
    main()
