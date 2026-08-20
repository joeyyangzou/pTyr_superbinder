#!/usr/bin/env python3
"""Summarize primary versus fixed-test Hamming-buffer classification results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-results-dir",
        default=(
            "results/holdout_10fold_analysis/"
            "classification_80_20_10fold_results"
        ),
    )
    parser.add_argument(
        "--hamming-results-dir",
        default="results/hamming_buffer_sensitivity_rerun/classification_results",
    )
    parser.add_argument(
        "--hamming-split-manifest",
        default=(
            "results/hamming_buffer_sensitivity_rerun/"
            "buffer_partitions/split_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/hamming_buffer_sensitivity_rerun/summary",
    )
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ci_text(summary, metric):
    item = summary["bootstrap_95_ci"][metric]
    return "%.3f (%.3f-%.3f)" % (
        float(item["estimate"]),
        float(item["ci_2.5"]),
        float(item["ci_97.5"]),
    )


def load_predictions(results_dir):
    path = Path(results_dir) / "repeated_training" / "ensemble_test_predictions.tsv"
    frame = pd.read_csv(path, sep="\t")
    required = {"sequence", "label", "calibrated_probability_mean"}
    if not required.issubset(frame.columns):
        raise ValueError("Unexpected ensemble prediction columns in %s" % path)
    return frame


def main():
    args = parse_args()
    primary_dir = Path(args.primary_results_dir)
    hamming_dir = Path(args.hamming_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_summary = load_json(primary_dir / "repeated_training" / "ensemble_summary.json")
    hamming_summary = load_json(hamming_dir / "repeated_training" / "ensemble_summary.json")
    hamming_manifest = load_json(args.hamming_split_manifest)
    primary_manifest = load_json(primary_dir / "splits" / "split_manifest.json")

    primary_predictions = load_predictions(primary_dir)
    hamming_predictions = load_predictions(hamming_dir)
    primary_pairs = set(zip(primary_predictions["sequence"], primary_predictions["label"].astype(int)))
    hamming_pairs = set(zip(hamming_predictions["sequence"], hamming_predictions["label"].astype(int)))
    if primary_pairs != hamming_pairs:
        raise ValueError("The primary and Hamming-buffer analyses did not use the same test set")

    rows = []
    for name, summary, n_development, n_test, minimum_hamming, excluded in [
        (
            "Primary random 80:20",
            primary_summary,
            int(primary_manifest["n_development"]),
            int(primary_manifest["n_independent_test"]),
            1,
            0,
        ),
        (
            "Fixed-test Hamming buffer",
            hamming_summary,
            int(hamming_manifest["development_rows"]),
            int(hamming_manifest["independent_test_rows"]),
            int(hamming_manifest["hamming_audit"]["minimum_development_test_hamming"]),
            int(hamming_manifest["excluded_rows"]),
        ),
    ]:
        row = {
            "analysis": name,
            "n_development": n_development,
            "n_independent_test": n_test,
            "minimum_development_test_Hamming": minimum_hamming,
            "n_excluded_by_Hamming_buffer": excluded,
        }
        for metric in ["AUROC", "AUPRC", "Accuracy", "F1", "MCC", "Brier", "ECE"]:
            row[metric + "_95CI"] = ci_text(summary, metric)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(
        output_dir / "Supplementary_Table_S6_hamming_buffer.tsv",
        sep="\t",
        index=False,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#1f77b4", "#d62728"]
    for name, predictions, summary, color in [
        ("Primary random", primary_predictions, primary_summary, colors[0]),
        ("Hamming buffer", hamming_predictions, hamming_summary, colors[1]),
    ]:
        labels = predictions["label"].to_numpy(dtype=int)
        probabilities = predictions["calibrated_probability_mean"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label="%s (AUROC %.3f)" % (name, summary["metrics"]["AUROC"]),
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label="%s (AUPRC %.3f)" % (name, summary["metrics"]["AUPRC"]),
        )
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC comparison")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall comparison")
    for axis in axes:
        axis.legend(loc="best", fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output_dir / "Supplementary_Figure_S6_hamming_buffer.png", dpi=300)
    figure.savefig(output_dir / "Supplementary_Figure_S6_hamming_buffer.pdf")
    plt.close(figure)

    primary_auc = primary_summary["metrics"]["AUROC"]
    hamming_auc = hamming_summary["metrics"]["AUROC"]
    primary_accuracy = primary_summary["metrics"]["Accuracy"]
    hamming_accuracy = hamming_summary["metrics"]["Accuracy"]
    text = """# Fixed-test Hamming-buffer result

The frozen independent test set was identical in both analyses (n = {n_test}).
After excluding every non-test sequence within Hamming distance 1 of a test
sequence, {n_development} development sequences remained and {n_excluded}
sequences were excluded. The minimum development-test Hamming distance was
{minimum_hamming}.

The ten-seed ensemble achieved AUROC = {hamming_auc}, compared with
{primary_auc} in the primary random-split analysis (delta AUROC = {delta_auc}).
Accuracy was {hamming_accuracy}, compared with {primary_accuracy} in the
primary analysis. Bootstrap confidence intervals and additional metrics are
reported in Supplementary_Table_fixed_test_hamming_buffer.tsv.

Interpretation: the test distribution was held constant, but Hamming filtering
substantially reduced development-set coverage. Therefore, the difference
reflects both removal of near-neighbor information and reduced training data.
""".format(
        n_test=hamming_manifest["independent_test_rows"],
        n_development=hamming_manifest["development_rows"],
        n_excluded=hamming_manifest["excluded_rows"],
        minimum_hamming=hamming_manifest["hamming_audit"]["minimum_development_test_hamming"],
        hamming_auc="%.3f" % hamming_auc,
        primary_auc="%.3f" % primary_auc,
        delta_auc="%.3f" % (hamming_auc - primary_auc),
        hamming_accuracy="%.3f" % hamming_accuracy,
        primary_accuracy="%.3f" % primary_accuracy,
    )
    (output_dir / "analysis_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
