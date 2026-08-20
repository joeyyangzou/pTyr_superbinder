#!/usr/bin/env python3
"""Validate the ANCHOR public workflow package."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require(relative):
    path = ROOT / relative
    if not path.exists():
        raise AssertionError(f"Missing required artifact: {relative}")
    return path


def read_json(relative):
    with require(relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def public_paths():
    for path in ROOT.rglob("*"):
        if ".git" not in path.relative_to(ROOT).parts:
            yield path


def main():
    required = [
        "README.md",
        "VERSION",
        "config/model_hyperparameters.json",
        "environment/environment.yml",
        "environment/requirements_analysis.txt",
        "docs/END_TO_END_WORKFLOW.md",
        "src/ngs_preprocessing/5.fastq2fasta.sh",
        "src/ngs_preprocessing/stat_uniq_pep_num.pl",
        "src/dataset_preparation/split_train_test.py",
        "src/dataset_preparation/06_train_test_split.py",
        "src/model_training/03_CNN_classification.py",
        "src/model_training/14_make_fixed_test_hamming_buffer.py",
        "src/model_training/07_CNN_regression.py",
        "src/prediction/classification_Multi-thread_new.py",
        "src/prediction/regression_multi_thread.py",
        "data/processed/classification/positive.tsv",
        "data/processed/classification/negative.tsv",
        "data/processed/regression/regression_dataset.tsv",
        "models/latest_models/classification/saved_model/saved_model.pb",
        "models/latest_models/classification/platt_calibration.json",
        "models/latest_models/regression/saved_model/saved_model.pb",
        "results/holdout_10fold_analysis/summary/evaluation_metrics_summary.tsv",
        "results/hamming_buffer_sensitivity/buffer_partitions/split_manifest.json",
        "results/hamming_buffer_sensitivity/repeated_training/ensemble_summary.json",
        "results/hamming_buffer_sensitivity/summary/Supplementary_Table_S6_hamming_buffer.tsv",
        "results/hamming_buffer_sensitivity/summary/Supplementary_Figure_S6_hamming_buffer.png",
        "docs/FIXED_TEST_HAMMING_BUFFER.md",
        "run_80_20_10fold_analysis.sh",
        "run_fixed_test_hamming_buffer.sh",
        "check_80_20_splits.sh",
    ]
    for relative in required:
        require(relative)

    if require("VERSION").read_text(encoding="utf-8").strip() != "1.2.1":
        raise AssertionError("VERSION must be 1.2.1")

    model_files = [
        path.relative_to(ROOT).as_posix()
        for path in public_paths()
        if path.is_file() and path.suffix.lower() in {".pb", ".h5", ".keras"}
    ]
    expected_models = sorted(
        [
            "models/latest_models/classification/saved_model/saved_model.pb",
            "models/latest_models/regression/saved_model/saved_model.pb",
        ]
    )
    if sorted(model_files) != expected_models:
        raise AssertionError(
            "Only the two latest SavedModels may be distributed; observed "
            f"model files: {sorted(model_files)}"
        )

    classification_manifest = read_json(
        "models/latest_models/classification/single_model_manifest.json"
    )
    regression_manifest = read_json(
        "models/latest_models/regression/single_model_manifest.json"
    )
    if classification_manifest["task"] != "classification":
        raise AssertionError("Unexpected classification model manifest")
    if regression_manifest["task"] != "regression":
        raise AssertionError("Unexpected regression model manifest")
    if classification_manifest.get("selected_training_seed") != 5:
        raise AssertionError("The distributed classifier must be the selected seed-5 model")
    if classification_manifest.get("selected_epoch") != 132:
        raise AssertionError("The distributed classifier must use the selected 132 epochs")
    calibration = classification_manifest.get("calibration")
    if calibration != "models/latest_models/classification/platt_calibration.json":
        raise AssertionError("Unexpected classification calibration manifest")
    calibration_parameters = read_json(calibration)
    if calibration_parameters.get("fitted_on") != "pooled OOF predictions from the 80% development set":
        raise AssertionError("Classification calibration must be development-only")
    if regression_manifest.get("selected_training_seed") != 8:
        raise AssertionError("The distributed regressor must be the selected seed-8 model")
    if regression_manifest.get("selected_epoch") != 199:
        raise AssertionError("The distributed regressor must use the selected 199 epochs")
    require("models/latest_models/regression/target_scaler.json")

    hamming_manifest = read_json(
        "results/hamming_buffer_sensitivity/buffer_partitions/split_manifest.json"
    )
    if hamming_manifest.get("split_mode") != "fixed_test_hamming_buffer":
        raise AssertionError("Unexpected Hamming-buffer split mode")
    if hamming_manifest.get("development_rows") != 3452:
        raise AssertionError("Unexpected Hamming-buffer development-set size")
    if hamming_manifest.get("independent_test_rows") != 3384:
        raise AssertionError("Unexpected fixed independent-test size")
    hamming_audit = hamming_manifest.get("hamming_audit", {})
    if hamming_audit.get("minimum_development_test_hamming") != 2:
        raise AssertionError("Hamming-buffer minimum distance must be 2")
    if hamming_audit.get("test_sequences_with_development_neighbor_hamming_le_1") != 0:
        raise AssertionError("A distance-0/1 development-test pair remains")
    if hamming_manifest.get("test_resampled") is not False:
        raise AssertionError("The Hamming-buffer test set must remain frozen")

    hamming_summary = read_json(
        "results/hamming_buffer_sensitivity/repeated_training/ensemble_summary.json"
    )
    if hamming_summary.get("n_training_seeds") != 10:
        raise AssertionError("The Hamming-buffer ensemble must contain ten seeds")
    if round(hamming_summary["metrics"]["AUROC"], 3) != 0.950:
        raise AssertionError("Unexpected Hamming-buffer AUROC")

    excluded_public_terms = [
        "review" + "er",
        "re" + "produc",
        "homo" + "log",
        "leg" + "acy",
        "manu" + "script",
    ]
    prohibited_names = []
    for path in public_paths():
        relative = path.relative_to(ROOT).as_posix().lower()
        if any(term in relative for term in excluded_public_terms):
            prohibited_names.append(relative)
    if prohibited_names:
        raise AssertionError(f"Excluded public names remain: {prohibited_names}")

    oversized = [
        path.relative_to(ROOT)
        for path in public_paths()
        if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"Files at or above 100 MB: {oversized}")

    forbidden_fragments = [
        "/home/yangzou/",
        "C:\\Users\\yangzou",
        "D:\\github_repository\\",
        *excluded_public_terms,
    ]
    text_suffixes = {".md", ".json", ".yml", ".yaml", ".txt", ".py", ".sh", ".pl"}
    leaked = []
    for path in public_paths():
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(fragment.lower() in content.lower() for fragment in forbidden_fragments):
            leaked.append(str(path.relative_to(ROOT)))
    if leaked:
        raise AssertionError(f"Private paths or excluded wording remain in: {leaked}")

    print("ANCHOR release validation: PASS")
    print(f"Root: {ROOT}")
    print("Primary 80:20, ten-fold, repeated-seed, and bootstrap outputs: verified")
    print("Fixed-test Hamming-buffer sensitivity outputs: verified")
    print("Distributed models: one classifier and one regressor for downstream inference")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ANCHOR release validation: FAIL: {exc}", file=sys.stderr)
        raise
