#!/usr/bin/env python3
"""Validate the ANCHOR public reproducibility package."""

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


def main():
    required = [
        "README.md",
        "VERSION",
        "config/model_hyperparameters.json",
        "config/robustness_run_configuration.json",
        "environment/environment.yml",
        "environment/requirements_analysis.txt",
        "src/robustness_analysis/10_robustness_analysis.py",
        "src/robustness_analysis/robustness_utils.py",
        "data/processed/classification/positive.tsv",
        "data/processed/classification/negative.tsv",
        "data/processed/regression/regression_dataset.tsv",
        "models/latest_models/classification/saved_model/saved_model.pb",
        "models/latest_models/regression/saved_model/saved_model.pb",
        "results/robustness_analysis/combined_metrics_for_manuscript.csv",
        "results/robustness_analysis/combined_split_summary.csv",
    ]
    for relative in required:
        require(relative)

    if require("VERSION").read_text(encoding="utf-8").strip() != "1.2.0":
        raise AssertionError("VERSION must be 1.2.0")

    split_expectations = {
        ("classification", "random"): (11844, 1692, 3384, 0, 1),
        ("classification", "hamming"): (2734, 432, 3384, 10370, 2),
        ("regression", "random"): (3153, 451, 902, 0, 1),
        ("regression", "hamming"): (1568, 237, 902, 1799, 2),
    }
    for (task, design), expected in split_expectations.items():
        metadata = read_json(
            f"results/robustness_analysis/{task}/{design}/splits/split_metadata.json"
        )
        observed = (
            metadata["n_train"],
            metadata["n_validation"],
            metadata["n_test"],
            metadata["n_excluded_hamming_buffer"],
            metadata["homology_audit"]["minimum_nearest_hamming_distance"],
        )
        if observed != expected:
            raise AssertionError(
                f"Unexpected {task}/{design} split summary: {observed}; expected {expected}"
            )

    seed_directories = list(
        (ROOT / "results" / "robustness_analysis").glob("*/*/seed_*")
    )
    if len(seed_directories) != 40:
        raise AssertionError(
            f"Expected results for 40 independent training runs; observed {len(seed_directories)}"
        )

    model_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
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
    if classification_manifest["selected_seed"] != 7:
        raise AssertionError("Unexpected selected classification seed")
    if regression_manifest["selected_seed"] != 5:
        raise AssertionError("Unexpected selected regression seed")

    internal_term = "review" + "er"
    prohibited_names = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix().lower()
        if internal_term in relative:
            prohibited_names.append(relative)
    if prohibited_names:
        raise AssertionError(f"Internal response-oriented names remain: {prohibited_names}")

    oversized = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"Files at or above 100 MB: {oversized}")

    forbidden_fragments = [
        "/home/yangzou/",
        "C:\\Users\\yangzou",
        "D:\\项目\\",
        internal_term,
    ]
    text_suffixes = {".md", ".json", ".yml", ".yaml", ".txt", ".py", ".sh", ".pl"}
    leaked = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(fragment.lower() in content.lower() for fragment in forbidden_fragments):
            leaked.append(str(path.relative_to(ROOT)))
    if leaked:
        raise AssertionError(f"Private paths or internal wording remain in: {leaked}")

    print("ANCHOR release validation: PASS")
    print(f"Root: {ROOT}")
    print(f"Independent run result directories: {len(seed_directories)}")
    print("Distributed models: classification seed 7; regression seed 5")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ANCHOR release validation: FAIL: {exc}", file=sys.stderr)
        raise
