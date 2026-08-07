"""Utilities for the robustness and repeated-run analyses.

The functions in this module deliberately have no TensorFlow dependency so
that data splitting, homology auditing and confidence intervals can be tested
on a login node before GPU training is started.
"""

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


AMINO_ACIDS = "ILVFMCAGPTSYWQNHEDKR"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def set_global_seed(seed: int, tf=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        if hasattr(tf.keras.utils, "set_random_seed"):
            tf.keras.utils.set_random_seed(seed)
        else:
            tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def one_hot_encode(sequences) -> np.ndarray:
    seqs = [str(s).strip().upper() for s in sequences]
    if not seqs:
        raise ValueError("No sequences were supplied")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"All sequences must have equal length; observed {sorted(lengths)}")
    encoded = np.zeros((len(seqs), len(seqs[0]), len(AMINO_ACIDS)), dtype=np.float32)
    for row, seq in enumerate(seqs):
        for col, aa in enumerate(seq):
            if aa not in AA_TO_INDEX:
                raise ValueError(f"Unsupported amino acid {aa!r} in sequence {seq!r}")
            encoded[row, col, AA_TO_INDEX[aa]] = 1.0
    return encoded


def _components_within_one_substitution(sequences):
    """Connected components where edges join equal-length sequences at Hamming <= 1."""
    seqs = [str(s) for s in sequences]
    parent = list(range(len(seqs)))
    size = [1] * len(seqs)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    buckets = defaultdict(list)
    for i, seq in enumerate(seqs):
        buckets[(len(seq), "full", seq)].append(i)
        for pos in range(len(seq)):
            buckets[(len(seq), pos, seq[:pos] + seq[pos + 1 :])].append(i)
    for ids in buckets.values():
        first = ids[0]
        for other in ids[1:]:
            union(first, other)
    roots = [find(i) for i in range(len(seqs))]
    remap = {root: j for j, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([remap[root] for root in roots], dtype=int)


def _select_small_components_for_test(groups, target_n, seed):
    """Select whole small components; a giant component is retained for training."""
    counts = Counter(groups)
    rng = np.random.default_rng(seed)
    candidates = list(counts)
    rng.shuffle(candidates)
    # Small components first prevents a single library-wide connected component
    # from consuming the entire holdout set.
    candidates.sort(key=lambda g: counts[g])
    chosen, total = [], 0
    for group in candidates:
        group_n = counts[group]
        if group_n > target_n and len(counts) > 1:
            continue
        if total < target_n:
            chosen.append(group)
            total += group_n
    if not chosen:
        raise ValueError("A homology-separated test set could not be formed")
    return set(chosen)


def split_dataframe(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    split_mode: str,
    test_size: float,
    validation_size: float,
    split_seed: int,
):
    """Return train/validation/test frames with a frozen test split.

    In homology mode, connected components under Hamming distance <= 1 are
    kept intact. For an 8-mer library this corresponds to >=87.5% identity.
    """
    if task not in {"classification", "regression"}:
        raise ValueError(task)
    if split_mode == "random":
        stratify = df[target_col] if task == "classification" else _regression_bins(df[target_col])
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=split_seed, shuffle=True, stratify=stratify
        )
    elif split_mode == "homology":
        groups = _components_within_one_substitution(df["sequence"])
        chosen = _select_small_components_for_test(groups, int(round(len(df) * test_size)), split_seed)
        mask = np.asarray([g in chosen for g in groups])
        train_val, test = df.loc[~mask], df.loc[mask]
    else:
        raise ValueError("split_mode must be 'random' or 'homology'")

    val_fraction_of_remainder = validation_size / (1.0 - test_size)
    stratify_tv = (
        train_val[target_col]
        if task == "classification"
        else _regression_bins(train_val[target_col])
    )
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remainder,
        random_state=split_seed + 1,
        shuffle=True,
        stratify=stratify_tv,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _regression_bins(values, max_bins=10):
    values = pd.Series(values)
    for q in range(min(max_bins, values.nunique()), 1, -1):
        try:
            bins = pd.qcut(values, q=q, labels=False, duplicates="drop")
            counts = pd.Series(bins).value_counts()
            if len(counts) > 1 and counts.min() >= 2:
                return bins
        except ValueError:
            pass
    return None


def homology_audit(train_sequences, test_sequences):
    """Nearest train-set Hamming distance for each test sequence."""
    train = [str(s) for s in train_sequences]
    test = [str(s) for s in test_sequences]
    if not train or not test:
        raise ValueError("Both train and test sequences are required")
    if len({len(s) for s in train + test}) != 1:
        raise ValueError("Homology audit currently requires equal-length sequences")
    alphabet = {aa: i for i, aa in enumerate(sorted(set("".join(train + test))))}
    train_arr = np.asarray([[alphabet[a] for a in s] for s in train], dtype=np.int16)
    minimum = []
    for start in range(0, len(test), 256):
        chunk = np.asarray([[alphabet[a] for a in s] for s in test[start : start + 256]], dtype=np.int16)
        distances = np.count_nonzero(chunk[:, None, :] != train_arr[None, :, :], axis=2)
        minimum.extend(distances.min(axis=1).tolist())
    counts = Counter(minimum)
    return {
        "n_train": len(train),
        "n_test": len(test),
        "exact_overlap_test_sequences": int(sum(d == 0 for d in minimum)),
        "test_sequences_with_train_neighbor_hamming_le_1": int(sum(d <= 1 for d in minimum)),
        "fraction_with_train_neighbor_hamming_le_1": float(np.mean(np.asarray(minimum) <= 1)),
        "nearest_hamming_distance_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "minimum_nearest_hamming_distance": int(min(minimum)),
        "maximum_nearest_hamming_distance": int(max(minimum)),
    }


def minimum_hamming_distances(reference_sequences, query_sequences, chunk_size=256):
    """Return each query's minimum Hamming distance to the reference set."""
    reference = [str(s).strip().upper() for s in reference_sequences]
    query = [str(s).strip().upper() for s in query_sequences]
    if not reference or not query:
        raise ValueError("Both reference and query sequences are required")
    if len({len(s) for s in reference + query}) != 1:
        raise ValueError("Hamming distance requires equal-length sequences")
    alphabet = {aa: i for i, aa in enumerate(sorted(set("".join(reference + query))))}
    reference_array = np.asarray(
        [[alphabet[aa] for aa in sequence] for sequence in reference], dtype=np.int16
    )
    minimum = []
    for start in range(0, len(query), chunk_size):
        chunk = np.asarray(
            [[alphabet[aa] for aa in sequence] for sequence in query[start : start + chunk_size]],
            dtype=np.int16,
        )
        distances = np.count_nonzero(
            chunk[:, None, :] != reference_array[None, :, :], axis=2
        )
        minimum.extend(distances.min(axis=1).tolist())
    return np.asarray(minimum, dtype=int)


def make_robustness_split(
    df,
    target_col,
    task,
    split_mode,
    test_size,
    validation_size,
    split_seed,
    minimum_test_train_hamming=2,
):
    """Create leakage-safe partitions and return split metadata.

    Random mode uses distinct stratified train/validation/test sets. Cluster
    mode keeps Hamming<=1 connected components intact. Hamming mode freezes a
    representative test set and excludes every remaining sequence closer than
    the requested distance to any test sequence. Buffer sequences are never
    used for fitting, early stopping, calibration, or testing.
    """
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")
    if split_mode not in {"random", "cluster", "hamming"}:
        raise ValueError("split_mode must be 'random', 'cluster' or 'hamming'")
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")
    missing = {"sequence", target_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy().reset_index(drop=True)
    data["sequence"] = data["sequence"].astype(str).str.strip().str.upper()
    data["source_row"] = np.arange(len(data), dtype=int)
    duplicates = int(data["sequence"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate sequences; resolve them before splitting")
    excluded = data.iloc[0:0].copy()

    if split_mode == "random":
        train, validation, test = split_dataframe(
            data,
            target_col=target_col,
            task=task,
            split_mode="random",
            test_size=test_size,
            validation_size=validation_size,
            split_seed=split_seed,
        )
    elif split_mode == "cluster":
        groups = _components_within_one_substitution(data["sequence"])
        counts = Counter(groups)
        largest_group = max(counts, key=counts.get)
        candidate_mask = groups != largest_group
        candidate_groups = groups[candidate_mask]
        available_holdout_n = int(np.sum(candidate_mask))
        requested_test_n = int(round(len(data) * test_size))
        requested_validation_n = int(round(len(data) * validation_size))
        if available_holdout_n < requested_test_n + requested_validation_n:
            test_target = max(
                1,
                int(
                    round(
                        available_holdout_n
                        * test_size
                        / (test_size + validation_size)
                    )
                ),
            )
            validation_target = max(1, available_holdout_n - test_target)
        else:
            test_target = requested_test_n
            validation_target = requested_validation_n
        chosen_test = _select_small_components_for_test(
            candidate_groups, test_target, split_seed
        )
        remaining_candidate_groups = np.asarray(
            [group for group in candidate_groups if group not in chosen_test], dtype=int
        )
        if len(remaining_candidate_groups) == 0:
            raise ValueError("No sequence clusters remain for validation")
        chosen_validation = _select_small_components_for_test(
            remaining_candidate_groups,
            min(validation_target, len(remaining_candidate_groups)),
            split_seed + 1,
        )
        test_mask = np.asarray([group in chosen_test for group in groups])
        validation_mask = np.asarray([group in chosen_validation for group in groups])
        train_mask = ~(test_mask | validation_mask)
        train = data.loc[train_mask].reset_index(drop=True)
        validation = data.loc[validation_mask].reset_index(drop=True)
        test = data.loc[test_mask].reset_index(drop=True)
        excluded = data.iloc[0:0].copy()
    else:
        stratify = data[target_col] if task == "classification" else _regression_bins(data[target_col])
        remaining, test = train_test_split(
            data,
            test_size=test_size,
            random_state=split_seed,
            shuffle=True,
            stratify=stratify,
        )
        distance_to_test = minimum_hamming_distances(test["sequence"], remaining["sequence"])
        keep = distance_to_test >= minimum_test_train_hamming
        excluded_test_buffer = remaining.loc[~keep].copy()
        excluded_test_buffer["exclusion_reason"] = "too_close_to_test"
        excluded_test_buffer["nearest_test_hamming_distance"] = distance_to_test[~keep]
        remaining = remaining.loc[keep].copy()
        if len(remaining) < 10:
            raise ValueError(
                "Hamming buffering left too few development sequences; reduce test_size or distance"
            )
        validation_fraction_of_development = validation_size / (1.0 - test_size)
        validation_n = max(1, int(round(len(remaining) * validation_fraction_of_development)))
        validation_n = min(validation_n, max(1, len(remaining) // 3))
        stratify_remaining = (
            remaining[target_col]
            if task == "classification"
            else _regression_bins(remaining[target_col])
        )
        train_candidates, validation = train_test_split(
            remaining,
            test_size=validation_n,
            random_state=split_seed + 1,
            shuffle=True,
            stratify=stratify_remaining,
        )
        distance_to_validation = minimum_hamming_distances(
            validation["sequence"], train_candidates["sequence"]
        )
        keep_train = distance_to_validation >= minimum_test_train_hamming
        excluded_validation_buffer = train_candidates.loc[~keep_train].copy()
        excluded_validation_buffer["exclusion_reason"] = "too_close_to_validation"
        excluded_validation_buffer["nearest_validation_hamming_distance"] = (
            distance_to_validation[~keep_train]
        )
        train = train_candidates.loc[keep_train].copy()
        excluded = pd.concat(
            [excluded_test_buffer, excluded_validation_buffer], ignore_index=True, sort=False
        )
        if len(train) < 10:
            raise ValueError(
                "Hamming buffering around test and validation left too few training sequences"
            )
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)
        excluded = excluded.reset_index(drop=True)

    if task == "classification":
        for partition_name, partition in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items():
            if set(partition[target_col].unique()) != {0, 1}:
                raise ValueError(
                    f"The {partition_name} partition does not contain both classes under {split_mode} splitting"
                )

    audit = homology_audit(train["sequence"], test["sequence"])
    pairwise_audits = {
        "train_vs_test": audit,
        "train_vs_validation": homology_audit(train["sequence"], validation["sequence"]),
        "validation_vs_test": homology_audit(validation["sequence"], test["sequence"]),
    }
    metadata = {
        "split_mode": split_mode,
        "split_seed": int(split_seed),
        "requested_test_fraction": float(test_size),
        "requested_validation_fraction": float(validation_size),
        "minimum_requested_test_train_hamming": (
            int(minimum_test_train_hamming) if split_mode == "hamming" else None
        ),
        "n_total": int(len(data)),
        "n_train": int(len(train)),
        "n_validation": int(len(validation)),
        "n_test": int(len(test)),
        "n_excluded_hamming_buffer": int(len(excluded)),
        "actual_train_fraction": float(len(train) / len(data)),
        "actual_validation_fraction": float(len(validation) / len(data)),
        "actual_test_fraction": float(len(test) / len(data)),
        "homology_audit": audit,
        "pairwise_homology_audits": pairwise_audits,
    }
    if task == "classification":
        metadata["class_counts"] = {
            name: {
                str(key): int(value)
                for key, value in frame[target_col].value_counts().sort_index().items()
            }
            for name, frame in {
                "train": train,
                "validation": validation,
                "test": test,
                "excluded": excluded,
            }.items()
        }
    return train, validation, test, excluded, metadata


def expected_calibration_error(y_true, probability, n_bins=10):
    """Equal-width expected calibration error for binary predictions."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    assignments = np.digitize(probability, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    error = 0.0
    for bin_index in range(n_bins):
        mask = assignments == bin_index
        if np.any(mask):
            error += np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(probability[mask]))
    return float(error)


def classification_metrics(y_true, probability, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, probability),
        "AUPRC": average_precision_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "Brier": brier_score_loss(y_true, probability),
        "ECE": expected_calibration_error(y_true, probability, n_bins=10),
    }


def regression_metrics(y_true, prediction):
    from scipy.stats import pearsonr, spearmanr

    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "Pearson_r": pearsonr(y_true, prediction)[0],
        "Spearman_rho": spearmanr(y_true, prediction)[0],
        "R2": r2_score(y_true, prediction),
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, prediction)),
    }


def bootstrap_confidence_intervals(
    y_true,
    prediction,
    metric_function,
    n_bootstrap,
    seed,
    continuous_strata=10,
):
    """Stratified non-parametric percentile confidence intervals.

    Binary outcomes are resampled within each class. Continuous regression
    outcomes are resampled within target-value quantile strata so that each
    bootstrap replicate retains the approximate test-set response distribution.
    """
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    rng = np.random.default_rng(seed)
    samples = []
    unique = np.unique(y_true)
    is_binary = set(unique).issubset({0, 1}) and len(unique) == 2
    if is_binary:
        strata_indices = [np.flatnonzero(y_true == cls) for cls in (0, 1)]
    else:
        strata = _regression_bins(y_true, max_bins=continuous_strata)
        if strata is None:
            strata_indices = [np.arange(len(y_true))]
        else:
            strata = np.asarray(strata)
            strata_indices = [np.flatnonzero(strata == group) for group in np.unique(strata)]
    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in strata_indices]
        )
        samples.append(metric_function(y_true[idx], prediction[idx]))
    point = metric_function(y_true, prediction)
    rows = []
    for metric, value in point.items():
        distribution = np.asarray([sample[metric] for sample in samples], dtype=float)
        distribution = distribution[np.isfinite(distribution)]
        rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci_2.5%": np.percentile(distribution, 2.5),
                "ci_97.5%": np.percentile(distribution, 97.5),
                "bootstrap_replicates": len(distribution),
            }
        )
    return pd.DataFrame(rows)


def save_json(data, path):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=convert), encoding="utf-8"
    )

