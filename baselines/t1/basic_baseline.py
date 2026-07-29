#!/usr/bin/env python3
"""T1 Basic Baselines -- Majority Class and Random Prior.

Computes two trivial baselines for the Pre-Market Interest Forecasting task:

1. **Majority class**: always predict the most common training label.
2. **Random prior**: sample predictions from the empirical class distribution;
   reports the expected macro-F1 analytically.

Usage:
    python baselines/t1/basic_baseline.py
    python baselines/t1/basic_baseline.py --local-dir ./data

When a validation split is available, both validation and test metrics are
reported.  Legacy datasets with only train/test continue to work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1 basic baselines (majority / random)")
    parser.add_argument("--local-dir", default=None, help="Local data directory (skips HF)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output predictions JSONL path")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    local_dir: Optional[str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Load train, optional validation, and test without changing loader API.

    ``eventxbench.load_task("t1")`` intentionally continues to return the
    historical ``(train, test)`` pair.  Baselines opt in to validation by
    asking for it explicitly.
    """
    import eventxbench

    kwargs = {"local_dir": local_dir} if local_dir else {}
    train_df = eventxbench.load_task("t1", split="train", **kwargs)
    test_df = eventxbench.load_task("t1", split="test", **kwargs)
    try:
        validation_df = eventxbench.load_task("t1", split="validation", **kwargs)
    except KeyError:
        validation_df = None
    except ValueError as exc:
        if "has no 'validation' split" not in str(exc):
            raise
        validation_df = None

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise TypeError("T1 train/test splits must load as pandas DataFrames")
    if validation_df is not None and not isinstance(validation_df, pd.DataFrame):
        raise TypeError("T1 validation split must load as a pandas DataFrame")
    return train_df, validation_df, test_df


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def majority_baseline(
    train_labels: pd.Series, test_labels: pd.Series
) -> dict[str, float | str]:
    """Always predict the most frequent training label."""
    majority_label = train_labels.value_counts().idxmax()
    preds = [majority_label] * len(test_labels)
    return {
        "majority_label": majority_label,
        "accuracy": accuracy_score(test_labels, preds),
        "macro_f1": f1_score(
            test_labels, preds, labels=LABEL_ORDER, average="macro", zero_division=0
        ),
    }


def random_prior_expected_f1(label_counts: dict[str, int]) -> dict[str, float]:
    """Compute analytical expected macro-F1 under random-prior prediction.

    When predictions are sampled i.i.d. from the empirical class distribution,
    the expected F1 for class c equals p(c) (since precision = recall = p(c)),
    so expected macro-F1 = mean of class priors.
    """
    total = sum(label_counts.values())
    if total == 0:
        return {"expected_accuracy": 0.0, "expected_macro_f1": 0.0}

    priors = {k: float(v / total) for k, v in label_counts.items()}
    expected_acc = sum(p ** 2 for p in priors.values())
    expected_macro_f1 = sum(priors.values()) / len(priors)

    return {
        "class_priors": priors,
        "expected_accuracy": expected_acc,
        "expected_macro_f1": expected_macro_f1,
    }


def empirical_random_baseline(
    train_labels: pd.Series,
    target_labels: pd.Series,
    seed: int,
) -> dict[str, float]:
    """Sample from the training prior and evaluate on one named split."""
    if len(target_labels) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    priors = train_labels.value_counts(normalize=True)
    rng = np.random.RandomState(seed)
    preds = rng.choice(priors.index, size=len(target_labels), p=priors.values)
    return {
        "accuracy": accuracy_score(target_labels, preds),
        "macro_f1": f1_score(
            target_labels,
            preds,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        ),
    }


def _validated_labels(df: pd.DataFrame, split_name: str) -> pd.Series:
    if "interest_label" not in df.columns:
        raise ValueError(f"T1 {split_name} split has no 'interest_label' column")
    labels = df["interest_label"].astype(str)
    invalid = sorted(set(labels) - set(LABEL_ORDER))
    if invalid:
        raise ValueError(f"T1 {split_name} split has unknown labels: {invalid}")
    return labels


def _print_split_results(
    split_name: str,
    train_labels: pd.Series,
    target_labels: pd.Series,
    seed: int,
) -> None:
    majority = majority_baseline(train_labels, target_labels)
    random_result = empirical_random_baseline(train_labels, target_labels, seed)

    print("=" * 50)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 50)
    print(f"  Majority label:             {majority['majority_label']}")
    print(f"  Majority accuracy:          {majority['accuracy']:.4f}")
    print(f"  Majority macro-F1:          {majority['macro_f1']:.4f}")
    print(f"  Random-prior accuracy:      {random_result['accuracy']:.4f}")
    print(f"  Random-prior macro-F1:      {random_result['macro_f1']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_preds(path: Path, condition_ids: list[str], labels: list[str],
                  scores: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid, lbl, sc in zip(condition_ids, labels, scores):
            f.write(json.dumps({
                "condition_id": cid, "label": lbl, "scores": sc,
            }, ensure_ascii=False) + "\n")


def _build_scores(priors: dict[str, float], preds: list[str]) -> list[dict[str, float]]:
    return [{k: (1.0 if k == p else 0.0) for k in LABEL_ORDER} for p in preds]


def main() -> None:
    args = parse_args()

    print("Loading T1 data...")
    train_df, validation_df, test_df = load_data(args.local_dir)

    train_labels = _validated_labels(train_df, "train")
    test_labels = _validated_labels(test_df, "test")
    validation_labels = (
        _validated_labels(validation_df, "validation")
        if validation_df is not None
        else None
    )

    split_sizes = [f"Train: {len(train_labels)}"]
    if validation_labels is not None:
        split_sizes.append(f"Validation: {len(validation_labels)}")
    split_sizes.append(f"Test: {len(test_labels)}")
    print("  ".join(split_sizes))
    print(f"Train distribution:\n{train_labels.value_counts().to_string()}\n")
    if validation_labels is not None:
        print(
            "Validation distribution:\n"
            f"{validation_labels.value_counts().to_string()}\n"
        )
    print(f"Test distribution:\n{test_labels.value_counts().to_string()}\n")

    # The analytical expectation depends only on the fitted training prior.
    counts = dict(train_labels.value_counts())
    rp = random_prior_expected_f1(counts)
    print("=" * 50)
    print("RANDOM PRIOR BASELINE (analytical expectation)")
    print("=" * 50)
    print(f"  Class priors:       {rp.get('class_priors', {})}")
    print(f"  Expected accuracy:  {rp['expected_accuracy']:.4f}")
    print(f"  Expected macro-F1:  {rp['expected_macro_f1']:.4f}")
    print()

    if validation_labels is not None:
        _print_split_results("validation", train_labels, validation_labels, args.seed)
        print()
    else:
        print("Validation split: not available (legacy train/test layout).\n")
    _print_split_results("test", train_labels, test_labels, args.seed)

    # Write predictions JSONL if --output is provided
    if args.output:
        priors = dict(train_labels.value_counts(normalize=True))
        majority_label = train_labels.value_counts().idxmax()
        rng = np.random.RandomState(args.seed)

        test_cids = [str(cid) for cid in test_df["condition_id"]]

        # Majority predictions
        maj_preds = [majority_label] * len(test_cids)
        maj_scores = _build_scores(priors, maj_preds)
        base = Path(args.output)
        maj_path = base.parent / f"t1_majority_{base.name}"
        _write_preds(maj_path, test_cids, maj_preds, maj_scores)
        print(f"  Majority predictions -> {maj_path}")

        # Random predictions
        rand_preds = rng.choice(list(priors.keys()),
                                size=len(test_cids), p=list(priors.values())).tolist()
        rand_scores = _build_scores(priors, rand_preds)
        rand_path = base.parent / f"t1_random_{base.name}"
        _write_preds(rand_path, test_cids, rand_preds, rand_scores)
        print(f"  Random predictions   -> {rand_path}")


if __name__ == "__main__":
    main()
