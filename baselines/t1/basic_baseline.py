#!/usr/bin/env python3
"""T1 Basic Baselines -- Majority Class and Random Prior.

Computes two trivial baselines for the Pre-Market Interest Forecasting task:

1. **Majority class**: always predict the most common training label.
2. **Random prior**: sample predictions from the empirical class distribution;
   reports the expected macro-F1 analytically.

Usage:
    python baselines/t1/basic_baseline.py
    python baselines/t1/basic_baseline.py --local-dir ./data
"""
from __future__ import annotations

import argparse
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import eventxbench

    if local_dir:
        return eventxbench.load_task("t1", local_dir=local_dir)
    return eventxbench.load_task("t1")


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def majority_baseline(
    train_labels: pd.Series, test_labels: pd.Series
) -> dict[str, float]:
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

    priors = {k: v / total for k, v in label_counts.items()}
    expected_acc = sum(p ** 2 for p in priors.values())
    expected_macro_f1 = sum(priors.values()) / len(priors)

    return {
        "class_priors": priors,
        "expected_accuracy": expected_acc,
        "expected_macro_f1": expected_macro_f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print("Loading T1 data...")
    train_df, test_df = load_data(args.local_dir)

    train_labels = train_df["interest_label"].astype(str)
    test_labels = test_df["interest_label"].astype(str)

    print(f"Train: {len(train_labels)}  Test: {len(test_labels)}")
    print(f"Train distribution:\n{train_labels.value_counts().to_string()}\n")
    print(f"Test distribution:\n{test_labels.value_counts().to_string()}\n")

    # --- Majority baseline ---
    maj = majority_baseline(train_labels, test_labels)
    print("=" * 50)
    print("MAJORITY BASELINE")
    print("=" * 50)
    print(f"  Always predict: {maj['majority_label']}")
    print(f"  Accuracy:       {maj['accuracy']:.4f}")
    print(f"  Macro-F1:       {maj['macro_f1']:.4f}")

    # --- Random prior baseline ---
    counts = dict(train_labels.value_counts())
    rp = random_prior_expected_f1(counts)
    print()
    print("=" * 50)
    print("RANDOM PRIOR BASELINE (analytical expectation)")
    print("=" * 50)
    print(f"  Class priors:       {rp.get('class_priors', {})}")
    print(f"  Expected accuracy:  {rp['expected_accuracy']:.4f}")
    print(f"  Expected macro-F1:  {rp['expected_macro_f1']:.4f}")

    # --- Empirical random baseline (for comparison) ---
    rng = np.random.RandomState(args.seed)
    priors = train_labels.value_counts(normalize=True)
    random_preds = rng.choice(priors.index, size=len(test_labels), p=priors.values)
    emp_acc = accuracy_score(test_labels, random_preds)
    emp_f1 = f1_score(
        test_labels, random_preds, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    print()
    print("=" * 50)
    print(f"RANDOM PRIOR BASELINE (empirical, seed={args.seed})")
    print("=" * 50)
    print(f"  Accuracy:  {emp_acc:.4f}")
    print(f"  Macro-F1:  {emp_f1:.4f}")


if __name__ == "__main__":
    main()
