#!/usr/bin/env python3
"""T5 daily KDD basic baselines -- Drift Persistence Classification.

Evaluates majority-class and random-prior baselines for T5 decay
classification.  Reports Macro-F1.

Note: In the original codebase this task is referred to as T7 / task5+7,
but in the paper and public release it is T5.

Usage:
    python -m baselines.t5.basic_baseline
    python -m baselines.t5.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import eventxbench

DECAY_LABELS = ["transient", "sustained", "reversal"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p == lab)
        fp = sum(1 for a, p in zip(y_true, y_pred) if a != lab and p == lab)
        fn = sum(1 for a, p in zip(y_true, y_pred) if a == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _majority_baseline(y_train: list[str], y_test: list[str]) -> dict:
    counts = Counter(y_train)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_test)
    mf1 = _macro_f1(y_test, y_pred, DECAY_LABELS)

    return {
        "baseline": "majority",
        "majority_label": majority,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "macro_f1": mf1,
    }


def _random_baseline(
    y_train: list[str], y_test: list[str], seeds: list[int] | None = None
) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    counts = Counter(y_train)
    total = len(y_train)
    priors = np.array([counts.get(lab, 0) / total for lab in DECAY_LABELS])

    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(DECAY_LABELS, size=len(y_test), p=priors).tolist()
        f1_scores.append(_macro_f1(y_test, y_pred, DECAY_LABELS))

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mean_macro_f1": float(np.mean(f1_scores)),
        "per_seed_macro_f1": [round(f, 4) for f in f1_scores],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T5 basic baselines")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if not isinstance(data, tuple):
        raise ValueError("Canonical T5 baseline requires frozen train/test splits.")
    train_df, test_df = data
    train_df = train_df[train_df["decay_class"].isin(DECAY_LABELS)]
    test_df = test_df[test_df["decay_class"].isin(DECAY_LABELS)]
    y_train = train_df["decay_class"].tolist()
    y_test = test_df["decay_class"].tolist()

    print(f"T5 train/test samples: {len(y_train)}/{len(y_test)}")
    print(f"Train class distribution: {dict(Counter(y_train))}")
    print(f"Test class distribution: {dict(Counter(y_test))}")

    # Majority baseline
    maj = _majority_baseline(y_train, y_test)
    print(f"\n[Majority] always predict '{maj['majority_label']}'")
    print(f"  Test Macro-F1: {maj['macro_f1']:.4f}")

    # Random baseline
    rand = _random_baseline(y_train, y_test)
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Macro-F1: {rand['mean_macro_f1']:.4f}")
    print(f"  Per-seed: {rand['per_seed_macro_f1']}")


if __name__ == "__main__":
    main()
