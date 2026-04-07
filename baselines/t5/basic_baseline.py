#!/usr/bin/env python3
"""T5 Basic Baselines -- Impact Persistence (Decay Classification).

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
def _majority_baseline(y_true: list[str]) -> dict:
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    mf1 = _macro_f1(y_true, y_pred, DECAY_LABELS)

    # Analytical: F1(majority) = 2p/(1+p), other classes = 0
    total = len(y_true)
    p = counts[majority] / total
    num_classes = len(DECAY_LABELS)
    analytical_mf1 = (2.0 * p / (1.0 + p)) / num_classes

    return {
        "baseline": "majority",
        "majority_label": majority,
        "n": total,
        "macro_f1": mf1,
        "macro_f1_analytical": analytical_mf1,
    }


def _random_baseline(y_true: list[str], seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    counts = Counter(y_true)
    total = len(y_true)
    priors = np.array([counts.get(lab, 0) / total for lab in DECAY_LABELS])

    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(DECAY_LABELS, size=total, p=priors).tolist()
        f1_scores.append(_macro_f1(y_true, y_pred, DECAY_LABELS))

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n": total,
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
    if isinstance(data, tuple):
        _, df = data
    else:
        df = data

    df = df[df["decay_class"].isin(DECAY_LABELS)].reset_index(drop=True)
    y_true = df["decay_class"].tolist()

    print(f"T5 samples: {len(y_true)}")
    print(f"Class distribution: {dict(Counter(y_true))}")

    # Majority baseline
    maj = _majority_baseline(y_true)
    print(f"\n[Majority] always predict '{maj['majority_label']}'")
    print(f"  Macro-F1: {maj['macro_f1']:.4f} (analytical: {maj['macro_f1_analytical']:.4f})")

    # Random baseline
    rand = _random_baseline(y_true)
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Macro-F1: {rand['mean_macro_f1']:.4f}")
    print(f"  Per-seed: {rand['per_seed_macro_f1']}")


if __name__ == "__main__":
    main()
