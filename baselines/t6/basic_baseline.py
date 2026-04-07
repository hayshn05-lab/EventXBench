#!/usr/bin/env python3
"""T6 Basic Baselines -- Cross-Market Propagation.

Evaluates majority-class and random-prior baselines for T6 cross-market
propagation classification.  Reports Macro-F1.

Usage:
    python -m baselines.t6.basic_baseline
    python -m baselines.t6.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import eventxbench

LABEL_ORDER = ["no_cross_market_effect", "cross_market_effect", "insufficient_data"]


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
def _majority_baseline(y_true: list[str], labels: list[str]) -> dict:
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    mf1 = _macro_f1(y_true, y_pred, labels)
    return {
        "baseline": "majority",
        "majority_label": majority,
        "n": len(y_true),
        "macro_f1": mf1,
    }


def _random_baseline(y_true: list[str], labels: list[str], seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    counts = Counter(y_true)
    total = len(y_true)
    priors = np.array([counts.get(lab, 0) / total for lab in labels])

    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(labels, size=total, p=priors).tolist()
        f1_scores.append(_macro_f1(y_true, y_pred, labels))

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
    parser = argparse.ArgumentParser(description="T6 basic baselines")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--exclude-insufficient",
        action="store_true",
        help="Exclude rows with insufficient_data label",
    )
    args = parser.parse_args()

    data = eventxbench.load_task("t6", local_dir=args.local_dir)
    if isinstance(data, tuple):
        _, df = data
    else:
        df = data

    # Filter
    if "insufficient_data_flag" in df.columns:
        df = df[df["insufficient_data_flag"] == False].reset_index(drop=True)
    if "confound_flag" in df.columns:
        df = df[df["confound_flag"] == False].reset_index(drop=True)

    df = df[df["label"].isin(LABEL_ORDER)].reset_index(drop=True)

    if args.exclude_insufficient:
        df = df[df["label"] != "insufficient_data"].reset_index(drop=True)
        eval_labels = [l for l in LABEL_ORDER if l != "insufficient_data"]
    else:
        eval_labels = LABEL_ORDER

    y_true = df["label"].tolist()

    print(f"T6 samples: {len(y_true)}")
    print(f"Class distribution: {dict(Counter(y_true))}")

    # Majority baseline
    maj = _majority_baseline(y_true, eval_labels)
    print(f"\n[Majority] always predict '{maj['majority_label']}'")
    print(f"  Macro-F1: {maj['macro_f1']:.4f}")

    # Random baseline
    rand = _random_baseline(y_true, eval_labels)
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Macro-F1: {rand['mean_macro_f1']:.4f}")
    print(f"  Per-seed: {rand['per_seed_macro_f1']}")


if __name__ == "__main__":
    main()
