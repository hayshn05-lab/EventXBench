#!/usr/bin/env python3
"""T4 Basic Baselines -- Majority Class and Random Walk.

Computes trivial baselines for Market Movement Prediction:

1. **Majority class**: always predict the most common label.
2. **Random prior**: sample from empirical class distribution.
3. **Random walk**: predict delta = 0 (no change) for all horizons.

All baselines are evaluated across three tiers:
  - Tier 1: All data
  - Tier 2: Non-confounded only
  - Tier 3: Active signals (non-confounded + non-flat)

Usage:
    python baselines/t4/basic_baseline.py
    python baselines/t4/basic_baseline.py --local-dir ./data
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["up", "down", "flat"]
MAGNITUDE_LABELS = ["small", "medium", "large"]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 basic baselines")
    parser.add_argument("--local-dir", default=None, help="Local data directory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> pd.DataFrame:
    import eventxbench

    if local_dir:
        result = eventxbench.load_task("t4", local_dir=local_dir)
    else:
        result = eventxbench.load_task("t4")
    if isinstance(result, tuple):
        return pd.concat(result, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def majority_macro_f1(label_counts: dict[str, int], all_labels: list[str]) -> dict:
    """Compute majority-class accuracy and macro-F1."""
    total = sum(label_counts.values())
    if total == 0:
        return {"majority_label": None, "accuracy": 0.0, "macro_f1": 0.0}

    majority_label = max(label_counts, key=label_counts.get)
    majority_count = label_counts[majority_label]

    acc = majority_count / total
    # Majority predicts one class: F1 = 2p/(1+p) for that class, 0 for others
    p = majority_count / total
    f1_majority = 2 * p / (1 + p)
    macro_f1 = f1_majority / len(all_labels)

    return {"majority_label": majority_label, "accuracy": acc, "macro_f1": macro_f1}


def random_prior_f1(label_counts: dict[str, int]) -> dict:
    """Expected macro-F1 under random-prior sampling."""
    total = sum(label_counts.values())
    if total == 0:
        return {"expected_accuracy": 0.0, "expected_macro_f1": 0.0}
    priors = {k: v / total for k, v in label_counts.items()}
    exp_acc = sum(p ** 2 for p in priors.values())
    exp_f1 = sum(priors.values()) / len(priors)
    return {"expected_accuracy": exp_acc, "expected_macro_f1": exp_f1}


# ---------------------------------------------------------------------------
# Tier helpers
# ---------------------------------------------------------------------------


def build_tiers(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("Tier 1: All Data", df.copy()),
        ("Tier 2: Non-confounded", df[~df["confound_flag"].astype(bool)].copy()),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            df[
                (~df["confound_flag"].astype(bool)) & (df["direction_label"] != "flat")
            ].copy(),
        ),
    ]


def evaluate_random_walk_tier(tier_df: pd.DataFrame) -> dict:
    """Random walk: predict delta=0 for all horizons. Direction=flat, magnitude=small."""
    n = len(tier_df)
    if n == 0:
        return {"dir_acc": 0.0, "mag_f1": 0.0, "spearman": None}

    # Direction: always flat
    dir_true = tier_df["direction_label"].tolist()
    dir_pred = ["flat"] * n
    dir_acc = accuracy_score(dir_true, dir_pred)

    # Magnitude: always small (since |delta|=0 <= 0.02)
    mag_true = tier_df["magnitude_bucket"].tolist()
    mag_pred = ["small"] * n
    mag_f1 = f1_score(mag_true, mag_pred, labels=MAGNITUDE_LABELS, average="macro", zero_division=0)

    # Spearman: predict 0 for all deltas
    actual = []
    for h in ("delta_30m", "delta_2h", "delta_6h"):
        vals = tier_df[h].dropna().tolist()
        actual.extend([float(v) for v in vals])
    predicted = [0.0] * len(actual)
    spr = spearman(predicted, actual)

    return {"dir_acc": dir_acc, "mag_f1": mag_f1, "spearman": spr}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_row(tier_name: str, n: int, metrics: dict, target: str) -> None:
    if target == "direction":
        val = metrics.get("accuracy", metrics.get("dir_acc", 0.0))
        print(f"  {tier_name} (n={n}):  Accuracy = {val*100:.2f}%")
    elif target == "magnitude":
        val = metrics.get("macro_f1", metrics.get("mag_f1", 0.0))
        print(f"  {tier_name} (n={n}):  Macro-F1 = {val*100:.2f}%")
    elif target == "spearman":
        spr = metrics.get("spearman")
        txt = "N/A" if spr is None else f"{spr:.4f}"
        print(f"  {tier_name} (n={n}):  Spearman = {txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print("Loading T4 data...")
    df = load_data(args.local_dir)
    print(f"Loaded {len(df)} rows")

    tiers = build_tiers(df)

    # === Majority Baseline ===
    print_header("MAJORITY BASELINE -- Direction")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["direction_label"].value_counts())
        if "non-flat" in tier_name:
            labels = ["up", "down"]
        else:
            labels = DIRECTION_LABELS
        m = majority_macro_f1(counts, labels)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"Always predict '{m['majority_label']}'  "
              f"Acc={m['accuracy']*100:.2f}%  Macro-F1={m['macro_f1']*100:.2f}%")

    print_header("MAJORITY BASELINE -- Magnitude")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["magnitude_bucket"].value_counts())
        m = majority_macro_f1(counts, MAGNITUDE_LABELS)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"Always predict '{m['majority_label']}'  "
              f"Acc={m['accuracy']*100:.2f}%  Macro-F1={m['macro_f1']*100:.2f}%")

    # === Random Prior Baseline ===
    print_header("RANDOM PRIOR BASELINE -- Direction")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["direction_label"].value_counts())
        r = random_prior_f1(counts)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"E[Acc]={r['expected_accuracy']*100:.2f}%  E[F1]={r['expected_macro_f1']*100:.2f}%")

    print_header("RANDOM PRIOR BASELINE -- Magnitude")
    for tier_name, tier_df in tiers:
        counts = dict(tier_df["magnitude_bucket"].value_counts())
        r = random_prior_f1(counts)
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"E[Acc]={r['expected_accuracy']*100:.2f}%  E[F1]={r['expected_macro_f1']*100:.2f}%")

    # === Random Walk Baseline ===
    print_header("RANDOM WALK BASELINE (predict delta=0 everywhere)")
    for tier_name, tier_df in tiers:
        rw = evaluate_random_walk_tier(tier_df)
        spr_txt = "N/A" if rw["spearman"] is None else f"{rw['spearman']:.4f}"
        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"DirAcc={rw['dir_acc']*100:.2f}%  MagF1={rw['mag_f1']*100:.2f}%  "
              f"Spearman={spr_txt}")

    # === Empirical Random Baseline ===
    print_header(f"EMPIRICAL RANDOM BASELINE (seed={args.seed})")
    rng = np.random.RandomState(args.seed)
    for tier_name, tier_df in tiers:
        if len(tier_df) == 0:
            continue

        # Direction
        dir_counts = tier_df["direction_label"].value_counts(normalize=True)
        dir_preds = rng.choice(dir_counts.index, size=len(tier_df), p=dir_counts.values)
        dir_acc = accuracy_score(tier_df["direction_label"], dir_preds)

        # Magnitude
        mag_counts = tier_df["magnitude_bucket"].value_counts(normalize=True)
        mag_preds = rng.choice(mag_counts.index, size=len(tier_df), p=mag_counts.values)
        mag_f1 = f1_score(tier_df["magnitude_bucket"], mag_preds,
                          labels=MAGNITUDE_LABELS, average="macro", zero_division=0)

        print(f"  {tier_name} (n={len(tier_df)}):  "
              f"DirAcc={dir_acc*100:.2f}%  MagF1={mag_f1*100:.2f}%")


if __name__ == "__main__":
    main()
