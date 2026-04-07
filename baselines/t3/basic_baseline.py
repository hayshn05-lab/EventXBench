#!/usr/bin/env python3
"""T3 Basic Baselines -- Evidence Grading.

Evaluates majority-class and random baselines for T3 evidence grading.
Reports Spearman correlation and Quadratic Weighted Kappa (QWK).

Usage:
    python -m baselines.t3.basic_baseline
    python -m baselines.t3.basic_baseline --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import eventxbench


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * len(vals)
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

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], num_classes: int = 6) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1
    W = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / n
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _run_majority(y_true: list[int]) -> dict:
    counts = Counter(y_true)
    majority = counts.most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
    qwk = _quadratic_weighted_kappa(y_true, y_pred)
    return {
        "baseline": "majority",
        "majority_grade": majority,
        "n": len(y_true),
        "spearman": rho,
        "qwk": qwk,
    }


def _run_random(y_true: list[int], seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]

    counts = Counter(y_true)
    grades = sorted(counts.keys())
    total = sum(counts.values())
    priors = np.array([counts[g] / total for g in grades])

    rhos: list[float] = []
    qwks: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(grades, size=len(y_true), p=priors).tolist()
        rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
        qwk = _quadratic_weighted_kappa(y_true, y_pred)
        if rho is not None:
            rhos.append(rho)
        qwks.append(qwk)

    return {
        "baseline": "random_prior",
        "seeds": seeds,
        "n": len(y_true),
        "mean_spearman": float(np.mean(rhos)) if rhos else None,
        "mean_qwk": float(np.mean(qwks)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 basic baselines")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    y_true = df["final_grade"].astype(int).tolist()

    print(f"T3 samples: {len(y_true)}")
    print(f"Grade distribution: {dict(sorted(Counter(y_true).items()))}")

    # Majority baseline
    maj = _run_majority(y_true)
    rho_str = f"{maj['spearman']:.4f}" if maj["spearman"] is not None else "N/A"
    print(f"\n[Majority] always predict grade={maj['majority_grade']}")
    print(f"  Spearman={rho_str}, QWK={maj['qwk']:.4f}")

    # Random baseline
    rand = _run_random(y_true)
    rho_str = f"{rand['mean_spearman']:.4f}" if rand["mean_spearman"] is not None else "N/A"
    print(f"\n[Random Prior] sample from training distribution")
    print(f"  Mean Spearman={rho_str}, Mean QWK={rand['mean_qwk']:.4f}")


if __name__ == "__main__":
    main()
