#!/usr/bin/env python3
"""T3 Pre-check Pipeline Baseline -- Evidence Grading.

Evaluates majority-class, random, and pre-check pipeline baselines
for T3 evidence grading using a market-level 70/30 train-test split.
Reports Cohen's Kappa and Macro F1.

Usage:
    python t3_precheck_baseline.py
    python t3_precheck_baseline.py --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

import eventxbench


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def split_by_market(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    markets = df["condition_id"].unique()
    train_markets, test_markets = train_test_split(
        markets, test_size=test_size, random_state=random_state
    )
    train_df = df[df["condition_id"].isin(train_markets)].copy()
    test_df = df[df["condition_id"].isin(test_markets)].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _run_majority(y_true: np.ndarray) -> dict:
    counts = Counter(y_true.tolist())
    majority_class = counts.most_common(1)[0][0]
    y_pred = np.full(len(y_true), majority_class)

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "majority",
        "majority_class": majority_class,
        "n": len(y_true),
        "kappa": kappa,
        "macro_f1": f1,
    }


def _run_random(y_true: np.ndarray, random_state: int = 42) -> dict:
    """Single random_state=42 draw from the test-set class priors.

    Matches T3_Reproducible_Package's canonical Random baseline
    (metrics.md Phase 6: Kappa 0.0038, Macro F1 0.1709) exactly - a
    multi-seed average would not reproduce that reported figure.
    """
    counts = Counter(y_true.tolist())
    grades = sorted(counts.keys())
    total = sum(counts.values())
    priors = np.array([counts[g] / total for g in grades])

    rng = np.random.default_rng(random_state)
    y_pred = rng.choice(grades, size=len(y_true), p=priors)

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "random_prior",
        "random_state": random_state,
        "n": len(y_true),
        "kappa": kappa,
        "macro_f1": f1,
    }


def _run_precheck_pipeline(test_df: pd.DataFrame) -> dict:
    """Reconstruct the silver pipeline's own `assign_final_grade` rule
    (t3_pipeline_source_to_silver.ipynb) rather than a NaN->3 fallback:
    rows where all four deterministic checks passed (`candidate_grade`
    set) get that grade as-is (always 5); every other row uses
    `llm_grade`, capped at <=3 when `check_source == "fail"`.
    """
    df = test_df.copy()

    def _assign(row):
        if pd.notna(row["candidate_grade"]):
            return int(row["candidate_grade"])
        grade = int(row["llm_grade"])
        if row["check_source"] == "fail":
            grade = min(grade, 3)
        return grade

    y_pred = df.apply(_assign, axis=1).values
    y_true = df["final_grade"].values

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "precheck_pipeline",
        "n": len(y_true),
        "kappa": kappa,
        "macro_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 pre-check pipeline baselines")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    df.sort_values(by=["created_at"], ascending=True, inplace=True)

    # Split
    train_df, test_df = split_by_market(
        df, test_size=args.test_size, random_state=args.random_state
    )

    y_true = test_df["final_grade"].values

    print(f"T3 samples: {len(df)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size:  {len(test_df)}")
    print(f"Grade distribution (test): {dict(sorted(Counter(y_true.tolist()).items()))}")

    # Majority baseline
    maj = _run_majority(y_true)
    print(f"\n[Majority] always predict grade={maj['majority_class']}")
    print(f"  Kappa={maj['kappa']:.4f}, Macro F1={maj['macro_f1']:.4f}")

    # Random baseline
    rand = _run_random(y_true, random_state=args.random_state)
    print(f"\n[Random Prior] sample from test-set class priors (random_state={rand['random_state']})")
    print(f"  Kappa={rand['kappa']:.4f}, Macro F1={rand['macro_f1']:.4f}")

    # Pre-check pipeline baseline
    pre = _run_precheck_pipeline(test_df)
    print(f"\n[Pre-check Pipeline] candidate_grade where checks pass, else llm_grade (capped <=3 on source fail)")
    print(f"  Kappa={pre['kappa']:.4f}, Macro F1={pre['macro_f1']:.4f}")


if __name__ == "__main__":
    main()