#!/usr/bin/env python3
"""T6 Basic Baselines -- Cross-Market Co-Movement.

Evaluates explicitly named global-majority, per-horizon-majority, and
random-prior baselines for T6 cross-market co-movement classification.
The canonical v2 prediction file uses the per-horizon-majority strategy by
default so that evaluating the file reproduces the corresponding leaderboard
row.

Supports both legacy v1 data (labels no_cross_market_effect/primary_mover/
propagated_signal, one row per post) and KDD v2 bundle/horizon data (labels
no_effect/primary_only/cross_market, one row per (m,d,H)).

Usage:
    python -m baselines.t6.basic_baseline
    python -m baselines.t6.basic_baseline --local-dir KDD/data/t6_kdd_v2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import eventxbench

V1_LABELS = ["no_cross_market_effect", "primary_mover", "propagated_signal"]
V2_LABELS = ["no_effect", "primary_only", "cross_market"]


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


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return (
        sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
        / len(y_true)
        if y_true
        else 0.0
    )


def _majority(y_true: list[str], labels: list[str], train_labels: list[str]) -> tuple[str, float]:
    majority = Counter(train_labels).most_common(1)[0][0]
    y_pred = [majority] * len(y_true)
    return majority, _macro_f1(y_true, y_pred, labels)


def _random_prior(y_true: list[str], labels: list[str], train_labels: list[str],
                  seeds: list[int] | None = None) -> dict:
    if seeds is None:
        seeds = [13, 42, 123]
    counts = Counter(train_labels)
    total_train = len(train_labels)
    priors = np.array([counts.get(lab, 0) / total_train for lab in labels])
    # Normalize to handle the case where train labels are a subset of label set
    if priors.sum() == 0:
        priors = np.ones(len(labels)) / len(labels)
    else:
        priors = priors / priors.sum()
    f1_scores = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y_pred = rng.choice(labels, size=len(y_true), p=priors).tolist()
        f1_scores.append(_macro_f1(y_true, y_pred, labels))
    return {
        "seeds": seeds,
        "mean_macro_f1": float(np.mean(f1_scores)),
        "per_seed_macro_f1": [round(f, 4) for f in f1_scores],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 basic baselines")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--output", default=None,
                        help="Optional JSONL output path for predictions")
    parser.add_argument(
        "--prediction-strategy",
        choices=("per-horizon-majority", "global-majority"),
        default="per-horizon-majority",
        help=(
            "Strategy written to --output (default: per-horizon-majority, "
            "which matches the canonical v2 leaderboard row)"
        ),
    )
    args = parser.parse_args()

    data = eventxbench.load_task("t6", local_dir=args.local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
    else:
        df = data
        if "split" not in df.columns:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            train_df = df[df["split"] == "train"].copy()
            test_df = df[df["split"] == "test"].copy()

    # Detect label set
    if "label" not in train_df.columns and "headline_label" in train_df.columns:
        label_col = "headline_label"
    else:
        label_col = "label"
    unique_labels = set(train_df[label_col].dropna().astype(str).unique())
    if unique_labels & set(V2_LABELS):
        labels = V2_LABELS
    else:
        labels = V1_LABELS
    train_df = train_df[train_df[label_col].isin(labels)].copy()
    test_df = test_df[test_df[label_col].isin(labels)].copy()

    # Exclude confounded from test (v2 policy: confounded excluded from val/test)
    if "confound_flag" in test_df.columns:
        test_df = test_df[test_df["confound_flag"] == False].reset_index(drop=True)

    print(f"T6 train: {len(train_df)}, test: {len(test_df)}")
    print(f"Detected label set: {labels}")
    print(f"Test class distribution: {dict(Counter(test_df[label_col]))}")

    has_horizons = "horizon_days" in train_df.columns and "horizon_days" in test_df.columns

    if has_horizons:
        horizons = sorted(train_df["horizon_days"].unique())
        print(f"Horizons: {horizons}")

    # Per-horizon + overall evaluation
    print("\n=== Explicit majority strategies ===")
    horiz_results: list[dict] = []
    majority_by_horizon: dict[int, str] = {}
    eval_splits = []
    if has_horizons:
        for h in horizons:
            tr_h = train_df[train_df["horizon_days"] == h][label_col].tolist()
            te_h = test_df[test_df["horizon_days"] == h][label_col].tolist()
            if not te_h:
                continue
            maj, mf1 = _majority(te_h, labels, tr_h)
            majority_by_horizon[int(h)] = maj
            print(
                f"  PER-HORIZON H={h}d (n={len(te_h)}): "
                f"majority='{maj}'  Macro-F1={mf1:.4f}"
            )
            horiz_results.append({"horizon_days": int(h), "majority": maj, "macro_f1": mf1, "n": len(te_h)})
            eval_splits.append(("h" + str(h), tr_h, te_h))
    # Overall
    tr_all = train_df[label_col].tolist()
    te_all = test_df[label_col].tolist()
    maj_all, mf1_all = _majority(te_all, labels, tr_all)
    print(
        f"  GLOBAL MAJORITY (n={len(te_all)}): "
        f"majority='{maj_all}'  Macro-F1={mf1_all:.4f}"
    )
    if has_horizons:
        combined_horizon_predictions = [
            majority_by_horizon[int(horizon)]
            for horizon in test_df["horizon_days"]
        ]
        combined_horizon_accuracy = _accuracy(
            te_all, combined_horizon_predictions
        )
        combined_horizon_f1 = _macro_f1(
            te_all, combined_horizon_predictions, labels
        )
        print(
            "  PER-HORIZON MAJORITY COMBINED "
            f"(n={len(te_all)}): Accuracy={combined_horizon_accuracy:.4f}  "
            f"Macro-F1={combined_horizon_f1:.4f}"
        )
    eval_splits.append(("all", tr_all, te_all))
    horiz_results.append({"horizon_days": "all", "majority": maj_all, "macro_f1": mf1_all, "n": len(te_all)})

    print("\n=== Random prior baseline ===")
    rand_results: list[dict] = []
    for name, tr_h, te_h in eval_splits:
        rand = _random_prior(te_h, labels, tr_h)
        print(f"  {name} (n={len(te_h)}): mean Macro-F1={rand['mean_macro_f1']:.4f}  seeds={rand['per_seed_macro_f1']}")
        rand_results.append({"split": name, **rand})

    # Write predictions
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.prediction_strategy == "per-horizon-majority" and not has_horizons:
            print(
                "Legacy data has no horizons; falling back to global-majority "
                "for --output."
            )
        selected_strategy = (
            args.prediction_strategy
            if has_horizons
            else "global-majority"
        )
        # Random prior predictions on test overall (seed 42)
        counts = Counter(tr_all)
        priors = np.array([counts.get(lab, 0) / len(tr_all) for lab in labels])
        if priors.sum() == 0:
            priors = np.ones(len(labels)) / len(labels)
        else:
            priors = priors / priors.sum()
        rng = np.random.default_rng(42)
        rand_preds = rng.choice(labels, size=len(te_all), p=priors).tolist()
        with out_path.open("w", encoding="utf-8") as f:
            for row, rpred in zip(test_df.to_dict("records"), rand_preds):
                per_horizon_pred = (
                    majority_by_horizon.get(int(row["horizon_days"]), maj_all)
                    if has_horizons
                    else maj_all
                )
                mpred = (
                    per_horizon_pred
                    if selected_strategy == "per-horizon-majority"
                    else maj_all
                )
                rec = {
                    "instance_id": str(row.get("instance_id", "")),
                    "condition_id": str(row.get("condition_id", "")),
                    "bundle_day": row.get("bundle_day", ""),
                    "horizon_days": row.get("horizon_days", ""),
                    "prediction": mpred,
                    "strategy": selected_strategy,
                    "majority_label": mpred,
                    "random_prior_label": rpred,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"\nPredictions saved to {out_path} "
            f"(strategy={selected_strategy})"
        )


if __name__ == "__main__":
    main()
