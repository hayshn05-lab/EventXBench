#!/usr/bin/env python3
"""T1 LightGBM Baseline -- Pre-Market Interest Forecasting.

Trains a LightGBM classifier on numeric social-signal features to predict
market interest level (high / moderate / low).  Hyperparameters are tuned
with Optuna (10 trials by default).

Usage:
    python baselines/t1/lightgbm_baseline.py
    python baselines/t1/lightgbm_baseline.py --local-dir ./data --trials 20
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]
LABEL_MAP = {label: idx for idx, label in enumerate(LABEL_ORDER)}

NUMERIC_FEATURES = [
    "score",
    "cluster_count",
    "linked_tweet_count",
    "avg_link_confidence",
    "max_link_confidence",
    "text_similarity",
    "tweet_count",
    "unique_user_count",
    "burst_duration_hours",
    "max_author_followers",
    "mean_author_followers",
    "median_author_followers",
    "high_follower_author_count",
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1 LightGBM baseline")
    parser.add_argument("--local-dir", default=None, help="Local data directory (skips HF)")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="t1_lightgbm_predictions.jsonl")
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        raise SystemExit("Install lightgbm:  pip install lightgbm")
    try:
        import optuna
    except ImportError:
        raise SystemExit("Install optuna:  pip install optuna")

    # Silence Optuna and LightGBM verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading T1 data...")
    train_df, test_df = load_data(args.local_dir)

    # Select available numeric features
    available = [c for c in NUMERIC_FEATURES if c in train_df.columns and c in test_df.columns]
    if not available:
        raise SystemExit(
            f"No numeric features found. Expected some of: {NUMERIC_FEATURES}"
        )
    print(f"Features ({len(available)}): {available}")

    X_train = train_df[available].copy().astype(float)
    X_test = test_df[available].copy().astype(float)
    y_train = train_df["interest_label"].map(LABEL_MAP).values
    y_test = test_df["interest_label"].map(LABEL_MAP).values

    # Replace NaN with column medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    n_classes = len(LABEL_ORDER)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # ------------------------------------------------------------------
    # Optuna hyperparameter search
    # ------------------------------------------------------------------

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": args.seed,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y_train[train_idx])
            dval = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx], reference=dtrain)
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_train.iloc[val_idx])
            pred_labels = np.argmax(preds, axis=1)
            scores.append(f1_score(y_train[val_idx], pred_labels, average="macro", zero_division=0))
        return float(np.mean(scores))

    print(f"Running Optuna ({args.trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print(f"Best CV macro-F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # ------------------------------------------------------------------
    # Retrain on full training set with best params
    # ------------------------------------------------------------------

    best = study.best_params.copy()
    best.update({
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "random_state": args.seed,
    })

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(best, dtrain_full, num_boost_round=300)

    # ------------------------------------------------------------------
    # Predict and evaluate
    # ------------------------------------------------------------------

    proba = final_model.predict(X_test)
    pred_indices = np.argmax(proba, axis=1)
    pred_labels = [LABEL_ORDER[i] for i in pred_indices]
    gold_labels = [LABEL_ORDER[i] for i in y_test]

    acc = accuracy_score(y_test, pred_indices)
    macro_f1 = f1_score(y_test, pred_indices, average="macro", zero_division=0)

    print(f"\n--- Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(available, importance), key=lambda x: x[1], reverse=True)
    print("\nTop features (gain):")
    for name, imp in feat_imp[:5]:
        print(f"  {name}: {imp:.1f}")

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(test_df.iterrows()):
            rec = {
                "condition_id": str(row["condition_id"]),
                "gold_label": gold_labels[i],
                "pred_label": pred_labels[i],
                "confidence": float(proba[i].max()),
                "scores": {
                    LABEL_ORDER[j]: float(proba[i][j]) for j in range(n_classes)
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
