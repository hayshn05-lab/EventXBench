#!/usr/bin/env python3
"""T4 LightGBM Baseline -- Market Movement Prediction.

Trains LightGBM classifiers for direction (up/down/flat) and magnitude
(small/medium/large) using features available in the public label file
(price_t0, confound_flag). Hyperparameters tuned with Optuna.

Evaluation tiers:
  - Tier 1: All data
  - Tier 2: Non-confounded only
  - Tier 3: Active signals (non-confounded + non-flat)

Usage:
    python baselines/t4/lightgbm_baseline.py
    python baselines/t4/lightgbm_baseline.py --local-dir ./data --trials 20
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
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["flat", "up", "down"]
MAGNITUDE_LABELS = ["small", "medium", "large"]

# Features available in the public t4_labels.jsonl
# Users with rehydrated tweets can extend this list with engagement stats.
FEATURE_COLS = ["price_t0"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 LightGBM baseline")
    parser.add_argument("--local-dir", default=None, help="Local data directory")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", default="t4_lightgbm_predictions.jsonl")
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
# Training helpers
# ---------------------------------------------------------------------------


def train_lgbm_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_classes: int,
    n_trials: int,
    seed: int,
) -> "lgb.Booster":
    """Train LightGBM with Optuna hyperparameter search, return best model."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": seed,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        if n_classes == 2:
            params.update({"objective": "binary", "metric": "binary_logloss"})
        else:
            params.update({"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"})

        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx])
            dval = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx], reference=dtrain)
            model = lgb.train(
                params, dtrain, valid_sets=[dval], num_boost_round=500,
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_train.iloc[val_idx])
            if n_classes == 2:
                pred_labels = (np.array(preds) >= 0.5).astype(int)
            else:
                pred_labels = np.argmax(preds, axis=1)
            scores.append(f1_score(y_train[val_idx], pred_labels, average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params.copy()
    best.update({"verbosity": -1, "random_state": seed})
    if n_classes == 2:
        best.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        best.update({"objective": "multiclass", "num_class": n_classes, "metric": "multi_logloss"})

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    return lgb.train(best, dtrain_full, num_boost_round=300)


def predict_labels(model, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    preds = model.predict(X)
    if n_classes == 2:
        return (np.array(preds) >= 0.5).astype(int)
    return np.argmax(preds, axis=1)


# ---------------------------------------------------------------------------
# Tier evaluation
# ---------------------------------------------------------------------------


def evaluate_tier(
    tier_name: str,
    df_tier: pd.DataFrame,
    features: list[str],
    target_col: str,
    label_list: list[str],
    n_trials: int,
    seed: int,
    test_size: float,
) -> dict:
    """Train + evaluate on a single tier."""
    import lightgbm as lgb

    label_map = {lab: i for i, lab in enumerate(label_list)}
    y = df_tier[target_col].map(label_map)
    valid = y.notnull()
    df_tier = df_tier[valid].copy()
    y = y[valid].values.astype(int)
    X = df_tier[features].copy().astype(float)

    if len(X) < 10:
        return {"tier": tier_name, "target": target_col, "n": len(X),
                "accuracy": 0.0, "macro_f1": 0.0, "note": "too few samples"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    n_classes = len(label_list)
    model = train_lgbm_optuna(X_train, y_train, n_classes, n_trials, seed)
    pred = predict_labels(model, X_test, n_classes)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro", zero_division=0)

    return {
        "tier": tier_name,
        "target": target_col,
        "n": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": acc,
        "macro_f1": mf1,
    }


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

    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading T4 data...")
    df = load_data(args.local_dir)
    print(f"Loaded {len(df)} rows")

    # Detect available features
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        raise SystemExit(f"No usable features found. Expected: {FEATURE_COLS}")
    print(f"Features: {available}")

    # Build tiers
    tiers = [
        ("Tier 1: All Data", df.copy()),
        ("Tier 2: Non-confounded", df[~df["confound_flag"].astype(bool)].copy()),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            df[(~df["confound_flag"].astype(bool)) & (df["direction_label"] != "flat")].copy(),
        ),
    ]

    results = []

    # Direction evaluation
    print("\n=== DIRECTION ===")
    for tier_name, tier_df in tiers:
        # For Tier 3, only up/down labels exist
        if "non-flat" in tier_name:
            labels = ["up", "down"]
        else:
            labels = DIRECTION_LABELS
        r = evaluate_tier(tier_name, tier_df, available, "direction_label", labels,
                          args.trials, args.seed, args.test_size)
        results.append(r)
        print(f"  {tier_name}: n={r['n']}  Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']*100:.2f}%")

    # Magnitude evaluation
    print("\n=== MAGNITUDE ===")
    for tier_name, tier_df in tiers:
        r = evaluate_tier(tier_name, tier_df, available, "magnitude_bucket", MAGNITUDE_LABELS,
                          args.trials, args.seed, args.test_size)
        results.append(r)
        print(f"  {tier_name}: n={r['n']}  Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']*100:.2f}%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
