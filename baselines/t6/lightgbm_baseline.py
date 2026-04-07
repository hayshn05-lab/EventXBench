#!/usr/bin/env python3
"""T6 LightGBM Baseline -- Cross-Market Propagation.

Trains a LightGBM multiclass classifier on numeric features from the T6
label file with Optuna hyperparameter tuning.  Uses train/test splits from
the data loader and reports Macro-F1.

Usage:
    python -m baselines.t6.lightgbm_baseline
    python -m baselines.t6.lightgbm_baseline --n-trials 20 --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_sample_weight

import eventxbench

LABEL_ORDER = ["no_cross_market_effect", "cross_market_effect", "insufficient_data"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
RANDOM_STATE = 42
N_TRIALS = 20

# Numeric features expected in the T6 data.  The script will use whichever
# of these are present in the loaded DataFrame.
CANDIDATE_FEATURES = [
    "sibling_count",
    "moved_sibling_count",
    "primary_delta_h",
    "confound_flag",
    "like_count",
    "reply_count",
    "view_count",
    "follower_count",
    "price_t0",
    "volume_24h_baseline",
]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def _select_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of CANDIDATE_FEATURES that exist and are numeric."""
    available = []
    for col in CANDIDATE_FEATURES:
        if col in df.columns:
            available.append(col)
    # Also include any column that looks numeric and isn't an ID or label
    skip = {"tweet_id", "primary_condition_id", "condition_id", "label",
            "split", "insufficient_data_flag", "confound_flag_orig"}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in skip and col not in available:
            available.append(col)
    return available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T6 LightGBM cross-market baseline")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument(
        "--exclude-insufficient",
        action="store_true",
        help="Exclude rows with insufficient_data label",
    )
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    data = eventxbench.load_task("t6", local_dir=args.local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
    else:
        # Single DataFrame -- do 80/20 split
        df = data
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

    # Filter insufficient_data_flag if present
    if "insufficient_data_flag" in train_df.columns:
        train_df = train_df[train_df["insufficient_data_flag"] == False].copy()
        test_df = test_df[test_df["insufficient_data_flag"] == False].copy()

    if args.exclude_insufficient:
        train_df = train_df[train_df["label"] != "insufficient_data"].copy()
        test_df = test_df[test_df["label"] != "insufficient_data"].copy()
        label_order = [l for l in LABEL_ORDER if l != "insufficient_data"]
    else:
        label_order = LABEL_ORDER

    label_to_id = {lab: i for i, lab in enumerate(label_order)}

    # Filter confounded rows from test set
    if "confound_flag" in test_df.columns:
        test_df = test_df[test_df["confound_flag"] == False].copy()

    # Determine features
    feature_cols = _select_features(train_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in T6 data.")

    for frame in (train_df, test_df):
        frame[feature_cols] = frame[feature_cols].fillna(0.0)

    # Filter to valid labels
    train_df = train_df[train_df["label"].isin(label_order)].copy()
    test_df = test_df[test_df["label"].isin(label_order)].copy()

    y_train = train_df["label"].map(label_to_id)
    y_test = test_df["label"].map(label_to_id)
    X_train = train_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    print(f"Train class distribution:\n{train_df['label'].value_counts().to_string()}")

    sample_weights = compute_sample_weight("balanced", y_train)

    # -- Optuna tuning on train set with CV ---------------------------------
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        list(skf.split(X_train, y_train))
    except ValueError:
        skf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": len(label_order),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "random_state": RANDOM_STATE,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

        oof_preds = np.zeros((len(X_train), len(label_order)))
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            w_tr = sample_weights[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            gbm = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            oof_preds[val_idx] = gbm.predict(X_val)

        pred_labels = np.argmax(oof_preds, axis=1)
        return f1_score(y_train, pred_labels, average="macro", zero_division=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # -- Train final model on full training set -----------------------------
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "multiclass",
            "num_class": len(label_order),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "random_state": RANDOM_STATE,
        }
    )

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    # Use a small validation hold-out for early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_train, y_train, sample_weights,
        test_size=0.15, random_state=RANDOM_STATE, stratify=y_train,
    )
    dtrain_final = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dval_final = lgb.Dataset(X_val, label=y_val, reference=dtrain_final)

    model = lgb.train(
        best_params,
        dtrain_final,
        valid_sets=[dval_final],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # -- Evaluate on test set -----------------------------------------------
    test_prob = model.predict(X_test)
    pred_test = np.argmax(test_prob, axis=1)

    test_macro_f1 = f1_score(y_test, pred_test, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, pred_test)

    print(f"\n=== T6 LightGBM Results ===")
    print(f"  Test Macro-F1: {test_macro_f1:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Best params: {study.best_params}")

    # Feature importance
    feat_imp = sorted(
        zip(feature_cols, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1],
        reverse=True,
    )
    print("  Top 5 features:")
    for name, imp in feat_imp[:5]:
        print(f"    {name}: {imp:.1f}")


if __name__ == "__main__":
    main()
