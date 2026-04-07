#!/usr/bin/env python3
"""T5 LightGBM Baseline -- Impact Persistence (Decay Classification).

Extracts numeric features from price_impact and volume_multiplier JSON
fields, trains a LightGBM classifier with Optuna hyperparameter tuning
using stratified 5-fold cross-validation.  Reports Macro-F1.

Note: In the original codebase this task is referred to as T7 / task5+7,
but in the paper and public release it is T5.

Usage:
    python -m baselines.t5.lightgbm_baseline
    python -m baselines.t5.lightgbm_baseline --n-trials 20 --local-dir /path/to/data
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

DECAY_LABELS = ["transient", "sustained", "reversal"]
HORIZONS = ["15m", "30m", "1h", "2h", "6h"]
RANDOM_STATE = 42
N_TRIALS = 10


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _parse_json_col(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature DataFrame from price_impact and volume_multiplier columns."""
    feature_rows = []

    for _, row in df.iterrows():
        feats: dict[str, float] = {}

        # Extract horizon values from JSON or flat columns
        for prefix in ["price_impact", "volume_multiplier"]:
            json_col = f"{prefix}_json"
            if json_col in row.index:
                d = _parse_json_col(row[json_col])
            else:
                d = {}
            for h in HORIZONS:
                col_name = f"{prefix}_{h}"
                if col_name in row.index and pd.notna(row[col_name]):
                    feats[col_name] = float(row[col_name])
                elif h in d and d[h] is not None:
                    feats[col_name] = float(d[h])
                else:
                    feats[col_name] = 0.0

        # Derived features
        pi_vals = [feats[f"price_impact_{h}"] for h in HORIZONS]
        vm_vals = [feats[f"volume_multiplier_{h}"] for h in HORIZONS]

        feats["pi_max"] = max(pi_vals) if pi_vals else 0.0
        feats["pi_min"] = min(pi_vals) if pi_vals else 0.0
        feats["pi_range"] = feats["pi_max"] - feats["pi_min"]
        feats["pi_mean"] = np.mean(pi_vals)
        feats["pi_std"] = np.std(pi_vals)

        # Ratio of 2h to 15m (persistence signal)
        pi_15m = feats.get("price_impact_15m", 0.0)
        pi_2h = feats.get("price_impact_2h", 0.0)
        feats["pi_2h_over_15m"] = pi_2h / (abs(pi_15m) + 1e-9)

        feats["vm_max"] = max(vm_vals) if vm_vals else 0.0
        feats["vm_mean"] = np.mean(vm_vals)

        # Confound flag if available
        if "confound_flag" in row.index:
            feats["confound_flag"] = float(bool(row["confound_flag"]))

        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T5 LightGBM decay classification baseline")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = data

    df = df[df["decay_class"].isin(DECAY_LABELS)].reset_index(drop=True)
    print(f"T5 samples: {len(df)}")
    print(f"Class distribution:\n{df['decay_class'].value_counts().to_string()}")

    # -- Feature extraction -------------------------------------------------
    X = _extract_features(df)
    feature_cols = list(X.columns)
    label_to_id = {lab: i for i, lab in enumerate(DECAY_LABELS)}
    y = df["decay_class"].map(label_to_id)

    sample_weights = compute_sample_weight("balanced", y)

    # -- Stratified cross-validation with Optuna ----------------------------
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        list(skf.split(X, y))  # validate feasibility
    except ValueError:
        print("Warning: falling back to regular KFold.")
        skf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": len(DECAY_LABELS),
            "metric": "multi_logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": RANDOM_STATE,
        }

        oof_preds = np.zeros((len(X), len(DECAY_LABELS)))
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            w_tr = sample_weights[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

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
        return f1_score(y, pred_labels, average="macro", zero_division=0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # -- Final OOF pass with best params ------------------------------------
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "multiclass",
            "num_class": len(DECAY_LABELS),
            "metric": "multi_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
        }
    )

    oof_preds = np.zeros((len(X), len(DECAY_LABELS)))
    feat_imp_accum = np.zeros(len(feature_cols), dtype=float)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        w_tr = sample_weights[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        gbm = lgb.train(
            best_params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        oof_preds[val_idx] = gbm.predict(X_val)
        feat_imp_accum += gbm.feature_importance(importance_type="gain")

    final_preds = np.argmax(oof_preds, axis=1)
    macro_f1 = f1_score(y, final_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, final_preds, average="weighted", zero_division=0)
    acc = accuracy_score(y, final_preds)

    # -- Report -------------------------------------------------------------
    print(f"\n=== T5 LightGBM Results (5-fold CV) ===")
    print(f"  Samples: {len(df)}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Best params: {study.best_params}")

    feat_imp = sorted(
        zip(feature_cols, feat_imp_accum / skf.get_n_splits()),
        key=lambda x: x[1],
        reverse=True,
    )
    print("  Top 5 features:")
    for name, imp in feat_imp[:5]:
        print(f"    {name}: {imp:.1f}")


if __name__ == "__main__":
    main()
