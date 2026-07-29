#!/usr/bin/env python3
"""T5 LightGBM Baseline -- Impact Persistence (Decay Classification).

Uses only prediction-time market-day features, tunes a LightGBM classifier on
the frozen training split with stratified cross-validation, and evaluates once
on the frozen test split.

Note: In the original codebase this task is referred to as T7 / task5+7,
but in the paper and public release it is T5.

Usage:
    python -m baselines.t5.lightgbm_baseline
    python -m baselines.t5.lightgbm_baseline --n-trials 20 --local-dir /path/to/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import eventxbench

DECAY_LABELS = ["transient", "sustained", "reversal"]
HORIZONS = ["1d", "3d", "7d"]
RANDOM_STATE = 42
N_TRIALS = 10

# Continuous targets are future outcomes, never input features.
CONTINUOUS_TARGETS = {
    **{f"drift_magnitude_{h}": f"drift_magnitude_{h}" for h in HORIZONS},
    **{f"volume_multiplier_{h}": f"volume_multiplier_{h}" for h in HORIZONS},
}

FEATURE_COLS = [
    "price_d",
    "n_posts",
    "volume_baseline_14d",
    "volume_baseline_n_obs",
]


# ---------------------------------------------------------------------------
# Spearman rho (pure-Python, ties-aware)
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


def _spearman(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / ((vx * vy) ** 0.5)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features observable by the end of bundle day d."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        raise ValueError(f"No prediction-time features found; expected {FEATURE_COLS}")
    features = df[available].apply(pd.to_numeric, errors="coerce").copy()

    # Calendar seasonality is known at prediction time and does not encode
    # future endpoints or labels.
    if "bundle_day" in df.columns:
        day = pd.to_datetime(df["bundle_day"], errors="coerce", utc=True)
        dow = day.dt.dayofweek.astype(float)
        month = day.dt.month.astype(float)
        features["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

    return features


def _classification_groups(df: pd.DataFrame) -> pd.Series:
    """Return leakage-safe CV groups, preferring event clusters per row."""
    groups = pd.Series(pd.NA, index=df.index, dtype="string")
    if "event_cluster_id" in df.columns:
        event_groups = df["event_cluster_id"].astype("string")
        usable = event_groups.notna() & event_groups.str.strip().ne("")
        groups.loc[usable] = "event:" + event_groups.loc[usable]
    if "condition_id" in df.columns:
        condition_groups = df["condition_id"].astype("string")
        missing = groups.isna() | groups.str.strip().eq("")
        groups.loc[missing] = "condition:" + condition_groups.loc[missing]

    missing = groups.isna() | groups.str.strip().eq("")
    if missing.any():
        raise ValueError(
            "Classification CV requires event_cluster_id or condition_id "
            f"for every row; {int(missing.sum())} row(s) have neither"
        )
    return groups.astype(str).reset_index(drop=True)


def _classification_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    max_splits: int = 5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Build grouped classification folds and verify zero group overlap."""
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError(
            f"Classification CV requires at least 2 groups; found {n_groups}"
        )
    expected_classes = set(y.astype(int).tolist())

    def valid_class_support(
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        return all(
            set(y.iloc[train_idx].astype(int)) == expected_classes
            and set(y.iloc[validation_idx].astype(int)) == expected_classes
            for train_idx, validation_idx in splits
        )

    splits = []
    splitter_name = ""
    for n_splits in range(min(max_splits, n_groups), 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        try:
            candidate = list(splitter.split(X, y, groups))
        except ValueError:
            continue
        if valid_class_support(candidate):
            splits = candidate
            splitter_name = "StratifiedGroupKFold"
            break

    if not splits:
        for n_splits in range(min(max_splits, n_groups), 1, -1):
            splitter = GroupKFold(n_splits=n_splits)
            candidate = list(splitter.split(X, y, groups))
            if valid_class_support(candidate):
                splits = candidate
                splitter_name = "GroupKFold"
                break
    if not splits:
        raise ValueError(
            "Could not construct group-disjoint classification folds with "
            "all classes present in every train and validation fold"
        )

    for train_idx, validation_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        validation_groups = set(groups.iloc[validation_idx])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(
                "Grouped CV leaked classification groups across folds: "
                f"{sorted(overlap)[:5]}"
            )
    return splits, splitter_name


def _prediction_records(
    test_df: pd.DataFrame,
    continuous_predictions: dict[str, np.ndarray],
    decay_predictions: dict[int, str],
) -> list[dict]:
    """Serialize every test row, adding decay predictions only when labeled."""
    records: list[dict] = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        rec = {
            "instance_id": str(row["instance_id"]),
            "condition_id": str(row["condition_id"]),
            "bundle_day": str(row["bundle_day"]),
        }
        if i in decay_predictions:
            rec["decay_class"] = decay_predictions[i]
        for target_name, predictions in continuous_predictions.items():
            value = predictions[i]
            if not np.isnan(value):
                rec[target_name] = float(value)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T5 LightGBM decay classification baseline")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--output", default=None,
                        help="Optional frozen-test predictions JSONL for evaluate.py")
    parser.add_argument("--metrics-output", default=None,
                        help="Optional JSON summary of frozen-test metrics")
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    data = eventxbench.load_task("t5", local_dir=args.local_dir)
    if not isinstance(data, tuple):
        raise SystemExit(
            "T5 LightGBM requires frozen train/test files; received an unsplit dataset"
        )
    train_full_df, test_full_df = (
        frame.reset_index(drop=True) for frame in data
    )
    train_decay_mask = train_full_df["decay_class"].isin(DECAY_LABELS)
    test_decay_mask = test_full_df["decay_class"].isin(DECAY_LABELS)
    train_df = train_full_df.loc[train_decay_mask].reset_index(drop=True)
    test_df = test_full_df.loc[test_decay_mask].reset_index(drop=True)
    test_decay_indices = np.flatnonzero(test_decay_mask.to_numpy())
    print(
        "T5 full train/test samples: "
        f"{len(train_full_df)}/{len(test_full_df)}"
    )
    print(
        "T5 decay-labeled train/test samples: "
        f"{len(train_df)}/{len(test_df)}"
    )
    print(
        "Train class distribution:\n"
        + train_df["decay_class"].value_counts().to_string()
    )
    print(
        "Test class distribution:\n"
        + test_df["decay_class"].value_counts().to_string()
    )

    # -- Feature extraction -------------------------------------------------
    X_train_full = _extract_features(train_full_df)
    X_test_full = _extract_features(test_full_df)
    feature_cols = list(X_train_full.columns)
    X_test_full = X_test_full.reindex(columns=feature_cols)
    X_train = X_train_full.loc[train_decay_mask].reset_index(drop=True)
    X_test = X_test_full.loc[test_decay_mask].reset_index(drop=True)
    label_to_id = {lab: i for i, lab in enumerate(DECAY_LABELS)}
    y_train = train_df["decay_class"].map(label_to_id).astype(int)
    y_test = test_df["decay_class"].map(label_to_id).astype(int)

    sample_weights = compute_sample_weight("balanced", y_train)

    # -- Train-only cross-validation with Optuna -----------------------------
    classification_groups = _classification_groups(train_df)
    cv_splits, cv_name = _classification_cv_splits(
        X_train, y_train, classification_groups
    )
    print(
        f"Classification CV: {cv_name}, folds={len(cv_splits)}, "
        f"groups={classification_groups.nunique()}"
    )

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

        oof_preds = np.zeros((len(X_train), len(DECAY_LABELS)))
        for tr_idx, val_idx in cv_splits:
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            w_tr = sample_weights[tr_idx]
            X_val = X_train.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval = lgb.Dataset(
                X_val, label=y_train.iloc[val_idx], reference=dtrain
            )
            gbm = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            oof_preds[val_idx] = gbm.predict(X_val)

        pred_labels = np.argmax(oof_preds, axis=1)
        return f1_score(
            y_train, pred_labels, average="macro", zero_division=0
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # -- Final frozen-train model and frozen-test evaluation ----------------
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

    dtrain_full = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    final_model = lgb.train(best_params, dtrain_full, num_boost_round=300)
    test_probabilities = final_model.predict(X_test)
    final_preds = np.argmax(test_probabilities, axis=1)
    macro_f1 = f1_score(
        y_test, final_preds, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, final_preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_test, final_preds)

    # -- Report -------------------------------------------------------------
    print(f"\n=== T5 LightGBM Results (frozen test) ===")
    print(f"  Train/test: {len(train_df)}/{len(test_df)}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Best params: {study.best_params}")

    feat_imp = sorted(
        zip(
            feature_cols,
            final_model.feature_importance(importance_type="gain"),
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    print("  Top 5 features:")
    for name, imp in feat_imp[:5]:
        print(f"    {name}: {imp:.1f}")

    # ----------------------------------------------------------------------
    # Continuous targets: train on frozen train, evaluate on frozen test.
    # ----------------------------------------------------------------------
    print(f"\n=== T5 Continuous Targets (frozen-test Spearman rho) ===")
    reg_params = {
        "verbosity": -1,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
    }

    test_cont: dict[str, np.ndarray] = {}
    continuous_metrics: dict[str, dict[str, float | int | None]] = {}
    for target_name, target_col in CONTINUOUS_TARGETS.items():
        train_target = pd.to_numeric(
            train_full_df[target_col], errors="coerce"
        )
        test_target = pd.to_numeric(
            test_full_df[target_col], errors="coerce"
        )
        train_valid = train_target.notnull()
        test_valid = test_target.notnull()

        full_pred = np.full(len(test_full_df), np.nan)
        if train_valid.sum() < 10:
            print(
                f"  {target_name}: too few train samples "
                f"({train_valid.sum()}), skipped"
            )
            test_cont[target_name] = full_pred
            continue

        dtrain_reg = lgb.Dataset(
            X_train_full.loc[train_valid],
            label=train_target.loc[train_valid].values.astype(float),
        )
        regressor = lgb.train(
            reg_params, dtrain_reg, num_boost_round=300
        )
        full_pred = regressor.predict(X_test_full)
        test_cont[target_name] = full_pred

        actual = test_target.loc[test_valid].values.astype(float)
        evaluated_predictions = full_pred[test_valid.to_numpy()]
        rho = (
            _spearman(evaluated_predictions.tolist(), actual.tolist())
            if test_valid.sum() >= 2
            else None
        )
        continuous_metrics[target_name] = {
            "spearman_rho": rho,
            "n_test": int(test_valid.sum()),
        }
        rho_txt = "N/A" if rho is None else f"{rho:.4f}"
        print(
            f"  {target_name}: rho={rho_txt} "
            f"(n={int(test_valid.sum())})"
        )

    # -- Optional per-instance predictions for evaluate.py -------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        decay_predictions = {
            int(full_index): DECAY_LABELS[int(prediction)]
            for full_index, prediction in zip(test_decay_indices, final_preds)
        }
        prediction_records = _prediction_records(
            test_full_df,
            test_cont,
            decay_predictions,
        )
        with out_path.open("w", encoding="utf-8") as f:
            for rec in prediction_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nPredictions saved to {out_path}")

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "protocol": "tier2.temporal.v2",
            "n_train": len(train_full_df),
            "n_test": len(test_full_df),
            "n_decay_train": len(train_df),
            "n_decay_test": len(test_df),
            "features": feature_cols,
            "decay_class": {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            },
            "classification_cv": {
                "splitter": cv_name,
                "folds": len(cv_splits),
                "groups": int(classification_groups.nunique()),
                "group_key": "event_cluster_id with row-wise condition_id fallback",
            },
            "continuous": continuous_metrics,
            "best_params": study.best_params,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
