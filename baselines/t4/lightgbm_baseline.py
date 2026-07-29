#!/usr/bin/env python3
"""T4 LightGBM Baseline -- Market Movement Prediction.

Trains LightGBM classifiers for direction and magnitude using prediction-time
features available in the public market-day bundle. Hyperparameters are tuned
on the frozen training split only; the frozen test split is used once.

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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTION_LABELS = ["flat", "up", "down"]
MAGNITUDE_LABELS = ["small", "medium", "large"]
DELTA_COLS = ["delta_1d", "delta_3d", "delta_7d"]
HORIZON_NAMES = ["1d", "3d", "7d"]

# All features are observable at the end of bundle day d. Forward prices,
# deltas, labels, confound flags, and fitted label cut points are excluded.
FEATURE_COLS = [
    "price_d",
    "sigma_14d",
    "sigma_n_obs",
    "n_posts",
    "followers_max",
    "engagement_sum",
    "engagement_max",
    "max_final_grade",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T4 LightGBM baseline")
    parser.add_argument("--local-dir", default=None, help="Local data directory")
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", default="t4_lightgbm_predictions.jsonl",
                        help="Per-instance predictions JSONL for evaluate.py")
    parser.add_argument("--metrics-output", default=None,
                        help="Optional JSON summary of tier metrics")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(local_dir: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import eventxbench

    if local_dir:
        result = eventxbench.load_task("t4", local_dir=local_dir)
    else:
        result = eventxbench.load_task("t4")
    if not isinstance(result, tuple):
        raise ValueError(
            "T4 LightGBM requires frozen train/test files; received an unsplit dataset"
        )
    train_df, test_df = result
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _group_values(df: pd.DataFrame) -> np.ndarray:
    """Use event clusters when present, with a row-wise market fallback."""
    groups = []
    for _, row in df.iterrows():
        event_id = row.get("event_cluster_id")
        condition_id = row.get("condition_id")
        if pd.notna(event_id) and str(event_id).strip():
            groups.append(f"event:{event_id}")
        elif pd.notna(condition_id) and str(condition_id).strip():
            groups.append(f"condition:{condition_id}")
        else:
            raise ValueError(
                "Group-safe CV requires event_cluster_id or condition_id "
                "for every row"
            )
    return np.asarray(groups)


def _make_stratified_group_cv(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    max_splits: int = 5,
) -> StratifiedGroupKFold:
    """Build the largest feasible stratified, group-disjoint CV splitter."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    if len(y) != len(groups):
        raise ValueError("CV labels and groups must have the same length")
    if len(classes) < 2:
        raise ValueError("Group-safe stratified CV requires at least two classes")

    n_groups = len(np.unique(groups))
    groups_per_class = min(
        len(np.unique(groups[y == label])) for label in classes
    )
    upper = min(max_splits, n_groups, groups_per_class)
    for n_splits in range(upper, 1, -1):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        try:
            splits = list(cv.split(np.zeros(len(y)), y, groups))
        except ValueError:
            continue
        if all(
            len(np.unique(y[train_idx])) == len(classes)
            and len(np.unique(y[val_idx])) == len(classes)
            for train_idx, val_idx in splits
        ):
            return cv
    raise ValueError(
        "Need at least two group-disjoint folds with every class in training"
    )


def _make_group_cv(groups: np.ndarray, max_splits: int = 5) -> GroupKFold:
    """Build the largest feasible group-disjoint splitter for regression."""
    n_splits = min(max_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Group-safe CV requires at least two distinct groups")
    return GroupKFold(n_splits=n_splits)


def train_lgbm_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_trials: int,
    seed: int,
) -> "lgb.Booster":
    """Train LightGBM with Optuna hyperparameter search, return best model."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = _make_stratified_group_cv(y_train, groups, seed)

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
        for tr_idx, val_idx in cv.split(X_train, y_train, groups):
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
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


def spearman_rho(x: list[float], y: list[float]) -> Optional[float]:
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
# Regression (continuous delta prediction -> Spearman rho)
# ---------------------------------------------------------------------------


def train_lgbm_regressor_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    seed: int,
) -> "lgb.Booster":
    """Train a LightGBM regressor with Optuna (MAE objective)."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = _make_group_cv(groups)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "random_state": seed,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train, groups):
            dtrain = lgb.Dataset(
                X_train.iloc[tr_idx], label=y_train[tr_idx]
            )
            dval = lgb.Dataset(
                X_train.iloc[val_idx], label=y_train[val_idx],
                reference=dtrain,
            )
            gbm = lgb.train(
                params, dtrain, valid_sets=[dval], num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            pred = gbm.predict(X_train.iloc[val_idx])
            scores.append(
                float(np.mean(np.abs(y_train[val_idx] - pred)))
            )
        return -float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params.copy()
    best.update({
        "verbosity": -1,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mae",
        "random_state": seed,
    })
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    return lgb.train(best, dtrain_full, num_boost_round=150)


# ---------------------------------------------------------------------------
# Tier evaluation
# ---------------------------------------------------------------------------


def evaluate_tier(
    tier_name: str,
    train_tier: pd.DataFrame,
    test_tier: pd.DataFrame,
    features: list[str],
    target_col: str,
    label_list: list[str],
    n_trials: int,
    seed: int,
    test_size: float,
) -> tuple[dict, pd.DataFrame]:
    """Tune on frozen train data and evaluate on the frozen test split."""

    label_map = {lab: i for i, lab in enumerate(label_list)}
    y_train = train_tier[target_col].map(label_map)
    y_test = test_tier[target_col].map(label_map)
    train_valid = y_train.notnull()
    test_valid = y_test.notnull()
    train_clean = train_tier[train_valid].copy()
    test_clean = test_tier[test_valid].copy()
    y_train = y_train[train_valid].values.astype(int)
    y_test = y_test[test_valid].values.astype(int)
    X_train = train_clean[features].copy().astype(float)
    X_test = test_clean[features].copy().astype(float)

    if len(X_train) < 10 or len(X_test) < 1:
        metrics = {
            "tier": tier_name,
            "target": target_col,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "note": "too few samples",
        }
        return metrics, pd.DataFrame()

    n_classes = len(label_list)
    train_groups = _group_values(train_clean)
    model = train_lgbm_optuna(
        X_train, y_train, train_groups, n_classes, n_trials, seed
    )
    pred = predict_labels(model, X_test, n_classes)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro", zero_division=0)

    metrics = {
        "tier": tier_name,
        "target": target_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": acc,
        "macro_f1": mf1,
    }
    id_cols = [
        c for c in ("instance_id", "condition_id", "bundle_day")
        if c in test_clean.columns
    ]
    pred_rows = test_clean[id_cols].copy()
    pred_rows[target_col] = [label_list[i] for i in pred]
    return metrics, pred_rows


def evaluate_regression_tier(
    tier_name: str,
    train_tier: pd.DataFrame,
    test_tier: pd.DataFrame,
    features: list[str],
    n_trials: int,
    seed: int,
    test_size: float,
) -> tuple[dict, pd.DataFrame]:
    """Train on the frozen train split and score each daily test horizon."""
    all_pred: list[float] = []
    all_actual: list[float] = []
    per_horizon: dict[str, Optional[float]] = {}
    n_train_by_horizon: dict[str, int] = {}
    n_test_by_horizon: dict[str, int] = {}
    notes_by_horizon: dict[str, str] = {}
    id_cols = [
        c for c in ("instance_id", "condition_id", "bundle_day")
        if c in test_tier.columns
    ]
    pred_rows = test_tier[id_cols].copy().reset_index(drop=True)
    for delta_col in DELTA_COLS:
        pred_rows[delta_col] = np.nan

    for delta_col, horizon in zip(DELTA_COLS, HORIZON_NAMES):
        train_valid = train_tier[delta_col].notnull()
        test_valid = test_tier[delta_col].notnull()
        train_clean = train_tier[train_valid].copy()
        n_train_by_horizon[horizon] = int(train_valid.sum())
        n_test_by_horizon[horizon] = int(test_valid.sum())

        if train_clean.empty or test_tier.empty:
            per_horizon[horizon] = None
            notes_by_horizon[horizon] = (
                "no train rows for target or no test rows"
            )
            continue

        X_train = train_clean[features].copy().astype(float)
        X_test = test_tier[features].copy().astype(float)
        train_groups = _group_values(train_clean)
        y_train = train_clean[delta_col].values.astype(float)

        model = train_lgbm_regressor_optuna(
            X_train, y_train, train_groups, n_trials, seed
        )
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        pred_rows[delta_col] = y_pred

        if test_valid.any():
            valid_mask = test_valid.to_numpy()
            y_test = test_tier.loc[test_valid, delta_col].values.astype(float)
            valid_pred = y_pred[valid_mask]
            per_horizon[horizon] = spearman_rho(
                valid_pred.tolist(), y_test.tolist()
            )
            all_pred.extend(valid_pred.tolist())
            all_actual.extend(y_test.tolist())
        else:
            per_horizon[horizon] = None

    flat_rho = spearman_rho(all_pred, all_actual)
    metrics = {
        "tier": tier_name,
        "target": "delta_curve",
        "n_train": len(train_tier),
        "n_test": len(test_tier),
        "n_train_by_horizon": n_train_by_horizon,
        "n_test_by_horizon": n_test_by_horizon,
        "spearman_by_horizon": per_horizon,
        "spearman_flat": flat_rho,
    }
    if notes_by_horizon:
        metrics["notes_by_horizon"] = notes_by_horizon
    if all(pred_rows[col].isna().all() for col in DELTA_COLS):
        metrics["note"] = "no horizon had enough training data"
    return metrics, pred_rows


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
    train_df, test_df = load_data(args.local_dir)
    print(f"Loaded train/test: {len(train_df)}/{len(test_df)} rows")

    # Detect available features
    available = [
        c for c in FEATURE_COLS
        if c in train_df.columns and c in test_df.columns
    ]
    if not available:
        raise SystemExit(f"No usable features found. Expected: {FEATURE_COLS}")
    print(f"Features: {available}")

    # Build tiers
    tiers = [
        ("Tier 1: All Data", train_df.copy(), test_df.copy()),
        (
            "Tier 2: Non-confounded",
            train_df[~train_df["confound_flag"].astype(bool)].copy(),
            test_df[~test_df["confound_flag"].astype(bool)].copy(),
        ),
        (
            "Tier 3: Active (non-confounded + non-flat)",
            train_df[
                (~train_df["confound_flag"].astype(bool))
                & (train_df["direction_label"] != "flat")
            ].copy(),
            test_df[
                (~test_df["confound_flag"].astype(bool))
                & (test_df["direction_label"] != "flat")
            ].copy(),
        ),
    ]

    results = []
    tier1_direction: pd.DataFrame | None = None
    tier1_magnitude: pd.DataFrame | None = None

    # Direction evaluation
    print("\n=== DIRECTION ===")
    for tier_name, tier_train, tier_test in tiers:
        # For Tier 3, only up/down labels exist
        if "non-flat" in tier_name:
            labels = ["up", "down"]
        else:
            labels = DIRECTION_LABELS
        r, pred_rows = evaluate_tier(
            tier_name, tier_train, tier_test, available, "direction_label",
            labels, args.trials, args.seed, args.test_size,
        )
        results.append(r)
        print(
            f"  {tier_name}: train={r['n_train']} test={r['n_test']}  "
            f"Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']*100:.2f}%"
        )
        if tier_name.startswith("Tier 1"):
            tier1_direction = pred_rows

    # Magnitude evaluation
    print("\n=== MAGNITUDE ===")
    for tier_name, tier_train, tier_test in tiers:
        labels = MAGNITUDE_LABELS if "non-flat" in tier_name else [
            "none", *MAGNITUDE_LABELS
        ]
        r, pred_rows = evaluate_tier(
            tier_name, tier_train, tier_test, available, "magnitude_bucket",
            labels, args.trials, args.seed, args.test_size,
        )
        results.append(r)
        print(
            f"  {tier_name}: train={r['n_train']} test={r['n_test']}  "
            f"Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']*100:.2f}%"
        )
        if tier_name.startswith("Tier 1"):
            tier1_magnitude = pred_rows

    # Continuous delta regression -> Spearman rho on the horizon curve
    print("\n=== CONTINUOUS DELTA (Spearman rho) ===")
    tier1_pred_rows: pd.DataFrame | None = None
    for tier_name, tier_train, tier_test in tiers:
        metrics, pred_rows = evaluate_regression_tier(
            tier_name, tier_train, tier_test, available, args.trials,
            args.seed, args.test_size,
        )
        results.append(metrics)
        if metrics.get("note"):
            print(f"  {tier_name}: {metrics['note']}")
            continue
        parts = []
        for h in HORIZON_NAMES:
            rho = metrics["spearman_by_horizon"].get(h)
            parts.append(f"{h}={'N/A' if rho is None else f'{rho:.4f}'}")
        flat = metrics["spearman_flat"]
        parts.append(f"flat={'N/A' if flat is None else f'{flat:.4f}'}")
        print(
            f"  {tier_name}: train={metrics['n_train']} "
            f"test={metrics['n_test']}  " + "  ".join(parts)
        )
        if tier_name.startswith("Tier 1"):
            tier1_pred_rows = pred_rows

    # Save per-instance predictions (Tier 1) for evaluation/evaluate.py
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_cols = [
        c for c in ("instance_id", "condition_id", "bundle_day")
        if c in test_df.columns
    ]
    merged_predictions = test_df[id_cols].copy()
    prediction_frames = [
        frame for frame in (tier1_direction, tier1_magnitude, tier1_pred_rows)
        if frame is not None and len(frame)
    ]
    for frame in prediction_frames:
        merged_predictions = merged_predictions.merge(
            frame, on=id_cols, how="left"
        )
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in merged_predictions.iterrows():
            rec = {
                "instance_id": str(row["instance_id"]),
                "condition_id": str(row["condition_id"]),
                "bundle_day": str(row["bundle_day"]),
                "direction_label": (
                    None if pd.isna(row.get("direction_label"))
                    else row.get("direction_label")
                ),
                "magnitude_bucket": (
                    None if pd.isna(row.get("magnitude_bucket"))
                    else row.get("magnitude_bucket")
                ),
            }
            for delta_col in DELTA_COLS:
                value = row.get(delta_col)
                rec[delta_col] = None if pd.isna(value) else float(value)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to {output_path}")

    # Save tier metrics summary
    if args.metrics_output:
        mpath = Path(args.metrics_output)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with mpath.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Metrics saved to {mpath}")


if __name__ == "__main__":
    main()
