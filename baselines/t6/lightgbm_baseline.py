#!/usr/bin/env python3
"""T6 LightGBM Baseline -- Cross-Market Co-Movement.

Trains a LightGBM multiclass classifier on the T6 market-day bundle /
horizon data with Optuna hyperparameter tuning.  Supports KDD v2 bundle
labels (no_effect/primary_only/cross_market) and per-horizon evaluation,
and falls back to legacy v1 data (no_cross_market_effect/primary_mover/
propagated_signal).

The model uses only prediction-time features (allowed by the manifest's
``feature_target_boundary.prediction_time_fields``), never label-time or
post-decision fields (primary_z_h, primary_moved, sibling_move_count,
cascade_size, confound_*, headline_label, four_way_label, etc.).

Usage:
    python -m baselines.t6.lightgbm_baseline
    python -m baselines.t6.lightgbm_baseline --local-dir KDD/data/t6_kdd_v2 --n-trials 20
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import eventxbench

V1_LABELS = ["no_cross_market_effect", "primary_mover", "propagated_signal"]
V2_LABELS = ["no_effect", "primary_only", "cross_market"]
RANDOM_STATE = 42
N_TRIALS = 20

# Only prediction-time features allowed per the KDD v2 manifest's
# feature_target_boundary.prediction_time_fields. We encode first_post_time
# as epoch seconds; categorical fields (domain, condition_id, bundle_day)
# are excluded from LightGBM input unless label-encoded.
ALLOWED_NUM_FEATURES = [
    "n_posts",
    "followers_max",
    "engagement_sum",
    "engagement_max",
    "max_final_grade",
    "num_siblings_visible_d",
    # horizon_days is a model input in the default joint model
    "horizon_days",
]

# v1 candidate features (legacy) -- numeric columns from the post-level format.
LEGACY_V1_CANDIDATE_FEATURES = [
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


def _select_features(df: pd.DataFrame, is_v2: bool) -> list[str]:
    """Return the subset of allowed/predict-time features present in df."""
    if is_v2:
        return [c for c in ALLOWED_NUM_FEATURES if c in df.columns]
    # Legacy v1: use v1 candidate features + any numeric non-ID column
    available = [c for c in LEGACY_V1_CANDIDATE_FEATURES if c in df.columns]
    skip = {"tweet_id", "primary_condition_id", "condition_id", "label",
            "split", "insufficient_data_flag", "confound_flag_orig"}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in skip and col not in available:
            available.append(col)
    return available


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
    y: pd.Series,
    groups: np.ndarray,
    seed: int = RANDOM_STATE,
    max_splits: int = 5,
) -> StratifiedGroupKFold:
    """Build the largest feasible stratified, group-disjoint CV splitter."""
    y_values = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y_values)
    if len(y_values) != len(groups):
        raise ValueError("CV labels and groups must have the same length")
    if len(classes) < 2:
        raise ValueError("Group-safe stratified CV requires at least two classes")

    n_groups = len(np.unique(groups))
    groups_per_class = min(
        len(np.unique(groups[y_values == label])) for label in classes
    )
    upper = min(max_splits, n_groups, groups_per_class)
    for n_splits in range(upper, 1, -1):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        try:
            splits = list(
                cv.split(np.zeros(len(y_values)), y_values, groups)
            )
        except ValueError:
            continue
        if all(
            len(np.unique(y_values[train_idx])) == len(classes)
            and len(np.unique(y_values[val_idx])) == len(classes)
            for train_idx, val_idx in splits
        ):
            return cv
    raise ValueError(
        "Need at least two group-disjoint folds with every class in training"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="T6 LightGBM cross-market baseline")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--per-horizon", action="store_true",
                        help=("Train a separate model for each horizon and omit "
                              "horizon_days (default: one joint model)"))
    parser.add_argument("--output", default=None,
                        help="Optional JSONL output path for predictions")
    parser.add_argument("--metrics-output", default=None,
                        help="Optional JSON metrics output")
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    data = eventxbench.load_task("t6", local_dir=args.local_dir)
    if isinstance(data, tuple):
        train_df, test_df = data
    else:
        df = data
        if "split" not in df.columns:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)
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
        is_v2 = True
    else:
        labels = V1_LABELS
        is_v2 = False
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # Filter to labels and (for v2) exclude confounded from test
    train_df = train_df[train_df[label_col].isin(labels)].reset_index(drop=True)
    test_df = test_df[test_df[label_col].isin(labels)].reset_index(drop=True)
    if is_v2 and "confound_flag" in test_df.columns:
        test_df = test_df[test_df["confound_flag"] == False].reset_index(drop=True)

    feature_cols = _select_features(train_df, is_v2)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found.")
    print(f"Labels: {labels}  (is_v2={is_v2})")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train class distribution:\n{train_df[label_col].value_counts().to_string()}")

    for frame in (train_df, test_df):
        frame[feature_cols] = frame[feature_cols].fillna(0.0).astype(float)

    has_horizons = is_v2 and "horizon_days" in train_df.columns
    horizons = (
        sorted(train_df["horizon_days"].unique())
        if has_horizons and args.per_horizon
        else [None]
    )

    def _train_eval(
        tr_x, tr_y, tr_groups, te_x, te_y, feature_names,
        n_trials=args.n_trials,
    ):
        sample_w = compute_sample_weight("balanced", tr_y)
        cv = _make_stratified_group_cv(tr_y, tr_groups)

        def objective(trial):
            params = {
                "objective": "multiclass",
                "num_class": len(labels),
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "feature_pre_filter": False,
                "random_state": RANDOM_STATE,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            oof = np.zeros((len(tr_x), len(labels)))
            for tr_idx, val_idx in cv.split(tr_x, tr_y, tr_groups):
                dtr = lgb.Dataset(tr_x.iloc[tr_idx], label=tr_y.iloc[tr_idx],
                                  weight=sample_w[tr_idx])
                dval = lgb.Dataset(tr_x.iloc[val_idx], label=tr_y.iloc[val_idx],
                                   reference=dtr)
                gbm = lgb.train(params, dtr, valid_sets=[dval], num_boost_round=500,
                                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
                oof[val_idx] = gbm.predict(tr_x.iloc[val_idx])
            pred = np.argmax(oof, axis=1)
            return f1_score(tr_y, pred, average="macro", zero_division=0)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params.copy()
        best.update({"objective": "multiclass", "num_class": len(labels),
                     "metric": "multi_logloss", "verbosity": -1,
                     "boosting_type": "gbdt", "feature_pre_filter": False,
                     "random_state": RANDOM_STATE})

        # Train on full train
        dtrain = lgb.Dataset(tr_x, label=tr_y, weight=sample_w)
        model = lgb.train(best, dtrain, num_boost_round=study.best_params.get("num_boost_round", 300))
        te_prob = model.predict(te_x)
        te_pred = np.argmax(te_prob, axis=1)
        macro = f1_score(te_y, te_pred, average="macro", zero_division=0)
        acc = accuracy_score(te_y, te_pred)
        return {
            "best_params": study.best_params, "best_cv_f1": float(study.best_value),
            "macro_f1": float(macro), "accuracy": float(acc),
            "predictions": te_pred.tolist(), "scores": te_prob.tolist(),
            "feature_importance": dict(zip(
                feature_names,
                model.feature_importance(importance_type="gain").tolist(),
            )),
        }

    results: list[dict] = []
    all_predictions: list[dict] = []
    for h in horizons:
        if h is not None:
            tr_h = train_df[train_df["horizon_days"] == h].reset_index(drop=True)
            te_h = test_df[test_df["horizon_days"] == h].reset_index(drop=True)
        else:
            tr_h, te_h = train_df, test_df
        if tr_h.empty or te_h.empty:
            print(f"  H={h}: too few samples (train={len(tr_h)}, test={len(te_h)}), skipped")
            continue

        # A joint model uses horizon_days; separate horizon models omit it.
        feats = (
            [c for c in feature_cols if c != "horizon_days"]
            if h is not None else feature_cols
        )
        tr_y = tr_h[label_col].map(label_to_id)
        te_y = te_h[label_col].map(label_to_id)
        tr_groups = _group_values(tr_h)
        mode = f"H={int(h)}d" if h is not None else "joint horizons"
        print(
            f"\n--- {mode} (train={len(tr_h)}, test={len(te_h)}, "
            f"feats={len(feats)}) ---"
        )
        r = _train_eval(
            tr_h[feats], tr_y, tr_groups, te_h[feats], te_y, feats,
            n_trials=args.n_trials,
        )
        print(f"  Best CV Macro-F1: {r['best_cv_f1']:.4f}")
        print(f"  Test Macro-F1:    {r['macro_f1']:.4f}")
        print(f"  Test Accuracy:   {r['accuracy']:.4f}")
        print(f"  Best params:     {r['best_params']}")
        fi = sorted(r["feature_importance"].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 features:  {fi}")
        r["horizon_days"] = int(h) if h is not None else None
        r["training_mode"] = "per_horizon" if h is not None else "joint"
        r["n_train"] = len(tr_h)
        r["n_test"] = len(te_h)
        results.append(r)
        # Save predictions
        if args.output:
            for row, pred, scores in zip(te_h.to_dict("records"),
                                         r["predictions"], r["scores"]):
                all_predictions.append({
                    "condition_id": str(row.get("condition_id", "")),
                    "bundle_day": row.get("bundle_day", ""),
                    "horizon_days": (
                        int(row["horizon_days"])
                        if row.get("horizon_days") is not None
                        else (int(h) if h is not None else None)
                    ),
                    "instance_id": str(row.get("instance_id", "")),
                    "prediction": labels[int(pred)],
                    "pred_label": labels[int(pred)],
                    "scores": {labels[i]: float(scores[i]) for i in range(len(labels))},
                })

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in all_predictions:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nPredictions saved to {out}")
    if args.metrics_output:
        mp = Path(args.metrics_output)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"Metrics saved to {mp}")


if __name__ == "__main__":
    main()
