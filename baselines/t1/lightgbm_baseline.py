#!/usr/bin/env python3
"""Leakage-safe T1 LightGBM baselines.

Two feature rungs are supported:

``market_only``
    Market question/description text, category/domain, causal text-length
    features, and optional precomputed market-embedding PCA components.

``market_social``
    All market-only inputs plus pre-market link/count/burst summaries.

Feature selection is an allowlist.  Targets and future-derived fields such as
volume, percentile rank, price, outcome, engagement, and follower counts are
never selected just because they happen to be numeric.

The hyperparameter search uses only the training split.  If group IDs are
available, every cross-validation fold keeps each group atomic.  A dedicated
validation split is reported before the final model is retrained on
train+validation and evaluated on test.  Legacy train/test datasets remain
supported.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - supported sklearn versions provide it
    StratifiedGroupKFold = None  # type: ignore[assignment]


LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]
LABEL_MAP = {label: idx for idx, label in enumerate(LABEL_ORDER)}

MARKET_NUMERIC_FEATURES = (
    "question_char_len",
    "description_char_len",
)
MARKET_CATEGORICAL_FEATURES = ("category", "domain")
MARKET_TEXT_FEATURES = ("question", "description")

SOCIAL_NUMERIC_FEATURES = (
    "score",
    "linked_tweet_count",
    "avg_link_confidence",
    "max_link_confidence",
    "text_similarity",
    "tweet_count",
    "unique_user_count",
    "burst_duration_hours",
    "in_3d_tweet_count",
    "in_3d_share",
)

GROUP_COLUMNS = ("market_group_id", "event_group_id", "group_id")
PCA_COLUMN_RE = re.compile(r"^market_embedding_pca_\d+$")


@dataclass(frozen=True)
class FeatureSpec:
    """Columns used by one feature rung, grouped by transformer type."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    text: tuple[str, ...]

    @property
    def source_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1 leakage-safe LightGBM baseline")
    parser.add_argument("--local-dir", default=None, help="Local data directory (skips HF)")
    parser.add_argument(
        "--feature-rung",
        choices=("market_only", "market_social"),
        default="market_social",
        help="Causal feature set to train",
    )
    parser.add_argument("--trials", type=int, default=10, help="Optuna trials; 0 uses defaults")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--tfidf-max-features", type=int, default=3000)
    parser.add_argument("--tfidf-min-df", type=int, default=1)
    parser.add_argument("--tfidf-ngram-max", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap search at 2 trials, 3 folds, 80 trees, and 1000 TF-IDF terms",
    )
    parser.add_argument("--output", default="t1_lightgbm_predictions.jsonl")
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional JSON path for CV/validation/test metrics and provenance",
    )
    return parser.parse_args()


def load_data(
    local_dir: Optional[str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Load train, optional validation, and test explicitly."""
    import eventxbench

    kwargs = {"local_dir": local_dir} if local_dir else {}
    train_df = eventxbench.load_task("t1", split="train", **kwargs)
    test_df = eventxbench.load_task("t1", split="test", **kwargs)
    try:
        validation_df = eventxbench.load_task("t1", split="validation", **kwargs)
    except KeyError:
        validation_df = None
    except ValueError as exc:
        if "has no 'validation' split" not in str(exc):
            raise
        validation_df = None

    if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise TypeError("T1 train/test splits must load as pandas DataFrames")
    if validation_df is not None and not isinstance(validation_df, pd.DataFrame):
        raise TypeError("T1 validation split must load as a pandas DataFrame")
    return train_df, validation_df, test_df


def _add_causal_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive text lengths from causal market text when exports omit them."""
    result = df.copy()
    for text_col, length_col in (
        ("question", "question_char_len"),
        ("description", "description_char_len"),
    ):
        if text_col in result.columns and length_col not in result.columns:
            result[length_col] = result[text_col].fillna("").astype(str).str.len()
    return result


def select_feature_spec(
    frames: Sequence[pd.DataFrame], feature_rung: str
) -> FeatureSpec:
    """Select only allowlisted columns shared by every modeled split."""
    if feature_rung not in {"market_only", "market_social"}:
        raise ValueError(f"Unknown feature rung: {feature_rung}")
    if not frames:
        raise ValueError("At least one DataFrame is required for feature selection")

    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)

    numeric = [col for col in MARKET_NUMERIC_FEATURES if col in common]
    numeric.extend(sorted(col for col in common if PCA_COLUMN_RE.fullmatch(col)))
    if feature_rung == "market_social":
        numeric.extend(col for col in SOCIAL_NUMERIC_FEATURES if col in common)

    categorical = [col for col in MARKET_CATEGORICAL_FEATURES if col in common]
    text = [col for col in MARKET_TEXT_FEATURES if col in common]
    spec = FeatureSpec(tuple(numeric), tuple(categorical), tuple(text))
    if not spec.source_columns:
        raise ValueError(
            f"No usable {feature_rung} features found. Expected market text/category, "
            "causal length/PCA fields, or allowlisted pre-market social summaries."
        )
    return spec


def prepare_model_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Coerce selected values and combine market text for preprocessing."""
    result = pd.DataFrame(index=df.index)
    for col in spec.numeric:
        result[col] = pd.to_numeric(df[col], errors="coerce")
    for col in spec.categorical:
        result[col] = df[col].fillna("__missing__").astype(str)
    if spec.text:
        text_parts = [df[col].fillna("").astype(str) for col in spec.text]
        combined = text_parts[0]
        for part in text_parts[1:]:
            combined = combined.str.cat(part, sep=" ")
        result["_market_text"] = combined.str.strip()
    return result


def build_pipeline(
    spec: FeatureSpec,
    model_params: dict,
    *,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_ngram_max: int,
) -> Pipeline:
    """Build a fold-fitted preprocessing + LightGBM pipeline."""
    from lightgbm import LGBMClassifier

    transformers: list[tuple] = []
    if spec.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                list(spec.numeric),
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(spec.categorical),
            )
        )
    if spec.text:
        transformers.append(
            (
                "market_text",
                TfidfVectorizer(
                    max_features=tfidf_max_features,
                    min_df=tfidf_min_df,
                    ngram_range=(1, tfidf_ngram_max),
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
                "_market_text",
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**model_params)),
        ]
    )


def encode_labels(df: pd.DataFrame, split_name: str) -> np.ndarray:
    if "interest_label" not in df.columns:
        raise ValueError(f"T1 {split_name} split has no 'interest_label' column")
    labels = df["interest_label"].astype(str)
    invalid = sorted(set(labels) - set(LABEL_ORDER))
    if invalid:
        raise ValueError(f"T1 {split_name} split has unknown labels: {invalid}")
    return labels.map(LABEL_MAP).to_numpy(dtype=int)


def _group_values(df: pd.DataFrame) -> tuple[Optional[str], Optional[np.ndarray]]:
    for column in GROUP_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].astype("object").copy()
        missing = values.isna() | values.astype(str).str.strip().eq("")
        if missing.any():
            # A missing group is an independent singleton, not one giant group.
            replacements = [f"__row_{idx}" for idx in df.index[missing]]
            values.loc[missing] = replacements
        return column, values.astype(str).to_numpy()
    return None, None


def make_cv_splits(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    requested_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, Optional[str]]:
    """Create feasible stratified CV, preserving groups whenever available."""
    if requested_splits < 2:
        raise ValueError("--cv-splits must be at least 2")
    class_counts = np.bincount(y_train, minlength=len(LABEL_ORDER))
    if np.any(class_counts == 0):
        missing = [LABEL_ORDER[i] for i, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Training split is missing required labels: {missing}")

    group_col, groups = _group_values(train_df)
    if groups is not None:
        n_splits = min(requested_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError(f"Need at least two distinct values in {group_col} for grouped CV")
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            try:
                splits = list(splitter.split(np.zeros(len(y_train)), y_train, groups))
                return splits, "stratified_group", group_col
            except ValueError:
                pass
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros(len(y_train)), y_train, groups))
        return splits, "group", group_col

    n_splits = min(requested_splits, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least two training examples per class for CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y_train)), y_train))
    return splits, "stratified", None


def _base_model_params(seed: int, n_jobs: int, num_boost_round: int) -> dict:
    return {
        "objective": "multiclass",
        "num_class": len(LABEL_ORDER),
        "class_weight": "balanced",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": n_jobs,
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 1e-3,
        "reg_lambda": 1e-3,
    }


def cross_validated_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    spec: FeatureSpec,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    model_params: dict,
    *,
    tfidf_max_features: int,
    tfidf_min_df: int,
    tfidf_ngram_max: int,
) -> float:
    scores: list[float] = []
    for train_idx, validation_idx in splits:
        model = build_pipeline(
            spec,
            model_params,
            tfidf_max_features=tfidf_max_features,
            tfidf_min_df=tfidf_min_df,
            tfidf_ngram_max=tfidf_ngram_max,
        )
        model.fit(X_train.iloc[train_idx], y_train[train_idx])
        predictions = model.predict(X_train.iloc[validation_idx])
        scores.append(
            f1_score(
                y_train[validation_idx],
                predictions,
                labels=np.arange(len(LABEL_ORDER)),
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(scores))


def tune_params(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    spec: FeatureSpec,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[dict, float]:
    """Tune on train-only CV, or evaluate deterministic defaults at trials=0."""
    base = _base_model_params(args.seed, args.n_jobs, args.num_boost_round)
    score_kwargs = {
        "tfidf_max_features": args.tfidf_max_features,
        "tfidf_min_df": args.tfidf_min_df,
        "tfidf_ngram_max": args.tfidf_ngram_max,
    }
    if args.trials == 0:
        return base, cross_validated_score(
            X_train, y_train, spec, splits, base, **score_kwargs
        )
    if args.trials < 0:
        raise ValueError("--trials cannot be negative")

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = base.copy()
        params.update(
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-2, 0.2, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 3.0, log=True),
            }
        )
        return cross_validated_score(
            X_train, y_train, spec, splits, params, **score_kwargs
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = base.copy()
    best.update(study.best_params)
    return best, float(study.best_value)


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities = np.asarray(model.predict_proba(X))
    predictions = np.asarray(model.predict(X), dtype=int)
    metrics = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=np.arange(len(LABEL_ORDER)),
                average="macro",
                zero_division=0,
            )
        ),
    }
    return metrics, predictions, probabilities


def _fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    spec: FeatureSpec,
    params: dict,
    args: argparse.Namespace,
) -> Pipeline:
    model = build_pipeline(
        spec,
        params,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_ngram_max=args.tfidf_ngram_max,
    )
    model.fit(X, y)
    return model


def _print_metrics(split_name: str, metrics: dict) -> None:
    print(f"\n--- {split_name.title()} Results ---")
    print(f"Samples:   {metrics['n']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1:  {metrics['macro_f1']:.4f}")


def _write_predictions(
    path: Path,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    feature_rung: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_number, (_, row) in enumerate(test_df.iterrows()):
            predicted_label = LABEL_ORDER[int(predictions[row_number])]
            record = {
                "condition_id": str(row["condition_id"]),
                # ``label`` is the evaluator interface; ``pred_label`` is kept
                # for compatibility with earlier baseline outputs.
                "label": predicted_label,
                "pred_label": predicted_label,
                "confidence": float(probabilities[row_number].max()),
                "scores": {
                    LABEL_ORDER[index]: float(probabilities[row_number][index])
                    for index in range(len(LABEL_ORDER))
                },
                "feature_rung": feature_rung,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.trials = min(args.trials, 2)
        args.cv_splits = min(args.cv_splits, 3)
        args.num_boost_round = min(args.num_boost_round, 80)
        args.tfidf_max_features = min(args.tfidf_max_features, 1000)
    if args.tfidf_min_df < 1:
        raise SystemExit("--tfidf-min-df must be at least 1")
    if args.tfidf_ngram_max < 1:
        raise SystemExit("--tfidf-ngram-max must be at least 1")

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        raise SystemExit("Install lightgbm: pip install lightgbm")
    if args.trials > 0:
        try:
            import optuna  # noqa: F401
        except ImportError:
            raise SystemExit("Install optuna or pass --trials 0")

    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted.*",
        category=UserWarning,
    )

    print("Loading T1 data...")
    train_df, validation_df, test_df = load_data(args.local_dir)
    train_df = _add_causal_derived_columns(train_df)
    test_df = _add_causal_derived_columns(test_df)
    if validation_df is not None:
        validation_df = _add_causal_derived_columns(validation_df)

    modeled_frames = [train_df]
    if validation_df is not None:
        modeled_frames.append(validation_df)
    modeled_frames.append(test_df)
    spec = select_feature_spec(modeled_frames, args.feature_rung)

    X_train = prepare_model_frame(train_df, spec)
    X_test = prepare_model_frame(test_df, spec)
    X_validation = (
        prepare_model_frame(validation_df, spec)
        if validation_df is not None
        else None
    )
    y_train = encode_labels(train_df, "train")
    y_test = encode_labels(test_df, "test")
    y_validation = (
        encode_labels(validation_df, "validation")
        if validation_df is not None
        else None
    )

    splits, cv_strategy, group_column = make_cv_splits(
        train_df, y_train, args.cv_splits, args.seed
    )
    print(f"Feature rung: {args.feature_rung}")
    print(f"Features ({len(spec.source_columns)}): {list(spec.source_columns)}")
    print(
        f"Train: {len(train_df)}  "
        f"Validation: {len(validation_df) if validation_df is not None else 'not available'}  "
        f"Test: {len(test_df)}"
    )
    print(
        f"CV: {cv_strategy}, folds={len(splits)}"
        + (f", group={group_column}" if group_column else "")
    )

    best_params, cv_macro_f1 = tune_params(
        X_train, y_train, spec, splits, args
    )
    printable_params = {
        key: value
        for key, value in best_params.items()
        if key
        in {
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "n_estimators",
        }
    }
    print(f"Best train-CV macro-F1: {cv_macro_f1:.4f}")
    print(f"Best params: {printable_params}")

    validation_metrics = None
    if X_validation is not None and y_validation is not None:
        validation_model = _fit_model(X_train, y_train, spec, best_params, args)
        validation_metrics, _, _ = evaluate_model(
            validation_model, X_validation, y_validation
        )
        _print_metrics("validation", validation_metrics)
        X_final = pd.concat([X_train, X_validation], ignore_index=True)
        y_final = np.concatenate([y_train, y_validation])
    else:
        print("\nValidation split not available; fitting final model on train only.")
        X_final = X_train
        y_final = y_train

    final_model = _fit_model(X_final, y_final, spec, best_params, args)
    test_metrics, test_predictions, test_probabilities = evaluate_model(
        final_model, X_test, y_test
    )
    _print_metrics("test", test_metrics)

    output_path = Path(args.output)
    _write_predictions(
        output_path,
        test_df,
        test_predictions,
        test_probabilities,
        args.feature_rung,
    )
    print(f"\nPredictions saved to {output_path}")

    if args.metrics_output:
        metrics_record = {
            "feature_rung": args.feature_rung,
            "features": list(spec.source_columns),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df) if validation_df is not None else 0,
            "test_rows": len(test_df),
            "cv_strategy": cv_strategy,
            "cv_group_column": group_column,
            "cv_folds": len(splits),
            "cv_macro_f1": cv_macro_f1,
            "best_params": printable_params,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
