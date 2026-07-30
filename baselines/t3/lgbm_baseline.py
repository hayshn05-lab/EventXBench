#!/usr/bin/env python3
"""T3 LightGBM Baseline -- Evidence Grading (Inference-Only).

Loads precomputed tweet and predicate embeddings and trains a LightGBM
classifier on [requires_official, tweet_embedding, predicate_embedding].
Reports Cohen's Kappa and Macro F1. Predicate embeddings are deduplicated
(predicates repeat heavily across rows) and reconstructed per-row via
predicate_indices - see T3_Reproducible_Package/embeddings/.

Usage:
    python t3_lgbm_inference.py
    python t3_lgbm_inference.py --local-dir /path/to/data
    python t3_lgbm_inference.py --tweet-emb tweet_embeddings.npy \\
        --predicate-emb predicate_embeddings.npy --predicate-idx predicate_indices.npy
"""
from __future__ import annotations

import argparse
from collections import Counter
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

import eventxbench


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
# Matches T3_Reproducible_Package's canonical LightGBM baseline exactly
# (t3_baselines_majority_random_lightgbm.ipynb, cell 9): requires_official +
# tweet embeddings + predicate embeddings ONLY. The four deterministic
# check_* columns, candidate_grade, and needs_llm are deliberately excluded -
# they are what *produce* final_grade for the auto-labeled rows and are
# heavily correlated with it elsewhere, so including them as model features
# is close to label leakage rather than an honest baseline.
def build_features(
    df: pd.DataFrame,
    tweet_embeddings: np.ndarray,
    predicate_embeddings: np.ndarray,
    predicate_indices: np.ndarray,
) -> np.ndarray:
    requires_official = df["requires_official"].astype(int).values.reshape(-1, 1)
    predicate_matrix = predicate_embeddings[predicate_indices]
    return np.hstack([requires_official, tweet_embeddings, predicate_matrix])


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def split_by_market(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    markets = df["condition_id"].unique()
    train_markets, test_markets = train_test_split(
        markets, test_size=test_size, random_state=random_state
    )
    train_idx = df["condition_id"].isin(train_markets).values
    test_idx = df["condition_id"].isin(test_markets).values
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _run_lgbm(X_train, y_train, X_test, y_test) -> dict:
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return {
        "baseline": "lgbm",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "kappa": kappa,
        "macro_f1": f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 LightGBM baseline (Inference Only)")
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--tweet-emb", default="tweet_embeddings.npy")
    parser.add_argument(
        "--predicate-emb",
        default="predicate_embeddings.npy",
        help="Deduplicated predicate-text embeddings (one row per unique predicate).",
    )
    parser.add_argument(
        "--predicate-idx",
        default="predicate_indices.npy",
        help="Per-row index into --predicate-emb; predicate_emb[predicate_idx] "
        "reconstructs the full per-row predicate embedding matrix.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    print(f"T3 samples: {len(df)}")
    print(f"Grade distribution: {dict(sorted(Counter(df['final_grade'].tolist()).items()))}")

    # Load precomputed embeddings
    print(f"\nLoading embeddings from disk...")
    tweet_embeddings = np.load(args.tweet_emb)
    predicate_embeddings = np.load(args.predicate_emb)
    predicate_indices = np.load(args.predicate_idx)
    print(f"Tweet embeddings shape:     {tweet_embeddings.shape}")
    print(f"Predicate embeddings shape: {predicate_embeddings.shape} (deduplicated)")
    print(f"Predicate indices shape:    {predicate_indices.shape}")
    assert len(df) == tweet_embeddings.shape[0] == predicate_indices.shape[0], (
        "Row count mismatch between T3 data and embeddings - "
        "these must be in the same row order."
    )

    # Features
    X = build_features(df, tweet_embeddings, predicate_embeddings, predicate_indices)
    y = df["final_grade"].values
    print(f"Feature matrix shape: {X.shape}")

    # Split
    train_idx, test_idx = split_by_market(df, args.test_size, args.random_state)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train & evaluate
    result = _run_lgbm(X_train, y_train, X_test, y_test)
    print(f"\n[LightGBM Results]")
    print(f"  Kappa={result['kappa']:.4f}, Macro F1={result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()