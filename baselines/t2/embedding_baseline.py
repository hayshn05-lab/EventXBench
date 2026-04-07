#!/usr/bin/env python3
"""T2 Embedding Recall Baseline -- Post-to-Market Linking.

Encodes tweet texts and market questions with BGE-large-en-v1.5, builds a
FAISS cosine-similarity index over market embeddings, and retrieves the
top-K markets for each tweet.  Reports Accuracy@1 and MRR.

Note: This baseline requires tweet text.  For the public release (where text
is stripped for privacy), you must first rehydrate tweets via the Twitter API.

Usage:
    python -m baselines.t2.embedding_baseline
    python -m baselines.t2.embedding_baseline --top-k 20 --model BAAI/bge-large-en-v1.5
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

import eventxbench

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
TOP_K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 embedding recall baseline")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="Sentence-transformer model name")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda", help="Device for encoding (cuda or cpu)")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Path to local data directory (skips HuggingFace download)",
    )
    args = parser.parse_args()

    # -- Optional imports (heavy) -------------------------------------------
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "This baseline requires sentence-transformers and faiss-cpu/faiss-gpu. "
            "Install with: pip install sentence-transformers faiss-cpu"
        ) from e

    # -- Load data ----------------------------------------------------------
    df = eventxbench.load_task("t2", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    if "tweet_text" not in df.columns:
        raise ValueError(
            "Column 'tweet_text' is missing. "
            "This baseline requires tweet text -- rehydrate via the Twitter API."
        )

    # Build tweet and market sets
    gold_market: dict[str, str] = {}
    tweet_texts: dict[str, str] = {}
    market_questions: dict[str, str] = {}

    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        tweet_texts[tid] = str(row["tweet_text"])
        mid = str(row["market_id"])
        market_questions[mid] = str(row["market_question"])
        if row.get("gold", False) or row.get("is_gold", False):
            gold_market[tid] = mid

    tweet_ids = sorted(gold_market.keys())
    market_ids = sorted(market_questions.keys())
    mid_to_idx = {mid: i for i, mid in enumerate(market_ids)}

    print(f"Tweets: {len(tweet_ids)}, Markets: {len(market_ids)}, Model: {args.model}")

    # -- Encode -------------------------------------------------------------
    model = SentenceTransformer(args.model, device=args.device)

    market_texts = [market_questions[mid] for mid in market_ids]
    market_embs = model.encode(
        market_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    tweet_text_list = [tweet_texts[tid] for tid in tweet_ids]
    tweet_embs = model.encode(
        tweet_text_list,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # -- FAISS retrieval ----------------------------------------------------
    dim = market_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(market_embs)
    print(f"FAISS index: {index.ntotal} markets, dim={dim}")

    scores, indices = index.search(tweet_embs, args.top_k)

    # -- Evaluate -----------------------------------------------------------
    acc1_hits = 0
    rr_sum = 0.0

    for i, tid in enumerate(tweet_ids):
        gold_mid = gold_market[tid]
        gold_idx = mid_to_idx.get(gold_mid, -1)

        retrieved = indices[i].tolist()
        if gold_idx in retrieved:
            rank = retrieved.index(gold_idx) + 1
        else:
            rank = args.top_k + 1

        rr_sum += 1.0 / rank
        if rank == 1:
            acc1_hits += 1

    n = len(tweet_ids)
    acc1 = acc1_hits / n
    mrr = rr_sum / n

    print(f"\n=== Embedding Baseline ({args.model}, top-{args.top_k}) ===")
    print(f"  N={n}, Acc@1={acc1:.4f}, MRR={mrr:.4f}")


if __name__ == "__main__":
    main()
