#!/usr/bin/env python3
"""T2 LLM Reranking Baseline -- Post-to-Market Linking.

For each tweet, takes the top-K candidate markets (ranked by embedding
similarity) and asks an LLM to rerank them by relevance.  Evaluates
Accuracy@1 and Mean Reciprocal Rank (MRR) against the gold market.

Usage:
    python -m baselines.t2.llm_baseline --provider openai --model gpt-4o --shots 0
    python -m baselines.t2.llm_baseline --provider anthropic --model claude-sonnet-4-20250514 --shots 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict

import pandas as pd

import eventxbench

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "tweet": "Bitcoin just hit $50k. This is huge for crypto adoption.",
        "candidates": [
            "Will Bitcoin reach $50,000 by end of 2021?",
            "Will Ethereum surpass Bitcoin in market cap by 2022?",
            "Will Dogecoin reach $1 by end of 2021?",
        ],
        "ranking": "1,2,3",
    },
    {
        "tweet": "The Fed just raised rates by 75bps for the third time this year.",
        "candidates": [
            "Will the Fed raise rates more than 3 times in 2022?",
            "Will inflation exceed 10% in the US in 2022?",
            "Will the S&P 500 end 2022 in a bear market?",
        ],
        "ranking": "1,2,3",
    },
    {
        "tweet": "Zelensky just addressed the UN General Assembly, calling for more weapons.",
        "candidates": [
            "Will Ukraine join NATO by end of 2023?",
            "Will Zelensky address the UN General Assembly in September 2022?",
            "Will Russia use nuclear weapons in Ukraine by end of 2022?",
        ],
        "ranking": "2,1,3",
    },
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_prompt_0shot(tweet_text: str, candidate_questions: list[str]) -> str:
    k = len(candidate_questions)
    lines = [
        "You are given a tweet and a list of prediction market questions.",
        "Rank ALL markets from most to least relevant to the tweet.",
        f"Reply with ONLY a comma-separated list of all {k} numbers "
        "in ranked order. Example: 3,1,2. No explanation.",
        "",
        f"Tweet: {tweet_text}",
        "",
        "Candidate markets:",
    ]
    for i, q in enumerate(candidate_questions, 1):
        lines.append(f"{i}. {q}")
    lines += ["", "Ranking (most to least relevant):"]
    return "\n".join(lines)


def _build_prompt_3shot(tweet_text: str, candidate_questions: list[str]) -> str:
    k = len(candidate_questions)
    lines = [
        "You are given a tweet and a list of prediction market questions.",
        "Rank ALL markets from most to least relevant to the tweet.",
        f"Reply with ONLY a comma-separated list of all {k} numbers "
        "in ranked order. Example: 3,1,2. No explanation.",
        "",
        "=== Examples ===",
    ]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"\nTweet: {ex['tweet']}")
        lines.append("Candidate markets:")
        for i, q in enumerate(ex["candidates"], 1):
            lines.append(f"{i}. {q}")
        lines.append(f"Ranking (most to least relevant): {ex['ranking']}")
    lines += [
        "",
        "=== Now answer ===",
        "",
        f"Tweet: {tweet_text}",
        "",
        "Candidate markets:",
    ]
    for i, q in enumerate(candidate_questions, 1):
        lines.append(f"{i}. {q}")
    lines += ["", "Ranking (most to least relevant):"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------
def _make_client(provider: str):
    """Return an API client for the given provider."""
    if provider == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key)
    elif provider == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "xai":
        from openai import OpenAI
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the XAI_API_KEY environment variable.")
        return OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_llm(client, provider: str, model: str, prompt: str, max_tokens: int) -> str:
    """Send a prompt and return the raw text response."""
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        # Works for both openai and xai (OpenAI-compatible)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_ranking(raw: str, k: int) -> list[int] | None:
    """Parse a ranking string like '3,1,2' into a 0-indexed list of length k."""
    nums = [int(x) for x in re.findall(r"\d+", raw) if 1 <= int(x) <= k]
    seen: set[int] = set()
    deduped: list[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            deduped.append(n - 1)
    if not deduped:
        return None
    # Append any missing indices at the end
    seen_0idx = set(deduped)
    for j in range(k):
        if j not in seen_0idx:
            deduped.append(j)
    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T2 LLM reranking baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        default="openai",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="t2_llm_results.jsonl")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Path to local data directory (skips HuggingFace download)",
    )
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    df = eventxbench.load_task("t2", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]  # use test split

    # Expect columns: tweet_id, tweet_text, market_id, market_question,
    #                  embedding_score, gold (bool or 0/1)
    required = {"tweet_id", "market_question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in T2 data: {missing}")

    # Group candidates per tweet
    groups = defaultdict(list)
    gold_market: dict[str, str] = {}
    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        groups[tid].append(row)
        if row.get("gold", False) or row.get("is_gold", False):
            gold_market[tid] = str(row["market_id"])

    # Keep only tweets with a gold label
    tweet_ids = sorted(gold_market.keys())
    print(
        f"Tweets with gold label: {len(tweet_ids)}, "
        f"model: {args.model}, shots: {args.shots}, top-k: {args.top_k}"
    )

    if not args.dry_run:
        client = _make_client(args.provider)

    results: list[dict] = []
    acc1_hits = 0
    rr_sum = 0.0
    parse_errors = 0

    for i, tid in enumerate(tweet_ids):
        candidates = sorted(
            groups[tid],
            key=lambda r: float(r.get("embedding_score", 0)),
            reverse=True,
        )[: args.top_k]
        k = len(candidates)

        tweet_text = str(candidates[0].get("tweet_text", ""))
        candidate_questions = [str(c["market_question"]) for c in candidates]

        gold_mid = gold_market[tid]
        gold_idx = next(
            (j for j, c in enumerate(candidates) if str(c["market_id"]) == gold_mid),
            -1,
        )

        prompt = (
            _build_prompt_0shot(tweet_text, candidate_questions)
            if args.shots == 0
            else _build_prompt_3shot(tweet_text, candidate_questions)
        )

        if args.dry_run:
            print("=== SAMPLE PROMPT ===")
            print(prompt)
            print(f"\nGold position in top-{k}: {gold_idx + 1} (0 = not found)")
            return

        try:
            raw = _call_llm(client, args.provider, args.model, prompt, k * 4)
        except Exception as e:
            print(f"  [API ERROR] tweet {tid}: {e}")
            time.sleep(5)
            continue

        ranking = _parse_ranking(raw, k)
        if ranking is None:
            parse_errors += 1
            print(f"  [PARSE ERROR] tweet {tid}, raw='{raw}'")
            ranking = list(range(k))

        if gold_idx >= 0:
            gpt_gold_rank = ranking.index(gold_idx) + 1
        else:
            gpt_gold_rank = k + 1

        rr = 1.0 / gpt_gold_rank
        rr_sum += rr
        if gpt_gold_rank == 1:
            acc1_hits += 1

        results.append(
            {
                "tweet_id": tid,
                "gold_market": gold_mid,
                "predicted_top1": str(candidates[ranking[0]]["market_id"]),
                "llm_raw": raw,
                "gold_rank": gpt_gold_rank,
                "rr": rr,
            }
        )

        n = len(results)
        if n % 20 == 0 or n == len(tweet_ids):
            print(
                f"  [{n}/{len(tweet_ids)}] "
                f"Acc@1={acc1_hits / n:.4f}  MRR={rr_sum / n:.4f}"
            )
        time.sleep(args.delay)

    # -- Report -------------------------------------------------------------
    if results:
        n = len(results)
        acc1 = acc1_hits / n
        mrr = rr_sum / n
        print(f"\n=== Results ({args.model}, {args.shots}-shot, top-{args.top_k}) ===")
        print(f"  N={n}, Acc@1={acc1:.4f}, MRR={mrr:.4f}, parse_errors={parse_errors}")

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()
