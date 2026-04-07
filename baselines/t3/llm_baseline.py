#!/usr/bin/env python3
"""T3 LLM Grading Baseline -- Evidence Grading.

Prompts an LLM to assign an evidence grade (0-5) for each tweet-market
pair, then evaluates against the human-annotated final_grade using
Spearman correlation and Quadratic Weighted Kappa (QWK).

Usage:
    python -m baselines.t3.llm_baseline --provider openai --model gpt-4o --shots 0
    python -m baselines.t3.llm_baseline --provider anthropic --model claude-sonnet-4-20250514 --shots 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

import eventxbench

# ---------------------------------------------------------------------------
# Grade definitions (included in prompt)
# ---------------------------------------------------------------------------
GRADE_SCALE = """\
5 (resolving)       - directly confirms the resolution condition; source meets requirement; no ambiguity
4 (strong_direct)   - directly addresses the condition but fails one of: source authority, threshold, or timing
3 (indirect_report) - credible but second-hand; does not directly assert the condition occurred
2 (speculation)     - directionally relevant but no authoritative claim (rumor, prediction, opinion with data)
1 (reaction)        - commentary or reaction; no new factual claim about the condition
0 (noise)           - off-topic, uninformative, or unrelated to the condition"""

FEW_SHOT_EXAMPLES = [
    {
        "tweet": "BREAKING: SEC officially approves spot Bitcoin ETFs, per SEC filing.",
        "question": "Will the SEC approve a spot Bitcoin ETF before January 15, 2024?",
        "grade": 5,
    },
    {
        "tweet": "Sources tell me the Fed is leaning toward a 50bp cut next meeting.",
        "question": "Will the Federal Reserve cut rates by 50+ basis points at the September 2024 FOMC meeting?",
        "grade": 3,
    },
    {
        "tweet": "Wow, markets are going crazy today. Wild times!",
        "question": "Will the S&P 500 close above 5000 by end of Q1 2024?",
        "grade": 1,
    },
]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _build_prompt_0shot(tweet: str, question: str) -> str:
    return (
        "You are grading a tweet as evidence for a prediction market resolution.\n\n"
        f"Market question: \"{question}\"\n"
        f"Tweet: \"{tweet[:500]}\"\n\n"
        f"Grading scale:\n{GRADE_SCALE}\n\n"
        "Assign the HIGHEST grade fully supported by the tweet.\n"
        "Reply with ONLY a JSON object: {\"grade\": <0-5>}"
    )


def _build_prompt_3shot(tweet: str, question: str) -> str:
    lines = [
        "You are grading a tweet as evidence for a prediction market resolution.",
        "",
        f"Grading scale:\n{GRADE_SCALE}",
        "",
        "=== Examples ===",
    ]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"\nMarket question: \"{ex['question']}\"")
        lines.append(f"Tweet: \"{ex['tweet']}\"")
        lines.append(f"Grade: {{\"grade\": {ex['grade']}}}")
    lines += [
        "",
        "=== Now grade ===",
        "",
        f"Market question: \"{question}\"",
        f"Tweet: \"{tweet[:500]}\"",
        "",
        "Assign the HIGHEST grade fully supported by the tweet.",
        "Reply with ONLY a JSON object: {\"grade\": <0-5>}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------
def _make_client(provider: str):
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


def _call_llm(client, provider: str, model: str, prompt: str, max_tokens: int = 32) -> str:
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()


def _parse_grade(raw: str) -> int | None:
    """Extract grade 0-5 from LLM response."""
    # Try JSON parse first
    try:
        obj = json.loads(raw)
        g = int(obj.get("grade", -1))
        if 0 <= g <= 5:
            return g
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Fallback: find first digit 0-5
    match = re.search(r"\b([0-5])\b", raw)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return None

    def _rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * len(vals)
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

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], num_classes: int = 6) -> float:
    """Compute QWK for ordinal grades 0..num_classes-1."""
    n = len(y_true)
    if n == 0:
        return 0.0
    # Confusion matrix
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t][p] += 1

    # Weight matrix (quadratic)
    W = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i][j] = (i - j) ** 2 / (num_classes - 1) ** 2

    # Expected matrix under independence
    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / n

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T3 LLM grading baseline")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        default="openai",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--shots", type=int, default=0, choices=[0, 3])
    parser.add_argument("--output", default="t3_llm_results.jsonl")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-dir", default=None)
    args = parser.parse_args()

    # -- Load data ----------------------------------------------------------
    df = eventxbench.load_task("t3", local_dir=args.local_dir)
    if isinstance(df, tuple):
        df = df[1]

    required = {"tweet_id", "final_grade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in T3 data: {missing}")

    # Determine text column names (may vary between data versions)
    tweet_col = "tweet" if "tweet" in df.columns else "tweet_text"
    question_col = "question" if "question" in df.columns else "market_question"
    if tweet_col not in df.columns or question_col not in df.columns:
        raise ValueError(
            f"Expected '{tweet_col}' and '{question_col}' columns in T3 data. "
            f"Found: {sorted(df.columns)}"
        )

    print(f"T3 samples: {len(df)}, model: {args.model}, shots: {args.shots}")

    if not args.dry_run:
        client = _make_client(args.provider)

    results: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    parse_errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        tweet = str(row[tweet_col])
        question = str(row[question_col])
        gold = int(row["final_grade"])

        prompt = (
            _build_prompt_0shot(tweet, question)
            if args.shots == 0
            else _build_prompt_3shot(tweet, question)
        )

        if args.dry_run:
            print("=== SAMPLE PROMPT ===")
            print(prompt)
            print(f"\nGold grade: {gold}")
            return

        try:
            raw = _call_llm(client, args.provider, args.model, prompt)
        except Exception as e:
            print(f"  [API ERROR] row {i}: {e}")
            time.sleep(5)
            continue

        grade = _parse_grade(raw)
        if grade is None:
            parse_errors += 1
            grade = 2  # fallback to median grade

        y_true.append(gold)
        y_pred.append(grade)
        results.append(
            {
                "tweet_id": str(row["tweet_id"]),
                "condition_id": str(row.get("condition_id", "")),
                "gold_grade": gold,
                "predicted_grade": grade,
                "llm_raw": raw,
            }
        )

        n = len(results)
        if n % 50 == 0:
            rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
            rho_str = f"{rho:.4f}" if rho is not None else "N/A"
            print(f"  [{n}/{len(df)}] Spearman={rho_str}  parse_errors={parse_errors}")

        time.sleep(args.delay)

    # -- Report -------------------------------------------------------------
    if results:
        rho = _spearman([float(v) for v in y_true], [float(v) for v in y_pred])
        qwk = _quadratic_weighted_kappa(y_true, y_pred, num_classes=6)
        rho_str = f"{rho:.4f}" if rho is not None else "N/A"

        print(f"\n=== Results ({args.model}, {args.shots}-shot) ===")
        print(f"  N={len(results)}, Spearman={rho_str}, QWK={qwk:.4f}, parse_errors={parse_errors}")

        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved predictions -> {args.output}")


if __name__ == "__main__":
    main()
