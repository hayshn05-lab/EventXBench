#!/usr/bin/env python3
"""T1 LLM Baseline -- Pre-Market Interest Forecasting.

Classifies prediction-market questions into interest levels
(high_interest / moderate_interest / low_interest) using a hosted LLM.

Usage examples:
    # Zero-shot with GPT-4o
    python baselines/t1/llm_baseline.py --provider openai --model gpt-4o --shots 0

    # 3-shot with Claude
    python baselines/t1/llm_baseline.py --provider anthropic --model claude-3-5-sonnet-20241022 --shots 3

    # Dry-run (prints prompts, no API calls)
    python baselines/t1/llm_baseline.py --provider openai --dry-run --limit 5

API keys are read from environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]
VALID_LABELS = set(LABEL_ORDER)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

SYSTEM_PROMPT = """\
You are evaluating a benchmark task: Pre-Market Interest Forecasting.

You will be given:
- A target prediction market question.
- Pre-market social signals extracted from tweets before market creation.
- Optionally, a few labeled examples from the training set.

Your task is to predict the market interest label:
- high_interest: very strong later market interest / trading volume
- moderate_interest: meaningful but not top-tier later interest
- low_interest: relatively weak later market interest

Rules:
- Use only the information explicitly provided in the prompt.
- Do not use external knowledge or future information.
- Focus on whether the pre-market signal suggests later market attention.

Return strict JSON only (no explanation) in exactly this format:
{
  "label": "high_interest | moderate_interest | low_interest",
  "confidence": 0.0,
  "scores": {
    "high_interest": 0.0,
    "moderate_interest": 0.0,
    "low_interest": 0.0
  }
}
"""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T1 LLM baseline: classify market interest level"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        required=True,
        help="LLM provider",
    )
    parser.add_argument("--model", default="", help="Model name (defaults per provider)")
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of few-shot examples per class (0 = zero-shot)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Path to local EventX data directory (skips HF download)",
    )
    parser.add_argument("--output", default="t1_llm_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max test samples to evaluate")
    parser.add_argument("--resume", action="store_true", help="Skip already-predicted IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls")
    parser.add_argument("--timeout", type=float, default=120.0, help="API request timeout")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    local_dir: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test splits via eventxbench loader."""
    import eventxbench

    if local_dir:
        train_df, test_df = eventxbench.load_task("t1", local_dir=local_dir)
    else:
        train_df, test_df = eventxbench.load_task("t1")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "score",
    "cluster_count",
    "linked_tweet_count",
    "avg_link_confidence",
    "max_link_confidence",
    "text_similarity",
    "tweet_count",
    "unique_user_count",
    "burst_duration_hours",
    "max_author_followers",
    "mean_author_followers",
    "median_author_followers",
    "high_follower_author_count",
]


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return re.sub(r"\s+", " ", str(value).strip())


def _trim(text: Any, max_chars: int = 1200) -> str:
    s = _fmt(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _instance_block(row: dict[str, Any], feature_cols: list[str]) -> str:
    lines = [
        f"- question: {_fmt(row.get('question'))}",
        f"- event_group_label: {_fmt(row.get('event_group_label'))}",
        f"- event_text: {_trim(row.get('event_text'))}",
    ]
    if feature_cols:
        lines.append("- structured_features:")
        for col in feature_cols:
            if col in row:
                lines.append(f"    {col}: {_fmt(row.get(col))}")
    return "\n".join(lines)


def select_few_shot(
    train_df: pd.DataFrame, shots_per_class: int
) -> list[dict[str, Any]]:
    if shots_per_class <= 0:
        return []
    examples: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        sub = train_df[train_df["interest_label"].astype(str) == label]
        examples.extend(sub.head(shots_per_class).to_dict("records"))
    return examples


def build_user_prompt(
    row: dict[str, Any],
    feature_cols: list[str],
    few_shot: list[dict[str, Any]],
) -> str:
    parts = ["Task 1: Pre-Market Interest Forecasting\n"]

    if few_shot:
        parts.append("Labeled examples:")
        for i, ex in enumerate(few_shot, 1):
            block = _instance_block(ex, feature_cols)
            parts.append(f"Example {i}:\n{block}\n- label: {ex['interest_label']}")
        parts.append("")

    parts.append("Target market to classify:")
    parts.append(_instance_block(row, feature_cols))
    parts.append("")
    parts.append("Return strict JSON only.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# API callers  (stdlib only -- no SDK dependency)
# ---------------------------------------------------------------------------


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai(api_key: str, model: str, user_prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = _post_json(OPENAI_API_URL, headers, body, timeout)
    return resp["choices"][0]["message"]["content"].strip()


def call_anthropic(api_key: str, model: str, user_prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    resp = _post_json(ANTHROPIC_API_URL, headers, body, timeout)
    parts = [c["text"] for c in resp.get("content", []) if c.get("type") == "text"]
    return "\n".join(parts).strip()


def call_llm(provider: str, api_key: str, model: str, prompt: str, timeout: float) -> str:
    if provider == "anthropic":
        return call_anthropic(api_key, model, prompt, timeout)
    return call_openai(api_key, model, prompt, timeout)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_prediction(text: str) -> dict[str, Any]:
    """Extract label, confidence, and per-class scores from LLM JSON output."""
    candidate = text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])

    label = payload.get("label")
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    scores = payload.get("scores") or {}
    parsed = {k: max(0.0, float(scores.get(k, 0.0))) for k in LABEL_ORDER}
    total = sum(parsed.values())
    if total > 0:
        parsed = {k: v / total for k, v in parsed.items()}
    else:
        parsed = {k: (1.0 if k == label else 0.0) for k in LABEL_ORDER}

    return {
        "label": label,
        "confidence": float(payload.get("confidence", parsed[label])),
        "scores": parsed,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(gold: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABEL_ORDER, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    model = args.model or (
        DEFAULT_ANTHROPIC_MODEL if args.provider == "anthropic" else DEFAULT_OPENAI_MODEL
    )

    api_key = ""
    if not args.dry_run:
        env_var = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"ERROR: Set {env_var} environment variable.", file=sys.stderr)
            sys.exit(1)

    print(f"Provider: {args.provider}  Model: {model}  Shots/class: {args.shots}")
    train_df, test_df = load_data(args.local_dir)

    # Detect available feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]

    # Build few-shot examples
    few_shot = select_few_shot(train_df, args.shots)

    # Resume support
    output_path = Path(args.output)
    completed_ids: set[str] = set()
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            completed_ids.add(str(row.get("condition_id", "")))
        print(f"Resuming: {len(completed_ids)} predictions already cached.")

    # Build evaluation set
    records = test_df.to_dict("records")
    if args.limit > 0:
        records = records[: args.limit]

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    errors = 0

    for i, row in enumerate(records):
        cid = str(row["condition_id"])
        if cid in completed_ids:
            continue

        prompt = build_user_prompt(row, available_features, few_shot)

        result: dict[str, Any] = {
            "condition_id": cid,
            "provider": args.provider,
            "model": model,
            "gold_label": row.get("interest_label"),
        }

        if args.dry_run:
            result["user_prompt"] = prompt
            print(f"[{i+1}/{len(records)}] {cid} (dry-run)")
            append_jsonl(output_path, result)
            continue

        try:
            raw = call_llm(args.provider, api_key, model, prompt, args.timeout)
            parsed = parse_prediction(raw)
            result["prediction"] = parsed
            result["raw_output"] = raw
            gold_labels.append(str(row["interest_label"]))
            pred_labels.append(parsed["label"])
        except Exception as exc:
            result["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            errors += 1

        append_jsonl(output_path, result)

        if (i + 1) % 20 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] errors={errors}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nPredictions written to {output_path}")
    print(f"Total processed this run: {len(gold_labels)}  Errors: {errors}")

    if gold_labels and not args.dry_run:
        metrics = evaluate(gold_labels, pred_labels)
        print(f"\n--- Results ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
