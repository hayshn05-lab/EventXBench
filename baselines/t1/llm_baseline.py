#!/usr/bin/env python3
"""T1 KDD-v2 LLM baseline -- Pre-Market Interest Forecasting.

Uses the frozen ``t1.kdd.v2`` market-level train/test release and only the
feature rung declared in its manifest. Test access, including prompt preview,
is sealed behind ``--allow-test``.

Usage examples:
    python baselines/t1/llm_baseline.py \
        --provider openai --model MODEL --split train --dry-run --limit 1

    python baselines/t1/llm_baseline.py \
        --provider openai --base-url https://lum.id/llm \
        --api-key-env LUMID_API_KEY --model MODEL --split test --allow-test \
        --shots 3 --resume --output results/t1.MODEL.test.3shot.jsonl

API keys are read from environment variables (OPENAI_API_KEY / ANTHROPIC_API_KEY,
or the variable selected with --api-key-env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ORDER = ["high_interest", "moderate_interest", "low_interest"]
VALID_LABELS = set(LABEL_ORDER)
DATASET_VERSION = "t1.kdd.v2"
PROMPT_VERSION = "t1.kdd.v2.llm.r1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "KDD/data/t1_kdd_v2"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

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
        choices=["openai", "anthropic", "xai"],
        required=True,
        help="LLM provider",
    )
    parser.add_argument("--model", required=True, help="Exact model identifier")
    parser.add_argument(
        "--base-url",
        default="",
        help=(
            "Optional OpenAI-compatible base URL, for example "
            "https://lum.id/llm/v1. /chat/completions is appended when absent."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help=(
            "Environment variable containing the API key. Defaults to "
            "OPENAI_API_KEY or ANTHROPIC_API_KEY according to --provider."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Send chat_template_kwargs.enable_thinking=false to compatible "
            "OpenAI-style endpoints (recommended for Lumid Qwen models)."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        choices=[0, 3],
        help="Zero-shot or one train example per class",
    )
    parser.add_argument(
        "--data-dir",
        "--local-dir",
        dest="data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Frozen t1.kdd.v2 directory",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for any sealed-test access, including prompt preview",
    )
    parser.add_argument(
        "--feature-rung",
        choices=["market_only", "market_social"],
        default="market_social",
    )
    parser.add_argument("--output", default="t1_llm_predictions.jsonl")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Max test samples to evaluate")
    parser.add_argument("--resume", action="store_true", help="Skip already-predicted IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls")
    parser.add_argument("--timeout", type=float, default=120.0, help="API request timeout")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(
    data_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path]:
    """Load train plus one requested split from the frozen v2 release."""
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / f"{split}.jsonl"
    for path in (manifest_path, train_path, eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical T1 file: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected {DATASET_VERSION}, got {manifest.get('dataset_version')!r}"
        )
    train_df = pd.read_json(train_path, lines=True)
    eval_df = pd.read_json(eval_path, lines=True)
    expected = int(manifest["counts"]["by_split"][split])
    if len(eval_df) != expected:
        raise ValueError(
            f"{split}: manifest declares {expected} rows, found {len(eval_df)}"
        )
    return train_df, eval_df, manifest, manifest_path, eval_path


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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
    lines = []
    for col in feature_cols:
        value = _trim(row.get(col)) if col in {"question", "description"} else _fmt(
            row.get(col)
        )
        lines.append(f"- {col}: {value}")
    return "\n".join(lines)


def select_few_shot(
    train_df: pd.DataFrame, shots: int
) -> list[dict[str, Any]]:
    if shots == 0:
        return []
    examples: list[dict[str, Any]] = []
    for label in LABEL_ORDER:
        sub = train_df[train_df["interest_label"].astype(str) == label]
        if sub.empty:
            raise ValueError(f"Train split has no few-shot example for {label}")
        examples.append(sub.sort_values("condition_id").iloc[0].to_dict())
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
        raw = resp.read()
        if not raw:
            raise RuntimeError(
                f"Empty HTTP response (status={resp.status}, final_url={resp.url})"
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            content_type = resp.headers.get("Content-Type", "unknown")
            preview = raw[:200].decode("utf-8", errors="replace")
            raise RuntimeError(
                "Non-JSON HTTP response "
                f"(status={resp.status}, final_url={resp.url}, "
                f"content_type={content_type}, body_prefix={preview!r})"
            ) from exc


def openai_chat_url(base_url: str, provider: str = "openai") -> str:
    """Resolve an OpenAI-compatible base URL without altering official default."""
    if not base_url:
        return XAI_API_URL if provider == "xai" else OPENAI_API_URL
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def call_openai(
    api_key: str,
    model: str,
    user_prompt: str,
    timeout: float,
    api_url: str = OPENAI_API_URL,
    disable_thinking: bool = False,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = _post_json(api_url, headers, body, timeout)
    return resp["choices"][0]["message"]["content"].strip()


def call_anthropic(api_key: str, model: str, user_prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0.0,
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


def call_llm(
    provider: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    base_url: str = "",
    disable_thinking: bool = False,
) -> str:
    if provider == "anthropic":
        if base_url:
            raise ValueError("--base-url currently supports the OpenAI-compatible provider only")
        return call_anthropic(api_key, model, prompt, timeout)
    return call_openai(
        api_key,
        model,
        prompt,
        timeout,
        openai_chat_url(base_url, provider),
        disable_thinking,
    )


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

    confidence = float(payload.get("confidence", parsed[label]))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence outside [0,1]: {confidence}")
    return {
        "label": label,
        "confidence": confidence,
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


def compact_jsonl(
    path: Path,
    source_rows: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source in source_rows:
            condition_id = str(source["condition_id"])
            if condition_id in latest:
                handle.write(
                    json.dumps(latest[condition_id], ensure_ascii=False) + "\n"
                )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    gold: list[str],
    pred: list[str],
    scores: list[dict[str, float]] | None = None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABEL_ORDER, average="macro", zero_division=0),
    }
    if scores:
        ranked = sorted(
            zip(gold, scores),
            key=lambda item: item[1].get("high_interest", 0.0),
            reverse=True,
        )
        for k in (5, 10):
            use_k = min(k, len(ranked))
            metrics[f"high_interest_precision_at_{k}"] = (
                sum(label == "high_interest" for label, _ in ranked[:use_k])
                / use_k
                if use_k
                else None
            )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing sealed test run without --allow-test")
    if args.resume and args.overwrite:
        raise SystemExit("Choose only one of --resume and --overwrite")

    train_df, eval_df, manifest, manifest_path, eval_path = load_data(
        args.data_dir,
        args.split,
    )
    feature_cols = list(manifest["feature_rungs"][args.feature_rung])
    forbidden = set(manifest["forbidden_feature_columns"])
    leaked = forbidden & set(feature_cols)
    if leaked:
        raise ValueError(f"Manifest feature rung contains forbidden fields: {leaked}")
    missing = set(feature_cols) - set(eval_df.columns)
    if missing:
        raise ValueError(f"Evaluation split is missing features: {sorted(missing)}")
    for split_name, frame in (("train", train_df), (args.split, eval_df)):
        invalid = set(frame["interest_label"].astype(str)) - VALID_LABELS
        if invalid:
            raise ValueError(f"{split_name}: unknown labels {sorted(invalid)}")

    few_shot = select_few_shot(train_df, args.shots)
    config = {
        "version": "t1.llm.baseline.run.v2",
        "dataset_version": DATASET_VERSION,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "split": args.split,
        "shots": args.shots,
        "feature_rung": args.feature_rung,
        "input_fields": feature_cols,
        "few_shot_condition_ids": [
            str(row["condition_id"]) for row in few_shot
        ],
        "temperature": 0,
        "thinking_enabled": not args.disable_thinking,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    api_url = (
        ANTHROPIC_API_URL
        if args.provider == "anthropic" and not args.base_url
        else openai_chat_url(args.base_url, args.provider)
    )
    print(
        f"T1 v2: split={args.split} rows={len(eval_df)} provider={args.provider} "
        f"model={args.model} shots={args.shots} rung={args.feature_rung}"
    )
    print(f"Endpoint: {api_url}  Run ID: {run_id}")

    records = eval_df.to_dict("records")
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No rows selected")
    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== PROMPT PREVIEW ===")
        print(build_user_prompt(records[0], feature_cols, few_shot))
        return

    env_var = args.api_key_env or {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
    }[args.provider]
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise SystemExit(f"Set {env_var} environment variable")

    output_path = Path(args.output)
    if output_path.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Output exists: {output_path}; use --resume or --overwrite"
        )
    if args.overwrite and output_path.exists():
        output_path.unlink()
    cached = read_jsonl(output_path) if args.resume else []
    latest = {str(row["condition_id"]): row for row in cached}
    for row in latest.values():
        if row.get("run_id") != run_id:
            raise SystemExit("Existing output has a different run configuration")
    completed_ids = {
        cid for cid, row in latest.items() if not row.get("error")
    }
    print(f"Cached successes: {len(completed_ids)}")

    errors = 0
    for i, row in enumerate(records):
        cid = str(row["condition_id"])
        if cid in completed_ids:
            continue

        prompt = build_user_prompt(row, feature_cols, few_shot)
        result: dict[str, Any] = {
            "condition_id": cid,
            "split": args.split,
            "provider": args.provider,
            "model": args.model,
            "shots": args.shots,
            "feature_rung": args.feature_rung,
            "run_id": run_id,
        }

        try:
            raw = call_llm(
                args.provider,
                api_key,
                args.model,
                prompt,
                args.timeout,
                args.base_url,
                args.disable_thinking,
            )
            parsed = parse_prediction(raw)
            result["prediction"] = parsed
            result["raw_output"] = raw
        except Exception as exc:
            result["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            errors += 1

        append_jsonl(output_path, result)
        latest[cid] = result

        if (i + 1) % 20 == 0 or i + 1 == len(records):
            print(f"  [{i+1}/{len(records)}] errors={errors}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    wanted = {str(row["condition_id"]): row for row in records}
    compact_jsonl(output_path, records, latest)
    successful = {
        cid: row
        for cid, row in latest.items()
        if cid in wanted and not row.get("error")
    }
    ordered_ids = [str(row["condition_id"]) for row in records]
    missing_ids = [cid for cid in ordered_ids if cid not in successful]
    gold_labels = [
        str(wanted[cid]["interest_label"])
        for cid in ordered_ids
        if cid in successful
    ]
    pred_labels = [
        str(successful[cid]["prediction"]["label"])
        for cid in ordered_ids
        if cid in successful
    ]
    pred_scores = [
        successful[cid]["prediction"]["scores"]
        for cid in ordered_ids
        if cid in successful
    ]
    metrics = (
        evaluate(gold_labels, pred_labels, pred_scores)
        if gold_labels
        else {}
    )
    report = {
        **config,
        "run_id": run_id,
        "status": "complete" if not missing_ids else "incomplete",
        "requested_rows": len(records),
        "successful_rows": len(successful),
        "failed_or_missing_rows": len(missing_ids),
        "failed_or_missing_condition_ids": missing_ids,
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "evaluation_split": {
                "path": str(eval_path),
                "sha256": _sha256(eval_path),
            },
        },
        "metrics": metrics,
    }
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_suffix(".report.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nPredictions: {output_path}")
    print(f"Report: {metrics_path}")
    if metrics:
        print(
            f"Accuracy={metrics['accuracy']:.4f} "
            f"Macro-F1={metrics['macro_f1']:.4f} N={len(gold_labels)}"
        )
    if missing_ids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
