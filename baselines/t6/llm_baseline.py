#!/usr/bin/env python3
"""Paper-grade T6 KDD-v2 LLM baseline.

Forecasts daily cross-market co-movement from information available at the end
of bundle day ``d``.  The headline target is the frozen three-way label:
``no_effect`` / ``primary_only`` / ``cross_market``.  The four-way label and
cascade size are emitted as analysis-only auxiliary predictions.

The runner deliberately rejects legacy T6 data.  It supports validation-first
development, a sealed-test acknowledgement, deterministic train-only few-shot
examples, resumable JSONL checkpoints, token/cost accounting, input hashes,
and event-cluster bootstrap confidence intervals.

Examples:
    # Inspect a canonical validation prompt without calling a provider.
    python baselines/t6/llm_baseline.py \
        --provider openai --model MODEL --split validation --shots 0 \
        --dry-run

    # Run an OpenAI-compatible gateway and checkpoint every response.
    python baselines/t6/llm_baseline.py \
        --provider openai --base-url https://lum.id/llm \
        --api-key-env LUMID_API_KEY --model MODEL --split validation \
        --shots 3 --resume --output results/t6.MODEL.validation.3shot.jsonl

    # Test is sealed and requires explicit acknowledgement.
    python baselines/t6/llm_baseline.py \
        --provider openai --model MODEL --split test --allow-test \
        --shots 0 --output results/t6.MODEL.test.0shot.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "KDD/data/t6_kdd_v2"

DATASET_VERSION = "t6.kdd.v2"
SPLIT_VERSION = "tier2.temporal.v2"
PROMPT_VERSION = "t6.kdd.v2.llm.r1"
SEED = 20260727

HEADLINE_LABELS = ("no_effect", "primary_only", "cross_market")
FOUR_WAY_LABELS = ("no_effect", "primary_only", "sibling_only", "co_movement")

# These fields are forbidden even if a future release accidentally exposes one
# in a permissive loader.  ``num_siblings_total`` is also excluded because the
# experimental protocol treats retrospective graph degree as future leakage,
# despite its presence in the manifest's descriptive field list.
FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "num_siblings_total",
        "num_siblings_usable_h",
        "num_siblings_usable_H",
        "sibling_ids_usable_h",
        "sibling_ids_usable_H",
        "primary_max_abs_delta_h",
        "primary_z_h",
        "primary_moved",
        "primary_onset_lag_days",
        "primary_onset_lag_d",
        "sibling_move_count_h",
        "sibling_move_fraction_h",
        "max_sibling_z_h",
        "first_sibling_onset_lag_days",
        "sibling_onset_lag_d",
        "headline_label",
        "four_way_label",
        "cascade_size_h",
        "cascade_size_H",
        "confound_1d",
        "confound_h",
        "confound_flag",
        "label",
        "label_4way",
    }
)

# ``bundle_day`` and ``horizon_days`` describe the forecast decision and are
# present in every rung. Opaque IDs and raw sibling IDs are intentionally not
# serialized into prompts.
BASE_CONTEXT_FIELDS = ("bundle_day", "horizon_days")
FEATURE_RUNGS: dict[str, tuple[str, ...]] = {
    "market_only": ("domain",),
    "text_social_only": (
        "n_posts",
        "first_post_time",
        "followers_max",
        "engagement_sum",
        "engagement_max",
        "max_final_grade",
    ),
    "market_social": (
        "domain",
        "n_posts",
        "first_post_time",
        "followers_max",
        "engagement_sum",
        "engagement_max",
        "max_final_grade",
    ),
    "market_graph": ("domain", "num_siblings_visible_d"),
    "market_social_graph": (
        "domain",
        "n_posts",
        "first_post_time",
        "followers_max",
        "engagement_sum",
        "engagement_max",
        "max_final_grade",
        "num_siblings_visible_d",
    ),
}

FIELD_LABELS = {
    "bundle_day": "decision_day_utc",
    "horizon_days": "forecast_horizon_days",
    "domain": "market_domain",
    "n_posts": "posts_in_primary_market_day_bundle",
    "first_post_time": "first_bundle_post_time",
    "followers_max": "maximum_author_followers",
    "engagement_sum": "total_bundle_engagement",
    "engagement_max": "maximum_post_engagement",
    "max_final_grade": "maximum_bundle_evidence_grade",
    "num_siblings_visible_d": "causal_visible_sibling_count",
}

SYSTEM_PROMPT = """\
You forecast daily cross-market co-movement for prediction markets.

At the end of UTC bundle day d, predict what happens during the forward window
[d+1, d+H]. Daily candles cannot establish which market moved first, so do not
claim intraday propagation or causal ordering.

Headline labels:
- no_effect: neither the primary market nor a causally visible sibling moves.
- primary_only: the primary market moves and no visible sibling moves.
- cross_market: at least one visible sibling moves, whether or not the primary
  market also moves.

Analysis-only four-way labels:
- no_effect: neither side moves.
- primary_only: only the primary market moves.
- sibling_only: sibling market(s) move but the primary market does not.
- co_movement: both the primary market and sibling market(s) move.

Also forecast cascade_size, the non-negative number of visible siblings that
move. It must be 0 for no_effect/primary_only and at least 1 for cross_market.

Return strict JSON only:
{"headline_label":"no_effect|primary_only|cross_market",
 "four_way_label":"no_effect|primary_only|sibling_only|co_movement",
 "cascade_size":0,
 "confidence":0.0}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional OpenAI/Anthropic-compatible gateway root or endpoint.",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="API-key environment variable (provider default when omitted).",
    )
    parser.add_argument(
        "--data-dir",
        "--local-dir",
        dest="data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the frozen T6 v2 split files.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "val", "test"],
        default="validation",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement that the test configuration is frozen.",
    )
    parser.add_argument("--shots", type=int, choices=[0, 3], default=0)
    parser.add_argument(
        "--feature-rung",
        choices=sorted(FEATURE_RUNGS),
        default="market_social_graph",
    )
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=None,
        help="Optional prompt-token price for estimated cost reporting.",
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=None,
        help="Optional completion-token price for estimated cost reporting.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _normalized_split(split: str) -> str:
    return "validation" if split == "val" else split


def load_release(
    data_dir: Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict, dict, Path, Path]:
    """Load and validate the canonical split plus train-only demonstrations."""
    manifest_path = data_dir / "manifest.json"
    schema_path = data_dir / "schema.json"
    split_path = data_dir / f"{split}.jsonl"
    train_path = data_dir / "train.jsonl"
    for path in (manifest_path, schema_path, split_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical T6 release file: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected {DATASET_VERSION}, got {manifest.get('dataset_version')!r}"
        )
    if manifest.get("split_version") != SPLIT_VERSION:
        raise ValueError(
            f"Expected {SPLIT_VERSION}, got {manifest.get('split_version')!r}"
        )
    if schema.get("dataset_version") != DATASET_VERSION:
        raise ValueError("T6 schema version does not match the manifest")
    if not manifest.get("release_ready"):
        raise ValueError("T6 release is not marked release_ready")

    split_rows = read_jsonl(split_path)
    train_rows = read_jsonl(train_path)
    _validate_rows(split_rows, split, schema)
    _validate_rows(train_rows, "train", schema)

    expected = int(manifest.get("counts", {}).get(split, -1))
    if expected >= 0 and len(split_rows) != expected:
        raise ValueError(
            f"{split}: manifest declares {expected} rows, found {len(split_rows)}"
        )
    return split_rows, train_rows, manifest, schema, split_path, train_path


def _validate_rows(rows: Sequence[dict[str, Any]], split: str, schema: dict) -> None:
    required = {
        "condition_id",
        "bundle_day",
        "horizon_days",
        "instance_id",
        "event_cluster_id",
        "split",
        "headline_label",
        "four_way_label",
        "cascade_size_h",
    }
    allowed_schema_fields = set(schema.get("prediction_time_fields", {}))
    used_fields = set(BASE_CONTEXT_FIELDS)
    for fields in FEATURE_RUNGS.values():
        used_fields.update(fields)
    unauthorized = used_fields - allowed_schema_fields
    if unauthorized:
        raise ValueError(
            "Prompt design uses fields absent from the frozen prediction-time "
            f"schema: {sorted(unauthorized)}"
        )
    leaked = used_fields & FORBIDDEN_INPUT_FIELDS
    if leaked:
        raise ValueError(f"Prompt design contains forbidden fields: {sorted(leaked)}")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{split} row {index}: missing {sorted(missing)}")
        iid = str(row["instance_id"])
        if iid in seen:
            raise ValueError(f"{split}: duplicate instance_id {iid}")
        seen.add(iid)
        if str(row["split"]) != split:
            raise ValueError(
                f"{iid}: row split {row['split']!r} does not match {split!r}"
            )
        if row["headline_label"] not in HEADLINE_LABELS:
            raise ValueError(f"{iid}: invalid headline label")
        if row["four_way_label"] not in FOUR_WAY_LABELS:
            raise ValueError(f"{iid}: invalid four-way label")
        if int(row["horizon_days"]) not in (1, 3, 7):
            raise ValueError(f"{iid}: invalid horizon")
        if split in {"validation", "test"} and bool(row.get("confound_flag")):
            raise ValueError(f"{iid}: confounded row present in frozen {split}")


def selected_fields(feature_rung: str) -> tuple[str, ...]:
    fields = BASE_CONTEXT_FIELDS + FEATURE_RUNGS[feature_rung]
    if set(fields) & FORBIDDEN_INPUT_FIELDS:
        raise AssertionError("Internal error: forbidden field selected")
    return fields


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def serialize_features(row: dict[str, Any], feature_rung: str) -> dict[str, Any]:
    """Create the complete and auditable set of fields sent to the model."""
    fields = selected_fields(feature_rung)
    serialized = {
        FIELD_LABELS[field]: _clean_value(row.get(field))
        for field in fields
    }
    if set(fields) & FORBIDDEN_INPUT_FIELDS:
        raise AssertionError("Forbidden T6 field reached prompt serialization")
    return serialized


def format_features(row: dict[str, Any], feature_rung: str) -> str:
    features = serialize_features(row, feature_rung)
    return "\n".join(
        f"- {name}: {json.dumps(value, ensure_ascii=False)}"
        for name, value in features.items()
    )


def example_answer(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline_label": str(row["headline_label"]),
        "four_way_label": str(row["four_way_label"]),
        "cascade_size": int(row["cascade_size_h"]),
        "confidence": 1.0,
    }


def select_few_shot_examples(
    train_rows: Sequence[dict[str, Any]],
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    """Choose one train example per headline class for each horizon."""
    selected: dict[int, list[dict[str, Any]]] = {}
    for horizon in (1, 3, 7):
        horizon_rows = [
            row for row in train_rows if int(row["horizon_days"]) == horizon
        ]
        examples = []
        for label in HEADLINE_LABELS:
            eligible = [
                row for row in horizon_rows if row["headline_label"] == label
            ]
            if not eligible:
                raise ValueError(
                    f"No train example for horizon={horizon}, label={label}"
                )
            eligible.sort(
                key=lambda row: hashlib.sha256(
                    f"{seed}:{row['instance_id']}".encode("utf-8")
                ).hexdigest()
            )
            examples.append(eligible[0])
        selected[horizon] = examples
    return selected


def build_prompt(
    row: dict[str, Any],
    examples_by_horizon: dict[int, list[dict[str, Any]]],
    feature_rung: str,
) -> str:
    horizon = int(row["horizon_days"])
    parts = [
        f"Feature rung: {feature_rung}",
        "Use only the supplied prediction-time information.",
    ]
    examples = examples_by_horizon.get(horizon, [])
    if examples:
        parts.append("\nTraining examples from the frozen train split:")
        for number, example in enumerate(examples, 1):
            parts.append(
                f"\nExample {number}:\n{format_features(example, feature_rung)}\n"
                f"Answer: {json.dumps(example_answer(example), separators=(',', ':'))}"
            )
    parts.append(
        f"\nTarget instance:\n{format_features(row, feature_rung)}\n"
        "Return the required strict JSON object only."
    )
    return "\n".join(parts)


def endpoint(provider: str, base_url: str) -> str:
    if provider == "anthropic":
        if base_url:
            normalized = base_url.rstrip("/")
            if normalized.endswith("/messages"):
                return normalized
            if normalized.endswith("/v1"):
                return normalized + "/messages"
            return normalized + "/v1/messages"
        return "https://api.anthropic.com/v1/messages"
    if not base_url:
        return (
            "https://api.x.ai/v1/chat/completions"
            if provider == "xai"
            else "https://api.openai.com/v1/chat/completions"
        )
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def default_key_env(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
    }[provider]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"response contains no JSON object: {raw[:200]!r}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def parse_prediction(raw: str, visible_siblings: int | None) -> dict[str, Any]:
    payload = _extract_json_object(raw)
    headline = str(
        payload.get("headline_label", payload.get("label", ""))
    ).strip().lower()
    if headline not in HEADLINE_LABELS:
        raise ValueError(f"invalid headline_label: {headline!r}")

    four_way_raw = payload.get("four_way_label")
    four_way = str(four_way_raw).strip().lower() if four_way_raw is not None else None
    cascade_raw = payload.get("cascade_size")
    confidence_raw = payload.get("confidence")
    auxiliary_errors: list[str] = []

    if four_way not in FOUR_WAY_LABELS:
        auxiliary_errors.append("invalid_four_way_label")
        four_way = None
    elif headline in {"no_effect", "primary_only"} and four_way != headline:
        auxiliary_errors.append("headline_four_way_inconsistent")
        four_way = None
    elif headline == "cross_market" and four_way not in {
        "sibling_only",
        "co_movement",
    }:
        auxiliary_errors.append("headline_four_way_inconsistent")
        four_way = None

    cascade: int | None
    try:
        cascade_float = float(cascade_raw)
        if not cascade_float.is_integer():
            raise ValueError
        cascade = int(cascade_float)
    except (TypeError, ValueError):
        auxiliary_errors.append("invalid_cascade_size")
        cascade = None
    if cascade is not None:
        invalid = cascade < 0
        invalid = invalid or (
            headline in {"no_effect", "primary_only"} and cascade != 0
        )
        invalid = invalid or (headline == "cross_market" and cascade < 1)
        if visible_siblings is not None:
            invalid = invalid or cascade > max(0, int(visible_siblings))
        if invalid:
            auxiliary_errors.append("headline_cascade_inconsistent")
            cascade = None

    confidence: float | None
    try:
        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError
    except (TypeError, ValueError):
        confidence = None
        if confidence_raw is not None:
            auxiliary_errors.append("invalid_confidence")

    return {
        "prediction": headline,
        "four_way_prediction": four_way,
        "cascade_size_prediction": cascade,
        "confidence": confidence,
        "auxiliary_errors": auxiliary_errors,
    }


def call_model(
    args: argparse.Namespace,
    api_key: str,
    prompt: str,
    visible_siblings: int | None,
) -> dict[str, Any]:
    url = endpoint(args.provider, args.base_url)
    started = time.perf_counter()
    if args.provider == "anthropic":
        body: dict[str, Any] = {
            "model": args.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        if args.disable_thinking and args.base_url:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if args.base_url:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
    else:
        body = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        if args.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    response = requests.post(url, headers=headers, json=body, timeout=args.timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError(
            f"non-JSON response status={response.status_code} "
            f"body_prefix={response.text[:200]!r}"
        ) from exc

    if args.provider == "anthropic":
        raw = "".join(
            str(item.get("text", ""))
            for item in payload.get("content", [])
            if item.get("type") == "text"
        ).strip()
        usage = payload.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage.get("output_tokens", 0) or 0),
        }
        finish_reason = payload.get("stop_reason")
    else:
        choice = payload["choices"][0]
        message = choice["message"]
        raw = (
            message.get("content") or message.get("reasoning_content") or ""
        ).strip()
        usage = payload.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
        finish_reason = choice.get("finish_reason")
    token_usage["total_tokens"] = (
        token_usage["prompt_tokens"] + token_usage["completion_tokens"]
    )

    parsed = parse_prediction(raw, visible_siblings)
    return {
        **parsed,
        "raw_output": raw,
        "finish_reason": finish_reason,
        "usage": token_usage,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def run_config(
    args: argparse.Namespace,
    examples_by_horizon: dict[int, list[dict[str, Any]]],
    manifest: dict,
) -> dict[str, Any]:
    example_ids = {
        str(horizon): [str(row["instance_id"]) for row in examples]
        for horizon, examples in examples_by_horizon.items()
    }
    return {
        "version": "t6.llm.baseline.run.v2",
        "dataset_version": manifest["dataset_version"],
        "split_version": manifest["split_version"],
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "api_endpoint": endpoint(args.provider, args.base_url),
        "split": args.split,
        "shots": args.shots,
        "feature_rung": args.feature_rung,
        "input_fields": list(selected_fields(args.feature_rung)),
        "forbidden_input_fields_sha256": json_hash(
            sorted(FORBIDDEN_INPUT_FIELDS)
        ),
        "temperature": 0,
        "thinking_enabled": not args.disable_thinking,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "few_shot_instance_ids_by_horizon": example_ids,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def read_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["instance_id"])] = row
    return latest


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def compact_jsonl(
    path: Path,
    source_rows: Sequence[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source in source_rows:
            iid = str(source["instance_id"])
            if iid in latest:
                handle.write(json.dumps(latest[iid], ensure_ascii=False) + "\n")


def prediction_row(
    source: dict[str, Any],
    response: dict[str, Any],
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "instance_id": str(source["instance_id"]),
        "condition_id": str(source["condition_id"]),
        "bundle_day": str(source["bundle_day"]),
        "horizon_days": int(source["horizon_days"]),
        "event_cluster_id": str(source["event_cluster_id"]),
        "split": config["split"],
        "provider": config["provider"],
        "model": config["model"],
        "shots": config["shots"],
        "feature_rung": config["feature_rung"],
        "run_id": run_id,
        **response,
    }


def macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> float:
    scores = []
    for label in labels:
        tp = sum(
            1 for actual, pred in zip(gold, predicted)
            if actual == label and pred == label
        )
        fp = sum(
            1 for actual, pred in zip(gold, predicted)
            if actual != label and pred == label
        )
        fn = sum(
            1 for actual, pred in zip(gold, predicted)
            if actual == label and pred != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return statistics.fmean(scores) if scores else 0.0


def accuracy(gold: Sequence[str], predicted: Sequence[str]) -> float:
    if not gold:
        return 0.0
    return sum(a == p for a, p in zip(gold, predicted)) / len(gold)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[position]]
        ):
            end += 1
        average = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = average
        position = end
    return ranks


def spearman_rho(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) < 2 or len(actual) != len(predicted):
        return 0.0
    rank_actual = _average_ranks(actual)
    rank_predicted = _average_ranks(predicted)
    mean_actual = statistics.fmean(rank_actual)
    mean_predicted = statistics.fmean(rank_predicted)
    numerator = sum(
        (a - mean_actual) * (p - mean_predicted)
        for a, p in zip(rank_actual, rank_predicted)
    )
    denominator_a = sum((a - mean_actual) ** 2 for a in rank_actual) ** 0.5
    denominator_p = (
        sum((p - mean_predicted) ** 2 for p in rank_predicted) ** 0.5
    )
    if denominator_a == 0.0 or denominator_p == 0.0:
        return 0.0
    return numerator / (denominator_a * denominator_p)


def confusion_matrix(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {pred: 0 for pred in labels}
        for actual in labels
    }
    for actual, pred in zip(gold, predicted):
        matrix[actual][pred] += 1
    return matrix


def summarize_predictions(
    rows: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    if indices is None:
        indices = list(range(len(rows)))
    selected = [rows[index] for index in indices]
    predicted_rows = [predictions[str(row["instance_id"])] for row in selected]

    gold_headline = [str(row["headline_label"]) for row in selected]
    pred_headline = [str(row["prediction"]) for row in predicted_rows]
    result: dict[str, Any] = {
        "n": len(selected),
        "accuracy": accuracy(gold_headline, pred_headline),
        "macro_f1": macro_f1(
            gold_headline,
            pred_headline,
            HEADLINE_LABELS,
        ),
        "gold_distribution": dict(Counter(gold_headline)),
        "prediction_distribution": dict(Counter(pred_headline)),
        "confusion_matrix": confusion_matrix(
            gold_headline,
            pred_headline,
            HEADLINE_LABELS,
        ),
    }

    four_pairs = [
        (str(row["four_way_label"]), pred["four_way_prediction"])
        for row, pred in zip(selected, predicted_rows)
        if pred.get("four_way_prediction") in FOUR_WAY_LABELS
    ]
    result["n_four_way"] = len(four_pairs)
    result["four_way_macro_f1"] = (
        macro_f1(
            [pair[0] for pair in four_pairs],
            [pair[1] for pair in four_pairs],
            FOUR_WAY_LABELS,
        )
        if four_pairs
        else None
    )

    cascade_pairs = [
        (float(row["cascade_size_h"]), float(pred["cascade_size_prediction"]))
        for row, pred in zip(selected, predicted_rows)
        if pred.get("cascade_size_prediction") is not None
    ]
    result["n_cascade"] = len(cascade_pairs)
    result["cascade_mae"] = (
        statistics.fmean(abs(actual - pred) for actual, pred in cascade_pairs)
        if cascade_pairs
        else None
    )
    result["cascade_spearman_rho"] = (
        spearman_rho(
            [pair[0] for pair in cascade_pairs],
            [pair[1] for pair in cascade_pairs],
        )
        if len(cascade_pairs) >= 2
        else None
    )
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if replicates <= 0 or not rows:
        return {}
    cluster_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        cluster = str(
            row.get("event_cluster_id")
            or row.get("condition_id")
            or row["instance_id"]
        )
        cluster_indices.setdefault(cluster, []).append(index)
    clusters = sorted(cluster_indices)
    rng = random.Random(seed)
    metric_names = (
        "accuracy",
        "macro_f1",
        "four_way_macro_f1",
        "cascade_mae",
        "cascade_spearman_rho",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled_clusters = rng.choices(clusters, k=len(clusters))
        indices = [
            index
            for cluster in sampled_clusters
            for index in cluster_indices[cluster]
        ]
        summary = summarize_predictions(rows, predictions, indices)
        for name in metric_names:
            value = summary.get(name)
            if value is not None and math.isfinite(float(value)):
                values[name].append(float(value))
    return {
        name: {
            "low": _percentile(samples, 0.025),
            "high": _percentile(samples, 0.975),
        }
        for name, samples in values.items()
        if samples
    }


def evaluate(
    rows: Sequence[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bootstrap_unit": "event_cluster_id",
        "bootstrap_replicates": replicates,
        "overall": summarize_predictions(rows, predictions),
        "by_horizon": {},
    }
    result["overall"]["confidence_intervals_95"] = cluster_bootstrap_ci(
        rows,
        predictions,
        replicates,
        seed,
    )
    for offset, horizon in enumerate((1, 3, 7), 1):
        horizon_rows = [
            row for row in rows if int(row["horizon_days"]) == horizon
        ]
        summary = summarize_predictions(horizon_rows, predictions)
        summary["confidence_intervals_95"] = cluster_bootstrap_ci(
            horizon_rows,
            predictions,
            replicates,
            seed + offset,
        )
        result["by_horizon"][str(horizon)] = summary
    return result


def usage_summary(
    prediction_rows: Sequence[dict[str, Any]],
    input_price: float | None,
    output_price: float | None,
) -> dict[str, Any]:
    prompt_tokens = sum(
        int(row.get("usage", {}).get("prompt_tokens", 0))
        for row in prediction_rows
    )
    completion_tokens = sum(
        int(row.get("usage", {}).get("completion_tokens", 0))
        for row in prediction_rows
    )
    latencies = [
        float(row.get("latency_seconds", 0.0)) for row in prediction_rows
    ]
    estimated_cost = None
    if input_price is not None and output_price is not None:
        estimated_cost = (
            prompt_tokens * input_price + completion_tokens * output_price
        ) / 1_000_000
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds_sum": sum(latencies),
        "latency_seconds_mean": (
            statistics.fmean(latencies) if latencies else 0.0
        ),
        "latency_seconds_median": (
            statistics.median(latencies) if latencies else 0.0
        ),
        "input_cost_per_million_usd": input_price,
        "output_cost_per_million_usd": output_price,
        "estimated_cost_usd": estimated_cost,
    }


def input_provenance(
    data_dir: Path,
    split_path: Path,
    train_path: Path,
    shots: int,
) -> dict[str, Any]:
    paths = {
        "manifest": data_dir / "manifest.json",
        "schema": data_dir / "schema.json",
        "evaluation_split": split_path,
    }
    if shots:
        paths["few_shot_train_split"] = train_path
    return {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }


def main() -> None:
    args = parse_args()
    args.split = _normalized_split(args.split)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing sealed test run without --allow-test")
    if args.resume and args.overwrite:
        raise SystemExit("Choose only one of --resume and --overwrite")
    if args.workers < 1 or args.max_retries < 1:
        raise SystemExit("--workers and --max-retries must be positive")
    if args.limit < 0 or args.bootstrap_replicates < 0:
        raise SystemExit("--limit and --bootstrap-replicates cannot be negative")

    (
        canonical_rows,
        train_rows,
        manifest,
        schema,
        split_path,
        train_path,
    ) = load_release(args.data_dir, args.split)
    rows = canonical_rows[: args.limit] if args.limit > 0 else canonical_rows
    if not rows:
        raise SystemExit("No rows selected")

    examples_by_horizon = (
        select_few_shot_examples(train_rows, args.seed)
        if args.shots == 3
        else {}
    )
    config = run_config(args, examples_by_horizon, manifest)
    run_id = json_hash(config)
    print(
        f"T6 v2 LLM: split={args.split} rows={len(rows)}/{len(canonical_rows)} "
        f"model={args.model} shots={args.shots} rung={args.feature_rung}"
    )
    print(f"Endpoint: {config['api_endpoint']}")
    print(f"Run ID: {run_id}")
    print(f"Input fields: {', '.join(config['input_fields'])}")

    sample_prompt = build_prompt(
        rows[0],
        examples_by_horizon,
        args.feature_rung,
    )
    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== PROMPT PREVIEW ===")
        print(sample_prompt)
        print(f"\nPrompt characters: {len(sample_prompt)}")
        return

    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    env_name = args.api_key_env or default_key_env(args.provider)
    api_key = os.environ.get(env_name, "")
    if not api_key:
        raise SystemExit(f"Set {env_name} before running the baseline")

    if args.output.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Output exists: {args.output}; use --resume or --overwrite"
        )
    if args.overwrite and args.output.exists():
        args.output.unlink()

    latest = read_latest(args.output) if args.resume else {}
    for cached in latest.values():
        if cached.get("run_id") != run_id:
            raise SystemExit(
                "Existing output belongs to a different run configuration"
            )
    completed = {
        iid for iid, row in latest.items()
        if not row.get("error")
    }
    pending = [
        row for row in rows if str(row["instance_id"]) not in completed
    ]
    print(f"Cached successes: {len(completed)}  Pending: {len(pending)}")

    def work(row: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(row, examples_by_horizon, args.feature_rung)
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                visible = row.get("num_siblings_visible_d")
                response = call_model(args, api_key, prompt, visible)
                return prediction_row(row, response, config, run_id)
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        return {
            "instance_id": str(row["instance_id"]),
            "condition_id": str(row["condition_id"]),
            "bundle_day": str(row["bundle_day"]),
            "horizon_days": int(row["horizon_days"]),
            "event_cluster_id": str(row["event_cluster_id"]),
            "split": args.split,
            "provider": args.provider,
            "model": args.model,
            "shots": args.shots,
            "feature_rung": args.feature_rung,
            "run_id": run_id,
            "error": {
                "type": last_error.__class__.__name__,
                "message": str(last_error),
            },
        }

    processed = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        future_to_row = {pool.submit(work, row): row for row in pending}
        for future in concurrent.futures.as_completed(future_to_row):
            result = future.result()
            latest[str(result["instance_id"])] = result
            append_jsonl(args.output, result)
            processed += 1
            errors += int("error" in result)
            if args.delay > 0:
                time.sleep(args.delay)
            if processed % 50 == 0 or processed == len(pending):
                print(
                    f"  [{processed}/{len(pending)}] errors={errors}",
                    flush=True,
                )

    compact_jsonl(args.output, rows, latest)
    row_ids = {str(row["instance_id"]) for row in rows}
    successful = {
        iid: row
        for iid, row in latest.items()
        if iid in row_ids and not row.get("error")
    }
    missing = [
        str(row["instance_id"])
        for row in rows
        if str(row["instance_id"]) not in successful
    ]
    successful_source_rows = [
        row for row in rows if str(row["instance_id"]) in successful
    ]
    ordered_predictions = [
        successful[str(row["instance_id"])]
        for row in successful_source_rows
    ]

    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        **config,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_access_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "status": "complete" if not missing else "incomplete",
        "requested_rows": len(rows),
        "canonical_split_rows": len(canonical_rows),
        "successful_rows": len(successful),
        "failed_or_missing_rows": len(missing),
        "failed_or_missing_instance_ids": missing,
        "inputs": input_provenance(
            args.data_dir,
            split_path,
            train_path,
            args.shots,
        ),
        "schema_contract": {
            "model_input_rule": schema.get("model_input_rule"),
            "excluded_descriptive_graph_degree": "num_siblings_total",
            "known_release_limitation": (
                "primary_price_d/primary_sigma_14d are present in JSONL but "
                "absent from schema.prediction_time_fields, so this runner "
                "does not serialize them"
            ),
        },
        "execution_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "usage": usage_summary(
            ordered_predictions,
            args.input_cost_per_million,
            args.output_cost_per_million,
        ),
    }
    if successful_source_rows:
        report["metrics"] = evaluate(
            successful_source_rows,
            successful,
            args.bootstrap_replicates,
            args.seed,
        )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Predictions: {args.output}")
    print(f"Report: {report_path}")
    if missing:
        print(
            f"INCOMPLETE: {len(missing)} rows need --resume",
            file=sys.stderr,
        )
        raise SystemExit(2)

    overall = report["metrics"]["overall"]
    print(
        f"Macro-F1={overall['macro_f1']:.4f} "
        f"Accuracy={overall['accuracy']:.4f} N={overall['n']}"
    )
    for horizon, metrics in report["metrics"]["by_horizon"].items():
        print(
            f"  H={horizon}d Macro-F1={metrics['macro_f1']:.4f} "
            f"Accuracy={metrics['accuracy']:.4f} N={metrics['n']}"
        )


if __name__ == "__main__":
    main()
