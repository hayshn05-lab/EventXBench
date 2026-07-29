#!/usr/bin/env python3
"""Paper-grade T2 contextual LLM baseline.

The runner evaluates one fixed model/shot/split configuration against
``t2.gold.r3.contextual.v1``.  It supports official hosted providers and
OpenAI-compatible gateways such as Lumid, predicts ``NONE`` explicitly,
produces a full candidate ranking for MRR, checkpoints every response, and
writes a provenance-rich report.

Examples:
    # Inspect a validation prompt without making a request.
    python baselines/t2/llm_baseline.py --provider openai --model MODEL \
        --base-url https://lum.id/llm --api-key-env LUMID_API_KEY \
        --split val --shots 0 --dry-run

    # Lumid validation run. The runner accepts either the gateway root shown
    # here or the versioned base URL https://lum.id/llm/v1.
    python baselines/t2/llm_baseline.py --provider openai \
        --base-url https://lum.id/llm --api-key-env LUMID_API_KEY \
        --model nvidia/Gemma-4-26B-A4B-NVFP4 --disable-thinking \
        --split val --shots 0 --resume --output results/gemma.val.0shot.jsonl

Test labels are sealed.  A test run requires the explicit ``--allow-test``
flag and must use a configuration frozen before inspecting test results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_DIR = REPO_ROOT / "data/t2/gold_r3_contextual_final"
DEFAULT_TRAIN_CANDIDATES = (
    REPO_ROOT
    / "KDD/t2_recall_freeze_m3/frozen_candidates_retrospective.jsonl"
)
DEFAULT_TRAIN_LABELS = (
    REPO_ROOT
    / "data/t2/train_silver_contextual_q3_p2/train_labels_final.csv"
)
DEFAULT_SILVER_REPORT = (
    REPO_ROOT
    / "data/t2/train_silver_contextual_q3_p2/generation_report.json"
)

POLICY_VERSION = "contextual_entity_market_v1"
PROMPT_VERSION = "t2.contextual.llm.r1"
RECALL_VERSION = "r3.bge-m3.dt3.nofloor"
SEED = 20260721

SYSTEM_PROMPT = """You perform contextual post-to-prediction-market linking.

Choose a candidate when the post is meaningfully related to the same central
entity and market context and could provide useful contextual information.
The post does not need the same direction, claim, threshold, timeframe, or
formal proposition. Use those details only to decide which related candidate
is best. Shared wording without a meaningful semantic relationship is not
enough. Choose NONE only when no candidate is meaningfully related.

Rank every option from most to least relevant. Option 0 is NONE; candidate
markets are numbered 1 through N. Return strict JSON only:
{"ranking":[integer, ...], "confidence": number}
The ranking must contain every integer from 0 through N exactly once."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "anthropic", "xai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default="",
        help=(
            "Optional OpenAI-compatible base URL. Lumid accepts either "
            "https://lum.id/llm or https://lum.id/llm/v1."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable containing the API key (provider default when omitted).",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement that the sealed test configuration is frozen.",
    )
    parser.add_argument("--shots", type=int, choices=[0, 3], default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--train-labels", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--silver-report", type=Path, default=DEFAULT_SILVER_REPORT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prediction JSONL path; required unless --dry-run is used.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--rule-chars", type=int, default=800)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def load_split(gold_dir: Path, split: str, top_k: int) -> tuple[list[dict], Path, Path]:
    candidate_path = gold_dir / f"{split}_candidates.jsonl"
    label_path = gold_dir / f"{split}_labels.csv"
    if not candidate_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing canonical {split} files under {gold_dir}")

    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["instance_id"]: row for row in csv.DictReader(handle)}

    candidate_rows: dict[str, dict] = {}
    with candidate_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row["instance_id"])
            candidates = sorted(row["candidates"], key=lambda item: int(item["candidate_rank"]))
            row["candidates"] = candidates[:top_k]
            candidate_rows[iid] = row

    if set(labels) != set(candidate_rows):
        raise ValueError(f"{split}: label and candidate instance IDs do not match")

    rows = []
    for iid in sorted(candidate_rows):
        row = candidate_rows[iid]
        label = labels[iid]
        candidates = row["candidates"]
        ids = [str(item["condition_id"]) for item in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{iid}: duplicate condition IDs")
        expected_ranks = list(range(1, len(candidates) + 1))
        actual_ranks = [int(item["candidate_rank"]) for item in candidates]
        if actual_ranks != expected_ranks:
            raise ValueError(f"{iid}: non-contiguous candidate ranks {actual_ranks}")
        gold = str(label["final_choice"]).strip()
        if gold != "NONE" and gold not in ids:
            raise ValueError(f"{iid}: gold choice is absent from top-{top_k} candidates")
        rows.append(
            {
                "instance_id": iid,
                "tweet_id": str(row["tweet_id"]),
                "post_text": str(row["post_text"]),
                "candidate_setting": str(row.get("candidate_setting", "retrospective")),
                "recall_version": str(row.get("recall_version", RECALL_VERSION)),
                "candidates": candidates,
                "gold": gold,
            }
        )
    return rows, candidate_path, label_path


def load_few_shot_examples(candidate_path: Path, label_path: Path) -> list[dict]:
    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {
            row["instance_id"]: row
            for row in csv.DictReader(handle)
            if row.get("consensus_status") == "exact_consensus" and row.get("final_choice")
        }

    candidates = {}
    with candidate_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("market_split") == "train" and row["instance_id"] in labels:
                row["candidates"] = sorted(
                    row["candidates"], key=lambda item: int(item["candidate_rank"])
                )[:10]
                candidates[row["instance_id"]] = row

    eligible = []
    for iid, row in candidates.items():
        gold = labels[iid]["final_choice"]
        ids = [str(item["condition_id"]) for item in row["candidates"]]
        if gold == "NONE":
            gold_index = 0
        elif gold in ids:
            gold_index = ids.index(gold) + 1
        else:
            continue
        eligible.append(
            {
                "instance_id": iid,
                "post_text": str(row["post_text"]),
                "candidates": row["candidates"],
                "gold": gold,
                "gold_index": gold_index,
            }
        )

    none_rows = sorted(
        (row for row in eligible if row["gold"] == "NONE"),
        key=lambda row: (len(row["candidates"]), row["instance_id"]),
    )
    link_rows = sorted(
        (row for row in eligible if row["gold"] != "NONE"),
        key=lambda row: (len(row["candidates"]), row["instance_id"]),
    )
    if not none_rows or len(link_rows) < 2:
        raise ValueError("Need at least one NONE and two LINK exact-consensus train examples")
    return [none_rows[0], link_rows[0], link_rows[1]]


def _candidate_block(candidates: list[dict], rule_chars: int) -> list[str]:
    lines = []
    for index, candidate in enumerate(candidates, 1):
        lines.append(f"{index}. {str(candidate['question']).strip()}")
        rule = str(candidate.get("resolution_rule") or "").strip()
        if rule and rule_chars > 0:
            lines.append(f"   Resolution: {rule[:rule_chars]}")
    return lines


def _example_ranking(example: dict) -> list[int]:
    all_options = list(range(0, len(example["candidates"]) + 1))
    first = int(example["gold_index"])
    return [first] + [value for value in all_options if value != first]


def build_prompt(row: dict, examples: list[dict], rule_chars: int) -> str:
    lines = []
    if examples:
        lines.append("TRAINING EXAMPLES")
        for number, example in enumerate(examples, 1):
            lines.extend(["", f"Example {number}", f"POST:\n{example['post_text'].strip()}", "OPTIONS:", "0. NONE"])
            lines.extend(_candidate_block(example["candidates"], rule_chars))
            answer = {"ranking": _example_ranking(example), "confidence": 1.0}
            lines.append("ANSWER: " + json.dumps(answer, separators=(",", ":")))
        lines.extend(["", "TARGET"])
    lines.extend([f"POST:\n{row['post_text'].strip()}", "OPTIONS:", "0. NONE"])
    lines.extend(_candidate_block(row["candidates"], rule_chars))
    lines.append("ANSWER:")
    return "\n".join(lines)


def parse_ranking(raw: str, n_candidates: int) -> tuple[list[int], float | None, str]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model content")
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    confidence = None
    source = "full_ranking"
    values: list[int]
    if isinstance(payload, dict) and isinstance(payload.get("ranking"), list):
        values = [int(value) for value in payload["ranking"]]
        if payload.get("confidence") is not None:
            confidence = float(payload["confidence"])
    elif isinstance(payload, dict) and payload.get("choice") is not None:
        values = [int(payload["choice"])]
        source = "choice_plus_canonical_tail"
        if payload.get("confidence") is not None:
            confidence = float(payload["confidence"])
    else:
        values = [int(value) for value in re.findall(r"(?<![.\d])-?\d+(?![.\d])", text)]
        source = "text_numbers"

    valid = set(range(0, n_candidates + 1))
    ranking = []
    seen = set()
    for value in values:
        if value in valid and value not in seen:
            ranking.append(value)
            seen.add(value)
    if not ranking:
        raise ValueError(f"no valid option in response: {text[:200]!r}")
    ranking.extend(value for value in range(0, n_candidates + 1) if value not in seen)
    if set(ranking) != valid or len(ranking) != len(valid):
        raise ValueError(f"invalid completed ranking: {ranking}")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError(f"confidence outside [0,1]: {confidence}")
    return ranking, confidence, source


def _response_schema(n_candidates: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": n_candidates},
                "minItems": n_candidates + 1,
                "maxItems": n_candidates + 1,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["ranking", "confidence"],
        "additionalProperties": False,
    }


def call_model(args: argparse.Namespace, api_key: str, prompt: str, n_candidates: int) -> dict:
    url = endpoint(args.provider, args.base_url)
    started = time.perf_counter()
    if args.provider == "anthropic":
        body = {
            "model": args.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        if args.disable_thinking and args.base_url:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
        if args.base_url:
            # Lumid's Anthropic-compatible gateway authenticates with the same
            # bearer PAT used by its OpenAI-compatible endpoint.
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
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=body, timeout=args.timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        preview = response.text[:200]
        raise RuntimeError(
            f"non-JSON response status={response.status_code} body_prefix={preview!r}"
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
            "total_tokens": int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0),
        }
        finish_reason = payload.get("stop_reason")
    else:
        choice = payload["choices"][0]
        message = choice["message"]
        raw = (message.get("content") or message.get("reasoning_content") or "").strip()
        usage = payload.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        finish_reason = choice.get("finish_reason")

    ranking, confidence, ranking_source = parse_ranking(raw, n_candidates)
    return {
        "raw_output": raw,
        "ranking_indices": ranking,
        "confidence": confidence,
        "ranking_source": ranking_source,
        "finish_reason": finish_reason,
        "usage": token_usage,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def run_role(model: str, base_url: str, silver_report: Path) -> tuple[str, list[str]]:
    silver_models = []
    if silver_report.is_file():
        report = json.loads(silver_report.read_text(encoding="utf-8"))
        silver_models = sorted(
            str(value.get("model"))
            for value in report.get("models", {}).values()
            if value.get("model")
        )
    if model in silver_models:
        return "diagnostic_excluded_silver_judge", silver_models
    if base_url:
        return "auxiliary_custom_gateway", silver_models
    return "preregistered_hosted_llm", silver_models


def run_config(args: argparse.Namespace, examples: list[dict]) -> dict:
    role, silver_models = run_role(args.model, args.base_url, args.silver_report)
    return {
        "version": "t2.llm.baseline.run.v1",
        "policy_version": POLICY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "recall_version": RECALL_VERSION,
        "provider": args.provider,
        "model": args.model,
        "api_endpoint": endpoint(args.provider, args.base_url),
        "split": args.split,
        "shots": args.shots,
        "top_k": args.top_k,
        "rule_chars": args.rule_chars,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "thinking_enabled": not args.disable_thinking,
        "seed": args.seed,
        "run_role": role,
        "silver_judge_models": silver_models,
        "few_shot_instance_ids": [row["instance_id"] for row in examples],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }


def read_latest(path: Path) -> dict[str, dict]:
    latest = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["instance_id"])] = row
    return latest


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def compact_jsonl(path: Path, rows: list[dict], latest: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source in rows:
            iid = source["instance_id"]
            if iid in latest:
                handle.write(json.dumps(latest[iid], ensure_ascii=False) + "\n")


def prediction_row(source: dict, response: dict, config: dict, run_id: str) -> dict:
    candidate_ids = [str(item["condition_id"]) for item in source["candidates"]]
    ranking_indices = response["ranking_indices"]
    ranking_choices = ["NONE" if index == 0 else candidate_ids[index - 1] for index in ranking_indices]
    candidate_ranking = [choice for choice in ranking_choices if choice != "NONE"]
    return {
        "instance_id": source["instance_id"],
        "tweet_id": source["tweet_id"],
        "split": config["split"],
        "provider": config["provider"],
        "model": config["model"],
        "shots": config["shots"],
        "run_id": run_id,
        "run_role": config["run_role"],
        "candidate_ids": candidate_ids,
        "prediction": ranking_choices[0],
        "ranked_options": ranking_choices,
        "ranked_candidates": candidate_ranking,
        "ranking_indices": ranking_indices,
        "ranking_source": response["ranking_source"],
        "confidence": response["confidence"],
        "finish_reason": response["finish_reason"],
        "usage": response["usage"],
        "latency_seconds": response["latency_seconds"],
        "raw_output": response["raw_output"],
    }


def metric_arrays(rows: list[dict], predictions: dict[str, dict]) -> dict[str, np.ndarray]:
    accuracy = []
    rr = []
    gold_none = []
    pred_none = []
    for row in rows:
        pred_row = predictions[row["instance_id"]]
        gold = row["gold"]
        prediction = pred_row["prediction"]
        accuracy.append(gold == prediction)
        gold_none.append(gold == "NONE")
        pred_none.append(prediction == "NONE")
        if gold == "NONE":
            rr.append(1.0 if prediction == "NONE" else 0.0)
        else:
            ranking = pred_row["ranked_candidates"]
            rr.append(1.0 / (ranking.index(gold) + 1) if gold in ranking else 0.0)
    return {
        "accuracy": np.asarray(accuracy, dtype=float),
        "rr": np.asarray(rr, dtype=float),
        "gold_none": np.asarray(gold_none, dtype=bool),
        "pred_none": np.asarray(pred_none, dtype=bool),
    }


def summarize(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices = np.arange(len(arrays["accuracy"]))
    accuracy = arrays["accuracy"][indices]
    rr = arrays["rr"][indices]
    gold_none = arrays["gold_none"][indices]
    pred_none = arrays["pred_none"][indices]
    tp = int(np.sum(gold_none & pred_none))
    fp = int(np.sum(~gold_none & pred_none))
    fn = int(np.sum(gold_none & ~pred_none))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    none_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(indices)),
        "accuracy_at_1": float(np.mean(accuracy)),
        "mrr": float(np.mean(rr)),
        "none_precision": precision,
        "none_recall": recall,
        "none_f1": none_f1,
        "gold_none": int(np.sum(gold_none)),
        "predicted_none": int(np.sum(pred_none)),
    }


def bootstrap_ci(arrays: dict[str, np.ndarray], replicates: int, seed: int) -> dict:
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(arrays["accuracy"])
    values = {key: [] for key in ("accuracy_at_1", "mrr", "none_f1")}
    for _ in range(replicates):
        result = summarize(arrays, rng.integers(0, n, size=n))
        for key in values:
            values[key].append(result[key])
    return {
        key: {
            "low": float(np.quantile(samples, 0.025)),
            "high": float(np.quantile(samples, 0.975)),
        }
        for key, samples in values.items()
    }


def usage_summary(predictions: list[dict]) -> dict:
    latencies = [float(row["latency_seconds"]) for row in predictions]
    prompt_tokens = sum(int(row["usage"].get("prompt_tokens", 0)) for row in predictions)
    completion_tokens = sum(int(row["usage"].get("completion_tokens", 0)) for row in predictions)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds_sum": sum(latencies),
        "latency_seconds_mean": statistics.fmean(latencies) if latencies else 0.0,
        "latency_seconds_median": statistics.median(latencies) if latencies else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing sealed test run without --allow-test")
    if args.resume and args.overwrite:
        raise SystemExit("Choose only one of --resume and --overwrite")
    if args.workers < 1 or args.top_k < 1 or args.max_retries < 1:
        raise SystemExit("--workers, --top-k, and --max-retries must be positive")

    rows, candidate_path, label_path = load_split(args.gold_dir, args.split, args.top_k)
    total_rows = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    examples = (
        load_few_shot_examples(args.train_candidates, args.train_labels)
        if args.shots == 3
        else []
    )
    config = run_config(args, examples)
    run_id = json_hash(config)

    print(
        f"T2 contextual LLM: split={args.split} rows={len(rows)}/{total_rows} "
        f"model={args.model} shots={args.shots} workers={args.workers}"
    )
    print(f"Endpoint: {config['api_endpoint']}")
    print(f"Run role: {config['run_role']}  Run ID: {run_id}")
    if config["run_role"] == "diagnostic_excluded_silver_judge":
        print("WARNING: model generated train silver and is excluded from paper evaluation.")

    sample_prompt = build_prompt(rows[0], examples, args.rule_chars)
    if args.dry_run:
        print("\n=== PROMPT PREVIEW ===")
        print(SYSTEM_PROMPT)
        print("\n" + sample_prompt)
        print(f"\nPrompt characters: {len(sample_prompt)}")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")

    env_name = args.api_key_env or default_key_env(args.provider)
    api_key = os.environ.get(env_name, "")
    if not api_key:
        raise SystemExit(f"Set {env_name} before running the baseline")

    if args.output.exists() and not (args.resume or args.overwrite):
        raise SystemExit(f"Output exists: {args.output}; use --resume or --overwrite")
    if args.overwrite and args.output.exists():
        args.output.unlink()

    latest = read_latest(args.output) if args.resume else {}
    for cached in latest.values():
        if cached.get("run_id") != run_id:
            raise SystemExit("Existing output belongs to a different run configuration")
    completed = {iid for iid, row in latest.items() if not row.get("error")}
    pending = [row for row in rows if row["instance_id"] not in completed]
    print(f"Cached successes: {len(completed)}  Pending: {len(pending)}")

    def work(row: dict) -> dict:
        prompt = build_prompt(row, examples, args.rule_chars)
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = call_model(args, api_key, prompt, len(row["candidates"]))
                return prediction_row(row, response, config, run_id)
            except Exception as exc:  # checkpoint the final failure for safe resume
                last_error = exc
                if attempt < args.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        return {
            "instance_id": row["instance_id"],
            "tweet_id": row["tweet_id"],
            "split": args.split,
            "provider": args.provider,
            "model": args.model,
            "shots": args.shots,
            "run_id": run_id,
            "run_role": config["run_role"],
            "error": {"type": last_error.__class__.__name__, "message": str(last_error)},
        }

    processed = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_row = {pool.submit(work, row): row for row in pending}
        for future in concurrent.futures.as_completed(future_to_row):
            result = future.result()
            latest[result["instance_id"]] = result
            append_jsonl(args.output, result)
            processed += 1
            errors += int("error" in result)
            if args.delay > 0:
                time.sleep(args.delay)
            if processed % 50 == 0 or processed == len(pending):
                print(f"  [{processed}/{len(pending)}] errors={errors}", flush=True)

    compact_jsonl(args.output, rows, latest)
    successful = {
        iid: row
        for iid, row in latest.items()
        if iid in {source["instance_id"] for source in rows} and not row.get("error")
    }
    missing = [row["instance_id"] for row in rows if row["instance_id"] not in successful]
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        **config,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "incomplete",
        "requested_rows": len(rows),
        "canonical_split_rows": total_rows,
        "successful_rows": len(successful),
        "failed_or_missing_rows": len(missing),
        "failed_or_missing_instance_ids": missing,
        "mrr_convention": (
            "NONE RR=1 iff predicted NONE; linked gold uses reciprocal rank after "
            "removing NONE from the submitted full option ranking"
        ),
        "inputs": {
            "candidates": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
            "labels": {"path": str(label_path), "sha256": sha256(label_path)},
            "train_candidates": (
                {"path": str(args.train_candidates), "sha256": sha256(args.train_candidates)}
                if examples
                else None
            ),
            "train_labels": (
                {"path": str(args.train_labels), "sha256": sha256(args.train_labels)}
                if examples
                else None
            ),
        },
        "output": {"path": str(args.output), "sha256": sha256(args.output)},
    }
    if not missing:
        arrays = metric_arrays(rows, successful)
        metrics = summarize(arrays)
        metrics["ci95"] = bootstrap_ci(arrays, args.bootstrap_replicates, args.seed)
        report["metrics"] = metrics
        ordered_predictions = [successful[row["instance_id"]] for row in rows]
        report["usage"] = usage_summary(ordered_predictions)
        report["prediction_counts"] = dict(
            Counter(row["prediction"] if row["prediction"] == "NONE" else "LINK" for row in ordered_predictions)
        )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Predictions: {args.output}")
    print(f"Report: {report_path}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} rows need --resume", file=sys.stderr)
        raise SystemExit(2)
    metrics = report["metrics"]
    print(
        f"Accuracy@1={metrics['accuracy_at_1']:.4f}  "
        f"MRR={metrics['mrr']:.4f}  NONE-F1={metrics['none_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
