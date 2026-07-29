#!/usr/bin/env python3
"""T4 leakage-safe LLM baseline for canonical daily market-day bundles.

The model sees only fields available by the end of bundle day d. It predicts
the forward 1d/3d/7d price deltas. Direction and magnitude are derived with
the frozen T4 label rule; gold outcomes are never included in target prompts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

HORIZONS = ("1d", "3d", "7d")
DATASET_VERSION = "t4.kdd.v2"
SPLIT_VERSION = "tier2.temporal.v2"
PROMPT_VERSION = "t4.kdd.v2.llm.r1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "KDD/data/t4_kdd_v2"
INPUT_FEATURES = (
    "bundle_day",
    "price_d",
    "sigma_14d",
    "sigma_n_obs",
    "n_posts",
    "followers_max",
    "engagement_sum",
    "engagement_max",
    "max_final_grade",
)
FORBIDDEN_FEATURES = {
    "price_1d", "price_3d", "price_7d",
    "delta_1d", "delta_3d", "delta_7d",
    "z_1d", "z_3d", "z_7d",
    "direction_label", "magnitude_bucket",
    "confound_1d", "confound_3d", "confound_7d", "confound_flag",
    "magnitude_tercile_1", "magnitude_tercile_2",
}
SYSTEM_PROMPT = """\
You forecast prediction-market prices from information observable at the end
of a UTC market day. Predict the signed YES-price changes after 1, 3, and 7
days. Do not invent extra fields. Return only strict JSON:
{"delta_1d": 0.0, "delta_3d": 0.0, "delta_7d": 0.0}
Each future price must remain in [0, 1].
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T4 daily LLM baseline: market movement prediction"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "xai"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--shots", type=int, choices=[0, 3], default=0)
    parser.add_argument(
        "--data-dir", "--local-dir", dest="data_dir", type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--split", choices=["validation", "val", "test"], default="validation"
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", default="t4_llm_predictions.jsonl")
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(
    data_dir: Path, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path, Path]:
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / f"{split}.jsonl"
    for path in (manifest_path, train_path, eval_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical T4 file: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"Expected {DATASET_VERSION}")
    if manifest.get("split_version") != SPLIT_VERSION:
        raise ValueError(f"Expected {SPLIT_VERSION}")
    if not manifest.get("release_ready"):
        raise ValueError("T4 release is not release_ready")
    train_df = pd.read_json(train_path, lines=True)
    eval_df = pd.read_json(eval_path, lines=True)
    if len(eval_df) != int(manifest["counts"][split]):
        raise ValueError(f"{split}: row count differs from manifest")
    required = set(INPUT_FEATURES) | {
        "instance_id", "condition_id", "event_cluster_id",
        "direction_label", "magnitude_bucket",
    }
    for name, frame in (("train", train_df), (split, eval_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if set(INPUT_FEATURES) & FORBIDDEN_FEATURES:
        raise AssertionError("Forbidden T4 feature selected")
    return train_df, eval_df, manifest, manifest_path, eval_path


def instance_key(row: dict[str, Any]) -> str:
    if row.get("instance_id"):
        return str(row["instance_id"])
    return f"{row['condition_id']}::{row['bundle_day']}"


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def _feature_block(row: dict[str, Any]) -> str:
    price = _number(row, "price_d", 0.5)
    sigma = _number(row, "sigma_14d", 0.0)
    return (
        f"bundle_day: {row.get('bundle_day')}\n"
        f"current_yes_price: {price:.6f}\n"
        f"historical_14d_volatility: {sigma:.6f}\n"
        f"historical_volatility_observations: {int(_number(row, 'sigma_n_obs'))}\n"
        f"posts_observed_today: {int(_number(row, 'n_posts'))}\n"
        f"maximum_author_followers: {int(_number(row, 'followers_max'))}\n"
        f"total_observed_engagement: {int(_number(row, 'engagement_sum'))}\n"
        f"maximum_post_engagement: {int(_number(row, 'engagement_max'))}\n"
        f"maximum_evidence_grade: {int(_number(row, 'max_final_grade'))}\n"
        f"valid_delta_range: [{-price:.6f}, {1.0-price:.6f}]"
    )


def build_prompt(
    row: dict[str, Any], few_shot: list[dict[str, Any]]
) -> str:
    parts = ["Task: predict the daily forward YES-price change."]
    if few_shot:
        parts.append("\nTraining examples:")
        for i, example in enumerate(few_shot, 1):
            answer = {
                f"delta_{h}": _number(example, f"delta_{h}") for h in HORIZONS
            }
            parts.append(
                f"\nExample {i}:\n{_feature_block(example)}\n"
                f"Answer: {json.dumps(answer, allow_nan=False)}"
            )
    parts.append(f"\nTarget:\n{_feature_block(row)}")
    parts.append("\nReturn strict JSON only.")
    return "\n".join(parts)


def _endpoint(base_url: str, provider: str) -> str:
    if not base_url:
        if provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        if provider == "xai":
            return "https://api.x.ai/v1/chat/completions"
        return "https://api.openai.com/v1/chat/completions"
    base = base_url.rstrip("/")
    if provider == "anthropic":
        if base.endswith("/messages"):
            return base
        return f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_llm(
    args: argparse.Namespace, api_key: str, prompt: str
) -> str:
    url = _endpoint(args.base_url, args.provider)
    if args.provider == "anthropic":
        body = {
            "model": args.model,
            "max_tokens": 128,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if args.base_url:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        response = _post_json(url, headers, body, args.timeout)
        return "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        ).strip()

    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 128,
    }
    if args.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = _post_json(url, headers, body, args.timeout)
    return response["choices"][0]["message"]["content"].strip()


def parse_prediction(text: str, price_d: float) -> dict[str, float]:
    candidate = text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(candidate[start : end + 1])

    prediction: dict[str, float] = {}
    for horizon in HORIZONS:
        key = f"delta_{horizon}"
        value = float(payload[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {key}")
        prediction[key] = max(-price_d, min(1.0 - price_d, value))
    return prediction


def derive_labels(
    delta_1d: float, sigma_14d: float, cut1: float, cut2: float
) -> tuple[str, str]:
    z = delta_1d / sigma_14d if sigma_14d > 0 else 0.0
    if abs(z) < 1.0:
        return "flat", "none"
    direction = "up" if z > 0 else "down"
    magnitude = "small" if abs(z) <= cut1 else (
        "medium" if abs(z) <= cut2 else "large"
    )
    return direction, magnitude


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def compact_jsonl(
    path: Path,
    source_rows: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for source in source_rows:
            key = instance_key(source)
            if key in latest:
                handle.write(
                    json.dumps(latest[key], ensure_ascii=False, allow_nan=False)
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    args.split = "validation" if args.split == "val" else args.split
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing sealed test run without --allow-test")
    if args.resume and args.overwrite:
        raise SystemExit("Choose only one of --resume and --overwrite")
    train_df, test_df, manifest, manifest_path, eval_path = load_splits(
        args.data_dir, args.split
    )
    if args.limit > 0:
        test_df = test_df.iloc[: args.limit].copy()
    # JSON round-trip preserves public JSONL null semantics instead of
    # pandas' float NaN sentinel, which must never reach rank metrics.
    train_records = json.loads(
        train_df.to_json(orient="records", double_precision=15)
    )
    test_records = json.loads(
        test_df.to_json(orient="records", double_precision=15)
    )
    few_shot: list[dict[str, Any]] = []
    if args.shots:
        for label in ("flat", "up", "down"):
            candidates = sorted(
                (
                    row for row in train_records
                    if row.get("direction_label") == label
                ),
                key=instance_key,
            )
            if not candidates:
                raise ValueError(f"No train example for direction {label}")
            few_shot.append(candidates[0])
    config = {
        "version": "t4.llm.baseline.run.v2",
        "dataset_version": DATASET_VERSION,
        "split_version": SPLIT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "provider": args.provider,
        "model": args.model,
        "split": args.split,
        "shots": args.shots,
        "input_features": list(INPUT_FEATURES),
        "few_shot_instance_ids": [instance_key(row) for row in few_shot],
        "temperature": 0,
        "thinking_enabled": not args.disable_thinking,
        "system_prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    print(
        f"T4 v2: split={args.split} rows={len(test_records)}, "
        f"model={args.model}, shots={args.shots}, workers={args.workers}"
    )
    print(f"Endpoint: {_endpoint(args.base_url, args.provider)}  Run ID: {run_id}")
    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== PROMPT PREVIEW ===")
        print(build_prompt(test_records[0], few_shot))
        return

    env_name = args.api_key_env or {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "xai": "XAI_API_KEY",
    }[args.provider]
    api_key = os.environ.get(env_name, "")
    if not api_key and "localhost" not in args.base_url:
        raise SystemExit(f"Set {env_name}")

    output_path = Path(args.output)
    if output_path.exists() and not (args.resume or args.overwrite):
        raise SystemExit(
            f"Output exists: {output_path}; use --resume or --overwrite"
        )
    if args.overwrite and output_path.exists():
        output_path.unlink()
    cached = read_jsonl(output_path) if args.resume else []
    latest = {
        str(row["instance_id"]): row
        for row in cached
        if row.get("instance_id")
    }
    for row in latest.values():
        if row.get("run_id") != run_id:
            raise SystemExit("Existing output has a different run configuration")
    completed = {
        key for key, row in latest.items() if not row.get("error")
    }
    pending = [r for r in test_records if instance_key(r) not in completed]
    print(f"Pending: {len(pending)}; cached successful: {len(completed)}")

    def infer(row: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(row, few_shot)
        price = _number(row, "price_d", 0.5)
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                raw = call_llm(args, api_key, prompt)
                prediction = parse_prediction(raw, price)
                direction, magnitude = derive_labels(
                    prediction["delta_1d"],
                    _number(row, "sigma_14d"),
                    float(manifest["magnitude_train_terciles"][0]),
                    float(manifest["magnitude_train_terciles"][1]),
                )
                return {
                    "instance_id": instance_key(row),
                    "condition_id": str(row["condition_id"]),
                    "bundle_day": str(row["bundle_day"]),
                    "direction_label": direction,
                    "magnitude_bucket": magnitude,
                    **prediction,
                    "model": args.model,
                    "shots": args.shots,
                    "split": args.split,
                    "run_id": run_id,
                    "raw_output": raw,
                }
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(0.25 * (attempt + 1))
        return {
            "instance_id": instance_key(row),
            "condition_id": str(row["condition_id"]),
            "bundle_day": str(row["bundle_day"]),
            "split": args.split,
            "run_id": run_id,
            "error": {
                "type": last_error.__class__.__name__ if last_error else "Error",
                "message": str(last_error),
            },
        }

    errors = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {executor.submit(infer, row): row for row in pending}
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            result = future.result()
            errors += int("error" in result)
            append_jsonl(output_path, result)
            latest[str(result["instance_id"])] = result
            if done % 100 == 0 or done == len(pending):
                print(f"  [{done}/{len(pending)}] errors={errors}")
            if args.sleep:
                time.sleep(args.sleep)

    compact_jsonl(output_path, test_records, latest)
    predictions = [
        row for row in latest.values() if "direction_label" in row
    ]
    wanted = {instance_key(row) for row in test_records}
    predictions = [p for p in predictions if p["instance_id"] in wanted]

    eventxbench_root = Path(__file__).resolve().parents[2]
    if str(eventxbench_root) not in sys.path:
        sys.path.insert(0, str(eventxbench_root))
    from evaluation.evaluate import evaluate_t4

    metrics = evaluate_t4(predictions, test_records)
    metrics.update({
        **config,
        "run_id": run_id,
        "status": "complete" if len(predictions) == len(test_records) else "incomplete",
        "n_errors": errors,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "evaluation_split": {"path": str(eval_path), "sha256": sha256(eval_path)},
        },
    })
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output_path.with_suffix(".report.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Metrics saved to {metrics_path}")
    if len(predictions) != len(test_records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
