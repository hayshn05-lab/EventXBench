#!/usr/bin/env python3
"""Run frozen deterministic T2 baselines on contextual val/test gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SEED = 20260721
BOOTSTRAP_REPLICATES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "T2 artifact root containing gold_r3_contextual_final, "
            "t2_contextual_thresholds_q3_p2, and "
            "train_silver_contextual_q3_p2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/baselines_contextual_r3)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
        help="Bootstrap replicates for confidence intervals",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(split: str, gold_dir: Path) -> list[dict]:
    label_path = gold_dir / f"{split}_labels.csv"
    candidate_path = gold_dir / f"{split}_candidates.jsonl"
    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {row["instance_id"]: row for row in csv.DictReader(handle)}
    candidates = {}
    with candidate_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            candidates[row["instance_id"]] = sorted(
                row["candidates"], key=lambda candidate: candidate["candidate_rank"]
            )
    if set(labels) != set(candidates):
        raise ValueError(f"{split}: labels/candidates mismatch")
    rows = []
    for instance_id in sorted(labels):
        label = labels[instance_id]
        ranked = candidates[instance_id]
        rows.append({
            "instance_id": instance_id,
            "tweet_id": label["tweet_id"],
            "gold": label["final_choice"],
            "ranked_ids": [candidate["condition_id"] for candidate in ranked],
            "scores": [float(candidate["cosine_score"]) for candidate in ranked],
        })
    return rows


def train_link_prior(train_labels: Path) -> tuple[float, dict[str, int]]:
    with train_labels.open(encoding="utf-8-sig", newline="") as handle:
        labels = [row["binary_label"] for row in csv.DictReader(handle)]
    counts = Counter(label for label in labels if label in {"LINK", "NONE"})
    total = counts["LINK"] + counts["NONE"]
    return counts["LINK"] / total, dict(counts)


def make_predictions(rows: list[dict], baseline: str, threshold: float, link_prior: float) -> list[dict]:
    rng = np.random.default_rng(SEED + (0 if rows[0]["instance_id"].split("-")[2] == "val" else 1))
    output = []
    for row in rows:
        ranked = row["ranked_ids"]
        scores = row["scores"]
        if baseline == "always_none":
            prediction = "NONE"
            submitted_ranking = ["NONE"]
        elif baseline == "stratified_random_train_prior":
            if ranked and rng.random() < link_prior:
                shuffled = list(ranked)
                rng.shuffle(shuffled)
                prediction = shuffled[0]
                submitted_ranking = shuffled
            else:
                prediction = "NONE"
                submitted_ranking = ["NONE"]
        elif baseline == "bge_top1_no_none":
            prediction = ranked[0]
            submitted_ranking = ranked
        elif baseline in {"bge_top1_frozen_threshold", "bge_top1_high_precision"}:
            prediction = ranked[0] if scores[0] >= threshold else "NONE"
            # Preserve the full dense ordering for retrieval MRR. NONE receives
            # RR=1 only when it is the top-level prediction (see metrics()).
            submitted_ranking = ranked
        else:
            raise ValueError(baseline)
        output.append({
            "instance_id": row["instance_id"],
            "tweet_id": row["tweet_id"],
            "gold": row["gold"],
            "prediction": prediction,
            "ranked_predictions": submitted_ranking,
            "top_cosine": scores[0],
        })
    return output


def evaluator_prediction(row: dict) -> dict:
    """Convert an internal prediction row to the unified T2 evaluator schema."""
    prediction = str(row["prediction"])
    ranked_candidates = [
        str(candidate)
        for candidate in row["ranked_predictions"]
        if str(candidate) != "NONE"
    ]
    ranked_options = [prediction]
    ranked_options.extend(
        candidate for candidate in ranked_candidates if candidate != prediction
    )
    if "NONE" not in ranked_options:
        ranked_options.append("NONE")
    return {
        "instance_id": str(row["instance_id"]),
        "tweet_id": str(row["tweet_id"]),
        "prediction": prediction,
        "ranked_options": ranked_options,
        "ranked_candidates": ranked_candidates,
    }


def write_evaluator_predictions(path: Path, predictions: list[dict]) -> None:
    """Write prediction-only JSONL accepted by ``evaluation.evaluate_t2``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(evaluator_prediction(row), ensure_ascii=False) + "\n")


def write_evaluator_gold(path: Path, rows: list[dict]) -> None:
    """Write the matching gold JSONL needed for a local evaluator round trip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            gold = {
                "instance_id": str(row["instance_id"]),
                "tweet_id": str(row["tweet_id"]),
                "gold": str(row["gold"]),
                "candidate_ids": [
                    str(value)
                    for value in row.get(
                        "ranked_ids",
                        row.get("ranked_predictions", []),
                    )
                ],
            }
            handle.write(json.dumps(gold, ensure_ascii=False) + "\n")


def per_instance(predictions: list[dict]) -> dict[str, np.ndarray]:
    accuracy = []
    reciprocal_rank = []
    gold_none = []
    pred_none = []
    for row in predictions:
        gold = row["gold"]
        pred = row["prediction"]
        accuracy.append(gold == pred)
        gold_none.append(gold == "NONE")
        pred_none.append(pred == "NONE")
        if gold == "NONE":
            reciprocal_rank.append(1.0 if pred == "NONE" else 0.0)
        else:
            ranked = row["ranked_predictions"]
            reciprocal_rank.append(1.0 / (ranked.index(gold) + 1) if gold in ranked else 0.0)
    return {
        "accuracy": np.asarray(accuracy, dtype=float),
        "rr": np.asarray(reciprocal_rank, dtype=float),
        "gold_none": np.asarray(gold_none, dtype=bool),
        "pred_none": np.asarray(pred_none, dtype=bool),
    }


def summarize_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict:
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(indices)),
        "accuracy_at_1": float(np.mean(accuracy)),
        "mrr": float(np.mean(rr)),
        "none_precision": precision,
        "none_recall": recall,
        "none_f1": f1,
        "gold_none": int(np.sum(gold_none)),
        "predicted_none": int(np.sum(pred_none)),
    }


def bootstrap_ci(
    arrays: dict[str, np.ndarray], split_seed: int, replicates: int
) -> dict:
    rng = np.random.default_rng(split_seed)
    n = len(arrays["accuracy"])
    values = {key: [] for key in ("accuracy_at_1", "mrr", "none_f1")}
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        result = summarize_arrays(arrays, indices)
        for key in values:
            values[key].append(result[key])
    return {
        key: {
            "low": float(np.quantile(samples, 0.025)),
            "high": float(np.quantile(samples, 0.975)),
        }
        for key, samples in values.items()
    }


def candidate_recall(rows: list[dict]) -> dict:
    linked = [row for row in rows if row["gold"] != "NONE"]
    result = {"linked_gold_n": len(linked)}
    for k in (1, 3, 5, 10):
        result[f"recall_at_{k}"] = sum(row["gold"] in row["ranked_ids"][:k] for row in linked) / len(linked)
    return result


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    gold_dir = data_dir / "gold_r3_contextual_final"
    thresholds = (
        data_dir / "t2_contextual_thresholds_q3_p2/FROZEN_THRESHOLDS.json"
    )
    train_labels = (
        data_dir / "train_silver_contextual_q3_p2/train_labels_final.csv"
    )
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_dir / "baselines_contextual_r3"
    )
    required_paths = [
        gold_dir / "release_manifest.json",
        gold_dir / "val_candidates.jsonl",
        gold_dir / "val_labels.csv",
        gold_dir / "test_candidates.jsonl",
        gold_dir / "test_labels.csv",
        thresholds,
        train_labels,
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required T2 artifacts:\n{missing}")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be at least 1")

    threshold_config = json.loads(thresholds.read_text(encoding="utf-8"))
    if threshold_config.get("test_labels_used") is not False:
        raise ValueError("threshold provenance does not preserve the test seal")
    dense_threshold = float(threshold_config["dense_none_threshold"])
    high_threshold = float(threshold_config["cosine_high_95pct_binary_link_precision_min50"])
    link_prior, prior_counts = train_link_prior(train_labels)
    splits = {split: load_split(split, gold_dir) for split in ("val", "test")}

    baseline_config = {
        "always_none": None,
        "stratified_random_train_prior": None,
        "bge_top1_no_none": 0.0,
        "bge_top1_frozen_threshold": dense_threshold,
        "bge_top1_high_precision": high_threshold,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    prediction_files = {}
    gold_files = {}
    for split, rows in splits.items():
        gold_path = out_dir / f"t2.{split}.gold.jsonl"
        write_evaluator_gold(gold_path, rows)
        gold_files[split] = {
            "path": str(gold_path),
            "sha256": sha256(gold_path),
            "rows": len(rows),
        }
    for baseline, threshold in baseline_config.items():
        metrics[baseline] = {}
        for split, rows in splits.items():
            effective_threshold = float(threshold or 0.0)
            predictions = make_predictions(rows, baseline, effective_threshold, link_prior)
            arrays = per_instance(predictions)
            result = summarize_arrays(arrays)
            result["ci95"] = bootstrap_ci(
                arrays,
                SEED + len(metrics) * 10 + (0 if split == "val" else 1),
                args.bootstrap_replicates,
            )
            metrics[baseline][split] = result

            csv_path = out_dir / f"{baseline}.{split}.predictions.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "instance_id", "tweet_id", "gold", "prediction",
                    "ranked_predictions", "top_cosine",
                ])
                writer.writeheader()
                for row in predictions:
                    item = dict(row)
                    item["ranked_predictions"] = json.dumps(item["ranked_predictions"])
                    writer.writerow(item)
            jsonl_path = out_dir / f"{baseline}.{split}.predictions.jsonl"
            write_evaluator_predictions(jsonl_path, predictions)
            prediction_files[f"{baseline}.{split}"] = {
                "path": str(csv_path),
                "sha256": sha256(csv_path),
                "rows": len(predictions),
                "evaluator_jsonl": {
                    "path": str(jsonl_path),
                    "sha256": sha256(jsonl_path),
                    "rows": len(predictions),
                },
            }

    report = {
        "version": "t2.baselines.contextual.r3.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_version": "contextual_entity_market_v1",
        "recall_version": "r3.bge-m3.dt3.nofloor",
        "candidate_setting": "retrospective",
        "test_evaluation_status": "frozen_re-evaluation_after_train-label completion",
        "thresholds_frozen_before_test": True,
        "dense_none_threshold": dense_threshold,
        "high_precision_threshold": high_threshold,
        "train_binary_prior": {"counts": prior_counts, "link_probability": link_prior},
        "mrr_convention": "NONE RR=1 iff predicted NONE; linked gold uses reciprocal rank in the submitted candidate ranking",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": SEED,
            "unit": "post",
        },
        "candidate_recall": {split: candidate_recall(rows) for split, rows in splits.items()},
        "metrics": metrics,
        "inputs": {
            "release_manifest": {"path": str(gold_dir / "release_manifest.json"), "sha256": sha256(gold_dir / "release_manifest.json")},
            "thresholds": {"path": str(thresholds), "sha256": sha256(thresholds)},
            "train_labels": {"path": str(train_labels), "sha256": sha256(train_labels)},
            "val_labels": {"path": str(gold_dir / "val_labels.csv"), "sha256": sha256(gold_dir / "val_labels.csv")},
            "test_labels": {"path": str(gold_dir / "test_labels.csv"), "sha256": sha256(gold_dir / "test_labels.csv")},
        },
        "prediction_files": prediction_files,
        "evaluator_gold_files": gold_files,
    }
    report_path = out_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_path = out_dir / "BASELINE_SUMMARY.md"
    lines = [
        "# T2 contextual baseline results", "",
        f"Frozen dense NONE threshold: `{dense_threshold}`.  ",
        f"Frozen high-precision threshold: `{high_threshold}`.  ",
        "The test set was re-evaluated after the completed training-label merge; "
        "threshold selection remained validation-only.", "",
        "| Baseline | Split | Accuracy@1 | MRR | NONE F1 |", "|---|---:|---:|---:|---:|",
    ]
    for baseline in baseline_config:
        for split in ("val", "test"):
            row = metrics[baseline][split]
            lines.append(
                f"| `{baseline}` | {split} | {row['accuracy_at_1']:.4f} | "
                f"{row['mrr']:.4f} | {row['none_f1']:.4f} |"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "summary": str(summary_path),
        "dense_frozen": metrics["bge_top1_frozen_threshold"],
        "candidate_recall": report["candidate_recall"],
    }, indent=2))


if __name__ == "__main__":
    main()
