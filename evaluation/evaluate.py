#!/usr/bin/env python3
"""
EventXBench unified evaluation CLI.

Usage
-----
# Evaluate a single task against gold labels on disk:
python evaluation/evaluate.py --task t1 --predictions preds.jsonl --gold gold.jsonl

# Evaluate using HuggingFace-hosted gold labels (auto-downloaded):
python evaluation/evaluate.py --task t1 --predictions preds.jsonl

# Evaluate all tasks at once (expects one predictions file per task):
python evaluation/evaluate.py --task all --predictions-dir results/

Results are printed as JSON to stdout and optionally written to --output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.metrics import (
    accuracy,
    derive_direction_magnitude,
    direction_accuracy,
    macro_f1,
    mrr,
    quadratic_weighted_kappa,
    spearman_rho,
)

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

TASKS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]

PREDICTION_FILE_NAMES = {
    "t1": "t1_predictions.jsonl",
    "t2": "t2_predictions.jsonl",
    "t3": "t3_predictions.jsonl",
    "t4": "t4_predictions.jsonl",
    "t5": "t5_predictions.jsonl",
    "t6": "t6_predictions.jsonl",
    "t7": "t7_predictions.jsonl",
}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_gold(task: str, gold_path: Optional[str]) -> List[Dict[str, Any]]:
    """Load gold labels from a local file or from HuggingFace."""
    if gold_path is not None:
        return _load_jsonl(gold_path)

    # Fall back to HuggingFace via eventxbench loader
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eventxbench import load_task  # type: ignore

        ds = load_task(task, split="test")
        return [dict(row) for row in ds]
    except Exception as exc:
        print(
            f"ERROR: Could not load gold data for {task}. "
            f"Provide --gold or install eventxbench. ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)


# ------------------------------------------------------------------ #
#  Per-task evaluation                                                #
# ------------------------------------------------------------------ #

T1_LABELS = ["high_interest", "moderate_interest", "low_interest"]
T7_LABELS = ["transient", "sustained", "reversal"]
T6_LABELS = ["no_cross_market_effect", "primary_mover", "propagated_signal"]


def evaluate_t1(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    gold_map = {str(g["condition_id"]): g["interest_label"] for g in gold}
    y_true, y_pred = [], []
    for p in preds:
        cid = str(p["condition_id"])
        if cid in gold_map:
            y_true.append(gold_map[cid])
            y_pred.append(p["label"])
    return {
        "task": "t1",
        "n": len(y_true),
        "macro_f1": round(macro_f1(y_true, y_pred, labels=T1_LABELS), 4),
        "accuracy": round(accuracy(y_true, y_pred), 4),
    }


def evaluate_t2(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    gold_map = {str(g["tweet_id"]): str(g["market_id"]) for g in gold}
    ranked_lists: list[list] = []
    gold_ids: list[str] = []
    for p in preds:
        tid = str(p["tweet_id"])
        if tid in gold_map:
            ranked_lists.append([str(m) for m in p["ranked_market_ids"]])
            gold_ids.append(gold_map[tid])
    acc1 = sum(
        1 for rl, gi in zip(ranked_lists, gold_ids) if len(rl) > 0 and rl[0] == gi
    ) / max(len(ranked_lists), 1)
    return {
        "task": "t2",
        "n": len(ranked_lists),
        "acc_at_1": round(acc1, 4),
        "mrr": round(mrr(ranked_lists, gold_ids), 4),
    }


def evaluate_t3(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    # Build key -> grade map from gold
    gold_map: dict[str, int] = {}
    for g in gold:
        key = f"{g['tweet_id']}_{g['condition_id']}"
        gold_map[key] = int(g["final_grade"])

    y_true, y_pred = [], []
    for p in preds:
        key = f"{p['tweet_id']}_{p['condition_id']}"
        if key in gold_map:
            y_true.append(gold_map[key])
            y_pred.append(int(p["predicted_grade"]))

    num_classes = 6  # grades 0-5
    return {
        "task": "t3",
        "n": len(y_true),
        "spearman_rho": round(spearman_rho(y_true, y_pred), 4),
        "qwk": round(quadratic_weighted_kappa(y_true, y_pred, num_classes), 4),
    }


def evaluate_t4(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    gold_map: dict[str, dict] = {}
    for g in gold:
        gold_map[str(g["tweet_id"])] = g

    results: dict[str, Any] = {"task": "t4"}

    # Collect aligned pairs
    y_dir_true, y_dir_pred = [], []
    y_mag_true, y_mag_pred = [], []
    y_delta_true, y_delta_pred = [], []

    for p in preds:
        tid = str(p["tweet_id"])
        if tid not in gold_map:
            continue
        g = gold_map[tid]

        pred_dir, pred_mag = derive_direction_magnitude(float(p["delta_2h"]))
        gold_dir, gold_mag = derive_direction_magnitude(float(g["delta_2h"]))

        y_dir_true.append(gold_dir)
        y_dir_pred.append(pred_dir)
        y_mag_true.append(gold_mag)
        y_mag_pred.append(pred_mag)
        y_delta_true.append(float(g["delta_2h"]))
        y_delta_pred.append(float(p["delta_2h"]))

    results["n"] = len(y_dir_true)
    results["direction_accuracy"] = round(direction_accuracy(y_dir_true, y_dir_pred), 4)
    results["magnitude_macro_f1"] = round(
        macro_f1(y_mag_true, y_mag_pred, labels=["small", "medium", "large"]), 4
    )
    results["spearman_rho"] = round(spearman_rho(y_delta_true, y_delta_pred), 4)

    return results


def evaluate_t5(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    """T5: Continuous prediction of price_impact and volume_multiplier."""
    gold_map: dict[str, dict] = {}
    for g in gold:
        key = f"{g['tweet_id']}_{g['condition_id']}"
        gold_map[key] = g

    pi_true, pi_pred = [], []
    vm_true, vm_pred = [], []
    for p in preds:
        key = f"{p['tweet_id']}_{p['condition_id']}"
        if key not in gold_map:
            continue
        g = gold_map[key]
        # price_impact: max absolute deviation from p0 (use 2h horizon as default)
        g_pi = g.get("price_impact_json", {})
        p_pi = p.get("price_impact")
        if p_pi is not None and g_pi:
            pi_true.append(float(g_pi.get("2h", 0)))
            pi_pred.append(float(p_pi))
        # volume_multiplier
        g_vm = g.get("volume_multiplier_json", {})
        p_vm = p.get("volume_multiplier")
        if p_vm is not None and g_vm:
            vm_true.append(float(g_vm.get("2h", 0)))
            vm_pred.append(float(p_vm))

    result: Dict[str, Any] = {"task": "t5", "n_price_impact": len(pi_true), "n_volume_multiplier": len(vm_true)}
    result["spearman_rho_price_impact"] = round(spearman_rho(pi_true, pi_pred), 4) if len(pi_true) >= 2 else None
    result["spearman_rho_volume_multiplier"] = round(spearman_rho(vm_true, vm_pred), 4) if len(vm_true) >= 2 else None
    return result


def evaluate_t7(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    """T7: Decay classification (transient/sustained/reversal)."""
    gold_map: dict[str, str] = {}
    for g in gold:
        key = f"{g['tweet_id']}_{g['condition_id']}"
        gold_map[key] = g["decay_class"]

    y_true, y_pred = [], []
    for p in preds:
        key = f"{p['tweet_id']}_{p['condition_id']}"
        if key in gold_map:
            y_true.append(gold_map[key])
            y_pred.append(p["label"])

    return {
        "task": "t7",
        "n": len(y_true),
        "macro_f1": round(macro_f1(y_true, y_pred, labels=T7_LABELS), 4),
    }


def evaluate_t6(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    gold_map: dict[str, str] = {}
    for g in gold:
        gold_map[str(g["tweet_id"])] = g["label"]

    y_true, y_pred = [], []
    for p in preds:
        tid = str(p["tweet_id"])
        if tid in gold_map:
            y_true.append(gold_map[tid])
            y_pred.append(p["label"])

    return {
        "task": "t6",
        "n": len(y_true),
        "macro_f1": round(macro_f1(y_true, y_pred, labels=T6_LABELS), 4),
    }


EVALUATORS = {
    "t1": evaluate_t1,
    "t2": evaluate_t2,
    "t3": evaluate_t3,
    "t4": evaluate_t4,
    "t5": evaluate_t5,
    "t6": evaluate_t6,
    "t7": evaluate_t7,
}


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EventXBench evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=TASKS + ["all"],
        help="Task to evaluate (t1-t6) or 'all'.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions JSONL (single-task mode).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory containing per-task prediction files (all-task mode).",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold JSONL. If omitted, loads from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    # Determine which tasks to run
    if args.task == "all":
        if args.predictions_dir is None:
            parser.error("--predictions-dir is required when --task all")
        tasks_to_run = TASKS
    else:
        if args.predictions is None:
            parser.error("--predictions is required for single-task evaluation")
        tasks_to_run = [args.task]

    all_results: list[dict] = []

    for task in tasks_to_run:
        # Load predictions
        if args.task == "all":
            pred_path = os.path.join(args.predictions_dir, PREDICTION_FILE_NAMES[task])
            if not os.path.exists(pred_path):
                print(f"SKIP {task}: {pred_path} not found", file=sys.stderr)
                continue
        else:
            pred_path = args.predictions

        preds = _load_jsonl(pred_path)

        # Load gold
        gold_path = args.gold if args.task != "all" else None
        gold = _load_gold(task, gold_path)

        # Evaluate
        result = EVALUATORS[task](preds, gold)
        all_results.append(result)

    # Output
    output = all_results if len(all_results) > 1 else (all_results[0] if all_results else {})
    output_str = json.dumps(output, indent=2)
    print(output_str)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_str + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
