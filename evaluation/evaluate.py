#!/usr/bin/env python3
"""
EventXBench unified evaluation CLI.

Usage
-----
# Evaluate a single task against gold labels on disk:
python evaluation/evaluate.py --task t1 --predictions preds.jsonl --gold gold.jsonl

# Explicitly opt into the hosted snapshot (currently legacy/gated):
python evaluation/evaluate.py --task t1 --predictions preds.jsonl --hosted-gold

# Evaluate all tasks against a directory of t1_gold.jsonl ... t6_gold.jsonl:
python evaluation/evaluate.py --task all --predictions-dir results/ --gold-dir gold/

Results are printed as JSON to stdout and optionally written to --output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Keep the documented ``python evaluation/evaluate.py`` invocation working.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import (
    accuracy,
    cohen_kappa,
    derive_direction_magnitude,
    direction_accuracy,
    macro_f1,
    precision_at_k,
    quadratic_weighted_kappa,
    spearman_rho,
)

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

TASKS = ["t1", "t2", "t3", "t4", "t5", "t6"]
LEGACY_TASKS = ["t7"]

PREDICTION_FILE_NAMES = {
    "t1": "t1_predictions.jsonl",
    "t2": "t2_predictions.jsonl",
    "t3": "t3_predictions.jsonl",
    "t4": "t4_predictions.jsonl",
    "t5": "t5_predictions.jsonl",
    "t6": "t6_predictions.jsonl",
    "t7": "t7_predictions.jsonl",
}

GOLD_FILE_NAMES = {
    task: f"{task}_gold.jsonl"
    for task in TASKS
}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_gold(
    task: str,
    gold_path: Optional[str],
    allow_hosted: bool = False,
) -> List[Dict[str, Any]]:
    """Load gold labels from a local file or from HuggingFace."""
    if gold_path is not None:
        return _load_jsonl(gold_path)

    if not allow_hosted:
        raise ValueError(
            "Canonical July 2026 evaluation requires explicit frozen gold. "
            "Provide --gold (single task) or --gold-dir (all tasks). "
            "Use --hosted-gold only when intentionally evaluating the "
            "currently gated legacy hosted snapshot."
        )

    # Explicitly requested HuggingFace fallback via the eventxbench loader.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eventxbench import load_task  # type: ignore

        # T3's held-out human adjudication is a distinct ``gold`` split.
        # Requesting ``test`` would either fail on the v2 builder or silently
        # score the legacy silver export as if it were human ground truth.
        hosted_split = "gold" if task == "t3" else "test"
        ds = load_task(task, split=hosted_split)
        if hasattr(ds, "to_dict"):
            try:
                records = ds.to_dict("records")
            except TypeError:
                records = ds.to_dict()
            if isinstance(records, list):
                return [dict(row) for row in records]
        return [dict(row) for row in ds]
    except Exception as exc:
        print(
            f"ERROR: Could not load gold data for {task}. "
            f"Provide --gold or install eventxbench. ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_finite_number(value: Any) -> bool:
    if _is_missing(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_complete_alignment(
    task: str,
    prediction_keys: Iterable[str],
    gold_keys: Iterable[str],
) -> None:
    """Reject duplicate, unknown, or incomplete submissions before scoring."""
    prediction_keys = list(prediction_keys)
    gold_keys = list(gold_keys)
    duplicate_gold = sorted(
        key for key, count in Counter(gold_keys).items() if count > 1
    )
    if duplicate_gold:
        raise ValueError(
            f"Duplicate {task} gold identifiers: {duplicate_gold[:3]}"
        )
    duplicate_predictions = sorted(
        key
        for key, count in Counter(prediction_keys).items()
        if count > 1
    )
    if duplicate_predictions:
        raise ValueError(
            f"Duplicate {task} prediction identifiers: "
            f"{duplicate_predictions[:3]}"
        )

    prediction_set = set(prediction_keys)
    gold_set = set(gold_keys)
    unknown = sorted(prediction_set - gold_set)
    missing = sorted(gold_set - prediction_set)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"{len(missing)} missing (examples: {missing[:3]})")
        if unknown:
            details.append(f"{len(unknown)} unknown (examples: {unknown[:3]})")
        raise ValueError(
            f"Incomplete {task} prediction coverage: " + "; ".join(details)
        )


def _coverage_metadata(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    return {
        "n_gold": len(gold),
        "n_predictions": len(preds),
        "coverage": 1.0,
    }


# ------------------------------------------------------------------ #
#  Per-task evaluation                                                #
# ------------------------------------------------------------------ #

T1_LABELS = ["high_interest", "moderate_interest", "low_interest"]
T7_LABELS = ["transient", "sustained", "reversal"]
T6_LABELS = ["no_effect", "primary_only", "cross_market"]
T6_LEGACY_LABELS = [
    "no_cross_market_effect",
    "primary_mover",
    "propagated_signal",
]


def evaluate_t1(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    gold_keys = [str(g["condition_id"]) for g in gold]
    prediction_keys = [str(p["condition_id"]) for p in preds]
    _validate_complete_alignment("t1", prediction_keys, gold_keys)
    gold_map = {str(g["condition_id"]): g["interest_label"] for g in gold}
    y_true, y_pred, y_scores = [], [], []
    for p in preds:
        cid = str(p["condition_id"])
        # Support both nested (LLM) and flat (LightGBM) output formats
        inner = p.get("prediction", p)
        predicted_label = inner["label"]
        if predicted_label not in T1_LABELS:
            raise ValueError(
                f"Invalid T1 prediction for {cid}: {predicted_label!r}"
            )
        y_true.append(gold_map[cid])
        y_pred.append(predicted_label)
        y_scores.append(inner.get("scores", {}))
    n = len(y_true)
    k5 = min(5, n) if n >= 1 else 1
    has_scores = n > 0 and all(
        isinstance(score, dict) and "high_interest" in score
        for score in y_scores
    )
    p_at_5 = (
        round(precision_at_k(y_true, y_scores, k=k5), 4)
        if has_scores
        else None
    )
    p_at_10 = (
        round(precision_at_k(y_true, y_scores, k=min(10, n)), 4)
        if has_scores and n >= 10
        else None
    )
    result: Dict[str, Any] = {
        "task": "t1",
        "n": n,
        **_coverage_metadata(preds, gold),
        "macro_f1": round(macro_f1(y_true, y_pred, labels=T1_LABELS), 4),
        "accuracy": round(accuracy(y_true, y_pred), 4),
    }
    if p_at_5 is not None:
        result["precision_at_5"] = p_at_5
    if p_at_10 is not None:
        result["precision_at_10"] = p_at_10
    return result


def evaluate_t2(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    def gold_choice(row: dict) -> Optional[str]:
        for field in ("market_id", "gold", "final_choice", "gold_choice", "label"):
            if not _is_missing(row.get(field)):
                return str(row[field])
        return None

    def candidate_ids(row: dict) -> Optional[set[str]]:
        for field in (
            "candidate_ids",
            "ranked_ids",
            "ranked_market_ids",
        ):
            values = row.get(field)
            if isinstance(values, list):
                return {str(value) for value in values}
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            ids = set()
            for candidate in candidates:
                if isinstance(candidate, dict):
                    value = next(
                        (
                            candidate.get(field)
                            for field in ("condition_id", "market_id", "id")
                            if not _is_missing(candidate.get(field))
                        ),
                        None,
                    )
                else:
                    value = candidate
                if value is not None:
                    ids.add(str(value))
            return ids
        return None

    labeled_gold = [
        (str(row["tweet_id"]), choice, candidate_ids(row))
        for row in gold
        if (choice := gold_choice(row)) is not None
    ]
    gold_keys = [tweet_id for tweet_id, _, _ in labeled_gold]
    prediction_keys = [str(p["tweet_id"]) for p in preds]
    _validate_complete_alignment("t2", prediction_keys, gold_keys)
    gold_map = {
        tweet_id: (choice, allowed)
        for tweet_id, choice, allowed in labeled_gold
    }
    ranked_lists: list[list[str]] = []
    top_choices: list[str] = []
    gold_ids: list[str] = []
    for p in preds:
        tid = str(p["tweet_id"])
        raw_ranked = p.get("ranked_market_ids")
        if raw_ranked is None:
            raw_ranked = p.get("ranked_candidates")
        if raw_ranked is None:
            raw_ranked = p.get("ranked_options", [])
        if not isinstance(raw_ranked, list):
            raise ValueError(f"Invalid T2 ranking for {tid}: expected a list")
        normalized_ranking = [str(market_id) for market_id in raw_ranked]
        if len(normalized_ranking) != len(set(normalized_ranking)):
            raise ValueError(f"Duplicate T2 ranking entries for {tid}")
        ranked = [
            market_id
            for market_id in normalized_ranking
            if market_id != "NONE"
        ]
        top_choice = p.get("prediction")
        if top_choice is None:
            options = p.get("ranked_options", [])
            top_choice = options[0] if options else (ranked[0] if ranked else "NONE")
        top_choice = str(top_choice)
        gold_id, allowed = gold_map[tid]
        if allowed is not None:
            invalid = sorted(set(ranked) - allowed)
            if invalid:
                raise ValueError(
                    f"Unknown T2 candidate IDs for {tid}: {invalid[:3]}"
                )
            if top_choice != "NONE" and top_choice not in allowed:
                raise ValueError(
                    f"Invalid T2 top prediction for {tid}: {top_choice!r}"
                )
            if gold_id != "NONE" and gold_id not in allowed:
                raise ValueError(
                    f"T2 gold choice is outside candidates for {tid}: "
                    f"{gold_id!r}"
                )
        if top_choice != "NONE" and top_choice not in ranked:
            raise ValueError(
                f"T2 top prediction is absent from ranking for {tid}: "
                f"{top_choice!r}"
            )
        ranked_lists.append(ranked)
        top_choices.append(top_choice)
        gold_ids.append(gold_id)
    acc1 = accuracy(gold_ids, top_choices)
    reciprocal_ranks = []
    for ranked, predicted, gold_id in zip(ranked_lists, top_choices, gold_ids):
        if gold_id == "NONE":
            reciprocal_ranks.append(1.0 if predicted == "NONE" else 0.0)
        else:
            reciprocal_ranks.append(
                1.0 / (ranked.index(gold_id) + 1) if gold_id in ranked else 0.0
            )
    mean_rr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )
    return {
        "task": "t2",
        "n": len(ranked_lists),
        "n_gold": len(labeled_gold),
        "n_predictions": len(preds),
        "coverage": 1.0,
        "accuracy_at_1": round(acc1, 4),
        "acc_at_1": round(acc1, 4),
        "mrr": round(mean_rr, 4),
        "none_f1": round(macro_f1(gold_ids, top_choices, labels=["NONE"]), 4),
    }


def evaluate_t3(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    # Prefer the human-adjudicated `gold_grade` (the 2,687-instance audit pool)
    # when present; fall back to the silver `final_grade` otherwise. The two
    # are NOT interchangeable ground truth - silver only agrees with gold at
    # kappa_w=0.582 (T3_Reproducible_Package metrics.md, Phase 6), below the
    # project's own 0.6 reliability bar. `gold_field` in the result records
    # which one was actually used, so results scored against each aren't
    # silently conflated.
    gold_map: dict[str, int] = {}
    gold_field = "gold_grade" if gold and "gold_grade" in gold[0] else "final_grade"
    for g in gold:
        key = f"{g['tweet_id']}_{g['condition_id']}"
        field = "gold_grade" if "gold_grade" in g else "final_grade"
        gold_map[key] = int(g[field])

    y_true, y_pred = [], []
    for p in preds:
        key = f"{p['tweet_id']}_{p['condition_id']}"
        if key in gold_map:
            y_true.append(gold_map[key])
            y_pred.append(int(p["predicted_grade"]))

    num_classes = 6  # grades 0-5
    return {
        "task": "t3",
        "gold_field": gold_field,
        "n": len(y_true),
        "kappa_unweighted": round(cohen_kappa(y_true, y_pred, num_classes), 4),
        "kappa_weighted": round(quadratic_weighted_kappa(y_true, y_pred, num_classes), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "spearman_rho": round(spearman_rho(y_true, y_pred), 4),
    }


def evaluate_t4(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    def row_key(row: dict) -> str:
        if "bundle_day" in row:
            return f"{row['condition_id']}_{row['bundle_day']}"
        return str(row["tweet_id"])

    gold_keys = [row_key(g) for g in gold]
    prediction_keys = [row_key(p) for p in preds]
    _validate_complete_alignment("t4", prediction_keys, gold_keys)
    gold_map: dict[str, dict] = {}
    for g in gold:
        gold_map[row_key(g)] = g

    results: dict[str, Any] = {
        "task": "t4",
        **_coverage_metadata(preds, gold),
    }
    y_dir_true, y_dir_pred = [], []
    y_mag_true, y_mag_pred = [], []
    delta_pairs = {h: ([], []) for h in ("1d", "3d", "7d")}

    for p in preds:
        key = row_key(p)
        g = gold_map[key]

        if "direction_label" in g:
            pred_dir = p.get("direction_label") or p.get("direction")
            pred_mag = p.get("magnitude_bucket") or p.get("magnitude")
            gold_dir = g["direction_label"]
            gold_mag = g["magnitude_bucket"]
            if pred_dir not in {"up", "down", "flat"}:
                raise ValueError(
                    f"Invalid T4 direction prediction for {key}: {pred_dir!r}"
                )
            if pred_mag not in {"none", "small", "medium", "large"}:
                raise ValueError(
                    f"Invalid T4 magnitude prediction for {key}: {pred_mag!r}"
                )
        else:
            if p.get("delta_2h") is None:
                raise ValueError(f"Missing T4 delta_2h prediction for {key}")
            pred_dir, pred_mag = derive_direction_magnitude(float(p["delta_2h"]))
            gold_dir, gold_mag = derive_direction_magnitude(float(g["delta_2h"]))

        if pred_dir is not None:
            y_dir_true.append(gold_dir)
            y_dir_pred.append(pred_dir)
        if pred_mag is not None:
            y_mag_true.append(gold_mag)
            y_mag_pred.append(pred_mag)
        for horizon in ("1d", "3d", "7d"):
            column = f"delta_{horizon}"
            if _is_finite_number(g.get(column)):
                if p.get(column) is None:
                    raise ValueError(f"Missing T4 {column} prediction for {key}")
                gold_value = float(g[column])
                pred_value = float(p[column])
                if not math.isfinite(pred_value):
                    raise ValueError(
                        f"Non-finite T4 {column} prediction for {key}"
                    )
                delta_pairs[horizon][0].append(gold_value)
                delta_pairs[horizon][1].append(pred_value)

    results["n"] = len(y_dir_true)
    results["direction_accuracy"] = round(direction_accuracy(y_dir_true, y_dir_pred), 4)
    results["magnitude_macro_f1"] = round(
        macro_f1(
            y_mag_true,
            y_mag_pred,
            labels=["none", "small", "medium", "large"],
        ),
        4,
    )
    for horizon, (y_true, y_pred) in delta_pairs.items():
        results[f"n_delta_{horizon}"] = len(y_true)
        results[f"spearman_rho_delta_{horizon}"] = (
            round(spearman_rho(y_true, y_pred), 4) if len(y_true) >= 2 else None
        )

    return results


def evaluate_t5(preds: List[dict], gold: List[dict]) -> Dict[str, Any]:
    """T5: per-horizon drift/volume regression plus decay classification."""
    def row_key(row: dict) -> str:
        if "bundle_day" in row:
            return f"{row['condition_id']}_{row['bundle_day']}"
        return f"{row['tweet_id']}_{row['condition_id']}"

    gold_keys = [row_key(g) for g in gold]
    prediction_keys = [row_key(p) for p in preds]
    _validate_complete_alignment("t5", prediction_keys, gold_keys)
    gold_map: dict[str, dict] = {}
    for g in gold:
        gold_map[row_key(g)] = g

    drift_pairs = {h: ([], []) for h in ("1d", "3d", "7d")}
    volume_pairs = {h: ([], []) for h in ("1d", "3d", "7d")}
    dc_true, dc_pred = [], []
    for p in preds:
        key = row_key(p)
        g = gold_map[key]
        g_drift_json = g.get("price_impact_json") or {}
        p_drift_json = p.get("price_impact_json") or {}
        g_volume_json = g.get("volume_multiplier_json") or {}
        p_volume_json = p.get("volume_multiplier_json") or {}
        for horizon in ("1d", "3d", "7d"):
            g_drift = g.get(f"drift_magnitude_{horizon}", g_drift_json.get(horizon))
            p_drift = p.get(f"drift_magnitude_{horizon}", p_drift_json.get(horizon))
            if _is_finite_number(g_drift):
                if p_drift is None:
                    raise ValueError(
                        f"Missing T5 drift_magnitude_{horizon} prediction for {key}"
                    )
                gold_value = float(g_drift)
                pred_value = float(p_drift)
                if not math.isfinite(pred_value):
                    raise ValueError(
                        f"Non-finite T5 drift_magnitude_{horizon} prediction "
                        f"for {key}"
                    )
                drift_pairs[horizon][0].append(gold_value)
                drift_pairs[horizon][1].append(pred_value)
            g_volume = g.get(
                f"volume_multiplier_{horizon}", g_volume_json.get(horizon)
            )
            p_volume = p.get(
                f"volume_multiplier_{horizon}", p_volume_json.get(horizon)
            )
            if _is_finite_number(g_volume):
                if p_volume is None:
                    raise ValueError(
                        f"Missing T5 volume_multiplier_{horizon} prediction "
                        f"for {key}"
                    )
                gold_value = float(g_volume)
                pred_value = float(p_volume)
                if not math.isfinite(pred_value):
                    raise ValueError(
                        f"Non-finite T5 volume_multiplier_{horizon} prediction "
                        f"for {key}"
                    )
                volume_pairs[horizon][0].append(gold_value)
                volume_pairs[horizon][1].append(pred_value)
        p_dc = p.get("decay_class") or p.get("label")
        g_dc = g.get("decay_class")
        if not _is_missing(g_dc):
            if p_dc not in T7_LABELS:
                raise ValueError(
                    f"Invalid T5 decay prediction for {key}: {p_dc!r}"
                )
            dc_true.append(g_dc)
            dc_pred.append(p_dc)

    result: Dict[str, Any] = {
        "task": "t5",
        "n": len(gold),
        **_coverage_metadata(preds, gold),
    }
    for horizon in ("1d", "3d", "7d"):
        drift_true, drift_pred = drift_pairs[horizon]
        volume_true, volume_pred = volume_pairs[horizon]
        result[f"n_drift_{horizon}"] = len(drift_true)
        result[f"spearman_rho_drift_{horizon}"] = (
            round(spearman_rho(drift_true, drift_pred), 4)
            if len(drift_true) >= 2
            else None
        )
        result[f"n_volume_multiplier_{horizon}"] = len(volume_true)
        result[f"spearman_rho_volume_multiplier_{horizon}"] = (
            round(spearman_rho(volume_true, volume_pred), 4)
            if len(volume_true) >= 2
            else None
        )
    result["n_decay"] = len(dc_true)
    result["decay_macro_f1"] = (
        round(macro_f1(dc_true, dc_pred, labels=T7_LABELS), 4)
        if dc_true
        else None
    )
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
    """Evaluate the canonical v2 market-day/horizon format or legacy v1 rows."""

    is_v2 = any("headline_label" in row for row in gold)
    if not is_v2:
        gold_keys = [str(g["tweet_id"]) for g in gold]
        prediction_keys = [str(p["tweet_id"]) for p in preds]
        _validate_complete_alignment("t6 legacy_v1", prediction_keys, gold_keys)
        gold_map = {str(g["tweet_id"]): g["label"] for g in gold}
        y_true, y_pred = [], []
        for p in preds:
            tid = str(p["tweet_id"])
            predicted_label = p["label"]
            if predicted_label not in T6_LEGACY_LABELS:
                raise ValueError(
                    f"Invalid T6 legacy_v1 prediction for {tid}: "
                    f"{predicted_label!r}"
                )
            y_true.append(gold_map[tid])
            y_pred.append(predicted_label)
        return {
            "task": "t6",
            "protocol": "legacy_v1",
            "n": len(y_true),
            **_coverage_metadata(preds, gold),
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "macro_f1": round(
                macro_f1(y_true, y_pred, labels=T6_LEGACY_LABELS), 4
            ),
        }

    def composite_key(row: dict) -> str:
        return "|".join(
            (
                str(row["condition_id"]),
                str(row["bundle_day"]),
                str(int(row["horizon_days"])),
            )
        )

    def instance_keys(row: dict) -> list[str]:
        keys = []
        if row.get("instance_id"):
            keys.append(str(row["instance_id"]))
        if all(
            field in row
            for field in ("condition_id", "bundle_day", "horizon_days")
        ):
            keys.append(composite_key(row))
        return keys

    canonical_gold_keys = [composite_key(row) for row in gold]
    _validate_complete_alignment(
        "t6 gold",
        canonical_gold_keys,
        canonical_gold_keys,
    )
    canonical_gold_map = {
        composite_key(row): row
        for row in gold
    }
    gold_aliases: dict[str, str] = {}
    for row in gold:
        canonical = composite_key(row)
        for alias in instance_keys(row):
            previous = gold_aliases.get(alias)
            if previous is not None and previous != canonical:
                raise ValueError(f"Ambiguous T6 gold identifier: {alias}")
            gold_aliases[alias] = canonical

    resolved_prediction_keys: list[str] = []
    for prediction in preds:
        aliases = instance_keys(prediction)
        resolved = {
            gold_aliases[alias]
            for alias in aliases
            if alias in gold_aliases
        }
        if len(resolved) > 1:
            raise ValueError(
                f"Conflicting T6 prediction identifiers: {aliases}"
            )
        resolved_prediction_keys.append(
            next(iter(resolved), f"<unknown:{'|'.join(aliases)}>")
        )
    _validate_complete_alignment(
        "t6",
        resolved_prediction_keys,
        canonical_gold_keys,
    )

    aligned: list[tuple[dict, dict, str]] = []
    for prediction, canonical in zip(preds, resolved_prediction_keys):
        gold_row = canonical_gold_map[canonical]
        predicted_label = (
            prediction.get("prediction")
            or prediction.get("pred_label")
            or prediction.get("label")
        )
        if predicted_label not in T6_LABELS:
            raise ValueError(
                "Invalid T6 v2 prediction for "
                f"{instance_keys(prediction)[0]}: {predicted_label!r}"
            )
        aligned.append((gold_row, prediction, str(predicted_label)))

    y_true = [str(gold_row["headline_label"]) for gold_row, _, _ in aligned]
    y_pred = [predicted_label for _, _, predicted_label in aligned]
    result: Dict[str, Any] = {
        "task": "t6",
        "protocol": "t6.kdd.v2",
        "n": len(aligned),
        **_coverage_metadata(preds, gold),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, labels=T6_LABELS), 4),
        "by_horizon": {},
    }
    for horizon in (1, 3, 7):
        horizon_rows = [
            (gold_row, predicted_label)
            for gold_row, _, predicted_label in aligned
            if int(gold_row["horizon_days"]) == horizon
        ]
        horizon_true = [
            str(gold_row["headline_label"]) for gold_row, _ in horizon_rows
        ]
        horizon_pred = [predicted_label for _, predicted_label in horizon_rows]
        result["by_horizon"][str(horizon)] = {
            "n": len(horizon_rows),
            "accuracy": round(accuracy(horizon_true, horizon_pred), 4),
            "macro_f1": round(
                macro_f1(horizon_true, horizon_pred, labels=T6_LABELS), 4
            ),
        }
    return result


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
        choices=TASKS + LEGACY_TASKS + ["all"],
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
        help="Path to frozen gold JSONL (single-task mode).",
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default=None,
        help=(
            "Directory containing t1_gold.jsonl ... t6_gold.jsonl "
            "(all-task mode)."
        ),
    )
    parser.add_argument(
        "--hosted-gold",
        action="store_true",
        help=(
            "Explicitly use the gated hosted snapshot. It currently exposes "
            "legacy schemas and is not the canonical July 2026 v2 source."
        ),
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
        if args.gold_dir is None and not args.hosted_gold:
            parser.error(
                "--gold-dir is required when --task all unless "
                "--hosted-gold is explicitly selected"
            )
        tasks_to_run = TASKS
        missing_files: list[str] = []
        for task in tasks_to_run:
            prediction_file = os.path.join(
                args.predictions_dir, PREDICTION_FILE_NAMES[task]
            )
            if not os.path.isfile(prediction_file):
                missing_files.append(prediction_file)
            if args.gold_dir is not None:
                gold_file = os.path.join(args.gold_dir, GOLD_FILE_NAMES[task])
                if not os.path.isfile(gold_file):
                    missing_files.append(gold_file)
        if missing_files:
            formatted = "\n".join(f"  - {path}" for path in missing_files)
            parser.error(
                "Missing required files for --task all:\n"
                f"{formatted}"
            )
    else:
        if args.predictions is None:
            parser.error("--predictions is required for single-task evaluation")
        if args.gold is None and not args.hosted_gold:
            parser.error(
                "--gold is required for canonical evaluation unless "
                "--hosted-gold is explicitly selected"
            )
        tasks_to_run = [args.task]

    all_results: list[dict] = []

    for task in tasks_to_run:
        # Load predictions
        if args.task == "all":
            pred_path = os.path.join(args.predictions_dir, PREDICTION_FILE_NAMES[task])
        else:
            pred_path = args.predictions

        preds = _load_jsonl(pred_path)

        # Load gold
        gold_path = args.gold
        if args.task == "all" and args.gold_dir is not None:
            gold_path = os.path.join(args.gold_dir, GOLD_FILE_NAMES[task])
        gold = _load_gold(task, gold_path, allow_hosted=args.hosted_gold)

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
