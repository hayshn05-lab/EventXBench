"""Regression tests for the July 2026 non-T3 benchmark refresh."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.t2 import contextual_baselines
from evaluation.evaluate import (
    _load_gold,
    evaluate_t1,
    evaluate_t2,
    evaluate_t3,
    evaluate_t4,
    evaluate_t5,
    evaluate_t6,
)
from eventxbench.loader import load_task


class EvaluationAlignmentTests(unittest.TestCase):
    def test_t1_accepts_nested_predictions_and_reports_precision_at_k(self) -> None:
        gold = [
            {"condition_id": "a", "interest_label": "high_interest"},
            {"condition_id": "b", "interest_label": "moderate_interest"},
        ]
        predictions = [
            {
                "condition_id": "a",
                "prediction": {
                    "label": "high_interest",
                    "scores": {"high_interest": 0.9},
                },
            },
            {
                "condition_id": "b",
                "prediction": {
                    "label": "moderate_interest",
                    "scores": {"high_interest": 0.1},
                },
            },
        ]

        result = evaluate_t1(predictions, gold)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["precision_at_5"], 0.5)

    def test_t1_omits_ranking_metrics_without_scores(self) -> None:
        result = evaluate_t1(
            [{"condition_id": "a", "label": "high_interest"}],
            [{"condition_id": "a", "interest_label": "high_interest"}],
        )

        self.assertNotIn("precision_at_5", result)
        self.assertNotIn("precision_at_10", result)

    def test_t1_rejects_incomplete_or_duplicate_predictions(self) -> None:
        gold = [
            {"condition_id": "a", "interest_label": "high_interest"},
            {"condition_id": "b", "interest_label": "low_interest"},
        ]

        with self.assertRaisesRegex(ValueError, "Incomplete t1 prediction coverage"):
            evaluate_t1(
                [{"condition_id": "a", "label": "high_interest"}],
                gold,
            )
        with self.assertRaisesRegex(ValueError, "Duplicate t1 prediction"):
            evaluate_t1(
                [
                    {"condition_id": "a", "label": "high_interest"},
                    {"condition_id": "a", "label": "high_interest"},
                ],
                gold,
            )

    def test_t4_uses_market_day_keys_and_daily_horizons(self) -> None:
        gold = [
            {
                "instance_id": "a-1",
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "direction_label": "up",
                "magnitude_bucket": "large",
                "delta_1d": 0.1,
                "delta_3d": 0.2,
                "delta_7d": 0.3,
            },
            {
                "condition_id": "b",
                "bundle_day": "2026-04-02",
                "direction_label": "down",
                "magnitude_bucket": "medium",
                "delta_1d": -0.1,
                "delta_3d": -0.2,
                "delta_7d": -0.3,
            },
        ]
        predictions = [dict(row) for row in gold]

        result = evaluate_t4(predictions, gold)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["direction_accuracy"], 1.0)
        self.assertEqual(result["spearman_rho_delta_1d"], 1.0)
        self.assertEqual(result["n_delta_7d"], 2)

    def test_t4_treats_nan_gold_targets_as_unavailable(self) -> None:
        gold = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "direction_label": "flat",
                "magnitude_bucket": "none",
                "delta_1d": 0.0,
                "delta_3d": float("nan"),
                "delta_7d": None,
            }
        ]
        predictions = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "direction_label": "flat",
                "magnitude_bucket": "none",
                "delta_1d": 0.0,
            }
        ]

        result = evaluate_t4(predictions, gold)

        self.assertEqual(result["n_delta_1d"], 1)
        self.assertEqual(result["n_delta_3d"], 0)
        self.assertEqual(result["n_delta_7d"], 0)

    def test_t5_reports_daily_continuous_and_decay_metrics(self) -> None:
        gold = [
            {
                "instance_id": "a-3",
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "drift_magnitude_1d": 0.1,
                "drift_magnitude_3d": 0.2,
                "drift_magnitude_7d": 0.3,
                "volume_multiplier_1d": 1.1,
                "volume_multiplier_3d": 1.2,
                "volume_multiplier_7d": 1.3,
                "decay_class": "transient",
            },
            {
                "condition_id": "b",
                "bundle_day": "2026-04-02",
                "drift_magnitude_1d": 0.2,
                "drift_magnitude_3d": 0.3,
                "drift_magnitude_7d": 0.4,
                "volume_multiplier_1d": 1.2,
                "volume_multiplier_3d": 1.3,
                "volume_multiplier_7d": 1.4,
                "decay_class": "sustained",
            },
        ]
        predictions = [dict(row) for row in gold]

        result = evaluate_t5(predictions, gold)

        self.assertEqual(result["n_drift_1d"], 2)
        self.assertEqual(result["spearman_rho_drift_7d"], 1.0)
        self.assertEqual(result["n_decay"], 2)

    def test_t5_requires_each_available_target(self) -> None:
        gold = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "drift_magnitude_1d": 0.1,
                "volume_multiplier_1d": 1.1,
            }
        ]
        predictions = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "volume_multiplier_1d": 1.1,
            }
        ]

        with self.assertRaisesRegex(ValueError, "Missing T5 drift_magnitude_1d"):
            evaluate_t5(predictions, gold)

    def test_t5_treats_nan_targets_and_decay_as_unavailable(self) -> None:
        gold = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "drift_magnitude_1d": 0.1,
                "drift_magnitude_3d": float("nan"),
                "volume_multiplier_1d": 1.1,
                "decay_class": float("nan"),
            }
        ]
        predictions = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "drift_magnitude_1d": 0.1,
                "volume_multiplier_1d": 1.1,
            }
        ]

        result = evaluate_t5(predictions, gold)

        self.assertEqual(result["n_drift_1d"], 1)
        self.assertEqual(result["n_drift_3d"], 0)
        self.assertEqual(result["n_decay"], 0)

    def test_t6_uses_v2_instance_key_and_headline_labels(self) -> None:
        gold = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "horizon_days": 1,
                "headline_label": "cross_market",
            },
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "horizon_days": 3,
                "headline_label": "no_effect",
            },
        ]
        predictions = [
            {
                "condition_id": row["condition_id"],
                "bundle_day": row["bundle_day"],
                "horizon_days": row["horizon_days"],
                "prediction": row["headline_label"],
            }
            for row in gold
        ]

        result = evaluate_t6(predictions, gold)

        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["by_horizon"]["1"]["n"], 1)

    def test_t6_rejects_invalid_headline_predictions(self) -> None:
        gold = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "horizon_days": 1,
                "headline_label": "no_effect",
            }
        ]
        predictions = [
            {
                "condition_id": "a",
                "bundle_day": "2026-04-01",
                "horizon_days": 1,
                "prediction": "legacy_propagated_signal",
            }
        ]

        with self.assertRaisesRegex(ValueError, "Invalid T6 v2 prediction"):
            evaluate_t6(predictions, gold)

    def test_t3_behavior_is_preserved(self) -> None:
        gold = [{"tweet_id": "1", "condition_id": "a", "final_grade": 4}]
        predictions = [
            {"tweet_id": "1", "condition_id": "a", "predicted_grade": 4}
        ]

        result = evaluate_t3(predictions, gold)

        self.assertEqual(result["task"], "t3")
        self.assertEqual(result["n"], 1)

    def test_t2_accepts_contextual_runner_output_and_none_convention(self) -> None:
        gold = [
            {
                "tweet_id": "1",
                "gold": "NONE",
                "candidate_ids": ["market-a"],
            },
            {
                "tweet_id": "2",
                "gold": "market-b",
                "candidate_ids": ["market-b"],
            },
        ]
        predictions = [
            {
                "tweet_id": "1",
                "prediction": "NONE",
                "ranked_options": ["NONE", "market-a"],
                "ranked_candidates": ["market-a"],
            },
            {
                "tweet_id": "2",
                "prediction": "market-b",
                "ranked_options": ["market-b", "NONE"],
                "ranked_candidates": ["market-b"],
            },
        ]

        result = evaluate_t2(predictions, gold)

        self.assertEqual(result["accuracy_at_1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["none_f1"], 1.0)

    def test_t2_rejects_duplicate_or_out_of_domain_rankings(self) -> None:
        gold = [
            {
                "tweet_id": "1",
                "gold": "market-a",
                "candidate_ids": ["market-a", "market-b"],
            }
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate T2 ranking"):
            evaluate_t2(
                [
                    {
                        "tweet_id": "1",
                        "prediction": "market-a",
                        "ranked_candidates": ["market-a", "market-a"],
                    }
                ],
                gold,
            )
        with self.assertRaisesRegex(ValueError, "Unknown T2 candidate"):
            evaluate_t2(
                [
                    {
                        "tweet_id": "1",
                        "prediction": "market-a",
                        "ranked_candidates": ["market-a", "not-a-candidate"],
                    }
                ],
                gold,
            )

    def test_t2_removes_none_from_every_linked_ranking_schema(self) -> None:
        result = evaluate_t2(
            [
                {
                    "tweet_id": "1",
                    "prediction": "market-b",
                    "ranked_market_ids": ["NONE", "market-b"],
                }
            ],
            [
                {
                    "tweet_id": "1",
                    "gold": "market-b",
                    "candidate_ids": ["market-b"],
                }
            ],
        )

        self.assertEqual(result["mrr"], 1.0)

    def test_t2_contextual_jsonl_round_trips_through_evaluator(self) -> None:
        predictions = [
            {
                "instance_id": "instance-none",
                "tweet_id": "1",
                "gold": "NONE",
                "prediction": "NONE",
                "ranked_predictions": ["market-a", "market-b"],
                "top_cosine": 0.4,
            },
            {
                "instance_id": "instance-linked",
                "tweet_id": "2",
                "gold": "market-b",
                "prediction": "market-b",
                "ranked_predictions": ["market-b", "market-a"],
                "top_cosine": 0.9,
            },
        ]
        expected_predictions = [
            {
                "instance_id": "instance-none",
                "tweet_id": "1",
                "prediction": "NONE",
                "ranked_options": ["NONE", "market-a", "market-b"],
                "ranked_candidates": ["market-a", "market-b"],
            },
            {
                "instance_id": "instance-linked",
                "tweet_id": "2",
                "prediction": "market-b",
                "ranked_options": ["market-b", "market-a", "NONE"],
                "ranked_candidates": ["market-b", "market-a"],
            },
        ]
        expected_gold = [
            {
                "instance_id": "instance-none",
                "tweet_id": "1",
                "gold": "NONE",
                "candidate_ids": ["market-a", "market-b"],
            },
            {
                "instance_id": "instance-linked",
                "tweet_id": "2",
                "gold": "market-b",
                "candidate_ids": ["market-b", "market-a"],
            },
        ]

        self.assertTrue(
            hasattr(contextual_baselines, "write_evaluator_predictions"),
            "contextual runner must expose an evaluator JSONL writer",
        )
        self.assertTrue(
            hasattr(contextual_baselines, "write_evaluator_gold"),
            "contextual runner must expose an evaluator gold JSONL writer",
        )
        with tempfile.TemporaryDirectory() as tmp:
            prediction_path = Path(tmp) / "predictions.jsonl"
            gold_path = Path(tmp) / "gold.jsonl"
            contextual_baselines.write_evaluator_predictions(
                prediction_path, predictions
            )
            contextual_baselines.write_evaluator_gold(gold_path, predictions)
            written_predictions = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
            ]
            written_gold = [
                json.loads(line)
                for line in gold_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(written_predictions, expected_predictions)
        self.assertEqual(written_gold, expected_gold)
        result = evaluate_t2(written_predictions, written_gold)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy_at_1"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["none_f1"], 1.0)

    def test_t2_contextual_cli_requires_explicit_artifact_root(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "baselines.t2.contextual_baselines",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data-dir", result.stdout)
        self.assertIn("--output-dir", result.stdout)

    def test_all_task_cli_rejects_incomplete_file_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions_dir = root / "predictions"
            gold_dir = root / "gold"
            predictions_dir.mkdir()
            gold_dir.mkdir()

            result = subprocess.run(
                [
                    sys.executable,
                    "evaluation/evaluate.py",
                    "--task",
                    "all",
                    "--predictions-dir",
                    str(predictions_dir),
                    "--gold-dir",
                    str(gold_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required files for --task all", result.stderr)
        self.assertIn("t1_predictions.jsonl", result.stderr)
        self.assertIn("t1_gold.jsonl", result.stderr)


class LoaderAlignmentTests(unittest.TestCase):
    def test_hosted_gold_loader_converts_dataframe_records(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            [{"condition_id": "a", "interest_label": "high_interest"}]
        )
        with patch("eventxbench.load_task", return_value=frame):
            records = _load_gold("t1", None, allow_hosted=True)

        self.assertEqual(
            records,
            [{"condition_id": "a", "interest_label": "high_interest"}],
        )

    def test_direct_kdd_directory_supports_validation_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"condition_id": "a", "split": "validation"}
            (root / "validation.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

            loaded = load_task("t4", local_dir=str(root), split="val")

        self.assertEqual(loaded.to_dict("records"), [row])


class GroupSafeCvTests(unittest.TestCase):
    def test_grouped_cv_is_disjoint_and_preserves_all_classes(self) -> None:
        try:
            import numpy as np
            import pandas as pd
            from baselines.t4.lightgbm_baseline import (
                _group_values,
                _make_stratified_group_cv,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional ML dependency unavailable: {exc}")

        rows = []
        labels = []
        for label in range(3):
            for group_index in range(3):
                for condition_index in range(2):
                    rows.append(
                        {
                            "event_cluster_id": f"cluster-{label}-{group_index}",
                            "condition_id": (
                                f"condition-{label}-{group_index}-{condition_index}"
                            ),
                        }
                    )
                    labels.append(label)
        frame = pd.DataFrame(rows)
        groups = _group_values(frame)
        y = np.asarray(labels)
        cv = _make_stratified_group_cv(y, groups, seed=42, max_splits=5)
        splits = list(cv.split(np.zeros(len(y)), y, groups))

        self.assertEqual(cv.n_splits, 3)
        for train_indices, validation_indices in splits:
            self.assertTrue(
                set(groups[train_indices]).isdisjoint(
                    set(groups[validation_indices])
                )
            )
            self.assertEqual(set(y[train_indices]), {0, 1, 2})
            self.assertEqual(set(y[validation_indices]), {0, 1, 2})

    def test_group_fallback_is_row_wise_and_namespaced(self) -> None:
        try:
            import pandas as pd
            from baselines.t4.lightgbm_baseline import _group_values
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional ML dependency unavailable: {exc}")

        frame = pd.DataFrame(
            [
                {
                    "event_cluster_id": "shared-event",
                    "condition_id": "condition-a",
                },
                {
                    "event_cluster_id": "shared-event",
                    "condition_id": "condition-b",
                },
                {
                    "event_cluster_id": None,
                    "condition_id": "condition-c",
                },
            ]
        )

        self.assertEqual(
            _group_values(frame).tolist(),
            [
                "event:shared-event",
                "event:shared-event",
                "condition:condition-c",
            ],
        )


if __name__ == "__main__":
    unittest.main()
