# EventX Evaluation

Unified evaluation module for the six canonical EventX tasks. The legacy `t7`
decay-only alias remains accepted for backward compatibility.

## Quick Start

```bash
# Single task
python evaluation/evaluate.py --task t1 --predictions preds.jsonl --gold gold.jsonl

# All tasks at once; gold/ contains t1_gold.jsonl ... t6_gold.jsonl
python evaluation/evaluate.py --task all \
  --predictions-dir results/ --gold-dir gold/
```

Results are printed as JSON to stdout. Use `--output results.json` to save.
The evaluator requires explicit frozen gold for canonical July 2026 results.
`--hosted-gold` is an intentional legacy opt-in only: the current hosted
snapshot is gated and has not yet migrated to the v2 T4--T6 schemas.

## Submission integrity checks

For canonical T1, T2, T4, T5, and T6 evaluation, a submission must contain
exactly one prediction for every labeled gold row. Duplicate identifiers,
unknown identifiers, missing rows, invalid class labels, non-finite values, and
missing predictions for an available T4/T5 target raise an error instead of
being silently skipped. Successful results include `n_gold`,
`n_predictions`, and `coverage` (which is therefore `1.0`). T3 keeps its
existing evaluator behavior because its separately maintained reproducibility
package is outside this refresh.

## Prediction Formats

Each task expects a JSONL file (one JSON object per line). The required fields are listed below.

### T1 -- Conditional Market Volume Prediction

```json
{"condition_id": "0x1234abcd", "label": "high_interest", "scores": {"high_interest": 0.8, "moderate_interest": 0.15, "low_interest": 0.05}}
```

Labels: `high_interest`, `moderate_interest`, `low_interest`

The LLM runner's nested `{"prediction": {"label": ..., "scores": ...}}` format
is also accepted.

**Primary metric:** Macro-F1 | **Secondary:** `high_interest` P@5/P@10

### T2 -- Post-to-Market Linking

```json
{"tweet_id": 123456789, "prediction": "0xaaa", "ranked_options": ["0xaaa", "NONE", "0xbbb"], "ranked_candidates": ["0xaaa", "0xbbb"]}
```

The legacy `ranked_market_ids` field is also accepted. For a `NONE` gold row,
MRR is 1 only when the top prediction is `NONE`; for a linked row, reciprocal
rank is computed after removing `NONE` from every supported ranking schema.
Canonical gold should include `candidate_ids`; when present, the evaluator
rejects duplicate ranking entries and predictions outside that frozen set.

**Primary metric:** Accuracy@1 | **Secondary:** MRR, `NONE` F1

### T3 -- Evidence Grading

```json
{"tweet_id": 123456789, "condition_id": "0x1234abcd", "predicted_grade": 3}
```

Grades: 0 (`noise`), 1 (`commentary_reaction`), 2 (`speculation_rumor`), 3 (`indirect_report`), 4 (`strong_direct`), 5 (`resolving`)

**Primary metric:** QWK (kappa) | **Secondary:** `resolving`-class precision, macro-F1

### T4 -- Market Movement Prediction

```json
{"condition_id": "0x1234abcd", "bundle_day": "2026-04-01", "direction_label": "up", "magnitude_bucket": "large", "delta_1d": 0.05, "delta_3d": 0.08, "delta_7d": 0.10}
```

Rows align by `(condition_id, bundle_day)`. Legacy tweet-level `delta_2h`
predictions remain supported when evaluated against legacy gold.

**Metrics:** Direction accuracy, magnitude macro-F1, Spearman rho at 1/3/7 days

### T5 -- Forward Drift and Persistence

```json
{"condition_id": "0x1234abcd", "bundle_day": "2026-04-01", "drift_magnitude_1d": 0.05, "drift_magnitude_3d": 0.08, "drift_magnitude_7d": 0.10, "volume_multiplier_1d": 1.2, "volume_multiplier_3d": 1.4, "volume_multiplier_7d": 1.6, "decay_class": "sustained"}
```

Decay labels: `transient`, `sustained`, `reversal`.

**Metrics:** Spearman rho for every continuous horizon, decay macro-F1

### T6 -- Cross-Market Co-Movement

```json
{"condition_id": "0x1234abcd", "bundle_day": "2026-04-01", "horizon_days": 3, "prediction": "cross_market"}
```

Labels: `no_effect`, `primary_only`, `cross_market`. Rows align by
`instance_id` when present, otherwise by
`(condition_id, bundle_day, horizon_days)`. `pred_label` and `label` are
accepted as compatibility aliases for `prediction`.

**Metrics:** Macro-F1 and accuracy overall and at 1/3/7 days

### T7 -- Legacy Impact Persistence Alias

```json
{"tweet_id": 123456789, "condition_id": "0x1234abcd", "label": "sustained"}
```

Labels: `transient`, `sustained`, `reversal`

This format is retained for earlier decay-only submissions. New results should
use the T5 format above.

## Using Metrics Programmatically

```python
from evaluation.metrics import macro_f1, spearman_rho, derive_direction_magnitude

score = macro_f1(y_true, y_pred, labels=["high_interest", "moderate_interest", "low_interest"])

direction, magnitude = derive_direction_magnitude(delta_2h=0.05)
# ("up", "medium")
```

All metric functions work with plain Python lists and have no hard dependency on numpy or scikit-learn.
