# EventX Evaluation

Unified evaluation module for all six EventX tasks.

## Quick Start

```bash
# Single task
python evaluation/evaluate.py --task t1 --predictions preds.jsonl --gold gold.jsonl

# Auto-load gold from HuggingFace
python evaluation/evaluate.py --task t1 --predictions preds.jsonl

# All tasks at once
python evaluation/evaluate.py --task all --predictions-dir results/
```

Results are printed as JSON to stdout. Use `--output results.json` to save.

## Prediction Formats

Each task expects a JSONL file (one JSON object per line). The required fields are listed below.

### T1 -- Conditional Market Volume Prediction

```json
{"condition_id": "0x1234abcd", "label": "high_interest"}
```

Labels: `high_interest`, `moderate_interest`, `low_interest`

**Primary metric:** Macro-F1 | **Secondary:** `high`-class P@K

### T2 -- Post-to-Market Linking

```json
{"tweet_id": 123456789, "ranked_market_ids": ["0xaaa", "0xbbb", "0xccc"]}
```

`ranked_market_ids` is an ordered list of candidate market IDs, most likely first.

**Primary metric:** Acc@1 | **Secondary:** MRR

### T3 -- Evidence Grading

```json
{"tweet_id": 123456789, "condition_id": "0x1234abcd", "predicted_grade": 3}
```

Grades: 0 (`noise`), 1 (`commentary_reaction`), 2 (`speculation_rumor`), 3 (`indirect_report`), 4 (`strong_direct`), 5 (`resolving`)

**Primary metric:** QWK (kappa) | **Secondary:** `resolving`-class precision, macro-F1

### T4 -- Market Movement Prediction

```json
{"tweet_id": 123456789, "delta_2h": 0.05}
```

Direction and magnitude labels are derived from `delta_2h` automatically:
- Direction: `up` (> 0.02), `down` (< -0.02), `flat` (otherwise)
- Magnitude: `small` (|d| <= 0.02), `medium` (0.02 < |d| <= 0.08), `large` (|d| > 0.08)

**Metrics:** Direction accuracy, Magnitude macro-F1, Spearman rho

### T5 -- Volume and Price Impact

```json
{"tweet_id": 123456789, "condition_id": "0x1234abcd", "decay_class": "sustained", "price_impact": 0.08, "volume_multiplier": 3.5}
```

Decay labels: `transient`, `sustained`, `reversal`. Continuous targets: `price_impact`, `volume_multiplier`.

**Primary metrics:** Spearman rho (price_impact), Spearman rho (volume_multiplier), decay macro-F1

### T6 -- Cross-Market Propagation

```json
{"tweet_id": 123456789, "label": "no_effect"}
```

Labels: `no_effect`, `primary_mover`, `propagated_signal`

**Primary metric:** Macro-F1 | **Secondary:** MAE on onset lag (minutes)

## Using Metrics Programmatically

```python
from evaluation.metrics import macro_f1, spearman_rho, derive_direction_magnitude

score = macro_f1(y_true, y_pred, labels=["high_interest", "moderate_interest", "low_interest"])

direction, magnitude = derive_direction_magnitude(delta_2h=0.05)
# ("up", "medium")
```

All metric functions work with plain Python lists and have no hard dependency on numpy or scikit-learn.
