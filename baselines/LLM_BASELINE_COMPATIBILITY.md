# LLM baseline compatibility: non-T3 refresh

This table records the canonical local releases updated by the July 2026
non-T3 refresh. T3 is maintained in its dedicated reproducible package and is
not modified here.

| Task | Canonical evaluation data | Headline output | Leakage-safe input source |
|---|---|---|---|
| T1 | `KDD/data/t1_kdd_v2` | `high_interest` / `moderate_interest` / `low_interest` | Selected `manifest.json.feature_rungs` |
| T2 | `data/t2/gold_r3_contextual_final` | Full candidate ranking including `NONE` | Contextual r3 candidates and raw market fields |
| T4 | `KDD/data/t4_kdd_v2` | 1/3/7-day signed price deltas | End-of-day price, volatility, and bundle/social aggregates |
| T5 | `KDD/data/t5_kdd_v2` | 1/3/7-day drift magnitudes, volume multipliers, decay | End-of-day price, volume baseline, bundle count |
| T6 | `KDD/data/t6_kdd_v2` | Three-way co-movement plus analysis auxiliaries | Frozen schema allowlist and causal-visible sibling count |

## Shared execution rules

- Use validation or development data before test.
- Test requires `--allow-test`, including dry-run prompt inspection.
- Few-shot demonstrations come from train/calibration only.
- Target prompts contain no label, post-decision diagnostic, future price, or
  future movement field.
- Temperature is zero.
- Responses are checkpointed in JSONL and resumable configurations are
  protected by a run hash.
- Invalid headline responses are errors rather than silently replaced by a
  majority prediction.
- Reports record the exact model, prompt/run configuration, and input hashes.

Provider choices are `openai`, `anthropic`, and `xai`. An OpenAI-compatible
gateway is selected by using `--provider openai --base-url ...`.

## Task-specific notes

### T1

The v2 release has train/test only. Use `--split train --dry-run` for prompt
development, then freeze the configuration before
`--split test --allow-test`. The LLM input now includes `question` and
`description` and no
longer references obsolete `event_group_label` or `event_text` fields.

### T2

The contextual r3 runner remains the reference implementation. Its default
few-shot labels now use `train_labels_final.csv`; the preliminary silver file
is no longer the default.

### T4

Validation is now loaded explicitly rather than silently evaluating test.
Prediction-time features follow the LightGBM allowlist. The frozen
train-derived magnitude terciles come from the manifest, not from target-row
columns.

### T5

The old runner predicted only `decay_class`. The updated runner predicts:

```text
drift_magnitude_1d / 3d / 7d
volume_multiplier_1d / 3d / 7d
decay_class
```

Rows with null decay remain in evaluation for their available continuous
targets; decay macro-F1 is computed only where the gold decay label exists.

### T6

The old propagation labels and future sibling-movement features were removed.
The updated design uses `no_effect`, `primary_only`, and `cross_market`,
reports the four-way/cascade auxiliaries, and bootstraps by event cluster.

The JSONL contains causally valid `primary_price_d` and
`primary_sigma_14d`, but the frozen schema omits them from its prediction-time
allowlist. T6 therefore excludes them until the schema is corrected. The
current `domain` column is also all-null, so T6's market-only rung is presently
a decision-context control.
