# T6 LLM baseline design

The T6 LLM baseline targets the frozen `t6.kdd.v2` release. It forecasts
cross-market co-movement for one `(primary market, bundle day, horizon)` row
using information available at the end of the bundle day.

> **Legacy note:** `data_utils.py` and `graph_baseline.py` implement the
> pre-v2 intraday propagation task and its old labels. They are retained only
> for reproducibility of legacy results and must not be used for `t6.kdd.v2`.
> The canonical v2 runners are `basic_baseline.py`, `lightgbm_baseline.py`,
> and `llm_baseline.py`.

## Basic-baseline strategies

`basic_baseline.py` reports the global-majority and per-horizon-majority
strategies separately. For v2 data, its default JSONL output is explicitly
tagged `strategy=per-horizon-majority`; evaluating that file reproduces the
leaderboard row (accuracy `0.5976`, overall macro-F1 `0.3989`). Use
`--prediction-strategy global-majority` when an overall single-label baseline
is required.

## Targets

The required headline prediction is:

- `no_effect`: neither the primary market nor a visible sibling moves.
- `primary_only`: the primary market moves and no visible sibling moves.
- `cross_market`: at least one visible sibling moves, with or without a
  primary-market move.

The runner also requests two analysis-only outputs:

- `four_way_label`: `no_effect`, `primary_only`, `sibling_only`, or
  `co_movement`.
- `cascade_size`: predicted number of visible siblings that move.

Daily data cannot resolve intraday ordering. The prompt therefore describes
co-movement and never asks the model to infer which market propagated a signal
to another.

## Leakage boundary

Only fields authorized by
`KDD/data/t6_kdd_v2/schema.json.prediction_time_fields` are serialized. The
default `market_social_graph` prompt contains:

```text
bundle_day
horizon_days
domain
n_posts
first_post_time
followers_max
engagement_sum
engagement_max
max_final_grade
num_siblings_visible_d
```

Future movement diagnostics, usable-at-horizon sibling sets, labels, cascade
size, confound flags, and onset lags are prohibited. `num_siblings_total` is
also prohibited because the experimental protocol identifies retrospective
graph degree as potential future leakage.

The release contains `primary_price_d`, `primary_sigma_14d`, and
`primary_sigma_n_obs`, which are logically known at decision time. They are
currently absent from the frozen schema's prediction-time allowlist, so this
baseline does not use them. The runner records this release limitation in each
report. The current release also has an all-null `domain` column, making the
`market_only` rung a decision-context-only control until the schema/data gap is
resolved.

## Feature rungs

The `--feature-rung` option supports the benchmark ablation ladder:

| Rung | Prompt information |
|---|---|
| `market_only` | Decision day, horizon, domain |
| `text_social_only` | Decision context and aggregate post/social features |
| `market_social` | Domain plus aggregate post/social features |
| `market_graph` | Domain plus causal-visible sibling count |
| `market_social_graph` | Domain, post/social features, and visible sibling count |

The public v2 release does not contain bundle text. Consequently,
`text_social_only` currently means structured bundle/social aggregates rather
than raw text.

## Evaluation protocol

- Validation is the default development split.
- Test requires `--allow-test`.
- Three-shot examples are selected deterministically from train only, with one
  example per headline class and target horizon.
- Temperature is fixed to zero.
- Every response is immediately appended to JSONL and can be resumed.
- Existing checkpoints must have the same run-configuration hash.
- Metrics are reported overall and separately at 1, 3, and 7 days.
- Headline metrics are macro-F1 and accuracy.
- Analysis metrics are four-way macro-F1, cascade-size MAE, and cascade-size
  Spearman correlation.
- Confidence intervals use event-cluster bootstrap resampling.
- Reports contain dataset hashes, prompt/configuration hashes, model access
  date, token counts, latency, and optional estimated API cost.

Malformed headline predictions are checkpointed as errors and make a run
incomplete; they are not silently replaced with a majority-class label.
Invalid auxiliary outputs do not discard an otherwise valid headline
prediction, but are excluded from the corresponding auxiliary metric and
recorded in `auxiliary_errors`.

## Usage

Run commands from the repository root after downloading/unpacking the frozen
`t6.kdd.v2` release. A clean Git checkout does not contain
`KDD/data/t6_kdd_v2`, so pass its location explicitly.

Prompt preview:

```bash
python baselines/t6/llm_baseline.py \
  --provider openai \
  --model MODEL \
  --data-dir KDD/data/t6_kdd_v2 \
  --split validation \
  --shots 0 \
  --dry-run
```

OpenAI-compatible gateway:

```bash
python baselines/t6/llm_baseline.py \
  --provider openai \
  --base-url https://lum.id/llm \
  --api-key-env LUMID_API_KEY \
  --model MODEL \
  --data-dir KDD/data/t6_kdd_v2 \
  --split validation \
  --shots 3 \
  --feature-rung market_social_graph \
  --resume \
  --output results/t6.MODEL.validation.3shot.jsonl
```

Sealed test after freezing the configuration on validation:

```bash
python baselines/t6/llm_baseline.py \
  --provider openai \
  --model MODEL \
  --data-dir KDD/data/t6_kdd_v2 \
  --split test \
  --allow-test \
  --shots 0 \
  --output results/t6.MODEL.test.0shot.jsonl
```

For estimated cost reporting, additionally pass
`--input-cost-per-million` and `--output-cost-per-million`.
