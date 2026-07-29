---
annotations_creators:
- machine-generated
- expert-generated
language:
- en
license: cc-by-nc-4.0
multilinguality: monolingual
pretty_name: EventX
size_categories:
- 1M<n<10M
source_datasets:
- original
tags:
- prediction-markets
- social-media
- multimodal
- financial-nlp
- twitter
- polymarket
task_categories:
- text-classification
- tabular-classification
task_ids:
- multi-class-classification
---

# EventX

A multimodal benchmark linking 9M Twitter/X posts to 11,952 Polymarket prediction markets (2021--2026).

## Dataset Description

EventX connects social media posts on Twitter/X to prediction-market dynamics on Polymarket. The current release provides six canonical tasks spanning two tiers:

- **Resolution tier** (human-annotated): Post-to-Market Linking (T2), Evidence Grading (T3)
- **Forecast tier** (causally constructed labels): Market Volume Prediction (T1), Daily Market Movement (T4), Forward Drift & Persistence (T5), Cross-Market Co-Movement (T6)

### Supported Tasks

| Config | Task | Rows | Description |
|--------|------|------|-------------|
| `t1` | Market Volume Prediction | 984 | Predict lifetime-volume interest from pre-market evidence |
| `t2` | Post-to-Market Linking | 5,544 | Rank a contextual market candidate or select `NONE` |
| `t3` | Evidence Grading | 342,552 | Grade tweet relevance to a market (0-5) |
| `t4` | Daily Market Movement | 10,934 | Predict direction, magnitude, and 1/3/7-day price deltas |
| `t5` | Forward Drift & Persistence | 3,342 | Predict drift, volume multiplier, and decay |
| `t6` | Cross-Market Co-Movement | 4,583 | Predict 1/3/7-day sibling-market co-movement |
| `t7` | Legacy decay alias | -- | Compatibility alias; canonical decay evaluation is T5 |
| `posts` | Tweet Metadata | ~9M | Tweet IDs (text stripped for privacy) |
| `markets` | Market Metadata | -- | Market questions, categories, resolution info |
| `ohlcv` | Market OHLCV | -- | Price/volume time series |

### Usage

> **Publishing status (2026-07-29):** the hosted
> `mlsys-io/EventXBench` repository is gated and still contains the earlier
> task schemas. The July 2026 v2 files described below are distributed in the
> dated maintainer release bundles and must be loaded from explicit local
> paths until the hosted configs are replaced.

```python
from eventxbench import load_task

# Canonical v2, after unpacking the dated release bundle
train_df, test_df = load_task("t1", local_dir="KDD/data/t1_kdd_v2")
validation_df = load_task(
    "t4",
    local_dir="KDD/data/t4_kdd_v2",
    split="validation",
)
```

After the v2 migration is published and its access conditions are accepted,
the equivalent hosted interface will be:

```python
from datasets import load_dataset

ds = load_dataset("mlsys-io/EventXBench", "t1")
```

### Release and Split Contracts

| Config | Version | Train | Validation | Test |
|--------|---------|------:|-----------:|-----:|
| `t1` | `t1.kdd.v2` | 709 | -- | 275 |
| `t2` | `t2.gold.r3.contextual.v1` | 544 | 2,500 | 2,500 |
| `t4` | `t4.kdd.v2` | 2,875 | 2,268 | 5,791 |
| `t5` | `t5.kdd.v2` | 889 | 692 | 1,761 |
| `t6` | `t6.kdd.v2` | 766 | 1,225 | 2,592 |

T1 uses a purged group-atomic temporal train/test split. T4--T6 use
`tier2.temporal.v2`; T6 is also event-cluster grouped. Validation is for model
selection, and sealed test access in the LLM runners requires `--allow-test`.

## Data Fields

### T1: Market Volume Prediction

- `event_group_id` (str): Cluster ID for the event group
- `condition_id` (str): Polymarket market condition ID
- `question` (str): Market question text
- `category` (str): Market category
- `tweet_count` (int): Number of tweets in the event cluster
- `unique_user_count` (int): Distinct authors
- `burst_duration_hours` (float): Duration of tweet burst
- `max_author_followers` (int): Max follower count in cluster
- `interest_label` (str): `high_interest`, `moderate_interest`, or `low_interest`
- Forbidden as inputs: lifetime/future volume, percentile, resolution, and outcome fields listed in the release manifest
- ... (see full schema in the dataset viewer)

### T2: Post-to-Market Linking

- `tweet_id` (int): Twitter post ID
- `tweet_text` (str): Tweet text content
- `instance_id` (str): Frozen benchmark instance ID
- `candidates` (list): Up to ten candidate markets with IDs, questions, resolution rules, domains, ranks, and dense scores
- Gold output: Candidate condition ID or `NONE`
- Splits: 544 mixed-provenance train rows; 2,500 three-reviewer validation and 2,500 three-reviewer test rows

### T3: Evidence Grading

- `tweet_id` (int): Twitter post ID
- `condition_id` (str): Polymarket condition ID
- `tweet` (str): Tweet text
- `market` (str): Market metadata
- `question` (str): Market question
- `final_grade` (int): Evidence grade 0-5
- `llm_grade` (int): LLM-assigned grade
- `llm_confidence` (float): LLM confidence score

### T4: Market Movement Prediction

- `condition_id` (str): Polymarket condition ID
- `bundle_day` (date): UTC market-day decision key
- `price_d` (float): End-of-day YES price available at decision time
- `delta_1d`, `delta_3d`, `delta_7d` (float): Forward price changes
- `direction_label` (str): `up`, `down`, or `flat`
- `magnitude_bucket` (str): `none`, `small`, `medium`, or `large`
- The manifest freezes the temporal split, train-derived thresholds, and prediction-time feature allowlist

### T5: Forward Drift and Persistence

- `condition_id` (str): Polymarket condition ID
- `bundle_day` (date): UTC market-day decision key
- `drift_magnitude_1d`, `drift_magnitude_3d`, `drift_magnitude_7d` (float)
- `volume_multiplier_1d`, `volume_multiplier_3d`, `volume_multiplier_7d` (float)
- `decay_class` (str): `transient`, `sustained`, or `reversal`
- Metrics: Spearman rho per continuous target/horizon and decay macro-F1

### T6: Cross-Market Co-Movement

- Row key: `condition_id`, `bundle_day`, `horizon_days`
- `headline_label` (str): `no_effect`, `primary_only`, or `cross_market`
- `num_siblings_visible_d` (int): Siblings causally visible at decision time
- Analysis-only targets: `four_way_label` and `cascade_size_h`
- Future movement, usable-at-horizon sibling, confound, and label fields are forbidden model inputs

### Posts (Tweet Metadata)

- `tweet_id` (int): Twitter post ID
- `text` (null): Set to NULL for privacy -- use Twitter API for rehydration
- Additional metadata fields (timestamps, user IDs, etc.)

## Privacy and Ethics

- **Tweet text**: Stripped from the public release per Twitter/X ToS. Tweet IDs are provided for authorized rehydration.
- **Market data**: Polymarket data is publicly available on-chain and included under fair use for research.
- **No PII**: User-level features are aggregated; no individual user profiles are released.

## License

CC BY-NC 4.0
