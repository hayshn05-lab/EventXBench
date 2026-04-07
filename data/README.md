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

EventX connects social media posts on Twitter/X to prediction market dynamics on Polymarket. It provides six tasks spanning two tiers:

- **Resolution tier** (human-annotated): Post-to-Market Linking (T2), Evidence Grading (T3)
- **Forecast tier** (deterministic labels): Market Volume Prediction (T1), Market Movement Prediction (T4), Impact Persistence (T5), Cross-Market Propagation (T6)

### Supported Tasks

| Config | Task | Rows | Description |
|--------|------|------|-------------|
| `t1` | Market Volume Prediction | 305 | Predict eventual trading volume from pre-market tweets |
| `t2` | Post-to-Market Linking | 815 | Match a tweet to the correct prediction market |
| `t3` | Evidence Grading | 342,552 | Grade tweet relevance to a market (0-5) |
| `t4` | Market Movement Prediction | 4,803 | Predict price direction and magnitude at 2h horizon |
| `t5` | Volume & Price Impact | 407 (268 clean) | Predict price_impact, volume_multiplier, and decay class |
| `t6` | Cross-Market Propagation | 4,006 | Predict spillover to sibling markets |
| `posts` | Tweet Metadata | ~9M | Tweet IDs (text stripped for privacy) |
| `markets` | Market Metadata | -- | Market questions, categories, resolution info |
| `ohlcv` | Market OHLCV | -- | Price/volume time series |

### Usage

```python
from datasets import load_dataset

# Load a task split
ds = load_dataset("mlsys-io/EventXBench", "t1")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

# Load market metadata
markets = load_dataset("mlsys-io/EventXBench", "markets")

# Load OHLCV time series
ohlcv = load_dataset("mlsys-io/EventXBench", "ohlcv")
```

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
- `interest_label` (str): Target label -- `high` (>80th pctl), `moderate` (40th--80th), `low` (<40th)
- ... (see full schema in the dataset viewer)

### T2: Post-to-Market Linking

- `tweet_id` (int): Twitter post ID
- `tweet_text` (str): Tweet text content
- `market_id` (str): Polymarket condition ID
- `market_question` (str): Market question text
- `embedding_score` (float): Semantic similarity score

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

- `tweet_id` (int): Twitter post ID
- `condition_id` (str): Polymarket condition ID
- `price_t0` (float): Price at tweet publication time
- `delta_2h` (float): Absolute price change at 2h horizon
- `direction_label` (str): `up`, `down`, or `flat`
- `magnitude_bucket` (str): `small`, `medium`, or `large`
- `confound_flag` (bool): Whether confounding events were detected

### T5: Volume and Price Impact

- `tweet_id` (int): Twitter post ID
- `condition_id` (str): Polymarket condition ID
- `price_impact_json` (dict): Max absolute deviation from p0 at horizons (15m, 30m, 1h, 2h, 6h)
- `volume_multiplier_json` (dict): Total volume / 24h baseline at multiple horizons
- `decay_class` (str): `transient`, `sustained`, or `reversal`
- `confound_flag` (bool): Whether confounding events were detected
- Metrics: Spearman rho for price_impact and volume_multiplier, decay macro-F1

### T6: Cross-Market Propagation

- `tweet_id` (int): Twitter post ID
- `primary_condition_id` (str): Primary market condition ID
- `label` (str): `no_effect`, `primary_mover`, `propagated_signal`
- `sibling_count` (int): Number of sibling markets
- `moved_sibling_count` (int): Number of siblings that moved
- `primary_delta_h` (float): Primary market price change
- `confound_flag` (bool): Whether confounding events were detected

### Posts (Tweet Metadata)

- `tweet_id` (int): Twitter post ID
- `text` (null): Set to NULL for privacy -- use Twitter API for rehydration
- Additional metadata fields (timestamps, user IDs, etc.)

## Privacy and Ethics

- **Tweet text**: Stripped from the public release per Twitter/X ToS. Tweet IDs are provided for authorized rehydration.
- **Market data**: Polymarket data is publicly available on-chain and included under fair use for research.
- **No PII**: User-level features are aggregated; no individual user profiles are released.

## Citation

```bibtex
@inproceedings{eventx2026,
  title     = {EventX: A Multimodal Benchmark Linking Social Media Posts to Prediction Market Dynamics},
  author    = {TODO},
  booktitle = {Proceedings of the ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
}
```

## License

CC BY-NC 4.0
