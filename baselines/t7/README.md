# T7 — Impact Persistence (Decay Classification)

Classify whether a tweet's initial market impact is `transient`, `sustained`, or `reversal`.

T7 shares the same underlying data as T5 (Volume & Price Impact). The difference:
- **T5** evaluates the continuous predictions (`price_impact`, `volume_multiplier`) via Spearman rho
- **T7** evaluates the `decay_class` classification via Macro-F1

In the original codebase, this uses `task5+7/` directories and `t7_` prefixes.

## Usage

```python
from eventxbench import load_task
train, test = load_task("t7")
```

## Baselines

The T7 classification baselines are the same as the T5 decay baselines in [`baselines/t5/`](../t5/):
- `baselines/t5/llm_baseline.py` — LLM classification of decay class
- `baselines/t5/lightgbm_baseline.py` — LightGBM on extracted features
- `baselines/t5/basic_baseline.py` — Majority / random baselines

When running these scripts, the decay classification metrics apply to T7.
