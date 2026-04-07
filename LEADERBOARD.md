# EventX Leaderboard

Results reported on the standard test splits. All classification metrics are macro-averaged.

## Resolution Tier

### T2: Post-to-Market Linking

| Model | Acc@1 | MRR |
|-------|-------|-----|
| BGE top-1 (retrieval only) | 0.379 | 0.525 |
| BM25 top-1 | 0.072 | 0.107 |
| GPT-4o (0-shot) | **0.659** | **0.779** |
| GPT-4o (3-shot) | 0.650 | 0.772 |
| Sonnet 4.5 (0-shot) | 0.625 | 0.779 |
| Sonnet 4.5 (3-shot) | 0.649 | 0.763 |
| Grok 4.1 (0-shot) | 0.564 | 0.728 |
| Grok 4.1 (3-shot) | 0.555 | 0.718 |
| Qwen3.5-4B (0-shot) | 0.320 | 0.577 |
| Qwen3.5-4B (3-shot) | 0.569 | 0.738 |
| Qwen3.5-27B (0-shot) | 0.620 | 0.757 |
| Qwen3.5-27B (3-shot) | 0.605 | 0.750 |

### T3: Evidence Grading

| Model | QWK (kappa) | Macro-F1 |
|-------|-------------|----------|
| Pre-check pipeline | 0.686 | 0.320 |
| LightGBM | **0.849** | **0.489** |
| GPT-4o (0-shot) | 0.080 | 0.274 |
| GPT-4o (3-shot) | 0.106 | 0.312 |
| GPT-4o + image | 0.103 | 0.170 |
| Sonnet 4.5 (0-shot) | 0.132 | 0.237 |
| Sonnet 4.5 (3-shot) | 0.172 | 0.285 |
| Grok 4.1 (0-shot) | 0.100 | 0.198 |
| Grok 4.1 (3-shot) | 0.174 | 0.301 |
| Qwen3.5-4B (0-shot) | 0.126 | 0.199 |
| Qwen3.5-4B (3-shot) | 0.192 | 0.287 |
| Qwen3.5-27B (0-shot) | 0.210 | 0.292 |
| Qwen3.5-27B (3-shot) | 0.214 | 0.315 |

## Forecast Tier

### T1: Conditional Market Volume Prediction

| Model | Macro-F1 | P@10 |
|-------|----------|------|
| LightGBM | **0.508** | **0.600** |
| GPT-4o (0-shot) | 0.266 | 0.600 |
| GPT-4o (3-shot) | 0.343 | 0.400 |
| Sonnet 4.5 (0-shot) | 0.362 | 0.600 |
| Sonnet 4.5 (3-shot) | 0.334 | 0.500 |
| Grok 4.1 (0-shot) | 0.329 | 0.300 |
| Grok 4.1 (3-shot) | 0.297 | 0.400 |
| Qwen3.5-4B (0-shot) | 0.304 | 0.400 |
| Qwen3.5-4B (3-shot) | 0.276 | 0.400 |
| Qwen3.5-27B (0-shot) | 0.240 | 0.300 |
| Qwen3.5-27B (3-shot) | 0.315 | 0.200 |

### T4: Market Movement Prediction

| Model | Dir-Acc | Mag Macro-F1 | Spearman rho |
|-------|---------|--------------|--------------|
| Random walk | 0.503 | 0.364 | 0.011 |
| LightGBM | **0.682** | 0.556 | 0.322 |
| GPT-4o (0-shot) | 0.605 | 0.370 | 0.121 |
| GPT-4o (3-shot) | 0.648 | 0.366 | 0.181 |
| GPT-4o + image | 0.590 | 0.373 | 0.084 |
| GPT-4o + prices | 0.638 | 0.373 | 0.120 |
| Sonnet 4.5 (0-shot) | 0.598 | 0.545 | 0.209 |
| Sonnet 4.5 (3-shot) | 0.604 | **0.554** | **0.356** |
| Grok 4.1 (0-shot) | 0.560 | 0.479 | 0.317 |
| Grok 4.1 (3-shot) | 0.585 | 0.447 | 0.295 |
| Qwen3.5-4B (0-shot) | 0.597 | 0.381 | -0.050 |
| Qwen3.5-4B (3-shot) | 0.581 | 0.263 | -0.045 |
| Qwen3.5-27B (0-shot) | 0.623 | 0.448 | 0.037 |
| Qwen3.5-27B (3-shot) | 0.619 | 0.449 | 0.118 |

### T5: Volume and Price Impact

| Model | rho (price) | rho (volume) | Decay Macro-F1 |
|-------|-------------|--------------|----------------|
| LightGBM | 0.343 | 0.363 | **0.518** |
| GPT-4o (0-shot) | 0.056 | -0.039 | 0.296 |
| GPT-4o + prices | 0.268 | -0.046 | 0.285 |
| Sonnet 4.5 (0-shot) | 0.308 | 0.073 | 0.277 |
| Sonnet 4.5 (3-shot) | **0.417** | **0.211** | 0.251 |
| Grok 4.1 (0-shot) | 0.176 | -0.054 | 0.284 |
| Grok 4.1 (3-shot) | 0.135 | 0.019 | 0.273 |
| Qwen3.5-4B (0-shot) | -0.038 | -0.122 | 0.309 |
| Qwen3.5-4B (3-shot) | 0.067 | 0.068 | 0.303 |
| Qwen3.5-27B (0-shot) | 0.179 | 0.078 | 0.290 |
| Qwen3.5-27B (3-shot) | 0.331 | 0.162 | 0.320 |

### T6: Cross-Market Propagation

| Model | Macro-F1 | MAE (min) |
|-------|----------|-----------|
| Graph heuristic | 0.250 | 24.49 |
| LightGBM | 0.274 | 24.56 |
| GPT-4o (0-shot) | 0.334 | -- |
| **GPT-4o (3-shot)** | **0.345** | -- |
| Sonnet 4.5 (0-shot) | 0.237 | -- |
| Sonnet 4.5 (3-shot) | 0.259 | -- |
| Grok 4.1 (0-shot) | 0.319 | -- |
| Grok 4.1 (3-shot) | 0.288 | -- |
| Qwen3.5-4B (0-shot) | 0.306 | -- |
| Qwen3.5-4B (3-shot) | 0.251 | -- |
| Qwen3.5-27B (0-shot) | 0.301 | -- |
| Qwen3.5-27B (3-shot) | 0.295 | -- |

## How to Submit

1. Run your model on the test split for each task
2. Format predictions per [`evaluation/README.md`](evaluation/README.md)
3. Evaluate locally: `python evaluation/evaluate.py --task all --predictions-dir results/`
4. Open a pull request adding your row with a link to your method
