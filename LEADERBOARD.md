# EventX Leaderboard

Results use the named release's frozen evaluation contract: `test` for the
non-T3 canonical tasks and the human-adjudicated `gold` audit split for T3.
The explicitly labeled T3 non-LLM references instead use the package's
documented 70/30 silver split. Values below are included only when a matching
frozen report or reproducible packaged prediction file is available; results
from superseded schemas are not carried forward.

## Resolution Tier

### T2: Post-to-Market Linking

Release: `t2.gold.r3.contextual.v1` (2,500 test rows).

| Model | Acc@1 | MRR | `NONE` F1 |
|-------|------:|----:|----------:|
| Always `NONE` | 0.5208 | 0.5208 | 0.6849 |
| Stratified random (train prior) | 0.1924 | 0.2642 | 0.3626 |
| BGE top-1, no `NONE` | 0.1512 | 0.2379 | 0.0000 |
| BGE top-1, frozen threshold | **0.5360** | 0.7355 | **0.7207** |
| BGE top-1, high-precision threshold | 0.5304 | **0.7543** | 0.6975 |

The contextual LLM runner is included in `baselines/t2`; no LLM row is copied
from the older T2 contract. The deterministic rows above are reproduced by
`python -m baselines.t2.contextual_baselines`.

### T3: Evidence Grading vs. Human Gold (2,687-instance gold audit pool)

Evaluated against the human-adjudicated `gold_grade` (T3_Reproducible_Package's
2,687-instance, rare-grade-enriched audit pool) - **not** silver `final_grade`.
The two are not interchangeable ground truth; see `data/README.md`.

**Anchors:** silver pipeline κ_w = 0.582 (fails the project's own 0.6 reliability
bar); human inter-annotator agreement κ_w = 0.775–0.848 (`metrics.md` Phase 5).
κ_w / κ_u = quadratic-weighted / unweighted Cohen's κ. Models served via
provider APIs.

| Model | 0-shot κ_w | 0-shot κ_u | 0-shot F1 | 3-shot κ_w | 3-shot κ_u | 3-shot F1 |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Gemma-4-26B | 0.652 | 0.230 | 0.331 | 0.688 | 0.323 | 0.417 |
| GPT-4o | 0.659 | 0.190 | 0.288 | 0.699 | 0.221 | 0.330 |
| Grok-4.5 | 0.637 | 0.197 | 0.306 | **0.774** | **0.416** | **0.447** |
| Qwen3.6-35B | 0.564 | 0.266 | 0.360 | 0.663 | 0.349 | 0.430 |
| DeepSeek-V3 | 0.645 | 0.213 | 0.304 | 0.618 | 0.331 | 0.421 |
| Claude Sonnet-5 | 0.564 | 0.247 | 0.328 | 0.539 | 0.264 | 0.344 |

Even the best 3-shot result (Grok-4.5, κ_w 0.774) sits below the human IAA
floor (0.775–0.848) and only modestly above the silver pipeline's own
agreement with gold (κ_w 0.582) - evidence grading remains unsolved at the
gold standard, and 3-shot doesn't reliably help (Claude Sonnet-5, DeepSeek-V3
both *regress* on κ_w with 3-shot).

#### Non-LLM baselines vs. silver (`final_grade`, 70/30 market-level split, `random_state=42`)

| Method | Kappa (unweighted) | Macro-F1 |
|--------|---------------------|----------|
| Majority | 0.0000 | 0.0853 |
| Random (single seed=42) | 0.0038 | 0.1709 |
| LightGBM (`requires_official` + tweet/predicate embeddings) | 0.4562 | 0.3963 |

Reference numbers from `T3_Reproducible_Package/metrics.md` Phase 6 - pending
re-run in this repo with the corrected `baselines/t3/basic_baseline.py` /
`lgbm_baseline.py` once the underlying data files are rebuilt. **The previous
"Pre-check pipeline: 0.686 QWK" / "LightGBM: 0.849 QWK" rows here were removed**
- they were produced by a mislabeled auto-grade shortcut and a feature set that
included near-label-leaking deterministic-check columns, both fixed on this
branch; those old numbers should not be cited.

## Forecast Tier

### T1: Conditional Market Volume Prediction

Release: `t1.kdd.v2` (275 test markets).

| Model | Accuracy | Macro-F1 | P@10 |
|-------|---------:|---------:|-----:|
| Majority class | 0.5855 | 0.2462 | -- |
| Random prior (seed 42) | 0.4400 | 0.3683 | -- |
| LightGBM (`market_social`) | **0.5855** | **0.5124** | **0.8000** |

### T4: Daily Market Movement Prediction

Release: `t4.kdd.v2` (5,791 test market-day bundles).

| Model | Dir-Acc | Mag Macro-F1 | rho 1d | rho 3d | rho 7d |
|-------|--------:|-------------:|-------:|-------:|-------:|
| Random walk | 0.6959 | 0.2052 | -- | -- | -- |
| LightGBM (Tier 1; supplied frozen report) | 0.6223 | **0.3029** | **0.1855** | **0.2198** | **0.2119** |
| Qwen3.6-35B-A3B (0-shot; supplied frozen report) | **0.6942** | 0.2074 | -0.0132 | -0.0444 | -0.0343 |

Tier 1 uses all rows. The package also reports non-confounded and active-signal
tiers; do not compare those filtered tiers directly with the headline rows
above. The current LightGBM runner corrects the supplied package's row-wise CV
and complete-curve filtering: it uses event-group-disjoint tuning, predicts
each continuous horizon independently, and produces one strict-evaluator row
per test instance. Therefore the two archived report rows above are retained
as provenance, not presented as newly rerun values.

### T5: Forward Drift Magnitude and Persistence

Release: `t5.kdd.v2` (1,761 test rows; metrics use rows with available
targets).

| Model | Drift rho 1d | 3d | 7d | Volume rho 1d | 3d | 7d | Decay Macro-F1 |
|-------|-------------:|---:|---:|--------------:|---:|---:|---------------:|
| Majority decay | -- | -- | -- | -- | -- | -- | 0.2516 |
| Random-prior decay | -- | -- | -- | -- | -- | -- | **0.3344** |
| LightGBM | 0.0260 | 0.0899 | 0.1167 | 0.0741 | 0.0790 | 0.0018 | 0.3097 |

The LightGBM row was reproduced with group-disjoint classification CV and the
corrected all-row continuous runner: 1,761/1,495/1,389 evaluable rows at
1/3/7 days and 1,495 decay-labeled rows. The updated T5 LLM runner predicts all
six continuous targets plus decay. Decay-only results from the legacy T7
schema are intentionally omitted.

### T6: Daily Cross-Market Co-Movement

Release: `t6.kdd.v2` (2,592 test market-day/horizon rows).

| Model | Accuracy | Overall Macro-F1 | H=1d | H=3d | H=7d |
|-------|---------:|-----------------:|-----:|-----:|-----:|
| Per-horizon majority class | 0.5976 | **0.3989** | 0.2068 | 0.2629 | 0.2834 |
| Random prior (mean of seeds 42/43/44) | -- | 0.3273 | **0.3166** | **0.3284** | **0.3350** |

The v2 LightGBM and LLM runners are packaged and the baseline suite has been
executed; their frozen report artifacts were not present in the supplied code
bundle, so this table does not guess or copy legacy values. The old graph
heuristic and onset-lag results use incompatible intraday v1 labels.

## How to Submit

1. Run your model on the test split for each task
2. Format predictions per [`evaluation/README.md`](evaluation/README.md)
3. Evaluate against the matching frozen gold, e.g.
   `python evaluation/evaluate.py --task t1 --predictions results/t1_predictions.jsonl --gold KDD/data/t1_kdd_v2/test.jsonl`
4. Open a pull request adding your row with a link to your method
