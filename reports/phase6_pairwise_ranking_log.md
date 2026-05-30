# Phase 6 Pairwise Ranking Optimization

## Objective

Move away from pointwise probability optimization and add direct ordering
signals for ROC-AUC. The rankers are not treated as calibrated probabilities;
their outputs are rank-normalized before blending.

## Pairwise Dataset Construction

Explicit positive-negative pairs are represented implicitly through ranking
queries:

- `RaceYearStint`
- `SyntheticBucket`
- `BoundaryBucket`

Each query contains positive and negative rows, allowing LambdaRank/XGBoost
rank-pairwise objectives to optimize within-query order. Rows in uncertain,
high-disagreement, and boundary probability regions are upweighted for
LightGBM LambdaRank.

## Ranking Models Run

### LightGBM LambdaRank

| Query | OOF AUC | Fold AUC |
|---|---:|---|
| RaceYearStint | 0.924391 | 0.929112, 0.925422, 0.918391 |
| SyntheticBucket | 0.929151 | 0.930188, 0.928067, 0.929271 |
| BoundaryBucket | 0.939478 | 0.939925, 0.938715, 0.939923 |

### XGBoost rank:pairwise

| Query | OOF AUC | Fold AUC |
|---|---:|---|
| RaceYearStint | 0.930978 | 0.930352, 0.931255, 0.931531 |
| BoundaryBucket | 0.943934 | 0.943889, 0.943468, 0.944568 |

XGBoost ranking does not accept per-row weights with group ranking, so it was
run unweighted. CatBoost ranking was not run in this pass because prior CatBoost
jobs already hit the local execution cap; the implemented Phase 6 path focuses
on rank objectives that completed reliably.

## Best OOF Rank Corrections

Best rank-fusion candidate:

`hard_lgbm_all_a0p08`

- OOF AUC: 0.949207
- Anchor: Phase 4.5 hard output
- Correction: 8% average LightGBM ranker score

Other high OOF variants:

- `hard_all_rankers_a0p08`: 0.949197
- `hard_lgbm_all_a0p05`: 0.949194
- `hard_rank_lgbm_SyntheticBucket_a0p05`: 0.949187

## Public-LB Realism

Because Phase 3/4.5 higher OOF did not transfer, public-facing probes should
include conservative Phase 2 anchors:

- `phase2_lgbm_all_a0p02`
- `phase2_lgbm_all_a0p03`
- `phase2_rank_lgbm_SyntheticBucket_a0p03`
- `phase2_rank_xgb_RaceYearStint_a0p03`
- `phase2_all_rankers_a0p03`

These sacrifice local OOF but test whether direct ranking corrections transfer
better than probability-specialist corrections.

## Generated Artifacts

- `src/phase6_pairwise_ranking.py`
- `src/phase6_rank_fusion.py`
- `reports/phase6_pairwise_ranking_report.md`
- `reports/phase6_rank_fusion_report.md`
- `experiments/20260519_153956_phase6_lambdarank/`
- `experiments/20260519_160100_phase6_xgbrank/`
- `experiments/20260519_161041_phase6_rank_fusion/`

## Recommended Submission Order

1. Conservative transfer probe:
   `submissions/20260519_161041_phase6_rank_fusion_phase2_lgbm_all_a0p03.csv`
2. Conservative ranker-specific probe:
   `submissions/20260519_161041_phase6_rank_fusion_phase2_rank_lgbm_SyntheticBucket_a0p03.csv`
3. XGB diversity probe:
   `submissions/20260519_161041_phase6_rank_fusion_phase2_rank_xgb_RaceYearStint_a0p03.csv`
4. High-OOF rank correction:
   `submissions/20260519_161041_phase6_rank_fusion_hard_lgbm_all_a0p08.csv`
5. Consensus high-OOF correction:
   `submissions/20260519_161041_phase6_rank_fusion_hard_all_rankers_a0p08.csv`
