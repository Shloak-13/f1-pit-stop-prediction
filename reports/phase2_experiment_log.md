# Phase 2 Experiment Log

## Key Discovery

The winning signal is not primarily normal model tuning. It is synthetic bucket
structure plus categorical ordered statistics.

Evidence:

- `Year x Stint` buckets range from near-zero target rate to ~0.57.
- `Race x Year` buckets reach ~0.76 target rate.
- Exact TyreLife values from 56-66 have target rates from ~0.74 to ~0.91.
- Best single threshold probe is `Stint <= 1`, AUC ~0.715.
- Temporal next-lap `PitStop` probes are weak, with best temporal-shift AUC only
  ~0.59. The data has sequential texture, but direct future-row leakage is not
  the main exploit.

## Completed Phase 2 Runs

| Experiment | Model | CV | OOF AUC | Notes |
|---|---|---|---:|---|
| `20260518_162337_phase2_table_stratified_hgb` | Empirical-Bayes tables + HGB | Stratified 5-fold | 0.939541 | Directly models synthetic conditional probabilities. |
| `20260518_174444_phase2_table_stratified_logistic` | Empirical-Bayes tables + logistic | Stratified 5-fold | 0.934096 | Stable diversity model. |
| `20260518_165505_phase2_cat_fast_stratified` | CatBoost categorical/bucket model | Stratified 3-fold | 0.946625 | Strongest single model. |
| `20260518_191817_phase2_table_racecv_race_hgb` | Empirical-Bayes tables + HGB | Race GroupKFold | 0.915594 | Confirms race holdout is harsher and public-LB correlation must be tracked. |

## Best Ensembles

| Ensemble | Members | OOF AUC | Recipe |
|---|---|---:|---|
| `20260518_162851_phase2_core` | LGBM + logistic + table-HGB | 0.940138 | Table-HGB dominant, small LGBM/logistic corrections. |
| `20260518_191305_phase2_cat_table` | CatBoost + table-HGB + other diversity | 0.946822 | CatBoost plus 12% table-HGB. |

Best current candidate:

`submissions/20260518_191305_phase2_cat_table_hill.csv`

Leaderboard probe variants:

- `submissions/20260518_191305_phase2_cat_table_hill_temp0.95.csv`
- `submissions/20260518_191305_phase2_cat_table_hill_temp1.05.csv`
- `submissions/20260518_191305_phase2_cat_table_logit.csv`
- `submissions/20260518_191305_phase2_cat_table_rank.csv`
- single model: `submissions/20260518_165505_phase2_cat_fast_stratified_catboost_0.946625.csv`

## Feature Importance Readout

CatBoost top features:

1. `DeltaBin`
2. `TyreLifeFrac`
3. `StintLapGap`
4. `TyreLifeBin5`
5. `PosChgBin`
6. `RaceLapsEst`
7. `TyreLifeBin3`
8. `CAT__Race__Year__Stint`
9. `PitStop`
10. `CAT__Race__Year__Stint__TyreLifeBin3`

This points to a latent strategy-state generator: stint phase, tyre exhaustion,
race length, and race-year-stint buckets.

## Next Attack Path

1. Submit `catboost`, `cat_table_hill`, `cat_table_hill_temp1.05`, and
   `cat_table_logit` to map public-LB calibration.
2. If public LB rewards sharper predictions, prioritize `temp1.05/temp1.15`;
   if it penalizes, use `temp0.95` or rank.
3. Run a longer CatBoost job overnight: 5 folds, 500-900 iterations.
4. Add KNN/local-density features around `TyreLifeFrac`, `StintLapGap`,
   `DeltaBin`, `RaceLapsEst`, and `Race-Year-Stint`.
5. Pseudo-label only extreme CatBoost/table-hill predictions and retrain
   CatBoost/table models.
