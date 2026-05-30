# Phase 3 Experiment Log

## Objective

Exploit local manifold continuity after Phase 2 saturated generic boosting.

The available competition files do not contain `GapToLeader_ms` or
`IntervalToPositionAhead_ms`, so the manifold uses available race-state
surrogates:

- `DeltaBin`
- `TyreLifeFrac`
- `StintLapGap`
- `TyreLifeBin3/5`
- `RaceLapsEst`
- `RaceProgress`
- `Position`
- `Compound`
- `Driver`
- `Race`
- `Stint`
- prior model predictions from CatBoost/table/LGBM/logistic

## Completed Runs

| Experiment | Description | OOF AUC | Notes |
|---|---|---:|---|
| `20260519_141027_phase3_knn_euclid_phase3` | PCA-32 Euclidean KNN + cluster priors + HGB stack | 0.947882 | Neighborhood and cluster features improved over Phase 2 CatBoost/table. |
| `20260519_143756_phase3_specialists` | Local specialists over high-signal regions | 0.948807 | Replaced Phase 3 predictions only where region OOF improved. |
| `20260519_144202_phase3_specialist_ensemble` | Specialist + neighborhood + CatBoost/table ensemble | 0.948983 | Hill recipe: specialist plus 28% neighborhood stack. |

## Specialist Diagnostics

| Region | Base AUC | Specialist AUC | Used |
|---|---:|---:|---|
| extreme tyre | 0.958773 | 0.960175 | yes |
| late race | 0.961638 | 0.961671 | no |
| high stint gap | 0.909874 | 0.910255 | yes |
| HARD | 0.928163 | 0.929300 | yes |
| MEDIUM | 0.950960 | 0.951687 | yes |
| SOFT | 0.925709 | 0.926983 | yes |
| Year 2023 | 0.927786 | 0.937221 | yes |
| Year 2024+ | 0.925721 | 0.926655 | yes |

## Best Submissions

Primary:

`submissions/20260519_144202_phase3_specialist_ensemble_hill.csv`

Probe variants:

- `submissions/20260519_144202_phase3_specialist_ensemble_hill_temp0.95.csv`
- `submissions/20260519_144202_phase3_specialist_ensemble_hill_temp1.05.csv`
- `submissions/20260519_144202_phase3_specialist_ensemble_logit.csv`
- `submissions/20260519_144202_phase3_specialist_ensemble_rank.csv`

## Pseudo Labeling

Conservative agreement pseudo labels were generated from CatBoost, table-HGB,
Phase 3 KNN, and specialist predictions.

Strict `low=0.02 high=0.98 max_std=0.035`:

- selected: 77,947
- positives: 0
- negatives: 77,947

Balanced-positive probe `low=0.015 high=0.95 max_std=0.03`:

- selected: 67,883
- positives: 208
- negatives: 67,675

The second file is:

`data/raw/train_phase3_pseudo_095.csv`

Use pseudo labels carefully: the stable agreement set is heavily negative, so a
blind retrain may improve calibration while damaging positive ranking.

## Current Conclusion

The remaining lift is coming from local corrections, not stronger global trees.
The best Phase 3 system improved from Phase 2 `0.946822` to `0.948983` OOF.

Next high-value work:

1. Submit the primary hill file plus `temp1.05`, `temp0.95`, rank, and logit.
2. If public LB rewards Phase 3, run a second KNN view with cosine or Manhattan
   overnight.
3. Add leaf-index embeddings from the CatBoost folds, then KNN in leaf space.
4. Retrain specialists using the `0.95` pseudo-label file with class-balanced
   weighting and compare OOF only in positive-heavy regions.
