# Experiment Log

## Environment

- System Python has `pandas`, `numpy`, `sklearn`, `lightgbm`, `xgboost`, and
  `torch`.
- `catboost`, `optuna`, and `shap` are not installed in the detected system
  Python, so the code supports them as optional upgrades but did not run them.
- Feature mode used for completed experiments: `balanced`, producing 243 base
  features before fold target encodings.

## Completed Runs

| Experiment | CV | Model | Fold AUC range | OOF AUC | Notes |
|---|---|---:|---:|---:|---|
| `20260518_145635_s0_stratified` | StratifiedKFold | LightGBM | 0.929554-0.935610 | 0.906572 | Strong fold scores, weaker pooled OOF suggests fold calibration/ranking shift. |
| `20260518_154412_s0_stratified` | StratifiedKFold | Logistic stack-style | 0.911795-0.913637 | 0.912429 | Very stable, stronger pooled OOF, useful leakage/linear-threshold detector. |

## Ensemble Results

| Ensemble tag | Members | Mean AUC | Rank AUC | Logit AUC | Hill AUC |
|---|---|---:|---:|---:|---:|
| `lgbm_logistic` | LGBM + Logistic | 0.918534 | 0.920314 | 0.920044 | 0.918684 |

Current best local OOF family is the rank ensemble:

`submissions/20260518_155711_lgbm_logistic_rank.csv`

Temperature variants were also emitted for public-LB probing:

- `temp0.85`
- `temp0.95`
- `temp1.05`
- `temp1.15`

## Next High-Value Iterations

1. Submit `rank`, `logit`, and `rank_temp1.05` from the `lgbm_logistic` family.
2. Run `race` and `driver` CV for logistic and LGBM to estimate shake-up risk.
3. Run XGBoost with the same features for a third tree-family member.
4. Install/enable CatBoost and Optuna, then tune categorical-heavy variants.
5. Run adversarial validation without `--skip-adv`; remove or downweight drifted
   features if train/test separation is high.
6. Use public LB feedback to choose calibration temperature and whether rank or
   logit averaging tracks the leaderboard better.
