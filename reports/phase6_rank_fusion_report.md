# Phase 6 Rank Fusion

## Top Rank Fusion Candidates
| name                                   | anchor   | ranker                                                       |   alpha |   oof_auc |   mean_abs_delta |
|:---------------------------------------|:---------|:-------------------------------------------------------------|--------:|----------:|-----------------:|
| hard_lgbm_all_a0p08                    | hard     | lgbm_all                                                     |    0.08 |  0.949207 |      0.00448478  |
| hard_all_rankers_a0p08                 | hard     | all_rankers                                                  |    0.08 |  0.949197 |      0.00377113  |
| hard_lgbm_all_a0p05                    | hard     | lgbm_all                                                     |    0.05 |  0.949194 |      0.00280299  |
| hard_rank_lgbm_SyntheticBucket_a0p05   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.05 |  0.949187 |      0.00435018  |
| hard_all_rankers_a0p05                 | hard     | all_rankers                                                  |    0.05 |  0.949184 |      0.00235695  |
| hard_rank_lgbm_SyntheticBucket_a0p03   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.03 |  0.949179 |      0.00261011  |
| hard_lgbm_all_a0p03                    | hard     | lgbm_all                                                     |    0.03 |  0.949176 |      0.00168179  |
| hard_rank_lgbm_RaceYearStint_a0p03     | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.03 |  0.949171 |      0.00257138  |
| hard_rank_lgbm_SyntheticBucket_a0p02   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.02 |  0.949169 |      0.00174007  |
| hard_all_rankers_a0p03                 | hard     | all_rankers                                                  |    0.03 |  0.949168 |      0.00141417  |
| hard_rank_lgbm_RaceYearStint_a0p05     | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.05 |  0.949168 |      0.00428564  |
| hard_rank_lgbm_SyntheticBucket_a0p08   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.08 |  0.949167 |      0.0069603   |
| hard_xgb_all_a0p08                     | hard     | xgb_all                                                      |    0.08 |  0.949165 |      0.00352904  |
| hard_rank_lgbm_RaceYearStint_a0p02     | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.02 |  0.949165 |      0.00171425  |
| hard_lgbm_all_a0p02                    | hard     | lgbm_all                                                     |    0.02 |  0.949164 |      0.00112119  |
| hard_xgb_all_a0p05                     | hard     | xgb_all                                                      |    0.05 |  0.949161 |      0.00220565  |
| hard_rank_xgb_RaceYearStint_a0p03      | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_RaceYearStint       |    0.03 |  0.94916  |      0.00188983  |
| hard_all_rankers_a0p02                 | hard     | all_rankers                                                  |    0.02 |  0.949158 |      0.000942781 |
| hard_rank_xgb_RaceYearStint_a0p05      | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_RaceYearStint       |    0.05 |  0.949158 |      0.00314972  |
| hard_rank_xgb_RaceYearStint_a0p02      | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_RaceYearStint       |    0.02 |  0.949155 |      0.00125989  |
| hard_rank_lgbm_SyntheticBucket_a0p01   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.01 |  0.949154 |      0.000870037 |
| hard_xgb_all_a0p03                     | hard     | xgb_all                                                      |    0.03 |  0.949154 |      0.00132339  |
| hard_rank_lgbm_RaceYearStint_a0p01     | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.01 |  0.949153 |      0.000857127 |
| hard_rank_lgbm_BoundaryBucket_a0p05    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.05 |  0.949151 |      0.00275618  |
| hard_rank_lgbm_BoundaryBucket_a0p03    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.03 |  0.94915  |      0.00165371  |
| hard_xgb_all_a0p02                     | hard     | xgb_all                                                      |    0.02 |  0.949149 |      0.00088226  |
| hard_rank_lgbm_BoundaryBucket_a0p02    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.02 |  0.949147 |      0.00110247  |
| hard_rank_xgb_RaceYearStint_a0p01      | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_RaceYearStint       |    0.01 |  0.949147 |      0.000629945 |
| hard_rank_lgbm_BoundaryBucket_a0p01    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.01 |  0.949142 |      0.000551236 |
| hard_rank_xgb_BoundaryBucket_a0p02     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.02 |  0.949137 |      0.00103282  |
| hard_rank_xgb_BoundaryBucket_a0p03     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.03 |  0.949137 |      0.00154922  |
| hard_rank_xgb_BoundaryBucket_a0p01     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.01 |  0.949137 |      0.000516408 |
| hard_rank_lgbm_BoundaryBucket_a0p08    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.08 |  0.949136 |      0.00440988  |
| hard_rank_xgb_BoundaryBucket_a0p05     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.05 |  0.949133 |      0.00258204  |
| hard_rank_xgb_RaceYearStint_a0p08      | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_RaceYearStint       |    0.08 |  0.949129 |      0.00503956  |
| hard_rank_lgbm_RaceYearStint_a0p08     | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.08 |  0.949122 |      0.00685702  |
| hard_rank_xgb_BoundaryBucket_a0p08     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.08 |  0.949121 |      0.00413126  |
| phase3_lgbm_all_a0p08                  | phase3   | lgbm_all                                                     |    0.08 |  0.94912  |      0.00451591  |
| phase3_all_rankers_a0p08               | phase3   | all_rankers                                                  |    0.08 |  0.949103 |      0.00380901  |
| phase3_rank_lgbm_SyntheticBucket_a0p08 | phase3   | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.08 |  0.949099 |      0.00698575  |
| hard_rank_xgb_BoundaryBucket_a0p12     | hard     | 20260519_160100_phase6_xgbrank__rank_xgb_BoundaryBucket      |    0.12 |  0.94909  |      0.0061969   |
| phase3_rank_lgbm_SyntheticBucket_a0p05 | phase3   | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.05 |  0.949088 |      0.00436609  |
| hard_rank_lgbm_BoundaryBucket_a0p12    | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_BoundaryBucket  |    0.12 |  0.949088 |      0.00661483  |
| phase3_lgbm_all_a0p05                  | phase3   | lgbm_all                                                     |    0.05 |  0.949084 |      0.00282244  |
| hard_rank_lgbm_SyntheticBucket_a0p12   | hard     | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.12 |  0.949079 |      0.0104404   |
| phase3_all_rankers_a0p05               | phase3   | all_rankers                                                  |    0.05 |  0.949068 |      0.00238063  |
| phase3_rank_lgbm_RaceYearStint_a0p05   | phase3   | 20260519_153956_phase6_lambdarank__rank_lgbm_RaceYearStint   |    0.05 |  0.949061 |      0.00430383  |
| phase3_xgb_all_a0p08                   | phase3   | xgb_all                                                      |    0.08 |  0.94906  |      0.00357614  |
| phase3_rank_lgbm_SyntheticBucket_a0p03 | phase3   | 20260519_153956_phase6_lambdarank__rank_lgbm_SyntheticBucket |    0.03 |  0.949059 |      0.00261966  |
| phase3_lgbm_all_a0p03                  | phase3   | lgbm_all                                                     |    0.03 |  0.949049 |      0.00169346  |

## Public-LB Strategy
- Use phase2/stable-cat anchors for conservative leaderboard probes.
- Use hard-anchor variants only if public LB rewards Phase 4.5 style corrections.
- Prefer alpha 0.02-0.05 unless a ranker variant clearly transfers.