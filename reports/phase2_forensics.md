# Phase 2 Forensics

## Temporal Reconstruction Probes
| keys                   | signal                  |   coverage |      auc |    exact |
|:-----------------------|:------------------------|-----------:|---------:|---------:|
| Year|Race|Driver|Stint | pit_shift_3             |   0.411837 | 0.59014  | 0.788349 |
| Year|Race|Driver|Stint | pit_shift_2             |   0.554932 | 0.582796 | 0.770301 |
| Year|Race|Driver|Stint | pit_shift_1             |   0.741388 | 0.567077 | 0.754049 |
| Race|Year|Position     | pit_shift_2             |   0.990966 | 0.55415  | 0.75271  |
| Race|Year|Position     | pit_shift_3             |   0.986576 | 0.552886 | 0.751545 |
| Race|Year|Position     | pit_shift_1             |   0.995437 | 0.551911 | 0.751677 |
| Year|Race|Driver       | pit_shift_1             |   0.906934 | 0.551021 | 0.745947 |
| Year|Race|Driver       | pit_shift_2             |   0.823068 | 0.550382 | 0.740824 |
| Race|Year|Position     | physical_lap_plus_1_pit |   0.990363 | 0.550358 | 0.81386  |
| Race|Year|Position     | pit_shift_-3            |   0.986576 | 0.549688 | 0.752154 |
| Race|Year|Position     | pit_shift_-2            |   0.990966 | 0.549471 | 0.751508 |
| Year|Race|Driver       | pit_shift_3             |   0.744694 | 0.5465   | 0.733436 |
| Race|Year|Position     | pit_shift_-1            |   0.995437 | 0.545748 | 0.748721 |
| Year|Race|Driver       | pit_shift_-1            |   0.906934 | 0.542937 | 0.751993 |
| Year|Race|Driver       | pit_shift_-2            |   0.823068 | 0.536386 | 0.754077 |
| Year|Race|Driver       | physical_lap_plus_1_pit |   0.289771 | 0.53548  | 0.809477 |
| Year|Race|Driver|Stint | pit_shift_-1            |   0.741388 | 0.534195 | 0.776093 |
| Year|Race|Driver       | pit_shift_-3            |   0.744694 | 0.530896 | 0.757428 |
| Year|Race|Driver|Stint | pit_shift_-2            |   0.554932 | 0.530784 | 0.80187  |
| Year|Race|Driver|Stint | physical_lap_plus_1_pit |   0.207421 | 0.526669 | 0.85516  |
| Year|Race|Driver|Stint | pit_shift_-3            |   0.411837 | 0.523378 | 0.823222 |
| Race|Driver            | pit_shift_3             |   0.906214 | 0.517429 | 0.718541 |
| Race|Driver            | pit_shift_2             |   0.935084 | 0.517009 | 0.721216 |
| Race|Driver            | pit_shift_1             |   0.965974 | 0.514633 | 0.723323 |
| Race|Driver            | pit_shift_-2            |   0.935084 | 0.511871 | 0.725197 |
| Race|Driver            | physical_lap_plus_1_pit |   0.715003 | 0.511442 | 0.773551 |
| Race|Driver            | pit_shift_-3            |   0.906214 | 0.51071  | 0.725336 |
| Race|Driver            | pit_shift_-1            |   0.965974 | 0.510025 | 0.724009 |

## Best Single-Threshold Leakage Probes
| feature                | op   |   threshold |      auc |   positive_rate |
|:-----------------------|:-----|------------:|---------:|----------------:|
| Stint                  | <=   |    1        | 0.715015 |        0.492526 |
| LapNumber              | <=   |   19        | 0.666441 |        0.509637 |
| RaceProgress           | <=   |    0.276316 | 0.652537 |        0.510263 |
| TyreLife               | <=   |   13        | 0.645948 |        0.543458 |
| Year                   | <=   | 2023        | 0.64415  |        0.499012 |
| LapTime_Delta          | <=   |   -1.08     | 0.617017 |        0.44004  |
| Cumulative_Degradation | <=   |  -22.63     | 0.608088 |        0.470023 |
| Position_Change        | <=   |    1        | 0.586611 |        0.730107 |
| LapTime (s)            | <=   |   83.889    | 0.546614 |        0.310029 |
| PitStop                | <=   |    0        | 0.520858 |        0.863882 |
| Position               | <=   |    7        | 0.516243 |        0.384281 |

## Exact/Bucket Value Spikes
| feature         | value              |   count |       mean |   lift_abs |
|:----------------|:-------------------|--------:|-----------:|-----------:|
| TyreLife        | 61.0               |      43 | 0.906977   |   0.707995 |
| TyreLife        | 56.0               |      84 | 0.869048   |   0.670066 |
| TyreLife        | 59.0               |      44 | 0.840909   |   0.641927 |
| TyreLife        | 65.0               |      49 | 0.836735   |   0.637753 |
| TyreLife        | 63.0               |      52 | 0.826923   |   0.627941 |
| TyreLife        | 64.0               |      65 | 0.815385   |   0.616403 |
| TyreLife        | 60.0               |      65 | 0.815385   |   0.616403 |
| TyreLife        | 66.0               |      58 | 0.810345   |   0.611363 |
| TyreLife        | 57.0               |      54 | 0.796296   |   0.597314 |
| TyreLife        | 58.0               |      42 | 0.738095   |   0.539113 |
| RaceProgress    | 0.6753246753246753 |     179 | 0.631285   |   0.432303 |
| RaceProgress    | 0.7631578947368421 |     216 | 0.615741   |   0.416759 |
| RaceProgress    | 0.5974025974025974 |     360 | 0.613889   |   0.414907 |
| RaceProgress    | 0.6493506493506493 |     417 | 0.611511   |   0.412529 |
| RaceProgress    | 0.7922077922077922 |      63 | 0.603175   |   0.404193 |
| RaceProgress    | 0.7051282051282052 |     234 | 0.598291   |   0.399308 |
| RaceProgress    | 0.5584415584415584 |     265 | 0.596226   |   0.397244 |
| RaceProgress    | 0.7402597402597403 |      92 | 0.586957   |   0.387974 |
| RaceProgress    | 0.4666666666666667 |      41 | 0.585366   |   0.386384 |
| RaceProgress    | 0.5324675324675324 |     260 | 0.584615   |   0.385633 |
| Driver          | VET                |     359 | 0.56546    |   0.366478 |
| Driver          | MSC                |     355 | 0.473239   |   0.274257 |
| Driver          | HAD                |     435 | 0.462069   |   0.263087 |
| Driver          | STR                |    1275 | 0.427451   |   0.228469 |
| Driver          | ANT                |     417 | 0.410072   |   0.21109  |
| Driver          | LAT                |     401 | 0.40399    |   0.205008 |
| Driver          | BEA                |     555 | 0.4        |   0.201018 |
| Driver          | D439               |      26 | 0          |   0.198982 |
| Stint           | 7.0                |     116 | 0          |   0.198982 |
| Driver          | D433               |      32 | 0          |   0.198982 |
| Driver          | D436               |      26 | 0          |   0.198982 |
| Position_Change | 16.0               |      94 | 0.393617   |   0.194635 |
| Stint           | 2.0                |  129536 | 0.391104   |   0.192122 |
| Position_Change | 14.0               |     625 | 0.3888     |   0.189818 |
| Race            | Chinese Grand Prix |    7311 | 0.388593   |   0.18961  |
| Year            | 2023.0             |  136147 | 0.00960726 |   0.189375 |
| LapNumber       | 36.0               |    6809 | 0.387135   |   0.188153 |
| LapNumber       | 50.0               |    4792 | 0.38293    |   0.183948 |
| LapNumber       | 38.0               |    6334 | 0.381118   |   0.182136 |
| Stint           | 6.0                |     728 | 0.0192308  |   0.179751 |

## Adversarial Train/Test Drift
Mean adversarial AUC: 0.999992
| feature                                              |   adv_importance |   adv_auc |
|:-----------------------------------------------------|-----------------:|----------:|
| id                                                   |      0.371       |  0.999992 |
| id_norm                                              |      0.0511667   |  0.999992 |
| Driver_Stint_hash                                    |      9.72222e-05 |  0.999992 |
| seq_Year__Race__Position_LapTime_Delta_roll3_mean    |      9.72222e-05 |  0.999992 |
| Driver_Race_hash                                     |      8.33333e-05 |  0.999992 |
| Driver_LapTime (s)_z                                 |      8.33333e-05 |  0.999992 |
| seq_Year__Race__Driver_Position_roll3_mean           |      6.94444e-05 |  0.999992 |
| Race__Year_LapNumber_z                               |      6.94444e-05 |  0.999992 |
| Driver_Compound_hash                                 |      5.55556e-05 |  0.999992 |
| seq_Year__Race__Position_Cumulative_Degradation_lag1 |      5.55556e-05 |  0.999992 |
| Driver__Race_Position_z                              |      5.55556e-05 |  0.999992 |
| seq_Year__Race__Driver_TyreLife_diff1                |      5.55556e-05 |  0.999992 |
| Driver__Race_LapTime_Delta_z                         |      5.55556e-05 |  0.999992 |
| Driver_freq                                          |      5.55556e-05 |  0.999992 |
| Driver__Compound_LapNumber_mean                      |      5.55556e-05 |  0.999992 |
| Race_Compound_hash                                   |      5.55556e-05 |  0.999992 |
| Driver_Year_hash                                     |      5.55556e-05 |  0.999992 |
| Driver__Race_TyreLife_z                              |      4.16667e-05 |  0.999992 |
| tyre_life_frac                                       |      4.16667e-05 |  0.999992 |
| Driver__Race_Cumulative_Degradation_z                |      4.16667e-05 |  0.999992 |
| Race_LapTime_Delta_z                                 |      4.16667e-05 |  0.999992 |
| Stint__Compound_Cumulative_Degradation_mean          |      4.16667e-05 |  0.999992 |
| Driver__Race_TyreLife_mean                           |      4.16667e-05 |  0.999992 |
| Race__Year__Compound_count                           |      4.16667e-05 |  0.999992 |
| Position_Change                                      |      4.16667e-05 |  0.999992 |
| Race__Year__Compound_LapTime_Delta_mean              |      4.16667e-05 |  0.999992 |
| race_compound_entropy                                |      4.16667e-05 |  0.999992 |
| laps_remaining_est                                   |      4.16667e-05 |  0.999992 |
| seq_Year__Race__Position_LapTime_Delta_ewm_a05       |      4.16667e-05 |  0.999992 |
| seq_Year__Race__Driver_PitStop_roll3_mean            |      4.16667e-05 |  0.999992 |
| seq_Year__Race__Position_order                       |      4.16667e-05 |  0.999992 |
| lap_sin                                              |      4.16667e-05 |  0.999992 |
| Race__Compound_Cumulative_Degradation_mean           |      2.77778e-05 |  0.999992 |
| Driver_Cumulative_Degradation_mean                   |      2.77778e-05 |  0.999992 |
| Race__Year_TyreLife_mean                             |      2.77778e-05 |  0.999992 |
| seq_Year__Race__Position_LapTime (s)_ewm_a05         |      2.77778e-05 |  0.999992 |
| Race__Year_LapTime (s)_mean                          |      2.77778e-05 |  0.999992 |
| Race__Year_LapTime_Delta_z                           |      2.77778e-05 |  0.999992 |
| Race__Compound_TyreLife_mean                         |      2.77778e-05 |  0.999992 |
| Race__Compound_LapTime_Delta_z                       |      2.77778e-05 |  0.999992 |