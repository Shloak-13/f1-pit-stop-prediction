# Phase 6 Pairwise Ranking Report

## Ranker OOF
| model   | query          |   oof_auc | fold_auc                                                     |
|:--------|:---------------|----------:|:-------------------------------------------------------------|
| xgb     | RaceYearStint  |  0.930978 | [0.9303521558806228, 0.9312554163310738, 0.9315312175261535] |
| xgb     | BoundaryBucket |  0.943934 | [0.9438893757167872, 0.9434676637711262, 0.9445681745879247] |

## Best Rank Blends
| candidate                            | anchor   | ranker                  |   alpha |   oof_auc |   mean_abs_delta |
|:-------------------------------------|:---------|:------------------------|--------:|----------:|-----------------:|
| hard_rank_xgb_RaceYearStint_a0p03    | hard     | rank_xgb_RaceYearStint  |    0.03 |  0.94916  |       0.00188983 |
| hard_rank_xgb_RaceYearStint_a0p05    | hard     | rank_xgb_RaceYearStint  |    0.05 |  0.949158 |       0.00314972 |
| hard_rank_xgb_BoundaryBucket_a0p03   | hard     | rank_xgb_BoundaryBucket |    0.03 |  0.949137 |       0.00154922 |
| hard_rank_xgb_BoundaryBucket_a0p05   | hard     | rank_xgb_BoundaryBucket |    0.05 |  0.949133 |       0.00258204 |
| hard_rank_xgb_RaceYearStint_a0p08    | hard     | rank_xgb_RaceYearStint  |    0.08 |  0.949129 |       0.00503956 |
| hard_rank_xgb_BoundaryBucket_a0p08   | hard     | rank_xgb_BoundaryBucket |    0.08 |  0.949121 |       0.00413126 |
| hard_rank_xgb_BoundaryBucket_a0p12   | hard     | rank_xgb_BoundaryBucket |    0.12 |  0.94909  |       0.0061969  |
| phase3_rank_xgb_RaceYearStint_a0p05  | phase3   | rank_xgb_RaceYearStint  |    0.05 |  0.949043 |       0.00317599 |
| hard_rank_xgb_RaceYearStint_a0p12    | hard     | rank_xgb_RaceYearStint  |    0.12 |  0.949039 |       0.00755934 |
| phase3_rank_xgb_RaceYearStint_a0p08  | phase3   | rank_xgb_RaceYearStint  |    0.08 |  0.949036 |       0.00508158 |
| phase3_rank_xgb_RaceYearStint_a0p03  | phase3   | rank_xgb_RaceYearStint  |    0.03 |  0.94903  |       0.00190559 |
| hard_rank_xgb_BoundaryBucket_a0p18   | hard     | rank_xgb_BoundaryBucket |    0.18 |  0.949012 |       0.00929534 |
| phase3_rank_xgb_BoundaryBucket_a0p08 | phase3   | rank_xgb_BoundaryBucket |    0.08 |  0.949005 |       0.00416235 |
| phase3_rank_xgb_BoundaryBucket_a0p05 | phase3   | rank_xgb_BoundaryBucket |    0.05 |  0.949004 |       0.00260147 |
| phase3_rank_xgb_BoundaryBucket_a0p03 | phase3   | rank_xgb_BoundaryBucket |    0.03 |  0.948999 |       0.00156088 |
| phase3_rank_xgb_BoundaryBucket_a0p12 | phase3   | rank_xgb_BoundaryBucket |    0.12 |  0.94899  |       0.00624352 |
| phase3_rank_xgb_RaceYearStint_a0p12  | phase3   | rank_xgb_RaceYearStint  |    0.12 |  0.948975 |       0.00762238 |
| phase3_rank_xgb_BoundaryBucket_a0p18 | phase3   | rank_xgb_BoundaryBucket |    0.18 |  0.948935 |       0.00936528 |
| hard_rank_xgb_BoundaryBucket_a0p25   | hard     | rank_xgb_BoundaryBucket |    0.25 |  0.948872 |       0.0129102  |
| phase3_rank_xgb_BoundaryBucket_a0p25 | phase3   | rank_xgb_BoundaryBucket |    0.25 |  0.948819 |       0.0130073  |
| hard_rank_xgb_RaceYearStint_a0p18    | hard     | rank_xgb_RaceYearStint  |    0.18 |  0.948791 |       0.011339   |
| phase3_rank_xgb_RaceYearStint_a0p18  | phase3   | rank_xgb_RaceYearStint  |    0.18 |  0.948766 |       0.0114336  |
| phase3_rank_xgb_RaceYearStint_a0p25  | phase3   | rank_xgb_RaceYearStint  |    0.25 |  0.948338 |       0.0158799  |
| hard_rank_xgb_RaceYearStint_a0p25    | hard     | rank_xgb_RaceYearStint  |    0.25 |  0.94832  |       0.0157486  |
| phase2_rank_xgb_RaceYearStint_a0p18  | phase2   | rank_xgb_RaceYearStint  |    0.18 |  0.94765  |       0.0136823  |
| phase2_rank_xgb_BoundaryBucket_a0p25 | phase2   | rank_xgb_BoundaryBucket |    0.25 |  0.947624 |       0.015051   |
| phase2_rank_xgb_RaceYearStint_a0p25  | phase2   | rank_xgb_RaceYearStint  |    0.25 |  0.947578 |       0.0190032  |
| phase2_rank_xgb_RaceYearStint_a0p12  | phase2   | rank_xgb_RaceYearStint  |    0.12 |  0.94753  |       0.00912152 |
| phase2_rank_xgb_BoundaryBucket_a0p18 | phase2   | rank_xgb_BoundaryBucket |    0.18 |  0.947492 |       0.0108367  |
| phase2_rank_xgb_RaceYearStint_a0p08  | phase2   | rank_xgb_RaceYearStint  |    0.08 |  0.94736  |       0.00608102 |

## Persistent Inversion Regions
| field           | group                         |    n |     target |   base_auc |   approx_inversion |
|:----------------|:------------------------------|-----:|-----------:|-----------:|-------------------:|
| SyntheticBucket | 2023_1_MEDIUM_7_17            |  556 | 0.00179856 | 0.00540541 |           1        |
| SyntheticBucket | 2023_2_HARD_2_18              |  450 | 0.00222222 | 0.0356347  |           1        |
| SyntheticBucket | 2023_2_HARD_2_19              |  236 | 0.00423729 | 0.0340426  |           1        |
| RaceYearStint   | British Grand Prix_2023_4     |  215 | 0.00465116 | 0.0934579  |           0.947368 |
| RaceYearStint   | Belgian Grand Prix_2023_4     |  131 | 0.00763359 | 0.138462   |           0.894737 |
| SyntheticBucket | 2023_3_HARD_3_22              |  321 | 0.00311526 | 0.16875    |           0.842105 |
| SyntheticBucket | 2025_1_MEDIUM_9_2             |  180 | 0.00555556 | 0.184358   |           0.842105 |
| SyntheticBucket | 2022_1_MEDIUM_8_24            |  128 | 0.015625   | 0.230159   |           0.797784 |
| SyntheticBucket | 2022_1_MEDIUM_7_6             |  158 | 0.00632911 | 0.203822   |           0.789474 |
| SyntheticBucket | 2023_1_SOFT_7_19              |  137 | 0.00729927 | 0.205882   |           0.789474 |
| SyntheticBucket | 2024_1_MEDIUM_6_0             |  401 | 0.00498753 | 0.253133   |           0.750693 |
| SyntheticBucket | 2022_1_MEDIUM_8_6             |  161 | 0.00621118 | 0.3        |           0.736842 |
| SyntheticBucket | 2022_1_MEDIUM_7_8             |  253 | 0.00395257 | 0.349206   |           0.684211 |
| SyntheticBucket | 2023_3_HARD_4_22              |  177 | 0.00564972 | 0.318182   |           0.684211 |
| SyntheticBucket | 2023_1_MEDIUM_9_18            |  490 | 0.00204082 | 0.368098   |           0.631579 |
| SyntheticBucket | 2023_2_HARD_2_20              |  137 | 0.00729927 | 0.360294   |           0.631579 |
| SyntheticBucket | 2022_1_MEDIUM_5_27            |  139 | 0.028777   | 0.418519   |           0.603878 |
| SyntheticBucket | 2023_2_HARD_4_21              | 1069 | 0.00280636 | 0.419637   |           0.592798 |
| SyntheticBucket | 2023_1_MEDIUM_9_17            |  608 | 0.00164474 | 0.415157   |           0.578947 |
| SyntheticBucket | 2023_2_HARD_5_20              |  174 | 0.00574713 | 0.439306   |           0.578947 |
| SyntheticBucket | 2023_1_MEDIUM_8_22            |  680 | 0.00294118 | 0.426254   |           0.570637 |
| RaceYearStint   | Mexico City Grand Prix_2023_4 |  437 | 0.00686499 | 0.442396   |           0.537396 |
| SyntheticBucket | 2022_1_MEDIUM_9_8             |  243 | 0.0246914  | 0.478903   |           0.512465 |
| SyntheticBucket | 2024_1_MEDIUM_9_12            |  194 | 0.0154639  | 0.469459   |           0.509695 |
| SyntheticBucket | 2025_1_MEDIUM_8_27            |  189 | 0.010582   | 0.483957   |           0.501385 |
| SyntheticBucket | 2023_2_HARD_3_13              |  327 | 0.00611621 | 0.486154   |           0.498615 |
| SyntheticBucket | 2022_1_MEDIUM_10_10           |  231 | 0.025974   | 0.504444   |           0.476454 |
| SyntheticBucket | 2023_3_HARD_3_17              |  304 | 0.00328947 | 0.531353   |           0.473684 |
| SyntheticBucket | 2022_1_MEDIUM_10_27           |  281 | 0.0284698  | 0.533883   |           0.468144 |
| SyntheticBucket | 2025_3_HARD_1_6               |  120 | 0.525      | 0.536619   |           0.459834 |
| SyntheticBucket | 2022_1_MEDIUM_6_11            |  240 | 0.025      | 0.536325   |           0.445983 |
| SyntheticBucket | 2024_2_HARD_1_26              |  128 | 0.21875    | 0.543571   |           0.445983 |
| SyntheticBucket | 2022_1_MEDIUM_8_9             |  150 | 0.02       | 0.519274   |           0.440443 |
| SyntheticBucket | 2024_3_HARD_2_5               |  140 | 0.664286   | 0.555251   |           0.440443 |
| SyntheticBucket | 2022_1_MEDIUM_6_3             |  276 | 0.0289855  | 0.543377   |           0.426593 |
| SyntheticBucket | 2022_1_MEDIUM_8_10            |  236 | 0.0338983  | 0.564145   |           0.423823 |
| SyntheticBucket | 2023_3_HARD_2_18              |  716 | 0.00139665 | 0.558042   |           0.421053 |
| SyntheticBucket | 2023_2_HARD_3_23              |  518 | 0.0019305  | 0.55706    |           0.421053 |
| SyntheticBucket | 2025_1_MEDIUM_6_19            |  266 | 0.0037594  | 0.596226   |           0.421053 |
| SyntheticBucket | 2023_1_MEDIUM_11_19           |  548 | 0.00547445 | 0.547401   |           0.404432 |
| SyntheticBucket | 2025_1_MEDIUM_11_2            |  137 | 0.189781   | 0.601178   |           0.401662 |
| SyntheticBucket | 2023_1_MEDIUM_5_23            |  272 | 0.0110294  | 0.594796   |           0.393352 |
| SyntheticBucket | 2024_1_MEDIUM_10_11           |  272 | 0.0183824  | 0.565543   |           0.393352 |
| SyntheticBucket | 2022_1_MEDIUM_8_3             |  265 | 0.0264151  | 0.578627   |           0.393352 |
| SyntheticBucket | 2022_2_HARD_2_29              |  229 | 0.371179   | 0.596977   |           0.393352 |
| SyntheticBucket | 2024_2_HARD_2_26              |  241 | 0.273859   | 0.5929     |           0.390582 |
| SyntheticBucket | 2022_1_MEDIUM_7_2             |  460 | 0.00434783 | 0.620087   |           0.387812 |
| SyntheticBucket | 2025_3_HARD_2_25              |  125 | 0.736      | 0.599473   |           0.387812 |
| SyntheticBucket | 2025_1_MEDIUM_7_26            |  540 | 0.00925926 | 0.565607   |           0.385042 |
| SyntheticBucket | 2024_3_HARD_3_28              |  184 | 0.88587    | 0.59889    |           0.385042 |

## Interpretation
- Ranker outputs are used as ordering scores, then rank-normalized before blending.
- Public-facing candidates use small alpha rank corrections over stable pointwise anchors.
- If high-alpha OOF winners fail public LB, submit lower-alpha phase2-anchor variants.