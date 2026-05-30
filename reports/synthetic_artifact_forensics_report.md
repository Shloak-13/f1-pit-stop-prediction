# Synthetic Artifact Forensics

## Quantization
| feature                |   top_exact_value |   top_exact_share | flag_gt5pct_same   |   round_share |   half_grid_share |   third_grid_share |   fifth_grid_share |   inferred_step |   inferred_step_share |   train_test_mean_drift_std |   artifact_score |   max_bin_share_0.001 |   n_bins_0.001 |   max_bin_share_0.01 |   n_bins_0.01 |   max_bin_share_0.1 |   n_bins_0.1 |
|:-----------------------|------------------:|------------------:|:-------------------|--------------:|------------------:|-------------------:|-------------------:|----------------:|----------------------:|----------------------------:|-----------------:|----------------------:|---------------:|---------------------:|--------------:|--------------------:|-------------:|
| PitStop                |             0     |       0.863882    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.000423032 |        1.56392   |           0.863882    |              2 |           0.863882   |             2 |           0.863882  |            2 |
| Stint                  |             1     |       0.492526    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00513113  |        1.19304   |           0.492526    |              8 |           0.492526   |             8 |           0.492526  |            8 |
| StintLapGap            |             0     |       0.454222    | True               |   0.999998    |        1          |         0.999998   |         0.999998   |           0.001 |             1         |                 0.00448262  |        1.15467   |           0.454222    |            106 |           0.454222   |           106 |           0.454222  |          106 |
| Position_Change        |             0     |       0.313495    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00131234  |        1.01363   |           0.313495    |             37 |           0.313495   |            37 |           0.313495  |           37 |
| PosChgBin              |             0     |       0.313495    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.000854037 |        1.01358   |           0.313495    |             21 |           0.313495   |            21 |           0.313495  |           21 |
| Year                   |          2023     |       0.310031    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00477761  |        1.01051   |           0.310031    |              4 |           0.310031   |             4 |           0.310031  |            4 |
| RaceLapsEst            |            72     |       0.213217    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00195305  |        0.913412  |           0.213217    |            123 |           0.213217   |           123 |           0.213217  |          123 |
| TyreLifeBin5           |             1     |       0.211003    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.000271992 |        0.911031  |           0.211003    |             16 |           0.211003   |            16 |           0.211003  |           16 |
| LapBin5                |             2     |       0.143515    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00383205  |        0.843898  |           0.143515    |             16 |           0.143515   |            16 |           0.143515  |           16 |
| TyreLifeFrac           |             1     |       0.454222    | True               |   0.463664    |        0.486057   |         0.48639    |         0.486448   |           0.001 |             0.573856  |                 0.00375541  |        0.838815  |           0.454222    |           1332 |           0.454222   |           282 |           0.467987  |           67 |
| TyreLifeBin3           |             2     |       0.130612    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 6.89077e-06 |        0.830613  |           0.130612    |             26 |           0.130612   |            26 |           0.130612  |           26 |
| RaceProgressBin20      |             1     |       0.113287    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.003745    |        0.813662  |           0.113287    |             20 |           0.113287   |            20 |           0.113287  |           20 |
| RacePhase20            |             1     |       0.113287    | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.003745    |        0.813662  |           0.113287    |             20 |           0.113287   |            20 |           0.113287  |           20 |
| TyreLifeFracBin12      |             5     |       0.0833333   | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 2.30927e-06 |        0.783334  |           0.0833333   |             12 |           0.0833333  |            12 |           0.0833333 |           12 |
| StintGapBin12          |             7     |       0.0833333   | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 2.30927e-06 |        0.783334  |           0.0833333   |             12 |           0.0833333  |            12 |           0.0833333 |           12 |
| Position               |             4     |       0.0575375   | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00488755  |        0.758026  |           0.0575375   |             20 |           0.0575375  |            20 |           0.0575375 |           20 |
| RacePhase50            |             7     |       0.0530628   | True               |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00376983  |        0.75344   |           0.0530628   |             50 |           0.0530628  |            50 |           0.0530628 |           50 |
| TyreLife               |             6     |       0.0470533   | False              |   0.999998    |        1          |         0.999998   |         0.999998   |           0.001 |             1         |                 0.000244265 |        0.747078  |           0.0470533   |             78 |           0.0470533  |            78 |           0.0470533 |           78 |
| LapNumber              |             1     |       0.0377055   | False              |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00328599  |        0.738034  |           0.0377055   |             78 |           0.0377055  |            78 |           0.0377055 |           78 |
| DegBin                 |            27     |       0.0333333   | False              |   1           |        1          |         1          |         1          |           0.001 |             1         |                 3.07002e-07 |        0.733333  |           0.0333333   |             30 |           0.0333333  |            30 |           0.0333333 |           30 |
| DeltaBin               |             8     |       0.0333333   | False              |   1           |        1          |         1          |         1          |           0.001 |             1         |                 3.07002e-07 |        0.733333  |           0.0333333   |             30 |           0.0333333  |            30 |           0.0333333 |           30 |
| LapsRemainingEst       |            65     |       0.026331    | False              |   1           |        1          |         1          |         1          |           0.001 |             1         |                 0.00381898  |        0.726713  |           0.026331    |            117 |           0.026331   |           117 |           0.026331  |          117 |
| LapTimeBin             |             4     |       0.0250011   | False              |   1           |        1          |         1          |         1          |           0.001 |             1         |                 6.90586e-07 |        0.725001  |           0.0250011   |             40 |           0.0250011  |            40 |           0.0250011 |           40 |
| LapTime_Delta          |             0     |       0.0488978   | False              |   0.0508843   |        0.0530571  |         0.0508843  |         0.0534987  |           0.001 |             0.99986   |                 0.00437444  |        0.559965  |           0.0488978   |          44438 |           0.0522339  |          9668 |           0.0945575 |         1472 |
| Cumulative_Degradation |             0     |       0.0238238   | False              |   0.0253162   |        0.0266792  |         0.0253162  |         0.0288217  |           0.001 |             0.99963   |                 0.0023323   |        0.529636  |           0.0238762   |          87362 |           0.0239263  |         26349 |           0.0247006 |         3995 |
| LapTime (s)            |            83.939 |       0.000582958 | False              |   0.00202932  |        0.00482222 |         0.00202932 |         0.00804872 |           0.001 |             0.999968  |                 0.00192861  |        0.50237   |           0.000582958 |          37690 |           0.00239559 |          6956 |           0.0118732 |          903 |
| TyreLife_RaceProgress  |            76     |       0.0538074   | True               |   0.541096    |        0.561747   |         0.567595   |         0.576804   |           0.005 |             0.647763  |                 0.0042147   |        0.493471  |           0.0986451   |          13851 |           0.0986451  |          7739 |           0.0986656 |         1523 |
| RaceProgress           |             0.5   |       0.0193332   | False              |   0.000883143 |        0.0202661  |         0.0136441  |         0.00842333 |           0.001 |             0.0795945 |                 0.00381323  |        0.063565  |           0.0193332   |            861 |           0.0297172  |           100 |           0.200471  |           11 |
| CurrentLap_RaceLapsEst |             0.5   |       0.0193424   | False              |   0.000886331 |        0.0202788  |         0.0136712  |         0.00838986 |           0.001 |             0.0793139 |                 0.00381002  |        0.0634361 |           0.0193424   |            876 |           0.0297172  |           100 |           0.200469  |           11 |

## Template Collisions Round 2
|                index |   count |   mean |   entropy |   purity |   round_decimals |   test_count | train_only   | test_only   | deterministic   | near_deterministic   |
|---------------------:|--------:|-------:|----------:|---------:|-----------------:|-------------:|:-------------|:------------|:----------------|:---------------------|
| 10000018714149332010 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000026776315991483 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000068491100247864 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000087890132588359 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000125839564291537 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000137306212065559 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000261294361049206 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000277328252513689 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000305957023223813 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000359703193422342 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000435262512179535 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000493695646405881 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000494752382386012 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000584688769712652 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000665586112443362 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000701012201588298 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000719889931784228 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000731034624657696 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
|  1000073761180232308 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000751025816927531 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000754950267384202 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000760010530244393 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000775028831469626 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000796609370266414 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000838274972049953 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000843359710808664 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000852832024542640 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000900911128691624 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000952579973957461 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10000969512577180248 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001003150543963496 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001049870507388092 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001053331923144786 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001058737847669694 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001071417909146143 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
|  1000107249234196323 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001124794139537763 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
|  1000112919977773116 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001131381552809967 |       1 |      1 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |
| 10001170885048102658 |       1 |      0 | 3.134e-08 |        1 |                2 |            0 | True         | False       | False           | False                |

## Template Collisions Round 3
|                index |   count |   mean |   entropy |   purity |   round_decimals |   test_count | train_only   | test_only   | deterministic   | near_deterministic   |
|---------------------:|--------:|-------:|----------:|---------:|-----------------:|-------------:|:-------------|:------------|:----------------|:---------------------|
| 10000005958792008065 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000027454105336591 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000047187274390573 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000138078211615539 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000192269802992420 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000373496844445504 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
|   100004178267205596 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000485257997780226 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000499797164146411 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000532086700826208 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000557902441709576 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000573154010975643 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
|  1000059555100053627 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000687634346071793 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000854175534115746 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000878627028546726 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10000966846208108589 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001142476292354506 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001155549444625371 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001189878422287298 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001224926773943462 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001240746187138943 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001281432190760517 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001300217661648764 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001328436397984820 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001376209739200194 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001420919603549751 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001448316235182864 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001529533011757724 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001591488468896059 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
|  1000161674567121020 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001663836300372705 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001665274610185164 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001877774823159181 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001880189017366336 |       1 |      1 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
|  1000191240490848742 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001938153240461922 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001950824534460896 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001976120903654882 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |
| 10001981434493210798 |       1 |      0 | 3.134e-08 |        1 |                3 |            0 | True         | False       | False           | False                |

## Conditional Entropy Pairs
| feature_a         | feature_b              |   bins |   conditional_entropy |   joint_auc |   n_bins |
|:------------------|:-----------------------|-------:|----------------------:|------------:|---------:|
| Year              | TyreLifeFracBin12      |     10 |              0.486369 |    0.871229 |       40 |
| Year              | TyreLife_RaceProgress  |     10 |              0.49787  |    0.865385 |       40 |
| Year              | Stint                  |     10 |              0.487257 |    0.859628 |       28 |
| Year              | StintGapBin12          |     10 |              0.506856 |    0.853533 |       40 |
| Year              | TyreLife_RaceProgress  |      5 |              0.516276 |    0.850641 |       20 |
| RaceLapsEst       | TyreLifeFracBin12      |     10 |              0.52197  |    0.848483 |       98 |
| RaceLapsEst       | TyreLife_RaceProgress  |     10 |              0.517868 |    0.847771 |      100 |
| Year              | Stint                  |      5 |              0.52261  |    0.841475 |       20 |
| DeltaBin          | TyreLifeFracBin12      |     10 |              0.52879  |    0.841028 |      100 |
| Stint             | RaceLapsEst            |     10 |              0.525014 |    0.840183 |       73 |
| Year              | CurrentLap_RaceLapsEst |     10 |              0.531944 |    0.839116 |       40 |
| Year              | RacePhase20            |     10 |              0.532786 |    0.838448 |       40 |
| Year              | RaceProgressBin20      |     10 |              0.532786 |    0.838448 |       40 |
| Year              | CurrentLap_RaceLapsEst |      5 |              0.534788 |    0.835519 |       20 |
| Year              | RacePhase20            |      5 |              0.535541 |    0.834829 |       20 |
| Year              | RaceProgressBin20      |      5 |              0.535541 |    0.834829 |       20 |
| RaceLapsEst       | StintGapBin12          |     10 |              0.539835 |    0.833114 |       99 |
| Stint             | TyreLife_RaceProgress  |     10 |              0.541889 |    0.832071 |       64 |
| Stint             | DeltaBin               |     10 |              0.531423 |    0.831705 |       76 |
| Year              | TyreLifeFracBin12      |      5 |              0.533508 |    0.831541 |       20 |
| Year              | Compound               |      5 |              0.539816 |    0.829615 |       20 |
| Year              | Compound               |     10 |              0.539816 |    0.829615 |       20 |
| Year              | StintGapBin12          |      5 |              0.542104 |    0.829042 |       20 |
| RaceLapsEst       | TyreLife_RaceProgress  |      5 |              0.543481 |    0.826992 |       25 |
| Year              | TyreLifeBin3           |     10 |              0.545646 |    0.826216 |       40 |
| DeltaBin          | TyreLife_RaceProgress  |     10 |              0.551245 |    0.824814 |      100 |
| DeltaBin          | StintGapBin12          |     10 |              0.548475 |    0.824361 |      100 |
| Year              | TyreLifeBin5           |     10 |              0.549845 |    0.822958 |       40 |
| Year              | TyreLifeBin3           |      5 |              0.549484 |    0.821913 |       20 |
| Year              | TyreLifeBin5           |      5 |              0.552346 |    0.819741 |       20 |
| TyreLifeFracBin12 | TyreLife_RaceProgress  |     10 |              0.562653 |    0.814724 |       82 |
| StintGapBin12     | TyreLife_RaceProgress  |     10 |              0.565404 |    0.812069 |       85 |
| Stint             | RaceLapsEst            |      5 |              0.565826 |    0.811761 |       25 |
| DeltaBin          | CurrentLap_RaceLapsEst |     10 |              0.568215 |    0.811136 |      100 |
| DeltaBin          | RaceProgressBin20      |     10 |              0.569411 |    0.810153 |      100 |
| RacePhase20       | DeltaBin               |     10 |              0.569411 |    0.810153 |      100 |
| RaceLapsEst       | CurrentLap_RaceLapsEst |     10 |              0.571662 |    0.807282 |      100 |
| Stint             | TyreLifeFracBin12      |     10 |              0.565964 |    0.807269 |       65 |
| RaceLapsEst       | RacePhase20            |     10 |              0.572443 |    0.806611 |      100 |
| RaceLapsEst       | RaceProgressBin20      |     10 |              0.572443 |    0.806611 |      100 |

## Deterministic Buckets
| feature                |     value |   count |   target_rate |   entropy | deterministic   | present_in_test   |   train_only_values |   test_only_values |
|:-----------------------|----------:|--------:|--------------:|----------:|:----------------|:------------------|--------------------:|-------------------:|
| RaceProgress           | 0.173077  |     434 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.173077  |     434 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.096     |     416 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0961538 |     414 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0961538 |     413 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.098     |     401 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0980392 |     385 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0980392 |     385 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.134615  |     384 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.135     |     384 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.134615  |     382 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.942     |     371 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.058     |     369 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.981     |     360 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.828     |     348 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0178571 |     347 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.827586  |     347 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0178571 |     347 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.827586  |     345 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0196078 |     321 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0196078 |     321 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.942308  |     319 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.942308  |     317 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.81      |     314 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.810345  |     313 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.810345  |     310 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.982     |     305 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.98      |     303 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.980769  |     289 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.980769  |     289 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.826923  |     277 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.826923  |     276 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0576923 |     270 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.839     |     267 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0576923 |     265 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.226     |     261 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.529412  |     258 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.529412  |     257 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.0689655 |     249 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0689655 |     248 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.961538  |     235 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.961538  |     235 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.129032  |     221 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.129032  |     221 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.982456  |     217 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.352941  |     216 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.0967742 |     215 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.352941  |     214 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.353     |     214 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.411765  |     213 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.412     |     213 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.411765  |     212 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.112903  |     210 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.112903  |     209 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.982456  |     209 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.16129   |     208 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.16129   |     207 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| CurrentLap_RaceLapsEst | 0.941176  |     205 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress_r3        | 0.941     |     202 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |
| RaceProgress           | 0.941176  |     201 |             0 | 3.134e-08 | True            | True              |                 nan |                nan |

## Interactions
| interaction                             |      auc |   n_bins |    entropy |   min_rate |   max_rate |   deterministic_bins |
|:----------------------------------------|---------:|---------:|-----------:|-----------:|-----------:|---------------------:|
| Race x Year x Stint                     | 0.898307 |      528 |   0.440359 |          0 |          1 |                   27 |
| Race x Year x RacePhase20               | 0.891171 |     1930 |   0.454089 |          0 |          1 |                   84 |
| Compound x DeltaBin x TyreLifeFracBin12 | 0.878266 |     1641 |   0.473797 |          0 |          1 |                  107 |
| Year x Compound x TyreLifeBin3          | 0.874416 |      357 |   0.483819 |          0 |          1 |                    8 |
| Year x Stint x CurrentLap_RaceLapsEst   | 0.874144 |      212 |   0.475685 |          0 |          1 |                   14 |
| Year x Compound x TyreLifeBin5          | 0.872694 |      225 |   0.486628 |          0 |          1 |                    5 |
| Compound x RacePhase20 x StintGapBin12  | 0.812175 |      880 |   0.564219 |          0 |          1 |                   48 |
| ratio::TyreLifeFrac                     | 0.766861 |       20 | nan        |        nan |        nan |                  nan |
| RaceProgressBin20 x Position x Compound | 0.764379 |     1743 |   0.612261 |          0 |          1 |                   82 |
| ratio::StintLapGap                      | 0.742686 |       20 | nan        |        nan |        nan |                  nan |
| ratio::TyreLife_RaceProgress            | 0.73266  |       20 | nan        |        nan |        nan |                  nan |
| ratio::CurrentLap_RaceLapsEst           | 0.714816 |       20 | nan        |        nan |        nan |                  nan |

## Additive Value vs CatBoost
| feature                    |   standalone_auc |   base_auc |   best_blend_alpha |   best_blend_auc |   additive_gain |
|:---------------------------|-----------------:|-----------:|-------------------:|-----------------:|----------------:|
| collision_r2_prior_oof     |         0.499994 |   0.946625 |               0.28 |         0.946625 |     1.20376e-08 |
| collision_r3_prior_oof     |         0.499994 |   0.946625 |               0.28 |         0.946625 |     1.20376e-08 |
| best_pair_prior_oof        |         0.876981 |   0.946625 |               0    |         0.946625 |     0           |
| best_interaction_prior_oof |         0.8974   |   0.946625 |               0    |         0.946625 |     0           |