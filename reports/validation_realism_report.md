# Validation Realism Reset

## System Comparison
| prediction       |   global_auc |   Year_macro_auc |   Year_p10_auc |   Year_std |   Year_n |   Race_macro_auc |   Race_p10_auc |   Race_std |   Race_n |   Driver_macro_auc |   Driver_p10_auc |   Driver_std |   Driver_n |   RaceYear_macro_auc |   RaceYear_p10_auc |   RaceYear_std |   RaceYear_n |   YearStint_macro_auc |   YearStint_p10_auc |   YearStint_std |   YearStint_n |   RaceYearStint_macro_auc |   RaceYearStint_p10_auc |   RaceYearStint_std |   RaceYearStint_n |   RaceYearCompound_macro_auc |   RaceYearCompound_p10_auc |   RaceYearCompound_std |   RaceYearCompound_n |   SyntheticBucket_macro_auc |   SyntheticBucket_p10_auc |   SyntheticBucket_std |   SyntheticBucket_n |
|:-----------------|-------------:|-----------------:|---------------:|-----------:|---------:|-----------------:|---------------:|-----------:|---------:|-------------------:|-----------------:|-------------:|-----------:|---------------------:|-------------------:|---------------:|-------------:|----------------------:|--------------------:|----------------:|--------------:|--------------------------:|------------------------:|--------------------:|------------------:|-----------------------------:|---------------------------:|-----------------------:|---------------------:|----------------------------:|--------------------------:|----------------------:|--------------------:|
| cat              |     0.946625 |         0.920252 |       0.912409 | 0.00721479 |        4 |         0.940487 |       0.913392 |  0.0184595 |       26 |           0.946403 |         0.929514 |    0.017069  |        458 |             0.904743 |           0.851629 |      0.0404939 |           94 |              0.872131 |            0.81544  |       0.04959   |            22 |                  0.83988  |                0.750725 |           0.089989  |               312 |                     0.863168 |                   0.779718 |              0.063769  |                  208 |                    0.765736 |                  0.640904 |              0.118632 |                 631 |
| phase2_cat_table |     0.946822 |         0.921105 |       0.912818 | 0.00755436 |        4 |         0.940752 |       0.913818 |  0.018451  |       26 |           0.94661  |         0.929807 |    0.0167965 |        458 |             0.905568 |           0.851729 |      0.039979  |           94 |              0.872755 |            0.816039 |       0.0501613 |            22 |                  0.840913 |                0.755145 |           0.0878231 |               312 |                     0.864376 |                   0.781163 |              0.063604  |                  208 |                    0.772235 |                  0.645812 |              0.11631  |                 631 |
| phase3_hill      |     0.948983 |         0.925534 |       0.915724 | 0.00929385 |        4 |         0.943049 |       0.916264 |  0.0178591 |       26 |           0.948976 |         0.932489 |    0.0160257 |        458 |             0.909094 |           0.854425 |      0.0393373 |           94 |              0.876311 |            0.816861 |       0.0512096 |            22 |                  0.846684 |                0.76032  |           0.0785027 |               312 |                     0.868305 |                   0.785789 |              0.063106  |                  208 |                    0.781726 |                  0.65641  |              0.115328 |                 631 |
| phase45_hard     |     0.949135 |         0.925643 |       0.915947 | 0.00915453 |        4 |         0.94319  |       0.916274 |  0.0177924 |       26 |           0.949135 |         0.932689 |    0.0161117 |        458 |             0.909089 |           0.853953 |      0.0394557 |           94 |              0.870532 |            0.813508 |       0.0593686 |            22 |                  0.846168 |                0.760628 |           0.0793244 |               312 |                     0.868184 |                   0.785297 |              0.0630555 |                  208 |                    0.781945 |                  0.655008 |              0.115186 |                 631 |

## LB-Aligned Candidates
| candidate        |   global_auc |   Year_macro_auc |   Year_p10_auc |   Year_std |   RaceYear_macro_auc |   RaceYear_p10_auc |   RaceYear_std |   YearStint_macro_auc |   YearStint_p10_auc |   YearStint_std |   RaceYearStint_macro_auc |   RaceYearStint_p10_auc |   RaceYearStint_std |   SyntheticBucket_macro_auc |   SyntheticBucket_p10_auc |   SyntheticBucket_std |   mean_abs_phase45_delta |   mean_abs_phase2_delta |
|:-----------------|-------------:|-----------------:|---------------:|-----------:|---------------------:|-------------------:|---------------:|----------------------:|--------------------:|----------------:|--------------------------:|------------------------:|--------------------:|----------------------------:|--------------------------:|----------------------:|-------------------------:|------------------------:|
| logit_stable     |     0.948038 |         0.924277 |       0.914475 | 0.00930775 |             0.908077 |           0.854942 |      0.0393607 |              0.874387 |            0.818378 |       0.0514945 |                  0.845084 |                0.756736 |           0.0798597 |                    0.781101 |                  0.651243 |              0.115687 |               0.0119654  |              0.00955141 |
| rank_stable      |     0.947904 |         0.924126 |       0.914251 | 0.00937997 |             0.907913 |           0.854636 |      0.0393942 |              0.873805 |            0.818195 |       0.0516387 |                  0.844648 |                0.75621  |           0.0804714 |                    0.780731 |                  0.653123 |              0.114971 |               0.301382   |              0.301735   |
| stable_mean      |     0.947732 |         0.923445 |       0.914058 | 0.00876501 |             0.907419 |           0.854518 |      0.0395719 |              0.874341 |            0.817908 |       0.0507462 |                  0.843824 |                0.756624 |           0.081932  |                    0.779394 |                  0.648934 |              0.115787 |               0.0137918  |              0.00571254 |
| phase45_shrink20 |     0.949042 |         0.925584 |       0.915814 | 0.00924669 |             0.909137 |           0.854349 |      0.0393376 |              0.87559  |            0.817012 |       0.0515708 |                  0.846694 |                0.760288 |           0.0784894 |                    0.781864 |                  0.656257 |              0.115418 |               0.00407889 |              0.0145083  |
| phase45_shrink40 |     0.949088 |         0.925622 |       0.915882 | 0.00921012 |             0.909155 |           0.854292 |      0.0393513 |              0.874708 |            0.813663 |       0.0524726 |                  0.846674 |                0.759943 |           0.0785186 |                    0.781906 |                  0.656144 |              0.115326 |               0.00305916 |              0.0149725  |
| phase3_shrink50  |     0.948274 |         0.924328 |       0.91477  | 0.00896681 |             0.908167 |           0.853753 |      0.0395377 |              0.875696 |            0.818098 |       0.0508484 |                  0.845209 |                0.757345 |           0.082086  |                    0.781359 |                  0.652309 |              0.115183 |               0.0102226  |              0.00705912 |
| phase3_shrink25  |     0.947674 |         0.923127 |       0.913943 | 0.00849254 |             0.907153 |           0.852905 |      0.039724  |              0.874703 |            0.817332 |       0.0504649 |                  0.843502 |                0.755228 |           0.08519   |                    0.778678 |                  0.649915 |              0.11574  |               0.0134099  |              0.00352956 |
| phase2_primary   |     0.946822 |         0.921105 |       0.912818 | 0.00755436 |             0.905568 |           0.851729 |      0.039979  |              0.872755 |            0.816039 |       0.0501613 |                  0.840913 |                0.755145 |           0.0878231 |                    0.772235 |                  0.645812 |              0.11631  |               0.0167258  |              0          |
| stable_cat_table |     0.946665 |         0.921193 |       0.912638 | 0.00781942 |             0.905574 |           0.851675 |      0.0398949 |              0.872366 |            0.815989 |       0.0504098 |                  0.840884 |                0.753798 |           0.0844802 |                    0.773199 |                  0.64536  |              0.115426 |               0.0181999  |              0.004514   |

## Top Changed Subgroups
| group                     |      n |   changed_rate |   mean_delta |     target | transition                    | field         |
|:--------------------------|-------:|---------------:|-------------:|-----------:|:------------------------------|:--------------|
| 2022                      |  82989 |      0.165781  |  0.0201442   | 0.266505   | phase2_cat_table->phase3_hill | Year          |
| 2025                      |  92894 |      0.128264  |  0.0182094   | 0.284389   | phase2_cat_table->phase3_hill | Year          |
| 2024                      | 127110 |      0.128007  |  0.018081    | 0.295319   | phase2_cat_table->phase3_hill | Year          |
| 2023                      | 136147 |      0.0144697 |  0.00395395  | 0.00960726 | phase2_cat_table->phase3_hill | Year          |
| SOFT                      |  38744 |      0.18839   |  0.0206144   | 0.193475   | phase2_cat_table->phase3_hill | Compound      |
| HARD                      | 170518 |      0.124192  |  0.0162952   | 0.327537   | phase2_cat_table->phase3_hill | Compound      |
| INTERMEDIATE              |  17382 |      0.120412  |  0.0166258   | 0.152284   | phase2_cat_table->phase3_hill | Compound      |
| MEDIUM                    | 211141 |      0.0630763 |  0.0110008   | 0.101131   | phase2_cat_table->phase3_hill | Compound      |
| WET                       |   1355 |      0.0199262 |  0.00802104  | 0.0250923  | phase2_cat_table->phase3_hill | Compound      |
| 4                         |  18903 |      0.14823   |  0.0172256   | 0.171666   | phase2_cat_table->phase3_hill | Stint         |
| 2                         | 129536 |      0.140378  |  0.0180459   | 0.391104   | phase2_cat_table->phase3_hill | Stint         |
| 3                         |  69238 |      0.118822  |  0.0153903   | 0.293105   | phase2_cat_table->phase3_hill | Stint         |
| 1                         | 216288 |      0.0669108 |  0.0112126   | 0.0598184  | phase2_cat_table->phase3_hill | Stint         |
| 5                         |   4281 |      0.0499883 |  0.00917397  | 0.053025   | phase2_cat_table->phase3_hill | Stint         |
| 7                         |    116 |      0.0258621 |  0.00751234  | 0          | phase2_cat_table->phase3_hill | Stint         |
| 6                         |    728 |      0.0137363 |  0.00721122  | 0.0192308  | phase2_cat_table->phase3_hill | Stint         |
| São Paulo Grand Prix      |  11497 |      0.21249   |  0.0222215   | 0.253718   | phase2_cat_table->phase3_hill | Race          |
| Spanish Grand Prix        |  20483 |      0.177025  |  0.0204081   | 0.319973   | phase2_cat_table->phase3_hill | Race          |
| Bahrain Grand Prix        |  19535 |      0.167187  |  0.0190963   | 0.287535   | phase2_cat_table->phase3_hill | Race          |
| French Grand Prix         |   3185 |      0.144741  |  0.0186629   | 0.257457   | phase2_cat_table->phase3_hill | Race          |
| Emilia Romagna Grand Prix |  15483 |      0.141639  |  0.0187391   | 0.272557   | phase2_cat_table->phase3_hill | Race          |
| Austrian Grand Prix       |  21223 |      0.138058  |  0.0168702   | 0.188051   | phase2_cat_table->phase3_hill | Race          |
| Japanese Grand Prix       |  12891 |      0.133582  |  0.0162839   | 0.204018   | phase2_cat_table->phase3_hill | Race          |
| Qatar Grand Prix          |  13817 |      0.115076  |  0.0150405   | 0.175581   | phase2_cat_table->phase3_hill | Race          |
| Hungarian Grand Prix      |  22481 |      0.110493  |  0.0153966   | 0.239269   | phase2_cat_table->phase3_hill | Race          |
| Azerbaijan Grand Prix     |  12126 |      0.110094  |  0.0144982   | 0.21458    | phase2_cat_table->phase3_hill | Race          |
| Canadian Grand Prix       |  21416 |      0.0989447 |  0.0144115   | 0.153857   | phase2_cat_table->phase3_hill | Race          |
| Australian Grand Prix     |  18406 |      0.0966533 |  0.013808    | 0.181571   | phase2_cat_table->phase3_hill | Race          |
| Dutch Grand Prix          |  24462 |      0.0940234 |  0.0153151   | 0.176069   | phase2_cat_table->phase3_hill | Race          |
| Mexico City Grand Prix    |  23672 |      0.0845302 |  0.0124282   | 0.0906556  | phase2_cat_table->phase3_hill | Race          |
| Pre-Season Testing        |  22492 |      0.081051  |  0.0115936   | 0.146541   | phase2_cat_table->phase3_hill | Race          |
| Belgian Grand Prix        |   9002 |      0.0808709 |  0.0133354   | 0.280382   | phase2_cat_table->phase3_hill | Race          |
| Italian Grand Prix        |  19854 |      0.0790269 |  0.0115812   | 0.131963   | phase2_cat_table->phase3_hill | Race          |
| United States Grand Prix  |  18045 |      0.0726517 |  0.0113144   | 0.114048   | phase2_cat_table->phase3_hill | Race          |
| Monaco Grand Prix         |  21539 |      0.0720089 |  0.0131132   | 0.357398   | phase2_cat_table->phase3_hill | Race          |
| Las Vegas Grand Prix      |  12479 |      0.069637  |  0.0118529   | 0.225339   | phase2_cat_table->phase3_hill | Race          |
| 9                         |  43914 |      0.210229  |  0.0228509   | 0.248964   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 1                         |  43914 |      0.165574  |  0.0181236   | 0.266498   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 2                         |  43914 |      0.164936  |  0.0188247   | 0.376509   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 3                         |  43914 |      0.130141  |  0.0174047   | 0.466389   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 4                         |  43914 |      0.0947534 |  0.0144591   | 0.315253   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 0                         |  43914 |      0.0927039 |  0.0144675   | 0.118277   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 5                         |  43914 |      0.0364576 |  0.00880282  | 0.050189   | phase2_cat_table->phase3_hill | TyreFracBin   |
| 6                         |  43914 |      0.036116  |  0.00880564  | 0.0482534  | phase2_cat_table->phase3_hill | TyreFracBin   |
| 7                         |  43914 |      0.034727  |  0.00870981  | 0.0502573  | phase2_cat_table->phase3_hill | TyreFracBin   |
| 8                         |  43914 |      0.0343626 |  0.00873362  | 0.0492326  | phase2_cat_table->phase3_hill | TyreFracBin   |
| 0                         |  43914 |      0.210548  |  0.0228842   | 0.249419   | phase2_cat_table->phase3_hill | StintGapBin   |
| 8                         |  43914 |      0.139523  |  0.0169899   | 0.325591   | phase2_cat_table->phase3_hill | StintGapBin   |
| 9                         |  43914 |      0.138657  |  0.0166328   | 0.191852   | phase2_cat_table->phase3_hill | StintGapBin   |
| 7                         |  43914 |      0.138156  |  0.0177248   | 0.39024    | phase2_cat_table->phase3_hill | StintGapBin   |
| 6                         |  43914 |      0.124425  |  0.0166523   | 0.359111   | phase2_cat_table->phase3_hill | StintGapBin   |
| 5                         |  43914 |      0.107141  |  0.0152804   | 0.275425   | phase2_cat_table->phase3_hill | StintGapBin   |
| 2                         |  43914 |      0.0363438 |  0.00882144  | 0.0484584  | phase2_cat_table->phase3_hill | StintGapBin   |
| 1                         |  43914 |      0.0360705 |  0.00876837  | 0.0501435  | phase2_cat_table->phase3_hill | StintGapBin   |
| 3                         |  43914 |      0.0345903 |  0.00870453  | 0.0500751  | phase2_cat_table->phase3_hill | StintGapBin   |
| 4                         |  43914 |      0.0345448 |  0.00872374  | 0.0495059  | phase2_cat_table->phase3_hill | StintGapBin   |
| 1                         |  14638 |      0.17721   |  0.0215972   | 0.327709   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 25                        |  14638 |      0.159926  |  0.0226253   | 0.323815   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 10                        |  14638 |      0.155417  |  0.0197717   | 0.300041   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 9                         |  14638 |      0.154871  |  0.0199853   | 0.294098   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 6                         |  14638 |      0.146946  |  0.0194127   | 0.274901   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 3                         |  14638 |      0.146468  |  0.0195015   | 0.296215   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 24                        |  14638 |      0.145375  |  0.022948    | 0.253587   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 7                         |  14638 |      0.142847  |  0.0190214   | 0.290613   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 4                         |  14638 |      0.137929  |  0.0190711   | 0.330578   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 11                        |  14638 |      0.13745   |  0.0186851   | 0.338093   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 29                        |  14638 |      0.135469  |  0.0188972   | 0.282416   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 0                         |  14638 |      0.13219   |  0.0177009   | 0.184042   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 28                        |  14638 |      0.130687  |  0.0189608   | 0.263424   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 8                         |  14638 |      0.130414  |  0.0181136   | 0.267591   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 2                         |  14638 |      0.126793  |  0.0177847   | 0.227763   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 27                        |  14638 |      0.123856  |  0.0183469   | 0.281118   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 5                         |  14638 |      0.119825  |  0.0172266   | 0.206722   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 12                        |  14638 |      0.117024  |  0.0173909   | 0.364531   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 26                        |  14638 |      0.0970078 |  0.0150373   | 0.187662   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 13                        |  14638 |      0.078631  |  0.0120138   | 0.182265   | phase2_cat_table->phase3_hill | DeltaBinGroup |
| 2024                      | 127110 |      0.141964  |  0.00708602  | 0.295319   | phase3_hill->phase45_hard     | Year          |
| 2025                      |  92894 |      0.138502  |  0.00704562  | 0.284389   | phase3_hill->phase45_hard     | Year          |
| 2022                      |  82989 |      0.137572  |  0.00701829  | 0.266505   | phase3_hill->phase45_hard     | Year          |
| 2023                      | 136147 |      0.0116492 |  0.000744501 | 0.00960726 | phase3_hill->phase45_hard     | Year          |

## Interpretation
- Later Phase 3/4.5 local corrections are evaluated by grouped macro/p10 AUC, not only global stratified OOF.
- Candidate submissions intentionally shrink high-OOF local deltas toward more stable CatBoost/table predictions.
- Use public LB to choose among stable shrink candidates rather than the highest local OOF candidate.