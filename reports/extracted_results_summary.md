# Extracted Model Results

- Best usable competition model: **Phase 4.5 hard-region specialist ensemble**
- Best recorded OOF ROC-AUC: **0.949135320**
- Source: `experiments/20260519_150804_phase45_hard/summary.json`

Note: synthetic-forensics files contain standalone artifact-probe AUC values, but the README uses the final model pipeline score rather than those diagnostic probes.

## Top Extracted AUC-like Metrics
| source                                                       | metric_path                   | metric         |    value |
|:-------------------------------------------------------------|:------------------------------|:---------------|---------:|
| experiments\20260519_163902_synthetic_forensics\summary.json | additive[1].standalone_auc    | standalone_auc | 1        |
| experiments\20260519_163902_synthetic_forensics\summary.json | additive[0].standalone_auc    | standalone_auc | 1        |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[1].specialist_auc | specialist_auc | 0.961671 |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[1].base_auc       | base_auc       | 0.961638 |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[0].specialist_auc | specialist_auc | 0.960175 |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[0].base_auc       | base_auc       | 0.958773 |
| experiments\20260519_163902_synthetic_forensics\summary.json | additive[1].best_blend_auc    | best_blend_auc | 0.953846 |
| experiments\20260519_163902_synthetic_forensics\summary.json | additive[0].best_blend_auc    | best_blend_auc | 0.953846 |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[4].specialist_auc | specialist_auc | 0.951687 |
| experiments\20260519_143756_phase3_specialists\summary.json  | specialists[4].base_auc       | base_auc       | 0.95096  |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[0].oof_auc                | oof_auc        | 0.949207 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[1].oof_auc                | oof_auc        | 0.949197 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[2].oof_auc                | oof_auc        | 0.949194 |
| experiments\20260519_153956_phase6_lambdarank\summary.json   | best_blends[0].oof_auc        | oof_auc        | 0.949187 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[3].oof_auc                | oof_auc        | 0.949187 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[4].oof_auc                | oof_auc        | 0.949184 |
| experiments\20260519_153956_phase6_lambdarank\summary.json   | best_blends[1].oof_auc        | oof_auc        | 0.949179 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[5].oof_auc                | oof_auc        | 0.949179 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[6].oof_auc                | oof_auc        | 0.949176 |
| experiments\20260519_161041_phase6_rank_fusion\summary.json  | top[7].oof_auc                | oof_auc        | 0.949171 |