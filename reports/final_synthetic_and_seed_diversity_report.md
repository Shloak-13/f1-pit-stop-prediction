# Synthetic Forensics and Seed Diversity Report

## Synthetic Artifact Forensics

Main validated finding:

- In-sample row-template collision priors looked perfect, but OOF validation
  reduced them to random signal. They should not be used as new features.

Strong standalone synthetic signals still match the known winning geometry:

| Artifact | Standalone AUC |
|---|---:|
| `Race x Year x Stint` | 0.898307 |
| `Race x Year x RacePhase20` | 0.891171 |
| `Compound x DeltaBin x TyreLifeFracBin12` | 0.878266 |
| `Year x Compound x TyreLifeBin3` | 0.874416 |
| `Year x Stint x CurrentLap_RaceLapsEst` | 0.874144 |

Top conditional entropy pairs:

| Pair | AUC |
|---|---:|
| `Year + TyreLifeFracBin12` | 0.871229 |
| `Year + TyreLife/RaceProgress` | 0.865385 |
| `Year + Stint` | 0.859628 |
| `Year + StintGapBin12` | 0.853533 |
| `RaceLapsEst + TyreLifeFracBin12` | 0.848483 |

Quantization findings:

- Several integer/bucket state columns have heavy repeated values, as expected:
  `PitStop`, `Stint`, `StintLapGap`, `Position_Change`, `Year`,
  `RaceLapsEst`.
- `RaceProgress` and `CurrentLap_RaceLapsEst` have many deterministic
  low-entropy exact rational buckets present in both train and test.
- Train/test drift exists in coverage of some exact rational values:
  `RaceProgress`, `CurrentLap_RaceLapsEst`, `RaceLapsEst`, and `RaceProgress_r3`
  have many train-only/test-only exact values.

OOF additive value versus CatBoost:

| Feature | OOF standalone AUC | Additive gain |
|---|---:|---:|
| collision round-2 prior | 0.499994 | ~0 |
| collision round-3 prior | 0.499994 | ~0 |
| best pair prior | 0.876981 | 0 |
| best interaction prior | 0.897400 | 0 |

Interpretation:

The artifact search did not discover a safe additive feature beyond the existing
CatBoost manifold. The apparent perfect template collision signal is in-sample
leakage, not a transferable feature.

## Seed Diversity Probe

Implemented:

- `src/catboost_seed_diversity.py`
- OOF/test storage per seed
- Pearson/Spearman correlation matrices
- greedy OOF ensemble builder
- low-correlation ensemble builder
- final rank blends with 50% Phase 3 specialist weight

Execution constraint:

- A bounded 2-model probe with 3 folds and 180 iterations hit the 20-minute tool
  cap after completing one model.

Completed seed:

| Model | OOF AUC | Fold AUC |
|---|---:|---|
| `cat_seed0_rs0p1_bt0p0` | 0.944314 | 0.944168, 0.943998, 0.944811 |

This probe confirms the script works, but the completed seed is weaker than the
existing Phase 2 CatBoost (`0.946625`). A full 16-24 model sweep needs a longer
uninterrupted runtime.

## Files

- `src/synthetic_artifact_forensics.py`
- `src/catboost_seed_diversity.py`
- `reports/synthetic_artifact_forensics_report.md`
- `experiments/20260519_164529_synthetic_forensics_oof/`
- `experiments/20260519_165532_cat_seed_diversity_probe/`

## Practical Recommendation

Do not add collision-template priors to the production model.

For leaderboard probing, the strongest currently available submissions remain
the stable Phase-reset and Phase-6 rank-fusion families. The seed-diversity
sweep is worth running overnight, but the partial probe does not yet justify a
new submission.
