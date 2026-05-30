# Phase 4.5 Hard Region Experiment Log

## Objective

Stop global tuning and correct only difficult local regions using OOF-gated
specialists.

Baseline:

- `20260519_144202_phase3_specialist_ensemble`
- OOF AUC: `0.948982918`
- Formula: `0.72 * specialist + 0.28 * neighborhood`

## Hard Region Signals

Hardness was computed from OOF-only prediction geometry:

- model prediction standard deviation
- prediction range
- entropy of the current baseline
- rank standard deviation across models
- CatBoost vs specialist disagreement
- neighborhood vs specialist disagreement

No target labels are used to define test hard masks.

## Selected Local Corrections

| Region | Train rows | Test rows | Base local AUC | Specialist local AUC | Blend alpha | Sequential AUC |
|---|---:|---:|---:|---:|---:|---:|
| extreme_tyre_uncertain | 40,022 | 16,626 | 0.825006 | 0.825616 | 0.60 | 0.949055917 |
| cat_specialist_disagree | 65,871 | 28,225 | 0.791981 | 0.792239 | 0.55 | 0.949092805 |
| disagreement_top25 | 109,785 | 47,042 | 0.830731 | 0.830755 | 0.50 | 0.949113168 |
| hard_compound_uncertain | 75,535 | 32,236 | 0.777355 | 0.777565 | 0.50 | 0.949115930 |
| high_gap_uncertain | 38,690 | 16,616 | 0.809016 | 0.808272 | 0.35 | 0.949125267 |
| year2023_hard | 5,335 | 2,445 | 0.904050 | 0.892916 | 0.20 | 0.949135320 |

Some specialists have lower standalone local AUC but improve blended ranking.
This is expected in a local reranking setup: they add a weak orthogonal ordering
signal rather than replacing the base model.

## Result

Final OOF AUC:

`0.949135320`

Net gain over Phase 3:

`+0.000152402`

## Primary Submission

`submissions/20260519_150804_phase45_hard_0.949135.csv`

Calibration/probe family:

- `submissions/20260519_150920_phase45_hard_calibrated_hill.csv`
- `submissions/20260519_150920_phase45_hard_calibrated_hill_temp0.95.csv`
- `submissions/20260519_150920_phase45_hard_calibrated_hill_temp1.05.csv`
- `submissions/20260519_150920_phase45_hard_calibrated_rank.csv`

## Files

- `src/phase45_hard_regions.py`
- `reports/phase45_hard_region_report.md`
- `reports/phase45_hard_region_characterization.csv`
- `experiments/20260519_150804_phase45_hard/hard_candidate_report.csv`
- `experiments/20260519_150804_phase45_hard/summary.json`

## Interpretation

The remaining leaderboard gains are tiny and localized. The useful regions are
not broad global failure modes; they are disagreement pockets where the current
CatBoost/table/neighborhood/specialist stack ranks rows inconsistently.

The best next moves are:

1. Probe `hill`, `hill_temp0.95`, and `hill_temp1.05`.
2. If public LB improves, run the same hard-region framework with a second
   manifold view such as cosine KNN or CatBoost leaf embeddings.
3. Avoid broad pseudo-label retraining unless the hard-region masks are preserved;
   otherwise the negative-heavy pseudo set may flatten positive ranking.
