# Phase Reset - Validation Realism

## Reason For Reset

Phase 3 and Phase 4.5 improved stratified OOF but did not improve public LB.
That means local corrections are overfitting fold geometry. The objective is now
leaderboard correlation and prediction stability, not raw OOF.

## Reconstructed Systems

| System | Definition | OOF AUC |
|---|---|---:|
| Phase 2 | `0.88 * CatBoost + 0.12 * table-HGB` | 0.946822 |
| Phase 3 | `0.72 * specialist + 0.28 * neighborhood` | 0.948983 |
| Phase 4.5 | hard-region corrected Phase 3 | 0.949135 |

## CV Realism Findings

Later local corrections are concentrated in specific subgroups:

- Phase 2 -> Phase 3 changed 16.6% of 2022 rows but only 1.4% of 2023 rows.
- Largest race changes include Sao Paulo, Spanish, Bahrain, French, Emilia
  Romagna, Austrian, and Japanese Grand Prix.
- Changes are strongest in SOFT, HARD, Stint 2-4, and extreme tyre-fraction bins.
- Phase 3 -> Phase 4.5 changes are much smaller but still concentrated in HARD,
  Stint 2-3, and non-2023 years.

This supports the overfitting hypothesis: improvements are not uniformly robust;
they are localized in synthetic fold-sensitive pockets.

## LB-Aligned Candidate Ranking

Candidates were ranked by grouped lower-tail metrics rather than global OOF.

Best RaceYear lower-tail candidate:

`logit_stable`

- Global OOF: 0.948038
- RaceYear p10 AUC: 0.854942
- Lower OOF than Phase 4.5, but best robustness proxy.

Best conservative high-OOF shrink candidates:

`phase45_shrink20`

- Global OOF: 0.949042
- RaceYear p10 AUC: 0.854349
- Keeps only 20% of the Phase 4.5 hard-region delta.

`phase45_shrink40`

- Global OOF: 0.949088
- RaceYear p10 AUC: 0.854292
- Keeps 40% of the Phase 4.5 delta.

## Recommended Submission Probes

Primary realism probes:

- `submissions/20260519_152759_validation_reset_logit_stable.csv`
- `submissions/20260519_152759_validation_reset_phase45_shrink20.csv`
- `submissions/20260519_152759_validation_reset_phase45_shrink40.csv`

Control submissions:

- `submissions/20260519_152759_validation_reset_phase2_primary.csv`
- `submissions/20260519_152759_validation_reset_stable_mean.csv`
- `submissions/20260519_152759_validation_reset_rank_stable.csv`

## Interpretation

If public LB rewards `logit_stable`, the public split prefers lower-variance
rank geometry even at lower OOF. If it rewards `phase45_shrink20/40`, then Phase
4.5 corrections contain useful signal but need shrinkage. If `phase2_primary`
matches or beats all later candidates, all local manifold corrections should be
considered fold-overfit until a new validation split proves otherwise.

## Generated Artifacts

- `src/validation_realism.py`
- `reports/validation_realism_report.md`
- `experiments/20260519_152759_validation_reset/cv_realism_summary.csv`
- `experiments/20260519_152759_validation_reset/lb_aligned_candidate_scores.csv`
- `experiments/20260519_152759_validation_reset/prediction_delta_subgroups.csv`
- `experiments/20260519_152759_validation_reset/prediction_stability_rows.csv`
