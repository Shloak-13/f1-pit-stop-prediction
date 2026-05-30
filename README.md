# Predicting F1 Pit Stops - Kaggle Winning Pipeline

This workspace is set up as a serious tabular competition lab for the ROC-AUC
`PitNextLap` objective. It is intentionally broader than a single model: feature
store, leakage diagnostics, multiple validation regimes, diverse models,
ensembling, calibration probes, and pseudo-labeling are separated so experiments
can be iterated without rewriting the stack.

## Data Intelligence Summary

Observed from the extracted competition files:

- Train: `439140 x 16`; test: `188165 x 15`.
- Target rate: `0.198982`.
- No missing values and no duplicate feature rows excluding `id`/target.
- Categorical columns: `Driver`, `Compound`, `Race`.
- Ordinal/state columns: `Year`, `PitStop`, `Stint`, `Position`.
- Temporal/progression columns: `LapNumber`, `TyreLife`, `RaceProgress`.
- Degradation/race-state proxies: `LapTime (s)`, `LapTime_Delta`,
  `Cumulative_Degradation`, `Position_Change`.
- `RaceProgress` appears to encode `LapNumber / race_laps`; exact rational
  artifacts and inferred race length are likely valuable.
- `Compound` is highly target-informative: HARD has a much higher pit-next-lap
  rate than MEDIUM, while WET is rare and low-rate.

## Structure

- `src/features.py` builds a transductive numeric feature store.
- `src/diagnostics.py` writes profile, MI, interaction, and optional adversarial
  drift reports.
- `src/train.py` runs fold-aware target encoding plus multiple model families.
- `src/ensemble.py` performs mean/rank/logit and hill-climbing ensembles.
- `src/pseudo_label.py` creates high-confidence pseudo-labeled train files.
- `src/make_probe_submissions.py` creates leaderboard calibration variants.
- `configs/default.json` controls seeds, CV, model list, and calibration variants.

## Run

```powershell
python src/diagnostics.py
python src/features.py
python src/train.py --cv stratified --models lgbm,xgb,hist_gbdt,extra_trees,logistic --tag s0
python src/train.py --cv race --models lgbm,xgb,hist_gbdt --tag race
python src/ensemble.py --experiments <exp_id_1>,<exp_id_2> --tag final
```

If optional libraries are installed, add `catboost` to the model list. The code
already includes hooks for CatBoost; Optuna/SHAP/TabNet/FT-Transformer/TabPFN
are called out as next expansion points because they are not all installed in
this environment.

## Validation Strategy

Run and compare:

- `stratified`: public-LB-like if the split is synthetic/random.
- `race`: stress-tests race/year generalization.
- `driver`: tests high-cardinality driver leakage dependence.
- `stint`: tests within-race stint leakage and temporal robustness.

The goal is not to blindly maximize one CV. Use the public leaderboard to map
which CV family tracks LB, then ensemble stable high-CV and high-LB families.

## Leaderboard Optimization Playbook

1. Train seed/model/CV families.
2. Submit one strong single model, one rank ensemble, one logit ensemble, and
   two temperature variants.
3. Record public LB beside `experiments/*/summary.json`.
4. Keep models whose OOF is strong and whose public LB is not redundant.
5. Use `src/make_probe_submissions.py` for controlled sharpening/smoothing.
6. Pseudo-label only extreme predictions and compare both stratified and race CV.

## Leakage Search Targets

- `PitStop`, `Stint`, `TyreLife`, and `RaceProgress` threshold rules.
- Reconstructed race length and lap remaining.
- Driver/race continuity after sorting by `Year/Race/Driver/LapNumber`.
- Synthetic fingerprints from `id`, exact `RaceProgress`, and hashed group IDs.
- Target reconstruction by high-cardinality groups such as
  `Race-Year-Driver`, `Driver-Compound`, and `Race-Stint`.
