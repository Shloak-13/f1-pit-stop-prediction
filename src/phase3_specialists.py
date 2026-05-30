from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from phase3_neighborhood import load_base_predictions
from table_model import add_bins
from features import TARGET, load_raw
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["DegradationPerTyre"] = d["Cumulative_Degradation"] / d["TyreLife"].replace(0, np.nan)
    d["DeltaPerTyre"] = d["LapTime_Delta"] / d["TyreLife"].replace(0, np.nan)
    d["TyreProgress"] = d["TyreLife"] * d["RaceProgress"]
    for c in ["Driver", "Race", "Compound"]:
        codes, _ = pd.factorize(d[c], sort=True)
        d[f"{c}_code"] = codes
        d[f"{c}_freq"] = d[c].map(d[c].value_counts()).astype(float)
    cols = [
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "TyreLife",
        "Position",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "RaceProgress",
        "Position_Change",
        "RaceLapsEst",
        "LapsRemainingEst",
        "RacePhase20",
        "RacePhase50",
        "TyreLifeBin3",
        "TyreLifeBin5",
        "LapBin5",
        "DegBin",
        "DeltaBin",
        "LapTimeBin",
        "PosChgBin",
        "TyreLifeFrac",
        "StintLapGap",
        "DegradationPerTyre",
        "DeltaPerTyre",
        "TyreProgress",
        "Driver_code",
        "Race_code",
        "Compound_code",
        "Driver_freq",
        "Race_freq",
        "Compound_freq",
    ]
    X = d[cols].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).astype("float32")


def masks(raw: pd.DataFrame) -> dict[str, pd.Series]:
    d = add_bins(raw)
    tyre_frac = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    stint_gap = d["LapNumber"] - d["TyreLife"]
    return {
        "extreme_tyre": tyre_frac > 0.80,
        "late_race": d["RaceProgress"] > 0.72,
        "high_stint_gap": stint_gap > stint_gap.quantile(0.85),
        "hard": d["Compound"].eq("HARD"),
        "medium": d["Compound"].eq("MEDIUM"),
        "soft": d["Compound"].eq("SOFT"),
        "year_2023": d["Year"].eq(2023),
        "year_2024plus": d["Year"].ge(2024),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3-exp", default="20260519_141027_phase3_knn_euclid_phase3")
    parser.add_argument("--tag", default="phase3_specialists")
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    train, test, sub = load_raw()
    y = train[TARGET].astype(int)
    X = make_features(train)
    XT = make_features(test)
    base_train, base_test = load_base_predictions()
    base_cols = [c for c in base_train.columns if c.startswith("base_")]
    X = pd.concat([X, base_train[base_cols]], axis=1).astype("float32")
    XT = pd.concat([XT, base_test[base_cols]], axis=1).astype("float32")
    phase_oof = pd.read_csv(EXP_DIR / args.phase3_exp / "oof_phase3.csv")["phase3_oof"].values
    phase_test = pd.read_csv(EXP_DIR / args.phase3_exp / "test_phase3.csv")[TARGET].values
    final_oof = phase_oof.copy()
    final_test = phase_test.copy()
    train_masks = masks(train)
    test_masks = masks(test)
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for name, m in train_masks.items():
        idx_all = np.where(m.values)[0]
        if len(idx_all) < 5000 or y.iloc[idx_all].nunique() < 2:
            continue
        folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260519).split(X.iloc[idx_all], y.iloc[idx_all]))
        pred = np.zeros(len(idx_all))
        test_pred = np.zeros(int(test_masks[name].sum()))
        XT_sub = XT.loc[test_masks[name].values]
        for fold, (tr_sub, va_sub) in enumerate(folds):
            tr_idx = idx_all[tr_sub]
            va_idx = idx_all[va_sub]
            model = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.04, random_state=fold)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred[va_sub] = model.predict_proba(X.iloc[va_idx])[:, 1]
            if len(XT_sub):
                test_pred += model.predict_proba(XT_sub)[:, 1] / len(folds)
            with open(MODEL_DIR / f"{exp_id}_{name}_fold{fold}.pkl", "wb") as f:
                pickle.dump(model, f)
        base_auc = auc(y.iloc[idx_all], phase_oof[idx_all])
        spec_auc = auc(y.iloc[idx_all], pred)
        use = spec_auc > base_auc + 0.0002
        if use:
            final_oof[idx_all] = pred
            final_test[test_masks[name].values] = test_pred
        report.append({"specialist": name, "n_train": int(len(idx_all)), "n_test": int(test_masks[name].sum()), "base_auc": base_auc, "specialist_auc": spec_auc, "used": bool(use)})
        print(report[-1])
    cv = auc(y, final_oof)
    pd.DataFrame({"id": train["id"], TARGET: y, "specialist_oof": final_oof}).to_csv(exp_dir / "oof_specialist.csv", index=False)
    pd.DataFrame({"id": test["id"], TARGET: final_test}).to_csv(exp_dir / "test_specialist.csv", index=False)
    sub.assign(**{TARGET: final_test}).to_csv(SUB_DIR / f"{exp_id}_{cv:.6f}.csv", index=False)
    pd.DataFrame(report).to_csv(exp_dir / "specialist_report.csv", index=False)
    write_json({"exp_id": exp_id, "phase3_exp": args.phase3_exp, "oof_auc": cv, "specialists": report}, exp_dir / "summary.json")
    print({"exp_id": exp_id, "oof_auc": cv})


if __name__ == "__main__":
    main()
