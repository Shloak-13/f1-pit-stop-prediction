from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from table_model import add_bins
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


CAT_BASE = [
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
    "Stint",
    "Position",
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
]

NUM_COLS = [
    "LapNumber",
    "TyreLife",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]

COMBO_CATS = [
    ["Year", "Stint"],
    ["Year", "Stint", "Compound"],
    ["Race", "Year"],
    ["Race", "Year", "Stint"],
    ["Race", "Year", "Compound"],
    ["Race", "Year", "Stint", "Compound"],
    ["Race", "Year", "Driver"],
    ["Driver", "Compound"],
    ["Driver", "Stint", "Compound"],
    ["RaceLapsEst", "LapNumber"],
    ["RaceLapsEst", "LapsRemainingEst"],
    ["Race", "Year", "RacePhase20"],
    ["Race", "Year", "Stint", "TyreLifeBin3"],
    ["Race", "Year", "Position", "Stint"],
]


def make_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = add_bins(df)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["DegradationPerTyre"] = d["Cumulative_Degradation"] / d["TyreLife"].replace(0, np.nan)
    d["DeltaPerTyre"] = d["LapTime_Delta"] / d["TyreLife"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["TyreProgress"] = d["TyreLife"] * d["RaceProgress"]
    extra_num = ["TyreLifeFrac", "DegradationPerTyre", "DeltaPerTyre", "StintLapGap", "TyreProgress"]
    cat_cols = CAT_BASE.copy()
    for keys in COMBO_CATS:
        name = "CAT__" + "__".join(keys)
        d[name] = d[keys].astype(str).agg("|".join, axis=1)
        cat_cols.append(name)
    use_cols = cat_cols + NUM_COLS + extra_num
    X = d[use_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__NA__")
    for c in NUM_COLS + extra_num:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan).fillna(X[c].median()).astype("float32")
    return X, cat_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default="stratified", choices=["stratified", "race", "driver"])
    parser.add_argument("--tag", default="cat_phase2")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1200)
    args = parser.parse_args()
    ensure_dirs()
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    train, test, sub = load_raw()
    X, cat_cols = make_frame(train.drop(columns=[]))
    X_test, _ = make_frame(test)
    y = train[TARGET].astype(int)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    if args.cv == "race":
        groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)
        folds = list(GroupKFold(args.splits).split(X, y, groups))
    elif args.cv == "driver":
        folds = list(GroupKFold(args.splits).split(X, y, train["Driver"].astype(str)))
    else:
        folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260518).split(X, y))

    exp_id = f"{timestamp()}_{args.tag}_{args.cv}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []
    importances = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = CatBoostClassifier(
            iterations=args.iterations,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            bootstrap_type="Bernoulli",
            subsample=0.85,
            random_seed=20260518 + fold,
            allow_writing_files=False,
            verbose=100,
            od_type="Iter",
            od_wait=120,
        )
        tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
        te_pool = Pool(X_test, cat_features=cat_idx)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(va_pool)[:, 1]
        test_pred += model.predict_proba(te_pool)[:, 1] / len(folds)
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(float(score))
        importances.append(model.get_feature_importance(prettified=True))
        with open(MODEL_DIR / f"{exp_id}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"catboost {args.cv} fold {fold}: {score:.6f}")
    cv = float(roc_auc_score(y, oof))
    pd.DataFrame({"id": train["id"], TARGET: y, "catboost_oof": oof}).to_csv(exp_dir / "oof_catboost.csv", index=False)
    pd.DataFrame({"id": test["id"], TARGET: test_pred}).to_csv(exp_dir / "test_catboost.csv", index=False)
    sub.assign(**{TARGET: test_pred}).to_csv(SUB_DIR / f"{exp_id}_catboost_{cv:.6f}.csv", index=False)
    pd.concat(importances).groupby("Feature Id", as_index=False)["Importances"].mean().sort_values("Importances", ascending=False).to_csv(
        exp_dir / "feature_importance_catboost.csv", index=False
    )
    summary = {"exp_id": exp_id, "cv": args.cv, "model": "catboost", "oof_auc": cv, "fold_auc": fold_scores, "cat_cols": cat_cols}
    write_json(summary, exp_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
