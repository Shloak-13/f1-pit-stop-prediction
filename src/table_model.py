from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    race_laps = out["LapNumber"] / out["RaceProgress"].replace(0, np.nan)
    out["RaceLapsEst"] = np.rint(race_laps).fillna(0).astype(int)
    out["LapsRemainingEst"] = (out["RaceLapsEst"] - out["LapNumber"]).astype(int)
    out["RacePhase20"] = np.floor(out["RaceProgress"] * 20).clip(0, 19).astype(int)
    out["RacePhase50"] = np.floor(out["RaceProgress"] * 50).clip(0, 49).astype(int)
    out["TyreLifeBin5"] = (out["TyreLife"] // 5).astype(int)
    out["TyreLifeBin3"] = (out["TyreLife"] // 3).astype(int)
    out["LapBin5"] = (out["LapNumber"] // 5).astype(int)
    out["DegBin"] = pd.qcut(out["Cumulative_Degradation"].rank(method="first"), 30, labels=False, duplicates="drop").astype(int)
    out["DeltaBin"] = pd.qcut(out["LapTime_Delta"].rank(method="first"), 30, labels=False, duplicates="drop").astype(int)
    out["LapTimeBin"] = pd.qcut(out["LapTime (s)"].rank(method="first"), 40, labels=False, duplicates="drop").astype(int)
    out["PosChgBin"] = out["Position_Change"].clip(-10, 10).astype(int)
    for col in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "TyreLife"]:
        scaled = np.round(out[col].astype(float), 3).astype(str)
        out[col.replace(" ", "_").replace("(", "").replace(")", "") + "_r3"] = scaled
    return out


COMBOS = [
    ["Year"],
    ["Stint"],
    ["Compound"],
    ["Race"],
    ["Driver"],
    ["Year", "Stint"],
    ["Year", "Compound"],
    ["Race", "Year"],
    ["Race", "Stint"],
    ["Race", "Compound"],
    ["Stint", "Compound"],
    ["Position", "Stint"],
    ["Year", "Stint", "Compound"],
    ["Race", "Year", "Stint"],
    ["Race", "Year", "Compound"],
    ["Race", "Stint", "Compound"],
    ["Race", "Year", "Stint", "Compound"],
    ["Race", "Year", "Driver"],
    ["Driver", "Compound"],
    ["Driver", "Stint"],
    ["Driver", "Year"],
    ["Driver", "Race"],
    ["Driver", "Stint", "Compound"],
    ["RaceLapsEst", "LapNumber"],
    ["RaceLapsEst", "LapsRemainingEst"],
    ["Year", "Stint", "TyreLifeBin3"],
    ["Year", "Stint", "TyreLifeBin5"],
    ["Compound", "TyreLifeBin3"],
    ["Compound", "RacePhase20"],
    ["Race", "Year", "RacePhase20"],
    ["Race", "Year", "RacePhase50"],
    ["Race", "Year", "Stint", "RacePhase20"],
    ["Race", "Year", "Stint", "TyreLifeBin3"],
    ["Race", "Year", "Compound", "TyreLifeBin3"],
    ["Race", "Year", "Position"],
    ["Race", "Year", "Position", "Stint"],
    ["Year", "PitStop", "Stint"],
    ["Race", "Year", "PitStop", "Stint"],
    ["Year", "Stint", "DegBin"],
    ["Year", "Stint", "DeltaBin"],
    ["Compound", "DegBin"],
    ["Compound", "DeltaBin"],
]


def smooth_map(frame: pd.DataFrame, y: pd.Series, keys: list[str], prior: float, alpha: float) -> pd.Series:
    stats = frame.assign(_y=y.values).groupby(keys, observed=True)["_y"].agg(["mean", "count"])
    return (stats["mean"] * stats["count"] + prior * alpha) / (stats["count"] + alpha)


def encode(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series, folds, alpha: float = 12.0):
    X_oof = pd.DataFrame(index=train.index)
    X_test_parts = []
    prior = float(y.mean())
    for keys in COMBOS:
        name = "p__" + "__".join(keys)
        X_oof[name] = prior
        for tr_idx, va_idx in folds:
            mp = smooth_map(train.iloc[tr_idx], y.iloc[tr_idx], keys, prior, alpha)
            vals = train.iloc[va_idx].set_index(keys).index.map(mp)
            X_oof.loc[va_idx, name] = pd.Series(vals, index=va_idx).fillna(prior).astype(float)
        full = smooth_map(train, y, keys, prior, alpha)
        X_test_parts.append(pd.Series(test.set_index(keys).index.map(full), name=name).fillna(prior).reset_index(drop=True))
        X_oof[name + "_logit"] = np.log(np.clip(X_oof[name], 1e-5, 1 - 1e-5) / np.clip(1 - X_oof[name], 1e-5, 1))
    X_test = pd.concat(X_test_parts, axis=1)
    for col in list(X_test.columns):
        X_test[col + "_logit"] = np.log(np.clip(X_test[col], 1e-5, 1 - 1e-5) / np.clip(1 - X_test[col], 1e-5, 1))
    X_test = X_test[X_oof.columns]
    return X_oof.astype("float32"), X_test.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default="stratified", choices=["stratified", "race"])
    parser.add_argument("--model", default="hgb", choices=["hgb", "logistic", "lgbm"])
    parser.add_argument("--tag", default="table")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    raw_train, raw_test, sub = load_raw()
    train = add_bins(raw_train)
    test = add_bins(raw_test)
    y = train[TARGET].astype(int)
    if args.cv == "race":
        groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)
        folds = list(GroupKFold(args.splits).split(train, y, groups))
    else:
        folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260518).split(train, y))

    X, X_test = encode(train, test, y, folds)
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model_factory = lambda seed: HistGradientBoostingClassifier(max_iter=450, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.03, random_state=seed)
    elif args.model == "lgbm":
        from lightgbm import LGBMClassifier

        model_factory = lambda seed: LGBMClassifier(n_estimators=900, learning_rate=0.025, num_leaves=48, min_child_samples=60, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5, random_state=seed, n_jobs=-1, verbosity=-1)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model_factory = lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.9, max_iter=1000, class_weight="balanced", random_state=seed))

    exp_id = f"{timestamp()}_{args.tag}_{args.cv}_{args.model}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = model_factory(20260518 + fold)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / len(folds)
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(float(score))
        with open(MODEL_DIR / f"{exp_id}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"table {args.model} {args.cv} fold {fold}: {score:.6f}")
    auc = float(roc_auc_score(y, oof))
    pd.DataFrame({"id": raw_train["id"], TARGET: y, f"{args.model}_oof": oof}).to_csv(exp_dir / f"oof_{args.model}.csv", index=False)
    pd.DataFrame({"id": raw_test["id"], TARGET: test_pred}).to_csv(exp_dir / f"test_{args.model}.csv", index=False)
    sub.assign(**{TARGET: test_pred}).to_csv(SUB_DIR / f"{exp_id}_{auc:.6f}.csv", index=False)
    X.mean().sort_values(ascending=False).to_csv(exp_dir / "encoding_feature_means.csv")
    summary = {"exp_id": exp_id, "model": args.model, "cv": args.cv, "oof_auc": auc, "fold_auc": fold_scores, "features": list(X.columns)}
    write_json(summary, exp_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
