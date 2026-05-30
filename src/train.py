from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import CAT_COLS, TARGET, build_features, load_raw
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, logit, read_json, seed_everything, sigmoid, timestamp, write_json


def auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def make_folds(raw_train: pd.DataFrame, y: pd.Series, strategy: str, n_splits: int, seed: int):
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(raw_train, y))
    if strategy == "race":
        groups = raw_train["Race"].astype(str) + "_" + raw_train["Year"].astype(str)
    elif strategy == "driver":
        groups = raw_train["Driver"].astype(str)
    elif strategy == "stint":
        groups = raw_train["Race"].astype(str) + "_" + raw_train["Year"].astype(str) + "_" + raw_train["Stint"].astype(str)
    else:
        raise ValueError(f"unknown CV strategy {strategy}")
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(raw_train, y, groups))


def target_encode(train_raw: pd.DataFrame, test_raw: pd.DataFrame, y: pd.Series, tr_idx, va_idx, cols: list[list[str]], alpha: float = 20.0):
    global_mean = y.iloc[tr_idx].mean()
    tr_te = pd.DataFrame(index=train_raw.index)
    va_te = pd.DataFrame(index=train_raw.index)
    test_te = pd.DataFrame(index=test_raw.index)
    for keys in cols:
        name = "te_" + "__".join(keys)
        stats = train_raw.iloc[tr_idx].assign(_y=y.iloc[tr_idx].values).groupby(keys, observed=True)["_y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)
        tr_te[name] = train_raw.set_index(keys).index.map(smooth).fillna(global_mean).astype("float32")
        va_te[name] = train_raw.set_index(keys).index.map(smooth).fillna(global_mean).astype("float32")
        test_te[name] = test_raw.set_index(keys).index.map(smooth).fillna(global_mean).astype("float32")
    return tr_te.iloc[tr_idx].reset_index(drop=True), va_te.iloc[va_idx].reset_index(drop=True), test_te.reset_index(drop=True)


def get_model(name: str, seed: int, scale_pos_weight: float):
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=2400,
            learning_rate=0.018,
            num_leaves=96,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.86,
            colsample_bytree=0.72,
            reg_alpha=0.15,
            reg_lambda=2.5,
            objective="binary",
            metric="auc",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=1500,
            max_depth=6,
            learning_rate=0.022,
            subsample=0.88,
            colsample_bytree=0.75,
            min_child_weight=20,
            reg_alpha=0.05,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=900,
            learning_rate=0.032,
            max_leaf_nodes=63,
            min_samples_leaf=45,
            l2_regularization=0.04,
            random_state=seed,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=650,
            min_samples_leaf=8,
            max_features=0.48,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=0.55, max_iter=1200, class_weight="balanced", random_state=seed))
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=2200, learning_rate=0.025, depth=7, loss_function="Logloss", eval_metric="AUC", random_seed=seed, verbose=False)
    raise ValueError(name)


def fit_predict(model, name: str, X_tr, y_tr, X_va, y_va, X_test):
    if name == "lgbm":
        try:
            from lightgbm import early_stopping, log_evaluation

            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=[early_stopping(120), log_evaluation(0)])
        except TypeError:
            model.fit(X_tr, y_tr)
    elif name == "xgb":
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except TypeError:
            model.fit(X_tr, y_tr)
    elif name == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1], model.predict_proba(X_test)[:, 1]


def run(config: dict, cv_strategy: str, models: list[str], tag: str) -> dict:
    seed = int(config["seed"])
    seed_everything(seed)
    ensure_dirs()
    train_raw, test_raw, sub = load_raw()
    X, y, X_test, feature_cols = build_features(train_raw, test_raw, level=config.get("feature_level", "balanced"))
    te_cols = [["Driver"], ["Race"], ["Compound"], ["Race", "Year"], ["Driver", "Compound"], ["Race", "Compound"], ["Stint", "Compound"], ["Race", "Year", "Driver"]]
    folds = make_folds(train_raw, y, cv_strategy, int(config["n_splits"]), seed)
    exp_id = f"{timestamp()}_{tag}_{cv_strategy}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
    summary = {"exp_id": exp_id, "cv_strategy": cv_strategy, "models": {}, "feature_count": len(feature_cols)}
    all_oof = {}
    all_test = {}
    for model_name in models:
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(folds):
            te_tr, te_va, te_test = target_encode(train_raw, test_raw, y, tr_idx, va_idx, te_cols)
            X_tr = pd.concat([X.iloc[tr_idx].reset_index(drop=True), te_tr], axis=1)
            X_va = pd.concat([X.iloc[va_idx].reset_index(drop=True), te_va], axis=1)
            XT = pd.concat([X_test.reset_index(drop=True), te_test], axis=1)
            model = get_model(model_name, seed + fold, scale_pos_weight)
            pred_va, pred_test = fit_predict(model, model_name, X_tr, y.iloc[tr_idx], X_va, y.iloc[va_idx], XT)
            oof[va_idx] = pred_va
            test_pred += pred_test / len(folds)
            score = auc(y.iloc[va_idx], pred_va)
            fold_scores.append(score)
            with open(MODEL_DIR / f"{exp_id}_{model_name}_fold{fold}.pkl", "wb") as f:
                pickle.dump(model, f)
            print(f"{model_name} {cv_strategy} fold {fold}: {score:.6f}")
        cv = auc(y, oof)
        all_oof[model_name] = oof
        all_test[model_name] = test_pred
        pd.DataFrame({"id": train_raw["id"], TARGET: y, f"{model_name}_oof": oof}).to_csv(exp_dir / f"oof_{model_name}.csv", index=False)
        pd.DataFrame({"id": test_raw["id"], TARGET: test_pred}).to_csv(exp_dir / f"test_{model_name}.csv", index=False)
        sub.assign(**{TARGET: test_pred}).to_csv(SUB_DIR / f"{exp_id}_{model_name}_{cv:.6f}.csv", index=False)
        summary["models"][model_name] = {"oof_auc": cv, "fold_auc": fold_scores, "fold_std": float(np.std(fold_scores))}
        print(f"{model_name} {cv_strategy} OOF: {cv:.6f}")
    write_json(summary, exp_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--cv", default="stratified", choices=["stratified", "race", "driver", "stint"])
    parser.add_argument("--models", default=None, help="comma-separated model list")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--n-splits", type=int, default=None)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.n_splits:
        config["n_splits"] = args.n_splits
    models = args.models.split(",") if args.models else config["models"]
    run(config, args.cv, models, args.tag)


if __name__ == "__main__":
    main()
