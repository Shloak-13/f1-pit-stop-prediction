from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from table_model import add_bins
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, logit, sigmoid, timestamp, write_json


BASE_EXPERIMENTS = {
    "cat": "20260518_165505_phase2_cat_fast_stratified",
    "table_hgb": "20260518_162337_phase2_table_stratified_hgb",
    "table_log": "20260518_174444_phase2_table_stratified_logistic",
    "lgbm": "20260518_145635_s0_stratified",
    "logistic": "20260518_154412_s0_stratified",
}


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def load_base_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for name, exp in BASE_EXPERIMENTS.items():
        exp_dir = EXP_DIR / exp
        oof_files = sorted(exp_dir.glob("oof_*.csv"))
        test_files = sorted(exp_dir.glob("test_*.csv"))
        if not oof_files or not test_files:
            continue
        oof = pd.read_csv(oof_files[0])
        tst = pd.read_csv(test_files[0])
        pred_col = [c for c in oof.columns if c.endswith("_oof")][0]
        train_parts.append(oof[["id", pred_col]].rename(columns={pred_col: f"base_{name}"}))
        test_parts.append(tst[["id", TARGET]].rename(columns={TARGET: f"base_{name}"}))
    train_base = train_parts[0]
    test_base = test_parts[0]
    for part in train_parts[1:]:
        train_base = train_base.merge(part, on="id", how="left")
    for part in test_parts[1:]:
        test_base = test_base.merge(part, on="id", how="left")
    return train_base, test_base


def make_state_frame(raw: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["DegradationPerTyre"] = d["Cumulative_Degradation"] / d["TyreLife"].replace(0, np.nan)
    d["DeltaPerTyre"] = d["LapTime_Delta"] / d["TyreLife"].replace(0, np.nan)
    d["TyreProgress"] = d["TyreLife"] * d["RaceProgress"]
    d["LateRace"] = (d["RaceProgress"] > 0.72).astype(int)
    d["ExtremeTyre"] = (d["TyreLifeFrac"] > 0.80).astype(int)
    d["HighGap"] = (d["StintLapGap"] > d["StintLapGap"].quantile(0.85)).astype(int)
    for c in ["Driver", "Race", "Compound"]:
        freq = d[c].value_counts(dropna=False)
        d[f"{c}_freq"] = d[c].map(freq).astype(float)
        codes, _ = pd.factorize(d[c], sort=True)
        d[f"{c}_code"] = codes.astype(float)
    return d


def make_embedding(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base, test_base = load_base_predictions()
    tr = make_state_frame(train_raw).merge(train_base, on="id", how="left")
    te = make_state_frame(test_raw).merge(test_base, on="id", how="left")
    numeric = [
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
        "Driver_freq",
        "Race_freq",
        "Compound_freq",
        "Driver_code",
        "Race_code",
        "Compound_code",
        "LateRace",
        "ExtremeTyre",
        "HighGap",
    ]
    base_cols = [c for c in tr.columns if c.startswith("base_")]
    for c in base_cols:
        tr[c + "_logit"] = logit(tr[c].fillna(tr[c].mean()).values)
        te[c + "_logit"] = logit(te[c].fillna(tr[c].mean()).values)
    cols = numeric + base_cols + [c + "_logit" for c in base_cols]
    all_df = pd.concat([tr[cols], te[cols]], axis=0, ignore_index=True)
    all_df = pd.get_dummies(
        pd.concat([all_df, pd.concat([tr[["Compound", "Race"]], te[["Compound", "Race"]]], axis=0).reset_index(drop=True)], axis=1),
        columns=["Compound", "Race"],
        dtype=float,
    )
    all_df = all_df.replace([np.inf, -np.inf], np.nan)
    all_df = all_df.fillna(all_df.median(numeric_only=True)).astype("float32")
    Xtr = all_df.iloc[: len(tr)].reset_index(drop=True)
    Xte = all_df.iloc[len(tr) :].reset_index(drop=True)
    return Xtr, Xte, list(Xtr.columns)


def summarize_neighbors(dist: np.ndarray, idx: np.ndarray, y_ref: np.ndarray, prefix: str, ks: list[int]) -> pd.DataFrame:
    out = {}
    eps = 1e-6
    for k in ks:
        labels = y_ref[idx[:, :k]]
        d = dist[:, :k]
        w = 1.0 / (d + eps)
        p = labels.mean(axis=1)
        wp = (labels * w).sum(axis=1) / w.sum(axis=1)
        var = labels.var(axis=1)
        ent = -(p * np.log(np.clip(p, eps, 1)) + (1 - p) * np.log(np.clip(1 - p, eps, 1)))
        out[f"{prefix}_k{k}_mean"] = p
        out[f"{prefix}_k{k}_wmean"] = wp
        out[f"{prefix}_k{k}_var"] = var
        out[f"{prefix}_k{k}_entropy"] = ent
        out[f"{prefix}_k{k}_agree"] = np.maximum(p, 1 - p)
        out[f"{prefix}_k{k}_dist_mean"] = d.mean(axis=1)
        out[f"{prefix}_k{k}_dist_min"] = d.min(axis=1)
    return pd.DataFrame(out).astype("float32")


def build_knn_features(X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series, folds, metrics: list[str], ks: list[int], pca_dim: int):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    max_k = max(ks)
    train_feats = pd.DataFrame(index=X.index)
    test_accum = []
    for metric in metrics:
        train_metric = pd.DataFrame(index=X.index)
        test_metric_sum = None
        for fold, (tr_idx, va_idx) in enumerate(folds):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X.iloc[tr_idx])
            Xva = scaler.transform(X.iloc[va_idx])
            Xte = scaler.transform(X_test)
            if pca_dim > 0 and pca_dim < Xtr.shape[1]:
                pca = PCA(n_components=pca_dim, random_state=20260518 + fold)
                Xtr = pca.fit_transform(Xtr)
                Xva = pca.transform(Xva)
                Xte = pca.transform(Xte)
            nn = NearestNeighbors(n_neighbors=max_k, metric=metric, algorithm="auto", n_jobs=-1)
            nn.fit(Xtr)
            d_va, i_va = nn.kneighbors(Xva, return_distance=True)
            d_te, i_te = nn.kneighbors(Xte, return_distance=True)
            va_feat = summarize_neighbors(d_va, i_va, y.iloc[tr_idx].values, f"knn_{metric}", ks)
            te_feat = summarize_neighbors(d_te, i_te, y.iloc[tr_idx].values, f"knn_{metric}", ks)
            train_metric.loc[va_idx, va_feat.columns] = va_feat.values
            test_metric_sum = te_feat if test_metric_sum is None else test_metric_sum.add(te_feat, fill_value=0)
            print(f"knn {metric} fold {fold} done")
        train_feats = pd.concat([train_feats, train_metric], axis=1)
        test_accum.append(test_metric_sum / len(folds))
    test_feats = pd.concat(test_accum, axis=1).reset_index(drop=True)
    return train_feats.astype("float32"), test_feats.astype("float32")


def cluster_features(X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series, folds, clusters: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    tr_out = pd.DataFrame(index=X.index)
    te_out = pd.DataFrame(index=X_test.index)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(pd.concat([X, X_test], axis=0))
    X_scaled = all_scaled[: len(X)]
    XT_scaled = all_scaled[len(X) :]
    for n in clusters:
        km = MiniBatchKMeans(n_clusters=n, random_state=20260518 + n, batch_size=8192, n_init="auto")
        labels_all = km.fit_predict(all_scaled)
        lab_tr = labels_all[: len(X)]
        lab_te = labels_all[len(X) :]
        name = f"cluster{n}"
        tr_out[name] = lab_tr
        te_out[name] = lab_te
        tr_out[f"{name}_dist"] = km.transform(X_scaled).min(axis=1)
        te_out[f"{name}_dist"] = km.transform(XT_scaled).min(axis=1)
        tr_out[f"{name}_prior"] = y.mean()
        for tr_idx, va_idx in folds:
            stats = pd.DataFrame({"lab": lab_tr[tr_idx], "y": y.iloc[tr_idx].values}).groupby("lab")["y"].agg(["mean", "count"])
            prior = y.iloc[tr_idx].mean()
            smooth = (stats["mean"] * stats["count"] + prior * 20) / (stats["count"] + 20)
            tr_out.loc[va_idx, f"{name}_prior"] = pd.Series(lab_tr[va_idx]).map(smooth).fillna(prior).values
        stats_full = pd.DataFrame({"lab": lab_tr, "y": y.values}).groupby("lab")["y"].agg(["mean", "count"])
        smooth_full = (stats_full["mean"] * stats_full["count"] + y.mean() * 20) / (stats_full["count"] + 20)
        te_out[f"{name}_prior"] = pd.Series(lab_te).map(smooth_full).fillna(y.mean()).values
        tr_out[f"{name}_entropy"] = -(tr_out[f"{name}_prior"] * np.log(np.clip(tr_out[f"{name}_prior"], 1e-6, 1)) + (1 - tr_out[f"{name}_prior"]) * np.log(np.clip(1 - tr_out[f"{name}_prior"], 1e-6, 1)))
        te_out[f"{name}_entropy"] = -(te_out[f"{name}_prior"] * np.log(np.clip(te_out[f"{name}_prior"], 1e-6, 1)) + (1 - te_out[f"{name}_prior"]) * np.log(np.clip(1 - te_out[f"{name}_prior"], 1e-6, 1)))
        print(f"cluster {n} done")
    return tr_out.astype("float32"), te_out.astype("float32")


def fit_stack(train_feat: pd.DataFrame, test_feat: pd.DataFrame, y: pd.Series, folds, tag: str):
    from sklearn.ensemble import HistGradientBoostingClassifier

    exp_id = f"{timestamp()}_{tag}_phase3"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(len(train_feat))
    test_pred = np.zeros(len(test_feat))
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.035,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=0.03,
            random_state=20260519 + fold,
        )
        model.fit(train_feat.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(train_feat.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(test_feat)[:, 1] / len(folds)
        scores.append(auc(y.iloc[va_idx], oof[va_idx]))
        with open(MODEL_DIR / f"{exp_id}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"phase3 stack fold {fold}: {scores[-1]:.6f}")
    cv = auc(y, oof)
    return exp_id, oof, test_pred, scores, cv, exp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--metrics", default="euclidean,cosine")
    parser.add_argument("--ks", default="7,25,80")
    parser.add_argument("--pca", type=int, default=32)
    parser.add_argument("--tag", default="neighborhood")
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.model_selection import StratifiedKFold

    train_raw, test_raw, sub = load_raw()
    y = train_raw[TARGET].astype(int)
    X, X_test, embed_cols = make_embedding(train_raw, test_raw)
    folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260519).split(X, y))
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    knn_tr, knn_te = build_knn_features(X, X_test, y, folds, metrics, ks, args.pca)
    clu_tr, clu_te = cluster_features(X, X_test, y, folds, [64, 128, 256])
    base_train, base_test = load_base_predictions()
    base_cols = [c for c in base_train.columns if c.startswith("base_")]
    meta_tr = pd.concat([base_train[base_cols], knn_tr, clu_tr], axis=1).replace([np.inf, -np.inf], np.nan)
    meta_te = pd.concat([base_test[base_cols], knn_te, clu_te], axis=1).replace([np.inf, -np.inf], np.nan)
    meta_tr = meta_tr.fillna(meta_tr.median(numeric_only=True)).astype("float32")
    meta_te = meta_te.fillna(meta_tr.median(numeric_only=True)).astype("float32")
    exp_id, oof, test_pred, scores, cv, exp_dir = fit_stack(meta_tr, meta_te, y, folds, args.tag)
    pd.DataFrame({"id": train_raw["id"], TARGET: y, "phase3_oof": oof}).to_csv(exp_dir / "oof_phase3.csv", index=False)
    pd.DataFrame({"id": test_raw["id"], TARGET: test_pred}).to_csv(exp_dir / "test_phase3.csv", index=False)
    sub.assign(**{TARGET: test_pred}).to_csv(SUB_DIR / f"{exp_id}_{cv:.6f}.csv", index=False)
    pd.DataFrame({"feature": meta_tr.columns}).to_csv(exp_dir / "phase3_feature_columns.csv", index=False)
    summary = {
        "exp_id": exp_id,
        "oof_auc": cv,
        "fold_auc": scores,
        "metrics": metrics,
        "ks": ks,
        "pca": args.pca,
        "embedding_columns": embed_cols,
        "meta_features": list(meta_tr.columns),
    }
    write_json(summary, exp_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
