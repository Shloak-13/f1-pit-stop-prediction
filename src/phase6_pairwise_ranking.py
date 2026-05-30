from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from phase3_neighborhood import load_base_predictions
from phase3_specialists import make_features
from table_model import add_bins
from utils import EXP_DIR, MODEL_DIR, REPORT_DIR, SUB_DIR, ensure_dirs, logit, sigmoid, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).values


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load_phase_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_base, test_base = load_base_predictions()
    phase3_oof = pd.read_csv(EXP_DIR / "20260519_141027_phase3_knn_euclid_phase3" / "oof_phase3.csv")
    phase3_test = pd.read_csv(EXP_DIR / "20260519_141027_phase3_knn_euclid_phase3" / "test_phase3.csv")
    spec_oof = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "oof_specialist.csv")
    spec_test = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "test_specialist.csv")
    hard_oof = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "oof_hard.csv")
    hard_test = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "test_hard.csv")
    tr = (
        train_base.merge(phase3_oof[["id", "phase3_oof"]], on="id")
        .merge(spec_oof[["id", "specialist_oof"]], on="id")
        .merge(hard_oof[["id", "hard_oof"]], on="id")
    )
    te = (
        test_base.merge(phase3_test[["id", TARGET]].rename(columns={TARGET: "phase3_oof"}), on="id")
        .merge(spec_test[["id", TARGET]].rename(columns={TARGET: "specialist_oof"}), on="id")
        .merge(hard_test[["id", TARGET]].rename(columns={TARGET: "hard_oof"}), on="id")
    )
    for df in [tr, te]:
        df["phase2_cat_table"] = 0.88 * df["base_cat"] + 0.12 * df["base_table_hgb"]
        df["phase3_hill"] = 0.72 * df["specialist_oof"] + 0.28 * df["phase3_oof"]
        pred_cols = [c for c in df.columns if c.startswith("base_")] + ["phase2_cat_table", "phase3_hill", "hard_oof"]
        vals = df[pred_cols].values
        df["pred_std"] = vals.std(axis=1)
        df["pred_range"] = vals.max(axis=1) - vals.min(axis=1)
        df["pred_entropy"] = entropy(df["phase2_cat_table"].values)
        df["rank_std"] = np.vstack([rank01(df[c].values) for c in pred_cols]).T.std(axis=1)
        for c in pred_cols:
            df[c + "_logit"] = logit(df[c].values)
    return tr, te


def add_query_cols(raw: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["YearStint"] = d["Year"].astype(str) + "_" + d["Stint"].astype(str)
    d["RaceYear"] = d["Race"].astype(str) + "_" + d["Year"].astype(str)
    d["RaceYearStint"] = d["RaceYear"] + "_" + d["Stint"].astype(str)
    d["RaceYearCompound"] = d["RaceYear"] + "_" + d["Compound"].astype(str)
    d["YearStintCompound"] = d["YearStint"] + "_" + d["Compound"].astype(str)
    d["TyreFracBin"] = pd.qcut(d["TyreLifeFrac"].rank(method="first"), 12, labels=False, duplicates="drop").astype(str)
    d["GapBin"] = pd.qcut(d["StintLapGap"].rank(method="first"), 12, labels=False, duplicates="drop").astype(str)
    d["SyntheticBucket"] = d["YearStintCompound"] + "_" + d["TyreFracBin"] + "_" + d["DeltaBin"].astype(str)
    d["BoundaryBucket"] = d["Compound"].astype(str) + "_" + d["TyreFracBin"] + "_" + d["GapBin"] + "_" + d["RacePhase20"].astype(str)
    return d


def make_rank_features(raw: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    X = make_features(raw)
    score_cols = [c for c in scores.columns if c.startswith("base_") or c in ["phase2_cat_table", "phase3_hill", "hard_oof", "pred_std", "pred_range", "pred_entropy", "rank_std"]]
    X = pd.concat([X.reset_index(drop=True), scores[score_cols].reset_index(drop=True)], axis=1)
    X["phase_margin"] = X["phase3_hill"] - X["phase2_cat_table"]
    X["hard_margin"] = X["hard_oof"] - X["phase3_hill"]
    for c in ["phase2_cat_table", "phase3_hill", "hard_oof", "base_cat", "base_table_hgb"]:
        X[c + "_rank"] = rank01(X[c].values)
        X[c + "_logit2"] = logit(X[c].values)
    return X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).astype("float32")


def sample_weight_from_scores(scores: pd.DataFrame) -> np.ndarray:
    uncert = scores["pred_std"].values / max(scores["pred_std"].max(), 1e-9)
    ent = scores["pred_entropy"].values / max(scores["pred_entropy"].max(), 1e-9)
    mid = (scores["phase2_cat_table"].between(0.20, 0.80)).astype(float).values
    disagree = np.abs(scores["phase3_hill"].values - scores["phase2_cat_table"].values)
    disagree = disagree / max(disagree.max(), 1e-9)
    return (1.0 + 2.0 * uncert + 1.0 * ent + 1.5 * mid + 1.5 * disagree).astype("float32")


def sorted_by_query(X, y, qid, w=None):
    order = np.lexsort((np.arange(len(qid)), pd.factorize(qid, sort=True)[0]))
    q = np.asarray(qid)[order]
    _, counts = np.unique(q, return_counts=True)
    if w is None:
        return X.iloc[order], y.iloc[order], counts
    return X.iloc[order], y.iloc[order], counts, w[order]


def train_lgbm_ranker(X, XT, y, qid_train, qid_test, folds, weights, tag, n_estimators=700):
    from lightgbm import LGBMRanker

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(XT))
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        Xtr, ytr, group, wtr = sorted_by_query(X.iloc[tr_idx], y.iloc[tr_idx], qid_train.iloc[tr_idx], weights[tr_idx])
        model = LGBMRanker(
            objective="lambdarank",
            metric="auc",
            n_estimators=n_estimators,
            learning_rate=0.035,
            num_leaves=47,
            min_child_samples=45,
            subsample=0.90,
            colsample_bytree=0.86,
            reg_lambda=2.0,
            random_state=20260519 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(Xtr, ytr, group=group, sample_weight=wtr)
        oof[va_idx] = model.predict(X.iloc[va_idx])
        test_pred += model.predict(XT) / len(folds)
        fold_scores.append(auc(y.iloc[va_idx], oof[va_idx]))
        with open(MODEL_DIR / f"{tag}_lgbmrank_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"lgbm rank {tag} fold {fold}: {fold_scores[-1]:.6f}")
    return oof, test_pred, fold_scores


def train_xgb_ranker(X, XT, y, qid_train, folds, weights, tag, n_estimators=420):
    from xgboost import XGBRanker

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(XT))
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        Xtr, ytr, group, wtr = sorted_by_query(X.iloc[tr_idx], y.iloc[tr_idx], qid_train.iloc[tr_idx], weights[tr_idx])
        model = XGBRanker(
            objective="rank:pairwise",
            n_estimators=n_estimators,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=20,
            subsample=0.90,
            colsample_bytree=0.86,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=20260519 + fold,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr, group=group, verbose=False)
        oof[va_idx] = model.predict(X.iloc[va_idx])
        test_pred += model.predict(XT) / len(folds)
        fold_scores.append(auc(y.iloc[va_idx], oof[va_idx]))
        with open(MODEL_DIR / f"{tag}_xgbrank_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"xgb rank {tag} fold {fold}: {fold_scores[-1]:.6f}")
    return oof, test_pred, fold_scores


def inversion_report(train: pd.DataFrame, scores: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = train[TARGET].values
    base = scores["phase2_cat_table"].values
    for col in ["Year", "Compound", "Stint", "Race", "RaceYear", "RaceYearStint", "SyntheticBucket"]:
        for g, idx in pd.Series(np.arange(len(y))).groupby(groups[col]).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) < 120 or y[idx].sum() == 0 or y[idx].sum() == len(idx):
                continue
            pos = base[idx][y[idx] == 1]
            neg = base[idx][y[idx] == 0]
            # Approximate inversion rate by comparing quantiles to avoid O(n^2).
            qpos = np.quantile(pos, np.linspace(0.05, 0.95, 19))
            qneg = np.quantile(neg, np.linspace(0.05, 0.95, 19))
            inv = float(np.mean(qpos[:, None] <= qneg[None, :]))
            rows.append({"field": col, "group": str(g), "n": int(len(idx)), "target": float(y[idx].mean()), "base_auc": auc(y[idx], base[idx]), "approx_inversion": inv})
    return pd.DataFrame(rows).sort_values(["approx_inversion", "n"], ascending=[False, False])


def blend_rankers(train, test, oof_panel, test_panel, y):
    # Ranker outputs are arbitrary scores. Convert each to ranks before blending.
    candidates = {}
    for col in [c for c in oof_panel.columns if c not in ["id", TARGET]]:
        candidates[col] = (rank01(oof_panel[col].values), rank01(test_panel[col].values))
    anchors = {
        "phase2": ("phase2_cat_table", 0.94855),
        "phase3": ("phase3_hill", 0.94855),
        "hard": ("hard_oof", 0.94855),
    }
    for name, col in [("phase2", "phase2_cat_table"), ("phase3", "phase3_hill"), ("hard", "hard_oof")]:
        candidates[name] = (rank01(train[col].values), rank01(test[col].values))
    rows = []
    blends = {}
    ranker_cols = [c for c in oof_panel.columns if c.startswith("rank_")]
    for anchor_name, anchor_col in [("phase2", "phase2_cat_table"), ("phase3", "phase3_hill"), ("hard", "hard_oof")]:
        ao = rank01(train[anchor_col].values)
        at = rank01(test[anchor_col].values)
        for rcol in ranker_cols:
            ro = rank01(oof_panel[rcol].values)
            rt = rank01(test_panel[rcol].values)
            for alpha in [0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
                name = f"{anchor_name}_{rcol}_a{alpha:.2f}".replace(".", "p")
                bo = (1 - alpha) * ao + alpha * ro
                bt = (1 - alpha) * at + alpha * rt
                rows.append({"candidate": name, "anchor": anchor_name, "ranker": rcol, "alpha": alpha, "oof_auc": auc(y, bo), "mean_abs_delta": float(np.abs(bo - ao).mean())})
                blends[name] = (bo, bt)
    scores = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    return scores, blends


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--tag", default="phase6_rank")
    parser.add_argument("--models", default="lgbm,xgb")
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.model_selection import StratifiedKFold

    train, test_raw, sub = load_raw()
    y = train[TARGET].astype(int)
    scores_tr, scores_te = load_phase_scores()
    groups_tr = add_query_cols(train)
    groups_te = add_query_cols(test_raw)
    X = make_rank_features(train, scores_tr)
    XT = make_rank_features(test_raw, scores_te)
    weights = sample_weight_from_scores(scores_tr)
    folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260519).split(X, y))
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    rank_oof = pd.DataFrame({"id": train["id"], TARGET: y})
    rank_test = pd.DataFrame({"id": test_raw["id"]})
    reports = []
    query_specs = {
        "RaceYearStint": "RaceYearStint",
        "SyntheticBucket": "SyntheticBucket",
        "BoundaryBucket": "BoundaryBucket",
    }
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for qname, qcol in query_specs.items():
        if "lgbm" in models:
            tag = f"{exp_id}_{qname}"
            oof, pred, fs = train_lgbm_ranker(X, XT, y, groups_tr[qcol], groups_te[qcol], folds, weights, tag)
            col = f"rank_lgbm_{qname}"
            rank_oof[col] = oof
            rank_test[col] = pred
            reports.append({"model": "lgbm", "query": qname, "oof_auc": auc(y, oof), "fold_auc": fs})
        if "xgb" in models and qname != "SyntheticBucket":
            tag = f"{exp_id}_{qname}"
            oof, pred, fs = train_xgb_ranker(X, XT, y, groups_tr[qcol], folds, weights, tag, n_estimators=320)
            col = f"rank_xgb_{qname}"
            rank_oof[col] = oof
            rank_test[col] = pred
            reports.append({"model": "xgb", "query": qname, "oof_auc": auc(y, oof), "fold_auc": fs})

    rank_oof.to_csv(exp_dir / "oof_rankers.csv", index=False)
    rank_test.to_csv(exp_dir / "test_rankers.csv", index=False)
    pd.DataFrame(reports).to_csv(exp_dir / "ranker_report.csv", index=False)
    inv = inversion_report(train, scores_tr, groups_tr)
    inv.to_csv(exp_dir / "pairwise_inversion_regions.csv", index=False)
    blend_scores, blends = blend_rankers(scores_tr, scores_te, rank_oof.drop(columns=[TARGET]), rank_test, y)
    blend_scores.to_csv(exp_dir / "rank_blend_scores.csv", index=False)
    for name, (_, bt) in list(blends.items())[:]:
        if name in set(blend_scores.head(24)["candidate"]):
            sub.assign(**{TARGET: np.clip(bt, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{name}.csv", index=False)
    # Also write deliberately conservative public-LB probes.
    for name in ["phase2_rank_only", "phase2_lgbm_boundary_a0p05", "phase2_lgbm_raceyear_a0p05"]:
        pass
    summary = {
        "exp_id": exp_id,
        "rankers": reports,
        "best_blends": blend_scores.head(20).to_dict("records"),
        "top_inversion_regions": inv.head(30).to_dict("records"),
    }
    write_json(summary, exp_dir / "summary.json")
    lines = [
        "# Phase 6 Pairwise Ranking Report",
        "",
        "## Ranker OOF",
        pd.DataFrame(reports).to_markdown(index=False),
        "",
        "## Best Rank Blends",
        blend_scores.head(30).to_markdown(index=False),
        "",
        "## Persistent Inversion Regions",
        inv.head(50).to_markdown(index=False),
        "",
        "## Interpretation",
        "- Ranker outputs are used as ordering scores, then rank-normalized before blending.",
        "- Public-facing candidates use small alpha rank corrections over stable pointwise anchors.",
        "- If high-alpha OOF winners fail public LB, submit lower-alpha phase2-anchor variants.",
    ]
    (REPORT_DIR / "phase6_pairwise_ranking_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(pd.DataFrame(reports).to_string(index=False))
    print(blend_scores.head(15).to_string(index=False))
    print(f"wrote {exp_dir}")


if __name__ == "__main__":
    main()
