from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_phase2 import make_frame
from features import TARGET, load_raw
from table_model import add_bins
from utils import EXP_DIR, MODEL_DIR, REPORT_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).values


def add_groups(raw: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["RaceYear"] = d["Race"].astype(str) + "_" + d["Year"].astype(str)
    d["YearStint"] = d["Year"].astype(str) + "_" + d["Stint"].astype(str)
    d["RaceYearStint"] = d["RaceYear"] + "_" + d["Stint"].astype(str)
    d["TyreLifeFracBin12"] = pd.qcut(d["TyreLifeFrac"].rank(method="first"), 12, labels=False, duplicates="drop")
    d["DeltaBinGroup"] = d["DeltaBin"].astype(str)
    d["SyntheticBucket"] = d["YearStint"] + "_" + d["Compound"].astype(str) + "_" + d["TyreLifeFracBin12"].astype(str) + "_" + d["DeltaBinGroup"]
    return d


def make_folds(raw: pd.DataFrame, X: pd.DataFrame, y: pd.Series, system: str, seed: int, n_splits: int):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if system == "stratified":
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
    groups = add_groups(raw)
    if system == "race":
        g = groups["RaceYear"]
        return list(GroupKFold(n_splits=min(n_splits, g.nunique())).split(X, y, g))
    if system == "year":
        g = groups["Year"].astype(str)
        return list(GroupKFold(n_splits=min(4, g.nunique())).split(X, y, g))
    if system == "cluster":
        cols = ["TyreLife", "LapNumber", "RaceProgress", "Position", "LapTime_Delta", "Cumulative_Degradation", "Stint", "RaceLapsEst", "TyreLifeFrac", "StintLapGap"]
        mat = groups[cols].replace([np.inf, -np.inf], np.nan).fillna(groups[cols].median())
        mat = StandardScaler().fit_transform(mat)
        lab = MiniBatchKMeans(n_clusters=24, random_state=seed, batch_size=8192, n_init="auto").fit_predict(mat)
        return list(GroupKFold(n_splits=n_splits).split(X, y, lab))
    raise ValueError(system)


def subgroup_scores(raw: pd.DataFrame, y: pd.Series, pred: np.ndarray) -> dict:
    groups = add_groups(raw)
    out = {}
    for col in ["Year", "Compound", "Stint", "RaceYear", "YearStint", "RaceYearStint"]:
        vals = []
        for _, idx in pd.Series(np.arange(len(y))).groupby(groups[col]).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) >= 200 and y.iloc[idx].nunique() == 2:
                vals.append(auc(y.iloc[idx], pred[idx]))
        if vals:
            out[f"{col}_macro_auc"] = float(np.mean(vals))
            out[f"{col}_std_auc"] = float(np.std(vals))
            out[f"{col}_p10_auc"] = float(np.quantile(vals, 0.10))
            out[f"{col}_n_groups"] = int(len(vals))
    return out


def model_configs(max_models: int | None = None) -> list[dict]:
    configs = []
    systems = ["stratified", "race", "year", "cluster"]
    rsm_values = [1.0, 0.95, 0.90, 0.85]
    for seed in range(32):
        configs.append(
            {
                "seed": seed,
                "fold_system": systems[seed % len(systems)],
                "random_strength": [0.1, 0.5, 1.0, 2.0][seed % 4],
                "bagging_temperature": [0.0, 0.5, 1.0, 2.0][(seed // 4) % 4],
                "rsm": rsm_values[(seed // 8) % len(rsm_values)],
            }
        )
    return configs if max_models is None else configs[:max_models]


def train_config(X, XT, y, raw, cat_idx, cfg, folds, iterations, exp_dir: Path, exp_id: str):
    from catboost import CatBoostClassifier, Pool

    name = f"seed{cfg['seed']:02d}_{cfg['fold_system']}_rs{cfg['random_strength']}_bt{cfg['bagging_temperature']}_rsm{cfg['rsm']}".replace(".", "p")
    oof_path = exp_dir / f"oof_{name}.csv"
    test_path = exp_dir / f"test_{name}.csv"
    meta_path = exp_dir / f"meta_{name}.json"
    if oof_path.exists() and test_path.exists() and meta_path.exists():
        print(f"skip completed {name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return name, pd.read_csv(oof_path)["prediction"].values, pd.read_csv(test_path)["prediction"].values, meta

    oof = np.zeros(len(X))
    pred = np.zeros(len(XT))
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=cfg["random_strength"],
            bootstrap_type="Bayesian",
            bagging_temperature=cfg["bagging_temperature"],
            rsm=cfg["rsm"],
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=cfg["seed"] * 1000 + fold,
            allow_writing_files=False,
            verbose=False,
            od_type="Iter",
            od_wait=80,
        )
        tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
        te_pool = Pool(XT, cat_features=cat_idx)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(va_pool)[:, 1]
        pred += model.predict_proba(te_pool)[:, 1] / len(folds)
        fs = auc(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(fs)
        with open(MODEL_DIR / f"{exp_id}_{name}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{name} fold {fold}: {fs:.6f}")
    meta = {
        "name": name,
        **cfg,
        "fold_scores": fold_scores,
        "oof_auc": auc(y, oof),
        "subgroup": subgroup_scores(raw, y, oof),
    }
    pd.DataFrame({"id": raw["id"], "prediction": oof}).to_csv(oof_path, index=False)
    # test ids are not in raw
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return name, oof, pred, meta


def greedy_rank_ensemble(y: np.ndarray, panel: pd.DataFrame, min_gain: float = 0.00005):
    scores = {c: auc(y, panel[c].values) for c in panel.columns}
    best = max(scores, key=scores.get)
    blend = rank01(panel[best].values)
    recipe = [{"model": best, "weight": 1.0, "auc": auc(y, blend)}]
    while True:
        best_step = None
        for c in panel.columns:
            rc = rank01(panel[c].values)
            for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
                cand = (1 - w) * blend + w * rc
                score = auc(y, cand)
                if best_step is None or score > best_step[0]:
                    best_step = (score, c, w, cand)
        if best_step is None or best_step[0] < recipe[-1]["auc"] + min_gain:
            break
        blend = best_step[3]
        recipe.append({"model": best_step[1], "weight": best_step[2], "auc": best_step[0]})
    return blend, recipe


def apply_recipe(test_panel: pd.DataFrame, recipe: list[dict]) -> np.ndarray:
    blend = rank01(test_panel[recipe[0]["model"]].values)
    for step in recipe[1:]:
        blend = (1 - step["weight"]) * blend + step["weight"] * rank01(test_panel[step["model"]].values)
    return blend


def lowcorr_rank_ensemble(y: np.ndarray, panel: pd.DataFrame, threshold: float = 0.97):
    corr = panel.rank().corr()
    order = sorted(panel.columns, key=lambda c: auc(y, panel[c].values), reverse=True)
    selected = []
    for c in order:
        if not selected or all(abs(corr.loc[c, s]) < threshold for s in selected):
            selected.append(c)
    if not selected:
        selected = [order[0]]
    blend = np.mean([rank01(panel[c].values) for c in selected], axis=0)
    return blend, selected


def rebuild_ensembles(exp_dir: Path, train: pd.DataFrame, test: pd.DataFrame, sub: pd.DataFrame, exp_id: str):
    y = train[TARGET].astype(int).values
    oof_files = sorted(exp_dir.glob("oof_*.csv"))
    test_files = sorted(exp_dir.glob("test_*.csv"))
    if not oof_files:
        return None
    oof_panel = {}
    test_panel = {}
    for p in oof_files:
        name = p.stem.replace("oof_", "")
        tp = exp_dir / f"test_{name}.csv"
        if not tp.exists():
            continue
        oof_panel[name] = pd.read_csv(p)["prediction"].values
        test_panel[name] = pd.read_csv(tp)["prediction"].values
    if not oof_panel:
        return None
    oof_df = pd.DataFrame(oof_panel)
    test_df = pd.DataFrame(test_panel)
    pearson = oof_df.corr()
    spearman = oof_df.rank().corr()
    pearson.to_csv(exp_dir / "pearson_correlation.csv")
    spearman.to_csv(exp_dir / "spearman_correlation.csv")
    greedy_oof, recipe = greedy_rank_ensemble(y, oof_df)
    greedy_test = apply_recipe(test_df, recipe)
    low_oof, low_models = lowcorr_rank_ensemble(y, oof_df)
    low_test = np.mean([rank01(test_df[c].values) for c in low_models], axis=0)

    phase3_oof = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "oof_specialist.csv")["specialist_oof"].values
    phase3_test = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "test_specialist.csv")[TARGET].values
    hard_oof = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "oof_hard.csv")["hard_oof"].values
    hard_test = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "test_hard.csv")[TARGET].values
    blends = {
        "greedy_rank": (greedy_oof, greedy_test),
        "lowcorr_rank": (low_oof, low_test),
        "final_50phase3_25greedy_25low": (
            0.50 * rank01(phase3_oof) + 0.25 * greedy_oof + 0.25 * low_oof,
            0.50 * rank01(phase3_test) + 0.25 * greedy_test + 0.25 * low_test,
        ),
        "final_50hard_25greedy_25low": (
            0.50 * rank01(hard_oof) + 0.25 * greedy_oof + 0.25 * low_oof,
            0.50 * rank01(hard_test) + 0.25 * greedy_test + 0.25 * low_test,
        ),
        "final_50phase3_35greedy_15low": (
            0.50 * rank01(phase3_oof) + 0.35 * greedy_oof + 0.15 * low_oof,
            0.50 * rank01(phase3_test) + 0.35 * greedy_test + 0.15 * low_test,
        ),
    }
    blend_rows = []
    for name, (bo, bt) in blends.items():
        score = auc(y, bo)
        blend_rows.append({"blend": name, "oof_auc": score})
        sub.assign(**{TARGET: np.clip(bt, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{name}_{score:.6f}.csv", index=False)
    pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False).to_csv(exp_dir / "blend_report.csv", index=False)
    return {
        "n_models": len(oof_df.columns),
        "best_single": max([{"model": c, "auc": auc(y, oof_df[c].values)} for c in oof_df.columns], key=lambda r: r["auc"]),
        "greedy_recipe": recipe,
        "greedy_auc": auc(y, greedy_oof),
        "lowcorr_models": low_models,
        "lowcorr_auc": auc(y, low_oof),
        "blend_report": blend_rows,
        "pearson_mean": float(pearson.where(~np.eye(len(pearson), dtype=bool)).stack().mean()) if len(pearson) > 1 else None,
        "spearman_mean": float(spearman.where(~np.eye(len(spearman), dtype=bool)).stack().mean()) if len(spearman) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default=None, help="resume/rebuild an existing experiment id")
    parser.add_argument("--max-models", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--tag", default="catboost_diversity")
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    train, test, sub = load_raw()
    X, cat_cols = make_frame(train)
    XT, _ = make_frame(test)
    y = train[TARGET].astype(int)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    exp_id = args.exp_id or f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_only:
        for cfg in model_configs(args.max_models):
            folds = make_folds(train, X, y, cfg["fold_system"], 20260520 + cfg["seed"], args.splits)
            name, oof, pred, meta = train_config(X, XT, y, train, cat_idx, cfg, folds, args.iterations, exp_dir, exp_id)
            pd.DataFrame({"id": train["id"], "prediction": oof}).to_csv(exp_dir / f"oof_{name}.csv", index=False)
            pd.DataFrame({"id": test["id"], "prediction": pred}).to_csv(exp_dir / f"test_{name}.csv", index=False)
            pd.DataFrame([meta]).to_csv(exp_dir / "model_metadata_append.csv", mode="a", header=not (exp_dir / "model_metadata_append.csv").exists(), index=False)
            sub.assign(**{TARGET: pred}).to_csv(SUB_DIR / f"{exp_id}_{name}_{meta['oof_auc']:.6f}.csv", index=False)
            print(f"completed {name}: {meta['oof_auc']:.6f}")

    ensemble = rebuild_ensembles(exp_dir, train, test, sub, exp_id)
    metas = []
    for p in sorted(exp_dir.glob("meta_*.json")):
        metas.append(json.loads(p.read_text(encoding="utf-8")))
    pd.DataFrame(metas).to_csv(exp_dir / "model_report.csv", index=False)
    summary = {"exp_id": exp_id, "completed_models": len(metas), "ensemble": ensemble, "models": metas}
    write_json(summary, exp_dir / "summary.json")
    lines = [
        "# CatBoost Diversity Engine Report",
        "",
        "## Completed Models",
        pd.DataFrame(metas).to_markdown(index=False) if metas else "No models completed.",
        "",
        "## Ensemble",
        json.dumps(ensemble, indent=2, default=str),
    ]
    (REPORT_DIR / "catboost_diversity_engine_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
