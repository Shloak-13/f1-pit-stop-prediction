from __future__ import annotations

import argparse
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

    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).values


def fast_configs(n: int = 12) -> list[dict]:
    random_strengths = [0.1, 0.5, 1.0]
    bagging_temperatures = [0.0, 0.5, 1.0, 2.0]
    rsm_values = [1.0, 0.95, 0.90]
    configs = []
    for seed in range(12):
        configs.append(
            {
                "seed": seed,
                "fold_seed": 20260520 + seed * 17,
                "random_strength": random_strengths[seed % len(random_strengths)],
                "bagging_temperature": bagging_temperatures[seed % len(bagging_temperatures)],
                "rsm": rsm_values[(seed // 4) % len(rsm_values)],
            }
        )
    return configs[:n]


def group_scores(raw: pd.DataFrame, y: pd.Series, pred: np.ndarray) -> dict:
    d = add_bins(raw)
    d["RaceYear"] = d["Race"].astype(str) + "_" + d["Year"].astype(str)
    d["YearStint"] = d["Year"].astype(str) + "_" + d["Stint"].astype(str)
    out = {}
    for col in ["Year", "Compound", "Stint", "RaceYear", "YearStint"]:
        vals = []
        for _, idx in pd.Series(np.arange(len(y))).groupby(d[col]).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) >= 200 and y.iloc[idx].nunique() == 2:
                vals.append(auc(y.iloc[idx], pred[idx]))
        if vals:
            out[f"{col}_macro"] = float(np.mean(vals))
            out[f"{col}_p10"] = float(np.quantile(vals, 0.10))
            out[f"{col}_std"] = float(np.std(vals))
    return out


def train_one(X, XT, y, raw_train, raw_test, sub, cat_idx, folds, cfg, iterations, exp_id, exp_dir):
    from catboost import CatBoostClassifier, Pool

    name = f"seed{cfg['seed']:02d}_fs{cfg['fold_seed']}_rs{cfg['random_strength']}_bt{cfg['bagging_temperature']}_rsm{cfg['rsm']}".replace(".", "p")
    oof_path = exp_dir / f"oof_{name}.csv"
    test_path = exp_dir / f"test_{name}.csv"
    meta_path = exp_dir / f"meta_{name}.json"
    if oof_path.exists() and test_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"skip {name}: {meta['oof_auc']:.6f}")
        return meta

    oof = np.zeros(len(X), dtype=np.float32)
    pred = np.zeros(len(XT), dtype=np.float32)
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
            od_wait=40,
            thread_count=-1,
        )
        tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
        te_pool = Pool(XT, cat_features=cat_idx)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(va_pool)[:, 1]
        pred += model.predict_proba(te_pool)[:, 1] / len(folds)
        fold_auc = auc(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(fold_auc)
        with open(MODEL_DIR / f"{exp_id}_{name}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{name} fold {fold}: {fold_auc:.6f}")

    meta = {
        "name": name,
        **cfg,
        "iterations": iterations,
        "fold_auc": fold_scores,
        "fold_std": float(np.std(fold_scores)),
        "oof_auc": auc(y, oof),
        "subgroup": group_scores(raw_train, y, oof),
    }
    pd.DataFrame({"id": raw_train["id"], "prediction": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({"id": raw_test["id"], "prediction": pred}).to_csv(test_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    sub.assign(**{TARGET: pred}).to_csv(SUB_DIR / f"{exp_id}_{name}_{meta['oof_auc']:.6f}.csv", index=False)
    print(f"completed {name}: {meta['oof_auc']:.6f}")
    return meta


def load_panels(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    oof_parts = []
    test_parts = []
    metas = []
    for meta_path in sorted(exp_dir.glob("meta_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        name = meta["name"]
        oof_path = exp_dir / f"oof_{name}.csv"
        test_path = exp_dir / f"test_{name}.csv"
        if not oof_path.exists() or not test_path.exists():
            continue
        oof_parts.append(pd.read_csv(oof_path).rename(columns={"prediction": name}))
        test_parts.append(pd.read_csv(test_path).rename(columns={"prediction": name}))
        metas.append(meta)
    if not oof_parts:
        return pd.DataFrame(), pd.DataFrame(), []
    oof = oof_parts[0]
    test = test_parts[0]
    for p in oof_parts[1:]:
        oof = oof.merge(p, on="id")
    for p in test_parts[1:]:
        test = test.merge(p, on="id")
    return oof, test, metas


def greedy_rank(y: np.ndarray, panel: pd.DataFrame, min_gain: float = 0.00005) -> tuple[np.ndarray, list[dict]]:
    cols = [c for c in panel.columns if c != "id"]
    best = max(cols, key=lambda c: auc(y, panel[c].values))
    blend = rank01(panel[best].values)
    recipe = [{"model": best, "weight": 1.0, "auc": auc(y, blend)}]
    while True:
        best_step = None
        for c in cols:
            r = rank01(panel[c].values)
            for w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
                cand = (1 - w) * blend + w * r
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


def lowcorr_rank(y: np.ndarray, panel: pd.DataFrame, threshold: float = 0.97) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in panel.columns if c != "id"]
    corr = panel[cols].rank().corr()
    order = sorted(cols, key=lambda c: auc(y, panel[c].values), reverse=True)
    selected = []
    for c in order:
        if not selected or all(abs(corr.loc[c, s]) < threshold for s in selected):
            selected.append(c)
    if not selected:
        selected = [order[0]]
    return np.mean([rank01(panel[c].values) for c in selected], axis=0), selected


def geom_rank(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    eps = 1e-6
    out = np.ones_like(arrays[0], dtype=float)
    for a, w in zip(arrays, weights):
        out *= np.clip(a, eps, 1) ** w
    return out


def trimmed_rank(arrays: list[np.ndarray]) -> np.ndarray:
    mat = np.vstack(arrays)
    if mat.shape[0] <= 2:
        return mat.mean(axis=0)
    return np.sort(mat, axis=0)[1:-1].mean(axis=0)


def rebuild(exp_id: str, exp_dir: Path, train, test_raw, sub):
    y = train[TARGET].astype(int).values
    oof, test, metas = load_panels(exp_dir)
    if oof.empty:
        return None
    cols = [c for c in oof.columns if c != "id"]
    pearson = oof[cols].corr()
    spearman = oof[cols].rank().corr()
    pearson.to_csv(exp_dir / "pearson_correlation.csv")
    spearman.to_csv(exp_dir / "spearman_correlation.csv")
    greedy_oof, recipe = greedy_rank(y, oof)
    greedy_test = apply_recipe(test, recipe)
    low_oof, low_models = lowcorr_rank(y, oof)
    low_test = np.mean([rank01(test[c].values) for c in low_models], axis=0)

    phase3_oof = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "oof_specialist.csv")["specialist_oof"].values
    phase3_test = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "test_specialist.csv")[TARGET].values
    hard_oof = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "oof_hard.csv")["hard_oof"].values
    hard_test = pd.read_csv(EXP_DIR / "20260519_150804_phase45_hard" / "test_hard.csv")[TARGET].values

    components_oof = [rank01(greedy_oof), rank01(low_oof), rank01(phase3_oof), rank01(hard_oof)]
    components_test = [rank01(greedy_test), rank01(low_test), rank01(phase3_test), rank01(hard_test)]
    blend_defs = {
        "rank_arith_25each": [0.25, 0.25, 0.25, 0.25],
        "rank_arith_50phase3_20hard_15g_15l": [0.15, 0.15, 0.50, 0.20],
        "rank_arith_50hard_20phase3_15g_15l": [0.15, 0.15, 0.20, 0.50],
        "rank_arith_50phase3_25g_25l": [0.25, 0.25, 0.50, 0.0],
        "rank_arith_50hard_25g_25l": [0.25, 0.25, 0.0, 0.50],
    }
    blend_rows = []
    for name, weights in blend_defs.items():
        bo = sum(w * a for w, a in zip(weights, components_oof))
        bt = sum(w * a for w, a in zip(weights, components_test))
        score = auc(y, bo)
        blend_rows.append({"candidate": name, "oof_auc": score, "type": "arithmetic"})
        sub.assign(**{TARGET: np.clip(bt, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{name}_{score:.6f}.csv", index=False)
        go = geom_rank(components_oof, weights)
        gt = geom_rank(components_test, weights)
        gscore = auc(y, go)
        blend_rows.append({"candidate": name.replace("rank_arith", "rank_geom"), "oof_auc": gscore, "type": "geometric"})
        sub.assign(**{TARGET: np.clip(gt, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{name.replace('rank_arith', 'rank_geom')}_{gscore:.6f}.csv", index=False)
    trim_oof = trimmed_rank(components_oof)
    trim_test = trimmed_rank(components_test)
    trim_score = auc(y, trim_oof)
    blend_rows.append({"candidate": "rank_trimmed_components", "oof_auc": trim_score, "type": "trimmed"})
    sub.assign(**{TARGET: np.clip(trim_test, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_rank_trimmed_components_{trim_score:.6f}.csv", index=False)

    model_rows = []
    for m in metas:
        row = {k: v for k, v in m.items() if k != "subgroup"}
        row.update({f"subgroup_{k}": v for k, v in m.get("subgroup", {}).items()})
        model_rows.append(row)
    pd.DataFrame(model_rows).sort_values("oof_auc", ascending=False).to_csv(exp_dir / "model_report.csv", index=False)
    pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False).to_csv(exp_dir / "blend_report.csv", index=False)
    summary = {
        "exp_id": exp_id,
        "completed_models": len(cols),
        "best_single": max([{"model": c, "oof_auc": auc(y, oof[c].values)} for c in cols], key=lambda r: r["oof_auc"]),
        "greedy_recipe": recipe,
        "greedy_auc": auc(y, greedy_oof),
        "lowcorr_models": low_models,
        "lowcorr_auc": auc(y, low_oof),
        "blend_report": blend_rows,
        "pearson_mean": float(pearson.where(~np.eye(len(pearson), dtype=bool)).stack().mean()) if len(cols) > 1 else None,
        "spearman_mean": float(spearman.where(~np.eye(len(spearman), dtype=bool)).stack().mean()) if len(cols) > 1 else None,
    }
    write_json(summary, exp_dir / "summary.json")
    report = [
        "# Fast CatBoost Diversity Report",
        "",
        "## Models",
        pd.DataFrame(model_rows).sort_values("oof_auc", ascending=False).to_markdown(index=False),
        "",
        "## Blends",
        pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False).to_markdown(index=False),
        "",
        "## Greedy Recipe",
        pd.DataFrame(recipe).to_markdown(index=False),
        "",
        f"Low-correlation models: {', '.join(low_models)}",
    ]
    (REPORT_DIR / "fast_catboost_diversity_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=140)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--tag", default="fast_catboost_diversity")
    parser.add_argument("--rebuild-only", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.model_selection import StratifiedKFold

    train, test_raw, sub = load_raw()
    X, cat_cols = make_frame(train)
    XT, _ = make_frame(test_raw)
    y = train[TARGET].astype(int)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    exp_id = args.exp_id or f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_only:
        for cfg in fast_configs(args.max_models):
            folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=cfg["fold_seed"]).split(X, y))
            train_one(X, XT, y, train, test_raw, sub, cat_idx, folds, cfg, args.iterations, exp_id, exp_dir)
            rebuild(exp_id, exp_dir, train, test_raw, sub)

    summary = rebuild(exp_id, exp_dir, train, test_raw, sub)
    print(summary)


if __name__ == "__main__":
    main()
