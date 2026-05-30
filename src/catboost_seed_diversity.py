from __future__ import annotations

import argparse
import itertools
import pickle

import numpy as np
import pandas as pd

from catboost_phase2 import make_frame
from features import TARGET, load_raw
from utils import EXP_DIR, MODEL_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).values


def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank().corr(method="pearson")


def train_one(X, X_test, y, folds, cat_idx, seed: int, random_strength: float, bagging_temperature: float, iterations: int, exp_id: str):
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=random_strength,
            bootstrap_type="Bayesian",
            bagging_temperature=bagging_temperature,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed * 100 + fold,
            allow_writing_files=False,
            verbose=False,
            od_type="Iter",
            od_wait=80,
        )
        tr_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
        va_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
        te_pool = Pool(X_test, cat_features=cat_idx)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(va_pool)[:, 1]
        test_pred += model.predict_proba(te_pool)[:, 1] / len(folds)
        scores.append(auc(y.iloc[va_idx], oof[va_idx]))
        with open(MODEL_DIR / f"{exp_id}_seed{seed}_rs{random_strength}_bt{bagging_temperature}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
    return oof, test_pred, scores


def greedy_ensemble(y: np.ndarray, preds: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    scores = {c: auc(y, preds[c].values) for c in preds.columns}
    best = max(scores, key=scores.get)
    blend = preds[best].values.copy()
    recipe = [{"model": best, "weight": 1.0, "auc": scores[best]}]
    while True:
        best_step = None
        for c in preds.columns:
            for w in np.linspace(0.05, 0.50, 10):
                cand = (1 - w) * blend + w * preds[c].values
                score = auc(y, cand)
                if best_step is None or score > best_step[0]:
                    best_step = (score, c, w, cand)
        if best_step is None or best_step[0] < recipe[-1]["auc"] + 0.0001:
            break
        blend = best_step[3]
        recipe.append({"model": best_step[1], "weight": float(best_step[2]), "auc": float(best_step[0])})
    return blend, recipe


def low_corr_ensemble(y: np.ndarray, preds: pd.DataFrame, threshold: float = 0.97) -> tuple[np.ndarray, list[str], float]:
    corr = preds.corr()
    order = sorted(preds.columns, key=lambda c: auc(y, preds[c].values), reverse=True)
    selected = []
    for c in order:
        if not selected or all(abs(corr.loc[c, s]) < threshold for s in selected):
            selected.append(c)
    if not selected:
        selected = [order[0]]
    blend = np.mean([rank01(preds[c].values) for c in selected], axis=0)
    return blend, selected, auc(y, blend)


def materialize_test_blend(test_preds: pd.DataFrame, recipe: list[dict]) -> np.ndarray:
    blend = test_preds[recipe[0]["model"]].values.copy()
    for step in recipe[1:]:
        w = step["weight"]
        blend = (1 - w) * blend + w * test_preds[step["model"]].values
    return blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-models", type=int, default=16)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--tag", default="cat_seed_diversity")
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.model_selection import StratifiedKFold

    train, test, sub = load_raw()
    X, cat_cols = make_frame(train)
    X_test, _ = make_frame(test)
    y = train[TARGET].astype(int)
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260519).split(X, y))
    grid = list(itertools.product(range(24), [0.1, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0]))
    # Deterministic coverage of seeds 0-23 before repeating parameter pairs too heavily.
    configs = []
    for seed in range(24):
        rs = [0.1, 0.5, 1.0, 2.0][seed % 4]
        bt = [0.0, 0.5, 1.0, 2.0][(seed // 4) % 4]
        configs.append((seed, rs, bt))
    configs = configs[: args.max_models]
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    oof_panel = pd.DataFrame({"id": train["id"], TARGET: y})
    test_panel = pd.DataFrame({"id": test["id"]})
    rows = []
    for seed, rs, bt in configs:
        name = f"cat_seed{seed}_rs{rs}_bt{bt}".replace(".", "p")
        print(f"training {name}")
        oof, pred, fold_scores = train_one(X, X_test, y, folds, cat_idx, seed, rs, bt, args.iterations, exp_id)
        oof_panel[name] = oof
        test_panel[name] = pred
        score = auc(y, oof)
        rows.append({"model": name, "seed": seed, "random_strength": rs, "bagging_temperature": bt, "oof_auc": score, "fold_auc": fold_scores})
        pd.DataFrame(rows).to_csv(exp_dir / "seed_model_report.csv", index=False)
        oof_panel.to_csv(exp_dir / "oof_seed_models.csv", index=False)
        test_panel.to_csv(exp_dir / "test_seed_models.csv", index=False)
        sub.assign(**{TARGET: pred}).to_csv(SUB_DIR / f"{exp_id}_{name}_{score:.6f}.csv", index=False)
        print(f"{name}: {score:.6f}")
    pred_cols = [c for c in oof_panel.columns if c not in ["id", TARGET]]
    pearson = oof_panel[pred_cols].corr()
    spearman = spearman_corr(oof_panel[pred_cols])
    pearson.to_csv(exp_dir / "pearson_oof_correlation.csv")
    spearman.to_csv(exp_dir / "spearman_oof_correlation.csv")
    greedy_oof, recipe = greedy_ensemble(y.values, oof_panel[pred_cols])
    greedy_test = materialize_test_blend(test_panel[pred_cols], recipe)
    low_oof, low_models, low_auc = low_corr_ensemble(y.values, oof_panel[pred_cols], threshold=0.97)
    low_test = np.mean([rank01(test_panel[c].values) for c in low_models], axis=0)
    # Original Phase 3 specialist geometry.
    phase3_oof_path = EXP_DIR / "20260519_143756_phase3_specialists" / "oof_specialist.csv"
    phase3_test_path = EXP_DIR / "20260519_143756_phase3_specialists" / "test_specialist.csv"
    phase3_oof = pd.read_csv(phase3_oof_path)["specialist_oof"].values if phase3_oof_path.exists() else oof_panel[pred_cols[0]].values
    phase3_test = pd.read_csv(phase3_test_path)[TARGET].values if phase3_test_path.exists() else test_panel[pred_cols[0]].values
    blends = {
        "greedy": (greedy_oof, greedy_test),
        "lowcorr": (low_oof, low_test),
        "rank_final_50_25_25": (
            0.50 * rank01(phase3_oof) + 0.25 * rank01(greedy_oof) + 0.25 * rank01(low_oof),
            0.50 * rank01(phase3_test) + 0.25 * rank01(greedy_test) + 0.25 * rank01(low_test),
        ),
        "rank_final_50_35_15": (
            0.50 * rank01(phase3_oof) + 0.35 * rank01(greedy_oof) + 0.15 * rank01(low_oof),
            0.50 * rank01(phase3_test) + 0.35 * rank01(greedy_test) + 0.15 * rank01(low_test),
        ),
        "rank_final_50_15_35": (
            0.50 * rank01(phase3_oof) + 0.15 * rank01(greedy_oof) + 0.35 * rank01(low_oof),
            0.50 * rank01(phase3_test) + 0.15 * rank01(greedy_test) + 0.35 * rank01(low_test),
        ),
    }
    blend_rows = []
    for name, (bo, bt) in blends.items():
        score = auc(y, bo)
        blend_rows.append({"blend": name, "oof_auc": score})
        sub.assign(**{TARGET: np.clip(bt, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{name}_{score:.6f}.csv", index=False)
    pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False).to_csv(exp_dir / "blend_report.csv", index=False)
    summary = {
        "exp_id": exp_id,
        "models_trained": len(pred_cols),
        "best_single": max(rows, key=lambda r: r["oof_auc"]) if rows else None,
        "greedy_recipe": recipe,
        "greedy_auc": auc(y, greedy_oof),
        "lowcorr_models": low_models,
        "lowcorr_auc": low_auc,
        "pearson_min": float(pearson.where(~np.eye(len(pearson), dtype=bool)).min().min()) if len(pearson) > 1 else None,
        "pearson_mean": float(pearson.where(~np.eye(len(pearson), dtype=bool)).stack().mean()) if len(pearson) > 1 else None,
        "spearman_min": float(spearman.where(~np.eye(len(spearman), dtype=bool)).min().min()) if len(spearman) > 1 else None,
        "blend_report": blend_rows,
    }
    write_json(summary, exp_dir / "summary.json")
    lines = [
        "# CatBoost Seed Diversity",
        "",
        "## Models",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "## Greedy Recipe",
        pd.DataFrame(recipe).to_markdown(index=False),
        "",
        "## Low-Correlation Ensemble",
        f"- Models: {', '.join(low_models)}",
        f"- OOF AUC: {low_auc:.6f}",
        "",
        "## Blends",
        pd.DataFrame(blend_rows).sort_values("oof_auc", ascending=False).to_markdown(index=False),
    ]
    (exp_dir / "seed_diversity_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "catboost_seed_diversity_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
