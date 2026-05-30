from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from utils import EXP_DIR, SUB_DIR, ensure_dirs, logit, sigmoid, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def rank_average(preds: np.ndarray) -> np.ndarray:
    ranks = np.vstack([pd.Series(preds[:, i]).rank(pct=True).values for i in range(preds.shape[1])]).T
    return ranks.mean(axis=1)


def hill_climb(y: np.ndarray, oof: np.ndarray, names: list[str], rounds: int = 80) -> tuple[np.ndarray, list[dict]]:
    best_idx = int(np.argmax([auc(y, oof[:, i]) for i in range(oof.shape[1])]))
    blend = oof[:, best_idx].copy()
    recipe = [{"feature": names[best_idx], "weight": 1.0, "auc": auc(y, blend)}]
    for _ in range(rounds):
        best = None
        for j in range(oof.shape[1]):
            for w in np.linspace(0.02, 0.50, 25):
                cand = (1 - w) * blend + w * oof[:, j]
                score = auc(y, cand)
                if best is None or score > best[0]:
                    best = (score, j, w, cand)
        if best is None or best[0] <= recipe[-1]["auc"] + 1e-7:
            break
        blend = best[3]
        recipe.append({"feature": names[best[1]], "weight": float(best[2]), "auc": float(best[0])})
    return blend, recipe


def discover_experiment_files(exp_dirs: list[str]) -> tuple[list[Path], list[Path]]:
    oof_files, test_files = [], []
    for exp in exp_dirs:
        p = EXP_DIR / exp
        oof_files.extend(sorted(p.glob("oof_*.csv")))
        test_files.extend(sorted(p.glob("test_*.csv")))
    return oof_files, test_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", required=True, help="comma-separated experiment ids under experiments/")
    parser.add_argument("--tag", default="ensemble")
    args = parser.parse_args()
    ensure_dirs()
    train, test, sub = load_raw()
    exp_dirs = [x.strip() for x in args.experiments.split(",") if x.strip()]
    oof_files, test_files = discover_experiment_files(exp_dirs)
    if not oof_files:
        raise SystemExit("No OOF files found")
    y = train[TARGET].values
    oofs, tests, names = [], [], []
    for oof_path in oof_files:
        model_name = oof_path.stem.replace("oof_", "")
        test_path = oof_path.with_name(f"test_{model_name}.csv")
        if not test_path.exists():
            continue
        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_path)
        pred_col = [c for c in oof_df.columns if c.endswith("_oof")][0]
        oofs.append(oof_df[pred_col].values)
        tests.append(test_df[TARGET].values)
        names.append(f"{oof_path.parent.name}:{model_name}")
    O = np.vstack(oofs).T
    T = np.vstack(tests).T
    corr = pd.DataFrame(O, columns=names).corr()
    corr.to_csv(EXP_DIR / f"{args.tag}_prediction_correlation.csv")

    avg = O.mean(axis=1)
    rank = rank_average(O)
    logit_avg = sigmoid(np.mean(logit(O), axis=1))
    blend, recipe = hill_climb(y, O, names)
    scores = {
        "mean_auc": auc(y, avg),
        "rank_auc": auc(y, rank),
        "logit_auc": auc(y, logit_avg),
        "hill_auc": auc(y, blend),
        "members": names,
        "hill_recipe": recipe,
    }
    out_id = f"{timestamp()}_{args.tag}"
    write_json(scores, EXP_DIR / f"{out_id}_ensemble_summary.json")

    submissions = {
        "mean": T.mean(axis=1),
        "rank": rank_average(T),
        "logit": sigmoid(np.mean(logit(T), axis=1)),
    }
    test_blend = T[:, names.index(recipe[0]["feature"])].copy()
    for step in recipe[1:]:
        j = names.index(step["feature"])
        w = step["weight"]
        test_blend = (1 - w) * test_blend + w * T[:, j]
    submissions["hill"] = test_blend
    for name, pred in submissions.items():
        sub.assign(**{TARGET: np.clip(pred, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{out_id}_{name}.csv", index=False)
        for temp in [0.85, 0.95, 1.05, 1.15]:
            cal = sigmoid(logit(np.clip(pred, 1e-6, 1 - 1e-6)) * temp)
            sub.assign(**{TARGET: cal}).to_csv(SUB_DIR / f"{out_id}_{name}_temp{temp}.csv", index=False)
    print(scores)


if __name__ == "__main__":
    main()
