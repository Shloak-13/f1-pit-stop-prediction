from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from features import TARGET, load_raw
from phase6_pairwise_ranking import load_phase_scores, rank01
from utils import EXP_DIR, REPORT_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def load_rankers(exp_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = None
    test = None
    for exp in exp_ids:
        odf = pd.read_csv(EXP_DIR / exp / "oof_rankers.csv")
        tdf = pd.read_csv(EXP_DIR / exp / "test_rankers.csv")
        rename_o = {c: f"{exp}__{c}" for c in odf.columns if c not in ["id", TARGET]}
        rename_t = {c: f"{exp}__{c}" for c in tdf.columns if c != "id"}
        odf = odf.rename(columns=rename_o)
        tdf = tdf.rename(columns=rename_t)
        keep_o = ["id"] + list(rename_o.values())
        keep_t = ["id"] + list(rename_t.values())
        oof = odf[keep_o] if oof is None else oof.merge(odf[keep_o], on="id")
        test = tdf[keep_t] if test is None else test.merge(tdf[keep_t], on="id")
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-exps", default="20260519_153956_phase6_lambdarank,20260519_160100_phase6_xgbrank")
    parser.add_argument("--tag", default="phase6_rank_fusion")
    args = parser.parse_args()
    ensure_dirs()
    train, _, sub = load_raw()
    y = train[TARGET].values
    base_oof, base_test = load_phase_scores()
    rank_oof, rank_test = load_rankers([x.strip() for x in args.rank_exps.split(",") if x.strip()])
    rank_cols = [c for c in rank_oof.columns if c != "id"]

    anchors = {
        "phase2": "phase2_cat_table",
        "stable_cat": "base_cat",
        "phase3": "phase3_hill",
        "hard": "hard_oof",
    }
    preferred = [c for c in rank_cols if "SyntheticBucket" in c or "RaceYearStint" in c or "BoundaryBucket" in c]
    recipes = []
    for anchor_name, anchor_col in anchors.items():
        ao = rank01(base_oof[anchor_col].values)
        at = rank01(base_test[anchor_col].values)
        for rcol in preferred:
            ro = rank01(rank_oof[rcol].values)
            rt = rank01(rank_test[rcol].values)
            for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
                po = (1 - alpha) * ao + alpha * ro
                pt = (1 - alpha) * at + alpha * rt
                recipes.append(
                    {
                        "name": f"{anchor_name}_{rcol.split('__')[-1]}_a{alpha:.2f}".replace(".", "p"),
                        "anchor": anchor_name,
                        "ranker": rcol,
                        "alpha": alpha,
                        "oof_auc": auc(y, po),
                        "mean_abs_delta": float(np.abs(po - ao).mean()),
                        "pred": pt,
                    }
                )
    # Multi-ranker consensus corrections: lower variance than one ranker.
    consensus_sets = {
        "lgbm_all": [c for c in preferred if "phase6_lambdarank" in c],
        "xgb_all": [c for c in preferred if "phase6_xgbrank" in c],
        "all_rankers": preferred,
    }
    for anchor_name, anchor_col in anchors.items():
        ao = rank01(base_oof[anchor_col].values)
        at = rank01(base_test[anchor_col].values)
        for set_name, cols in consensus_sets.items():
            if not cols:
                continue
            ro = np.mean([rank01(rank_oof[c].values) for c in cols], axis=0)
            rt = np.mean([rank01(rank_test[c].values) for c in cols], axis=0)
            for alpha in [0.02, 0.03, 0.05, 0.08]:
                po = (1 - alpha) * ao + alpha * ro
                pt = (1 - alpha) * at + alpha * rt
                recipes.append(
                    {
                        "name": f"{anchor_name}_{set_name}_a{alpha:.2f}".replace(".", "p"),
                        "anchor": anchor_name,
                        "ranker": set_name,
                        "alpha": alpha,
                        "oof_auc": auc(y, po),
                        "mean_abs_delta": float(np.abs(po - ao).mean()),
                        "pred": pt,
                    }
                )
    scores = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in recipes]).sort_values("oof_auc", ascending=False)
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(exp_dir / "rank_fusion_scores.csv", index=False)
    # Write top OOF plus deliberately conservative phase2/stable-cat probes.
    selected_names = set(scores.head(30)["name"])
    selected_names |= set(scores[(scores["anchor"].isin(["phase2", "stable_cat"])) & (scores["alpha"].isin([0.02, 0.03, 0.05]))].sort_values("oof_auc", ascending=False).head(24)["name"])
    for r in recipes:
        if r["name"] in selected_names:
            sub.assign(**{TARGET: np.clip(r["pred"], 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{r['name']}.csv", index=False)
    write_json({"exp_id": exp_id, "top": scores.head(40).to_dict("records"), "selected_count": len(selected_names)}, exp_dir / "summary.json")
    lines = [
        "# Phase 6 Rank Fusion",
        "",
        "## Top Rank Fusion Candidates",
        scores.head(50).to_markdown(index=False),
        "",
        "## Public-LB Strategy",
        "- Use phase2/stable-cat anchors for conservative leaderboard probes.",
        "- Use hard-anchor variants only if public LB rewards Phase 4.5 style corrections.",
        "- Prefer alpha 0.02-0.05 unless a ranker variant clearly transfers.",
    ]
    (REPORT_DIR / "phase6_rank_fusion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(scores.head(25).to_string(index=False))
    print(f"wrote {exp_dir}")


if __name__ == "__main__":
    main()
