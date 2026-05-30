from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from table_model import add_bins
from utils import EXP_DIR, REPORT_DIR, SUB_DIR, ensure_dirs, logit, sigmoid, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).values


def read_oof(exp: str, stem: str, pred_col: str | None = None) -> pd.DataFrame:
    path = EXP_DIR / exp / stem
    df = pd.read_csv(path)
    if pred_col is None:
        pred_col = [c for c in df.columns if c.endswith("_oof") or c in ["phase3_oof", "specialist_oof", "hard_oof"]][0]
    return df[["id", pred_col]].rename(columns={pred_col: exp})


def read_test(exp: str, stem: str) -> pd.DataFrame:
    df = pd.read_csv(EXP_DIR / exp / stem)
    return df[["id", TARGET]].rename(columns={TARGET: exp})


def load_system_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces_oof = {
        "cat": read_oof("20260518_165505_phase2_cat_fast_stratified", "oof_catboost.csv", "catboost_oof"),
        "table_hgb": read_oof("20260518_162337_phase2_table_stratified_hgb", "oof_hgb.csv"),
        "table_log": read_oof("20260518_174444_phase2_table_stratified_logistic", "oof_logistic.csv"),
        "phase3_knn": read_oof("20260519_141027_phase3_knn_euclid_phase3", "oof_phase3.csv", "phase3_oof"),
        "specialist": read_oof("20260519_143756_phase3_specialists", "oof_specialist.csv", "specialist_oof"),
        "phase45": read_oof("20260519_150804_phase45_hard", "oof_hard.csv", "hard_oof"),
    }
    pieces_test = {
        "cat": read_test("20260518_165505_phase2_cat_fast_stratified", "test_catboost.csv"),
        "table_hgb": read_test("20260518_162337_phase2_table_stratified_hgb", "test_hgb.csv"),
        "table_log": read_test("20260518_174444_phase2_table_stratified_logistic", "test_logistic.csv"),
        "phase3_knn": read_test("20260519_141027_phase3_knn_euclid_phase3", "test_phase3.csv"),
        "specialist": read_test("20260519_143756_phase3_specialists", "test_specialist.csv"),
        "phase45": read_test("20260519_150804_phase45_hard", "test_hard.csv"),
    }
    oof = None
    test = None
    for name, df in pieces_oof.items():
        df = df.rename(columns={df.columns[1]: name})
        oof = df if oof is None else oof.merge(df, on="id", how="inner")
    for name, df in pieces_test.items():
        df = df.rename(columns={df.columns[1]: name})
        test = df if test is None else test.merge(df, on="id", how="inner")

    oof["phase2_cat_table"] = 0.88 * oof["cat"] + 0.12 * oof["table_hgb"]
    test["phase2_cat_table"] = 0.88 * test["cat"] + 0.12 * test["table_hgb"]
    oof["phase3_hill"] = 0.72 * oof["specialist"] + 0.28 * oof["phase3_knn"]
    test["phase3_hill"] = 0.72 * test["specialist"] + 0.28 * test["phase3_knn"]
    oof["phase45_hard"] = oof["phase45"]
    test["phase45_hard"] = test["phase45"]
    return oof, test


def add_groups(raw: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["YearStint"] = d["Year"].astype(str) + "_" + d["Stint"].astype(str)
    d["RaceYear"] = d["Race"].astype(str) + "_" + d["Year"].astype(str)
    d["RaceYearStint"] = d["RaceYear"] + "_" + d["Stint"].astype(str)
    d["RaceYearCompound"] = d["RaceYear"] + "_" + d["Compound"].astype(str)
    d["TyreFracBin"] = pd.qcut(d["TyreLifeFrac"].rank(method="first"), 10, labels=False, duplicates="drop").astype(str)
    d["StintGapBin"] = pd.qcut(d["StintLapGap"].rank(method="first"), 10, labels=False, duplicates="drop").astype(str)
    d["DeltaBinGroup"] = d["DeltaBin"].astype(str)
    d["SyntheticBucket"] = d["YearStint"] + "_" + d["Compound"].astype(str) + "_" + d["TyreFracBin"] + "_" + d["DeltaBinGroup"]
    return d


def macro_group_auc(y: np.ndarray, pred: np.ndarray, groups: pd.Series, min_n: int = 150) -> dict:
    rows = []
    for g, idx in pd.Series(np.arange(len(y))).groupby(groups).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < min_n or len(np.unique(y[idx])) < 2:
            continue
        rows.append((g, len(idx), auc(y[idx], pred[idx])))
    if not rows:
        return {"macro_auc": np.nan, "std": np.nan, "p10": np.nan, "n_groups": 0}
    vals = np.array([r[2] for r in rows])
    return {"macro_auc": float(vals.mean()), "std": float(vals.std()), "p10": float(np.quantile(vals, 0.10)), "n_groups": int(len(vals))}


def compare_predictions(train: pd.DataFrame, groups: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = train[TARGET].values
    preds = ["cat", "phase2_cat_table", "phase3_hill", "phase45_hard"]
    rows = []
    for p in preds:
        base = {"prediction": p, "global_auc": auc(y, oof[p].values)}
        for g in ["Year", "Race", "Driver", "RaceYear", "YearStint", "RaceYearStint", "RaceYearCompound", "SyntheticBucket"]:
            stats = macro_group_auc(y, oof[p].values, groups[g])
            base[f"{g}_macro_auc"] = stats["macro_auc"]
            base[f"{g}_p10_auc"] = stats["p10"]
            base[f"{g}_std"] = stats["std"]
            base[f"{g}_n"] = stats["n_groups"]
        rows.append(base)
    summary = pd.DataFrame(rows)

    delta_rows = []
    for a, b in [("phase2_cat_table", "phase3_hill"), ("phase3_hill", "phase45_hard"), ("phase2_cat_table", "phase45_hard")]:
        delta = np.abs(oof[b].values - oof[a].values)
        changed = delta >= np.quantile(delta, 0.90)
        for col in ["Year", "Compound", "Stint", "Race", "TyreFracBin", "StintGapBin", "DeltaBinGroup"]:
            tmp = (
                pd.DataFrame({"group": groups[col].astype(str), "changed": changed, "delta": delta, "y": y, "pa": oof[a].values, "pb": oof[b].values})
                .groupby("group")
                .agg(n=("changed", "size"), changed_rate=("changed", "mean"), mean_delta=("delta", "mean"), target=("y", "mean"))
                .reset_index()
            )
            tmp = tmp[tmp["n"] >= 100]
            tmp["transition"] = f"{a}->{b}"
            tmp["field"] = col
            delta_rows.append(tmp.sort_values("changed_rate", ascending=False).head(20))
    deltas = pd.concat(delta_rows, ignore_index=True)
    return summary, deltas


def stability_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    cols = ["cat", "table_hgb", "table_log", "phase3_knn", "specialist", "phase2_cat_table", "phase3_hill", "phase45_hard"]
    ranks = np.vstack([rank01(oof[c].values) for c in cols]).T
    vals = oof[cols].values
    out = pd.DataFrame(
        {
            "id": oof["id"],
            "pred_std": vals.std(axis=1),
            "pred_range": vals.max(axis=1) - vals.min(axis=1),
            "rank_std": ranks.std(axis=1),
            "phase3_delta": np.abs(oof["phase3_hill"] - oof["phase2_cat_table"]),
            "phase45_delta": np.abs(oof["phase45_hard"] - oof["phase3_hill"]),
        }
    )
    return out


def make_aligned_candidates(oof: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Shrink later local corrections toward the more public-stable Phase 2/CatBoost base.
    recipes = {
        "stable_cat_table": {"cat": 0.75, "table_hgb": 0.25},
        "phase2_primary": {"cat": 0.88, "table_hgb": 0.12},
        "phase3_shrink25": {"phase2_cat_table": 0.75, "phase3_hill": 0.25},
        "phase3_shrink50": {"phase2_cat_table": 0.50, "phase3_hill": 0.50},
        "phase45_shrink20": {"phase3_hill": 0.80, "phase45_hard": 0.20},
        "phase45_shrink40": {"phase3_hill": 0.60, "phase45_hard": 0.40},
        "stable_mean": {"cat": 0.50, "table_hgb": 0.20, "phase3_hill": 0.20, "phase45_hard": 0.10},
        "rank_stable": {},
        "logit_stable": {},
    }
    o = pd.DataFrame({"id": oof["id"]})
    t = pd.DataFrame({"id": test["id"]})
    for name, weights in recipes.items():
        if name == "rank_stable":
            cols = ["cat", "table_hgb", "phase3_hill", "phase45_hard"]
            o[name] = np.mean([rank01(oof[c].values) for c in cols], axis=0)
            t[name] = np.mean([rank01(test[c].values) for c in cols], axis=0)
        elif name == "logit_stable":
            cols = ["cat", "table_hgb", "phase3_hill", "phase45_hard"]
            o[name] = sigmoid(np.mean([logit(oof[c].values) for c in cols], axis=0))
            t[name] = sigmoid(np.mean([logit(test[c].values) for c in cols], axis=0))
        else:
            o[name] = sum(w * oof[c].values for c, w in weights.items())
            t[name] = sum(w * test[c].values for c, w in weights.items())
    return o, t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="validation_reset")
    args = parser.parse_args()
    ensure_dirs()
    train, test_raw, sub = load_raw()
    groups = add_groups(train)
    oof, test = load_system_predictions()
    y = train[TARGET].values
    summary, deltas = compare_predictions(train, groups, oof)
    stability = stability_metrics(oof)
    cand_oof, cand_test = make_aligned_candidates(oof, test)
    cand_scores = []
    for c in [x for x in cand_oof.columns if x != "id"]:
        row = {"candidate": c, "global_auc": auc(y, cand_oof[c].values)}
        for g in ["Year", "RaceYear", "YearStint", "RaceYearStint", "SyntheticBucket"]:
            stats = macro_group_auc(y, cand_oof[c].values, groups[g])
            row[f"{g}_macro_auc"] = stats["macro_auc"]
            row[f"{g}_p10_auc"] = stats["p10"]
            row[f"{g}_std"] = stats["std"]
        row["mean_abs_phase45_delta"] = float(np.abs(cand_oof[c].values - oof["phase45_hard"].values).mean())
        row["mean_abs_phase2_delta"] = float(np.abs(cand_oof[c].values - oof["phase2_cat_table"].values).mean())
        cand_scores.append(row)
    cand_scores = pd.DataFrame(cand_scores).sort_values(["RaceYear_p10_auc", "SyntheticBucket_p10_auc", "global_auc"], ascending=False)

    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(exp_dir / "cv_realism_summary.csv", index=False)
    deltas.to_csv(exp_dir / "prediction_delta_subgroups.csv", index=False)
    stability.to_csv(exp_dir / "prediction_stability_rows.csv", index=False)
    cand_scores.to_csv(exp_dir / "lb_aligned_candidate_scores.csv", index=False)
    cand_oof.to_csv(exp_dir / "oof_lb_aligned_candidates.csv", index=False)
    cand_test.to_csv(exp_dir / "test_lb_aligned_candidates.csv", index=False)

    for c in [x for x in cand_test.columns if x != "id"]:
        sub.assign(**{TARGET: np.clip(cand_test[c].values, 1e-6, 1 - 1e-6)}).to_csv(SUB_DIR / f"{exp_id}_{c}.csv", index=False)
    write_json(
        {
            "exp_id": exp_id,
            "best_by_raceyear_p10": cand_scores.iloc[0].to_dict(),
            "phase2_oof_auc": float(summary.loc[summary["prediction"].eq("phase2_cat_table"), "global_auc"].iloc[0]),
            "phase3_oof_auc": float(summary.loc[summary["prediction"].eq("phase3_hill"), "global_auc"].iloc[0]),
            "phase45_oof_auc": float(summary.loc[summary["prediction"].eq("phase45_hard"), "global_auc"].iloc[0]),
        },
        exp_dir / "summary.json",
    )

    lines = [
        "# Validation Realism Reset",
        "",
        "## System Comparison",
        summary.to_markdown(index=False),
        "",
        "## LB-Aligned Candidates",
        cand_scores.to_markdown(index=False),
        "",
        "## Top Changed Subgroups",
        deltas.head(80).to_markdown(index=False),
        "",
        "## Interpretation",
        "- Later Phase 3/4.5 local corrections are evaluated by grouped macro/p10 AUC, not only global stratified OOF.",
        "- Candidate submissions intentionally shrink high-OOF local deltas toward more stable CatBoost/table predictions.",
        "- Use public LB to choose among stable shrink candidates rather than the highest local OOF candidate.",
    ]
    (REPORT_DIR / "validation_realism_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(cand_scores.head(10).to_string(index=False))
    print(f"wrote {exp_dir}")


if __name__ == "__main__":
    main()
