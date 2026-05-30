from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from phase3_neighborhood import load_base_predictions
from phase3_specialists import make_features
from utils import EXP_DIR, MODEL_DIR, REPORT_DIR, SUB_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def load_prediction_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_base, test_base = load_base_predictions()
    phase3_oof = pd.read_csv(EXP_DIR / "20260519_141027_phase3_knn_euclid_phase3" / "oof_phase3.csv")
    phase3_test = pd.read_csv(EXP_DIR / "20260519_141027_phase3_knn_euclid_phase3" / "test_phase3.csv")
    spec_oof = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "oof_specialist.csv")
    spec_test = pd.read_csv(EXP_DIR / "20260519_143756_phase3_specialists" / "test_specialist.csv")
    tr = train_base.merge(phase3_oof[["id", "phase3_oof"]], on="id").merge(spec_oof[["id", "specialist_oof"]], on="id")
    te = test_base.merge(phase3_test[["id", TARGET]].rename(columns={TARGET: "phase3_oof"}), on="id").merge(
        spec_test[["id", TARGET]].rename(columns={TARGET: "specialist_oof"}), on="id"
    )
    tr["baseline"] = 0.72 * tr["specialist_oof"] + 0.28 * tr["phase3_oof"]
    te["baseline"] = 0.72 * te["specialist_oof"] + 0.28 * te["phase3_oof"]
    pred_cols = [c for c in tr.columns if c.startswith("base_")] + ["phase3_oof", "specialist_oof", "baseline"]
    for df in [tr, te]:
        vals = df[pred_cols].values
        df["pred_mean"] = vals.mean(axis=1)
        df["pred_std"] = vals.std(axis=1)
        df["pred_range"] = vals.max(axis=1) - vals.min(axis=1)
        df["pred_entropy"] = entropy(df["baseline"].values)
        ranks = np.vstack([pd.Series(df[c]).rank(pct=True).values for c in pred_cols]).T
        df["rank_std"] = ranks.std(axis=1)
        df["cat_vs_specialist_abs"] = np.abs(df["base_cat"] - df["specialist_oof"])
        df["phase3_vs_specialist_abs"] = np.abs(df["phase3_oof"] - df["specialist_oof"])
        df["uncertainty_score"] = (
            0.35 * (df["pred_std"] / max(df["pred_std"].max(), 1e-9))
            + 0.25 * (df["pred_entropy"] / max(df["pred_entropy"].max(), 1e-9))
            + 0.20 * (df["rank_std"] / max(df["rank_std"].max(), 1e-9))
            + 0.20 * (df["phase3_vs_specialist_abs"] / max(df["phase3_vs_specialist_abs"].max(), 1e-9))
        )
    return tr, te


def add_region_columns(raw: pd.DataFrame) -> pd.DataFrame:
    from table_model import add_bins

    d = add_bins(raw)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["TyreLifeFracBin"] = pd.qcut(d["TyreLifeFrac"].rank(method="first"), 10, labels=False, duplicates="drop")
    d["StintGapBin"] = pd.qcut(d["StintLapGap"].rank(method="first"), 10, labels=False, duplicates="drop")
    d["ProgressBin"] = pd.qcut(d["RaceProgress"].rank(method="first"), 10, labels=False, duplicates="drop")
    return d


def hard_masks(panel: pd.DataFrame, raw_regions: pd.DataFrame) -> dict[str, np.ndarray]:
    q = panel.quantile(numeric_only=True)
    masks = {
        "disagreement_top15": panel["pred_std"] >= panel["pred_std"].quantile(0.85),
        "disagreement_top25": panel["pred_std"] >= panel["pred_std"].quantile(0.75),
        "uncertainty_top15": panel["uncertainty_score"] >= panel["uncertainty_score"].quantile(0.85),
        "rank_unstable_top15": panel["rank_std"] >= panel["rank_std"].quantile(0.85),
        "midprob_35_65": panel["baseline"].between(0.35, 0.65),
        "cat_specialist_disagree": panel["cat_vs_specialist_abs"] >= panel["cat_vs_specialist_abs"].quantile(0.85),
        "year2023_hard": raw_regions["Year"].eq(2023) & (panel["uncertainty_score"] >= panel["uncertainty_score"].quantile(0.60)),
        "hard_compound_uncertain": raw_regions["Compound"].eq("HARD") & (panel["uncertainty_score"] >= panel["uncertainty_score"].quantile(0.70)),
        "extreme_tyre_uncertain": (raw_regions["TyreLifeFrac"] > 0.80) & (panel["uncertainty_score"] >= panel["uncertainty_score"].quantile(0.70)),
        "high_gap_uncertain": (raw_regions["StintLapGap"] >= raw_regions["StintLapGap"].quantile(0.85)) & (panel["uncertainty_score"] >= panel["uncertainty_score"].quantile(0.60)),
    }
    return {k: v.fillna(False).values for k, v in masks.items()}


def characterize(train: pd.DataFrame, panel: pd.DataFrame, raw_regions: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    y = train[TARGET].values
    err = np.abs(panel["baseline"].values - y)
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "mask": name,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "target_rate": float(y[mask].mean()),
                "baseline_auc": auc(y[mask], panel.loc[mask, "baseline"]),
                "mean_abs_error": float(err[mask].mean()),
                "pred_std": float(panel.loc[mask, "pred_std"].mean()),
                "uncertainty": float(panel.loc[mask, "uncertainty_score"].mean()),
                "year_mode": str(raw_regions.loc[mask, "Year"].mode().iloc[0]),
                "compound_mode": str(raw_regions.loc[mask, "Compound"].mode().iloc[0]),
                "avg_tyre_frac": float(raw_regions.loc[mask, "TyreLifeFrac"].mean()),
                "avg_stint_gap": float(raw_regions.loc[mask, "StintLapGap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_error", ascending=False)


def train_candidate(name: str, mask_tr: np.ndarray, mask_te: np.ndarray, X: pd.DataFrame, XT: pd.DataFrame, y: pd.Series, folds, baseline_oof, baseline_test):
    from sklearn.ensemble import HistGradientBoostingClassifier

    oof_spec = np.full(len(X), np.nan)
    test_spec = np.zeros(mask_te.sum(), dtype=float)
    if mask_tr.sum() < 4000 or y.iloc[np.where(mask_tr)[0]].nunique() < 2:
        return None
    for fold, (tr_idx, va_idx) in enumerate(folds):
        fit_idx = np.intersect1d(tr_idx, np.where(mask_tr)[0], assume_unique=False)
        val_idx = np.intersect1d(va_idx, np.where(mask_tr)[0], assume_unique=False)
        if len(fit_idx) < 1000 or len(val_idx) < 100 or y.iloc[fit_idx].nunique() < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.035,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.04,
            random_state=20260519 + fold,
        )
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof_spec[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        if mask_te.sum():
            test_spec += model.predict_proba(XT.loc[mask_te])[:, 1] / len(folds)
        with open(MODEL_DIR / f"phase45_{name}_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)
    valid = mask_tr & np.isfinite(oof_spec)
    if valid.sum() < 500:
        return None
    base_local = auc(y.iloc[valid], baseline_oof[valid])
    spec_local = auc(y.iloc[valid], oof_spec[valid])
    best = {"alpha": 0.0, "global_auc": auc(y, baseline_oof), "local_auc": base_local}
    for alpha in np.linspace(0.05, 1.0, 20):
        cand = baseline_oof.copy()
        cand[valid] = (1 - alpha) * baseline_oof[valid] + alpha * oof_spec[valid]
        score = auc(y, cand)
        local = auc(y.iloc[valid], cand[valid])
        if score > best["global_auc"]:
            best = {"alpha": float(alpha), "global_auc": float(score), "local_auc": float(local)}
    test_pred = baseline_test.copy()
    if best["alpha"] > 0 and mask_te.sum():
        test_pred[mask_te] = (1 - best["alpha"]) * baseline_test[mask_te] + best["alpha"] * test_spec
    return {
        "name": name,
        "n_train": int(mask_tr.sum()),
        "n_valid_oof": int(valid.sum()),
        "n_test": int(mask_te.sum()),
        "base_local_auc": float(base_local),
        "spec_local_auc": float(spec_local),
        "best_alpha": best["alpha"],
        "best_global_auc": best["global_auc"],
        "best_local_blend_auc": best["local_auc"],
        "oof_spec": oof_spec,
        "test_pred": test_pred,
        "valid_mask": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--tag", default="phase45_hard")
    args = parser.parse_args()
    ensure_dirs()
    from sklearn.model_selection import StratifiedKFold

    train, test, sub = load_raw()
    y = train[TARGET].astype(int)
    panel_tr, panel_te = load_prediction_panel()
    raw_tr = add_region_columns(train)
    raw_te = add_region_columns(test)
    masks_tr = hard_masks(panel_tr, raw_tr)
    masks_te = hard_masks(panel_te, raw_te)
    char = characterize(train, panel_tr, raw_tr, masks_tr)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    char.to_csv(REPORT_DIR / "phase45_hard_region_characterization.csv", index=False)

    X = make_features(train)
    XT = make_features(test)
    base_train, base_test = load_base_predictions()
    base_cols = [c for c in base_train.columns if c.startswith("base_")]
    extra_tr = panel_tr[["phase3_oof", "specialist_oof", "baseline", "pred_std", "pred_range", "pred_entropy", "rank_std", "uncertainty_score"]]
    extra_te = panel_te[["phase3_oof", "specialist_oof", "baseline", "pred_std", "pred_range", "pred_entropy", "rank_std", "uncertainty_score"]]
    X = pd.concat([X, base_train[base_cols], extra_tr], axis=1).astype("float32")
    XT = pd.concat([XT, base_test[base_cols], extra_te], axis=1).astype("float32")
    folds = list(StratifiedKFold(args.splits, shuffle=True, random_state=20260519).split(X, y))
    baseline_oof = panel_tr["baseline"].values.copy()
    baseline_test = panel_te["baseline"].values.copy()
    candidates = []
    for name in masks_tr:
        print(f"training hard candidate {name}")
        res = train_candidate(name, masks_tr[name], masks_te[name], X, XT, y, folds, baseline_oof, baseline_test)
        if res is not None:
            candidates.append(res)
            print({k: v for k, v in res.items() if k not in ["oof_spec", "test_pred", "valid_mask"]})

    current_oof = baseline_oof.copy()
    current_test = baseline_test.copy()
    selected = []
    for res in sorted(candidates, key=lambda r: r["best_global_auc"], reverse=True):
        if res["best_alpha"] <= 0:
            continue
        valid = res["valid_mask"]
        alpha = res["best_alpha"]
        cand = current_oof.copy()
        cand[valid] = (1 - alpha) * current_oof[valid] + alpha * res["oof_spec"][valid]
        score = auc(y, cand)
        if score > auc(y, current_oof) + 1e-7:
            current_oof = cand
            current_test = res["test_pred"]
            selected.append({k: v for k, v in res.items() if k not in ["oof_spec", "test_pred", "valid_mask"]} | {"sequential_auc": float(score)})
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    final_auc = auc(y, current_oof)
    pd.DataFrame({"id": train["id"], TARGET: y, "hard_oof": current_oof}).to_csv(exp_dir / "oof_hard.csv", index=False)
    pd.DataFrame({"id": test["id"], TARGET: current_test}).to_csv(exp_dir / "test_hard.csv", index=False)
    sub.assign(**{TARGET: current_test}).to_csv(SUB_DIR / f"{exp_id}_{final_auc:.6f}.csv", index=False)
    cand_report = pd.DataFrame([{k: v for k, v in r.items() if k not in ["oof_spec", "test_pred", "valid_mask"]} for r in candidates])
    cand_report.to_csv(exp_dir / "hard_candidate_report.csv", index=False)
    char.to_csv(exp_dir / "hard_region_characterization.csv", index=False)
    summary = {
        "exp_id": exp_id,
        "baseline_auc": auc(y, baseline_oof),
        "final_auc": final_auc,
        "selected": selected,
    }
    write_json(summary, exp_dir / "summary.json")
    lines = ["# Phase 4.5 Hard Region Report", "", f"- Baseline AUC: {summary['baseline_auc']:.9f}", f"- Final AUC: {final_auc:.9f}", "", "## Characterization", char.to_markdown(index=False), "", "## Candidate Specialists", cand_report.to_markdown(index=False), "", "## Selected", pd.DataFrame(selected).to_markdown(index=False) if selected else "No specialist passed the OOF gate."]
    (REPORT_DIR / "phase45_hard_region_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
