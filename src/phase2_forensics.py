from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, build_features, load_raw
from utils import REPORT_DIR, ensure_dirs


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def temporal_reconstruction(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys in [["Year", "Race", "Driver"], ["Race", "Driver"], ["Year", "Race", "Driver", "Stint"], ["Race", "Year", "Position"]]:
        s = train.sort_values(keys + ["LapNumber", "TyreLife", "id"]).copy()
        g = s.groupby(keys, observed=True, sort=False)
        for shift in [-3, -2, -1, 1, 2, 3]:
            col = f"pit_shift_{shift}"
            s[col] = g["PitStop"].shift(shift)
            m = s[col].notna()
            if m.any():
                rows.append(
                    {
                        "keys": "|".join(keys),
                        "signal": col,
                        "coverage": float(m.mean()),
                        "auc": auc(s.loc[m, TARGET], s.loc[m, col]),
                        "exact": float((s.loc[m, TARGET].values == s.loc[m, col].values).mean()),
                    }
                )
        left = train[keys + ["LapNumber", TARGET]].copy()
        left["next_lap"] = left["LapNumber"] + 1
        right = train[keys + ["LapNumber", "PitStop"]].rename(columns={"LapNumber": "next_lap", "PitStop": "pit_at_lap_plus_1"})
        mrg = left.merge(right, on=keys + ["next_lap"], how="left")
        m = mrg["pit_at_lap_plus_1"].notna()
        if m.any():
            rows.append(
                {
                    "keys": "|".join(keys),
                    "signal": "physical_lap_plus_1_pit",
                    "coverage": float(m.mean()),
                    "auc": auc(mrg.loc[m, TARGET], mrg.loc[m, "pit_at_lap_plus_1"]),
                    "exact": float((mrg.loc[m, TARGET].values == mrg.loc[m, "pit_at_lap_plus_1"].values).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["auc", "coverage"], ascending=False)


def threshold_scan(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    num_cols = [c for c in train.columns if c not in ["id", TARGET, "Driver", "Compound", "Race"]]
    y = train[TARGET].values
    for col in num_cols:
        values = train[col].values.astype(float)
        qs = np.unique(np.nanquantile(values, np.linspace(0.01, 0.99, 99)))
        best = None
        for q in qs:
            for op in ["<=", ">"]:
                pred = (values <= q).astype(float) if op == "<=" else (values > q).astype(float)
                if pred.min() == pred.max():
                    continue
                score = auc(y, pred)
                score = max(score, 1 - score)
                if best is None or score > best["auc"]:
                    best = {"feature": col, "op": op, "threshold": float(q), "auc": float(score), "positive_rate": float(pred.mean())}
        if best:
            rows.append(best)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def exact_value_spikes(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    global_mean = train[TARGET].mean()
    for col in train.columns:
        if col in ["id", TARGET]:
            continue
        if train[col].nunique(dropna=False) > 5000:
            continue
        stats = train.groupby(col, observed=True)[TARGET].agg(["count", "mean"])
        stats = stats[stats["count"] >= 25].copy()
        if len(stats):
            stats["lift_abs"] = (stats["mean"] - global_mean).abs()
            top = stats.sort_values("lift_abs", ascending=False).head(10).reset_index()
            for _, r in top.iterrows():
                rows.append({"feature": col, "value": str(r[col]), "count": int(r["count"]), "mean": float(r["mean"]), "lift_abs": float(r["lift_abs"])})
    return pd.DataFrame(rows).sort_values("lift_abs", ascending=False)


def adversarial(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    Xtr, _, Xte, cols = build_features(train, test, level="balanced")
    X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    folds = StratifiedKFold(3, shuffle=True, random_state=20260518)
    aucs = []
    imp = np.zeros(X.shape[1])
    for tr_idx, va_idx in folds.split(X, y):
        model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=31, random_state=1)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], pred))
        sample = np.random.default_rng(1).choice(va_idx, min(12000, len(va_idx)), replace=False)
        pi = permutation_importance(model, X.iloc[sample], y[sample], n_repeats=2, random_state=1)
        imp += pi.importances_mean
    return pd.DataFrame({"feature": cols, "adv_importance": imp / 3, "adv_auc": np.mean(aucs)}).sort_values("adv_importance", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-adv", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    train, test, _ = load_raw()
    temporal = temporal_reconstruction(train, test)
    thresholds = threshold_scan(train)
    spikes = exact_value_spikes(train)
    temporal.to_csv(REPORT_DIR / "phase2_temporal_reconstruction.csv", index=False)
    thresholds.to_csv(REPORT_DIR / "phase2_threshold_scan.csv", index=False)
    spikes.to_csv(REPORT_DIR / "phase2_exact_value_spikes.csv", index=False)
    adv = pd.DataFrame() if args.skip_adv else adversarial(train, test)
    if len(adv):
        adv.to_csv(REPORT_DIR / "phase2_adversarial_validation.csv", index=False)
    lines = [
        "# Phase 2 Forensics",
        "",
        "## Temporal Reconstruction Probes",
        temporal.head(30).to_markdown(index=False),
        "",
        "## Best Single-Threshold Leakage Probes",
        thresholds.head(30).to_markdown(index=False),
        "",
        "## Exact/Bucket Value Spikes",
        spikes.head(40).to_markdown(index=False),
    ]
    if len(adv):
        lines += ["", "## Adversarial Train/Test Drift", f"Mean adversarial AUC: {adv['adv_auc'].iloc[0]:.6f}", adv.head(40).to_markdown(index=False)]
    (REPORT_DIR / "phase2_forensics.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote phase2 forensic reports")


if __name__ == "__main__":
    main()
