from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET, load_raw
from table_model import add_bins
from utils import EXP_DIR, REPORT_DIR, ensure_dirs, timestamp, write_json


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(p)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def entropy_from_rate(p: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def smooth_rates(keys: pd.Series, y: pd.Series, alpha: float = 20.0) -> pd.Series:
    prior = float(y.mean())
    stats = pd.DataFrame({"key": keys.astype(str), "y": y.values}).groupby("key")["y"].agg(["mean", "count"])
    sm = (stats["mean"] * stats["count"] + prior * alpha) / (stats["count"] + alpha)
    return keys.astype(str).map(sm).fillna(prior).astype(float)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = add_bins(df)
    d["TyreLifeFrac"] = d["TyreLife"] / d["LapNumber"].replace(0, np.nan)
    d["StintLapGap"] = d["LapNumber"] - d["TyreLife"]
    d["CurrentLap_RaceLapsEst"] = d["LapNumber"] / d["RaceLapsEst"].replace(0, np.nan)
    d["TyreLife_RaceProgress"] = d["TyreLife"] / d["RaceProgress"].replace(0, np.nan)
    d["RaceProgressBin20"] = np.floor(d["RaceProgress"] * 20).clip(0, 19).astype(int)
    d["TyreLifeFracBin12"] = pd.qcut(d["TyreLifeFrac"].rank(method="first"), 12, labels=False, duplicates="drop").astype(int)
    d["StintGapBin12"] = pd.qcut(d["StintLapGap"].rank(method="first"), 12, labels=False, duplicates="drop").astype(int)
    return d


def quantization_forensics(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    both = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)
    num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in ["id", TARGET]]
    candidate_steps = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.125, 0.2, 0.25, 0.5, 1.0])
    for col in num_cols:
        x = both[col].dropna().astype(float).values
        xt = train[col].dropna().astype(float).values
        if len(x) == 0:
            continue
        vc = pd.Series(xt).value_counts(normalize=True)
        max_share = float(vc.iloc[0]) if len(vc) else 0.0
        top_value = float(vc.index[0]) if len(vc) else np.nan
        decimals = np.round(np.abs(x - np.floor(x)), 6)
        round_share = float(np.mean(np.isclose(decimals, 0, atol=1e-9)))
        half_share = float(np.mean(np.isclose((x * 2) % 1, 0, atol=1e-9)))
        third_share = float(np.mean(np.isclose((x * 3) % 1, 0, atol=1e-9)))
        fifth_share = float(np.mean(np.isclose((x * 5) % 1, 0, atol=1e-9)))
        best_step, best_resid = None, -1.0
        for step in candidate_steps:
            resid = np.abs(x / step - np.round(x / step))
            share = float(np.mean(resid < 1e-6))
            if share > best_resid:
                best_step, best_resid = step, share
        # Hist concentration at requested resolutions.
        hist_stats = {}
        for res in [0.001, 0.01, 0.1]:
            binned = np.round(xt / res) * res
            hist_stats[f"max_bin_share_{res}"] = float(pd.Series(binned).value_counts(normalize=True).iloc[0])
            hist_stats[f"n_bins_{res}"] = int(pd.Series(binned).nunique())
        drift = abs(train[col].mean() - test[col].mean()) / (train[col].std() + 1e-9)
        score = max_share + 0.5 * best_resid + 0.2 * max(round_share, half_share, third_share, fifth_share) + 0.1 * drift
        rows.append(
            {
                "feature": col,
                "top_exact_value": top_value,
                "top_exact_share": max_share,
                "flag_gt5pct_same": bool(max_share > 0.05),
                "round_share": round_share,
                "half_grid_share": half_share,
                "third_grid_share": third_share,
                "fifth_grid_share": fifth_share,
                "inferred_step": best_step,
                "inferred_step_share": best_resid,
                "train_test_mean_drift_std": float(drift),
                "artifact_score": score,
                **hist_stats,
            }
        )
    return pd.DataFrame(rows).sort_values("artifact_score", ascending=False)


def collision_forensics(train: pd.DataFrame, test: pd.DataFrame, round_decimals: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in ["id", TARGET]]
    cat_cols = [c for c in ["Driver", "Compound", "Race", "Year", "PitStop", "Stint", "Position"] if c in train.columns]
    tr = train.copy()
    te = test.copy()
    for df in [tr, te]:
        for c in num_cols:
            df[c] = df[c].round(round_decimals)
    tr_key = pd.util.hash_pandas_object(tr[cat_cols + num_cols].astype(str), index=False).astype(str)
    te_key = pd.util.hash_pandas_object(te[cat_cols + num_cols].astype(str), index=False).astype(str)
    stats = pd.DataFrame({"key": tr_key, "y": train[TARGET]}).groupby("key")["y"].agg(["count", "mean"])
    stats["entropy"] = entropy_from_rate(stats["mean"])
    stats["purity"] = np.maximum(stats["mean"], 1 - stats["mean"])
    stats["round_decimals"] = round_decimals
    test_counts = te_key.value_counts().rename("test_count")
    stats = stats.join(test_counts, how="outer").fillna({"count": 0, "mean": np.nan, "test_count": 0, "entropy": np.nan, "purity": np.nan})
    stats["train_only"] = (stats["count"] > 0) & (stats["test_count"] == 0)
    stats["test_only"] = (stats["count"] == 0) & (stats["test_count"] > 0)
    stats["deterministic"] = (stats["count"] >= 5) & ((stats["mean"] == 0) | (stats["mean"] == 1))
    stats["near_deterministic"] = (stats["count"] >= 5) & (stats["purity"] >= 0.98)
    report = stats.reset_index().sort_values(["count", "purity"], ascending=False)
    tr_prior = tr_key.map(((stats["mean"] * stats["count"] + train[TARGET].mean() * 20) / (stats["count"] + 20))).fillna(train[TARGET].mean())
    te_prior = te_key.map(((stats["mean"] * stats["count"] + train[TARGET].mean() * 20) / (stats["count"] + 20))).fillna(train[TARGET].mean())
    tr_feat = pd.DataFrame({f"collision_r{round_decimals}_prior": tr_prior, f"collision_r{round_decimals}_count": tr_key.map(stats["count"]).fillna(0)})
    te_feat = pd.DataFrame({f"collision_r{round_decimals}_prior": te_prior, f"collision_r{round_decimals}_count": te_key.map(stats["count"]).fillna(0)})
    return report, tr_feat, te_feat


def conditional_entropy_mining(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = train[TARGET].astype(int)
    cols = [c for c in train.columns if c not in ["id", TARGET]]
    # Avoid huge pair explosion from raw high-card strings plus many rounded cols.
    preferred = [
        "Year",
        "Compound",
        "Race",
        "PitStop",
        "Stint",
        "Position",
        "RaceLapsEst",
        "RacePhase20",
        "TyreLifeBin3",
        "TyreLifeBin5",
        "DeltaBin",
        "DegBin",
        "TyreLifeFracBin12",
        "StintGapBin12",
        "RaceProgressBin20",
        "CurrentLap_RaceLapsEst",
        "TyreLife_RaceProgress",
    ]
    cols = [c for c in preferred if c in train.columns]
    for bins in [5, 10]:
        discretized = {}
        for c in cols:
            if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > bins:
                discretized[c] = pd.qcut(train[c].rank(method="first"), bins, labels=False, duplicates="drop").astype(str)
            else:
                discretized[c] = train[c].astype(str)
        for a, b in itertools.combinations(cols, 2):
            key = discretized[a] + "|" + discretized[b]
            pred = smooth_rates(key, y, alpha=30.0)
            stats = pd.DataFrame({"key": key, "y": y}).groupby("key")["y"].agg(["count", "mean"])
            cond_h = float(np.average(entropy_from_rate(stats["mean"]), weights=stats["count"]))
            rows.append({"feature_a": a, "feature_b": b, "bins": bins, "conditional_entropy": cond_h, "joint_auc": auc(y, pred), "n_bins": int(stats.shape[0])})
    return pd.DataFrame(rows).sort_values(["joint_auc", "conditional_entropy"], ascending=[False, True])


def deterministic_bucket_search(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = train[TARGET].astype(int)
    candidate_cols = [c for c in train.columns if c not in ["id", TARGET] and train[c].nunique(dropna=False) <= 2000]
    for c in candidate_cols:
        stats = train.groupby(c, observed=True)[TARGET].agg(["count", "mean"])
        test_values = set(test[c].dropna().astype(str).unique())
        train_values = set(train[c].dropna().astype(str).unique())
        for val, r in stats.iterrows():
            if r["count"] < 10:
                continue
            ent = float(entropy_from_rate([r["mean"]])[0])
            if r["mean"] in [0, 1] or ent < 0.05 or r["count"] >= 1000:
                rows.append(
                    {
                        "feature": c,
                        "value": str(val),
                        "count": int(r["count"]),
                        "target_rate": float(r["mean"]),
                        "entropy": ent,
                        "deterministic": bool(r["mean"] in [0, 1]),
                        "present_in_test": str(val) in test_values,
                    }
                )
        only_train = len(train_values - test_values)
        only_test = len(test_values - train_values)
        rows.append({"feature": c, "value": "__COVERAGE__", "count": int(stats["count"].sum()), "target_rate": np.nan, "entropy": np.nan, "deterministic": False, "present_in_test": True, "train_only_values": only_train, "test_only_values": only_test})
    return pd.DataFrame(rows).sort_values(["deterministic", "entropy", "count"], ascending=[False, True, False])


def interaction_forensics(train: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ["Year", "Compound", "TyreLifeBin3"],
        ["Year", "Compound", "TyreLifeBin5"],
        ["Race", "Year", "Stint"],
        ["Compound", "DeltaBin", "TyreLifeFracBin12"],
        ["RaceProgressBin20", "Position", "Compound"],
        ["Year", "Stint", "CurrentLap_RaceLapsEst"],
        ["Compound", "RacePhase20", "StintGapBin12"],
        ["Race", "Year", "RacePhase20"],
    ]
    rows = []
    y = train[TARGET].astype(int)
    for spec in specs:
        keys = []
        for c in spec:
            if c not in train.columns:
                break
            if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 30:
                keys.append(pd.qcut(train[c].rank(method="first"), 10, labels=False, duplicates="drop").astype(str))
            else:
                keys.append(train[c].astype(str))
        if len(keys) != len(spec):
            continue
        key = keys[0]
        for k in keys[1:]:
            key = key + "|" + k
        pred = smooth_rates(key, y, alpha=20.0)
        stats = pd.DataFrame({"key": key, "y": y}).groupby("key")["y"].agg(["count", "mean"])
        rows.append(
            {
                "interaction": " x ".join(spec),
                "auc": auc(y, pred),
                "n_bins": int(stats.shape[0]),
                "entropy": float(np.average(entropy_from_rate(stats["mean"]), weights=stats["count"])),
                "min_rate": float(stats["mean"].min()),
                "max_rate": float(stats["mean"].max()),
                "deterministic_bins": int(((stats["mean"].isin([0, 1])) & (stats["count"] >= 10)).sum()),
            }
        )
    ratio_cols = ["CurrentLap_RaceLapsEst", "TyreLife_RaceProgress", "TyreLifeFrac", "StintLapGap"]
    for c in ratio_cols:
        if c not in train.columns:
            continue
        key = pd.qcut(train[c].rank(method="first"), 20, labels=False, duplicates="drop").astype(str)
        pred = smooth_rates(key, y, alpha=20.0)
        rows.append({"interaction": f"ratio::{c}", "auc": auc(y, pred), "n_bins": int(key.nunique()), "entropy": np.nan, "min_rate": np.nan, "max_rate": np.nan, "deterministic_bins": np.nan})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def additive_value_report(train: pd.DataFrame, features: dict[str, pd.Series]) -> pd.DataFrame:
    # Compare artifact priors against the existing CatBoost OOF.
    cat_path = EXP_DIR / "20260518_165505_phase2_cat_fast_stratified" / "oof_catboost.csv"
    if not cat_path.exists():
        return pd.DataFrame()
    base = pd.read_csv(cat_path)["catboost_oof"]
    y = train[TARGET].astype(int)
    rows = []
    base_auc = auc(y, base)
    for name, pred in features.items():
        pred = pd.Series(pred).fillna(y.mean()).astype(float)
        best = {"alpha": 0.0, "auc": base_auc}
        for alpha in np.linspace(0.02, 0.30, 15):
            blend = (1 - alpha) * base + alpha * pred
            score = auc(y, blend)
            if score > best["auc"]:
                best = {"alpha": float(alpha), "auc": float(score)}
        rows.append({"feature": name, "standalone_auc": auc(y, pred), "base_auc": base_auc, "best_blend_alpha": best["alpha"], "best_blend_auc": best["auc"], "additive_gain": best["auc"] - base_auc})
    return pd.DataFrame(rows).sort_values("additive_gain", ascending=False)


def oof_prior_from_key(keys: pd.Series, y: pd.Series, alpha: float = 20.0, n_splits: int = 5) -> pd.Series:
    from sklearn.model_selection import StratifiedKFold

    out = pd.Series(np.nan, index=keys.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260519)
    for tr_idx, va_idx in skf.split(pd.DataFrame({"key": keys}), y):
        prior = float(y.iloc[tr_idx].mean())
        stats = pd.DataFrame({"key": keys.iloc[tr_idx].astype(str), "y": y.iloc[tr_idx].values}).groupby("key")["y"].agg(["mean", "count"])
        sm = (stats["mean"] * stats["count"] + prior * alpha) / (stats["count"] + alpha)
        out.iloc[va_idx] = keys.iloc[va_idx].astype(str).map(sm).fillna(prior).values
    return out.fillna(float(y.mean()))


def collision_key_frame(df: pd.DataFrame, round_decimals: int) -> pd.Series:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["id", TARGET]]
    cat_cols = [c for c in ["Driver", "Compound", "Race", "Year", "PitStop", "Stint", "Position"] if c in df.columns]
    tmp = df[cat_cols + num_cols].copy()
    for c in num_cols:
        tmp[c] = tmp[c].round(round_decimals)
    return pd.util.hash_pandas_object(tmp.astype(str), index=False).astype(str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="synthetic_forensics")
    args = parser.parse_args()
    ensure_dirs()
    train_raw, test_raw, _ = load_raw()
    train = enrich(train_raw)
    test = enrich(test_raw)
    exp_id = f"{timestamp()}_{args.tag}"
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    quant = quantization_forensics(train, test)
    coll2, coll2_tr, coll2_te = collision_forensics(train, test, 2)
    coll3, coll3_tr, coll3_te = collision_forensics(train, test, 3)
    ce = conditional_entropy_mining(train)
    det = deterministic_bucket_search(train, test)
    inter = interaction_forensics(train)
    y = train[TARGET].astype(int)
    collision2_oof = oof_prior_from_key(collision_key_frame(train, 2), y)
    collision3_oof = oof_prior_from_key(collision_key_frame(train, 3), y)
    best_pair_key = train[ce.iloc[0]["feature_a"]].astype(str) + "|" + train[ce.iloc[0]["feature_b"]].astype(str)
    best_pair_oof = oof_prior_from_key(best_pair_key, y)
    if " x " in inter.iloc[0]["interaction"]:
        spec = inter.iloc[0]["interaction"].split(" x ")
        key = train[spec[0]].astype(str)
        for c in spec[1:]:
            key = key + "|" + train[c].astype(str)
        best_inter_oof = oof_prior_from_key(key, y)
    else:
        best_inter_oof = pd.Series(np.full(len(train), y.mean()))
    additive = additive_value_report(
        train,
        {
            "collision_r2_prior_oof": collision2_oof,
            "collision_r3_prior_oof": collision3_oof,
            "best_pair_prior_oof": best_pair_oof,
            "best_interaction_prior_oof": best_inter_oof,
        },
    )

    quant.to_csv(exp_dir / "quantization_forensics.csv", index=False)
    coll2.to_csv(exp_dir / "template_collisions_round2.csv", index=False)
    coll3.to_csv(exp_dir / "template_collisions_round3.csv", index=False)
    ce.to_csv(exp_dir / "conditional_entropy_pairs.csv", index=False)
    det.to_csv(exp_dir / "deterministic_buckets.csv", index=False)
    inter.to_csv(exp_dir / "interaction_forensics.csv", index=False)
    additive.to_csv(exp_dir / "artifact_additive_value.csv", index=False)
    drift = quant[["feature", "train_test_mean_drift_std", "top_exact_share", "inferred_step", "inferred_step_share"]].copy()
    drift.to_csv(exp_dir / "train_test_drift_diagnostics.csv", index=False)

    summary = {
        "exp_id": exp_id,
        "top_quantization": quant.head(20).to_dict("records"),
        "top_collision_r2": coll2.head(20).to_dict("records"),
        "top_pairs": ce.head(20).to_dict("records"),
        "top_interactions": inter.head(20).to_dict("records"),
        "additive": additive.head(20).to_dict("records"),
    }
    write_json(summary, exp_dir / "summary.json")
    lines = [
        "# Synthetic Artifact Forensics",
        "",
        "## Quantization",
        quant.head(40).to_markdown(index=False),
        "",
        "## Template Collisions Round 2",
        coll2.head(40).to_markdown(index=False),
        "",
        "## Template Collisions Round 3",
        coll3.head(40).to_markdown(index=False),
        "",
        "## Conditional Entropy Pairs",
        ce.head(40).to_markdown(index=False),
        "",
        "## Deterministic Buckets",
        det.head(60).to_markdown(index=False),
        "",
        "## Interactions",
        inter.head(40).to_markdown(index=False),
        "",
        "## Additive Value vs CatBoost",
        additive.head(40).to_markdown(index=False),
    ]
    (REPORT_DIR / "synthetic_artifact_forensics_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Top quantization")
    print(quant.head(10).to_string(index=False))
    print("Top pairs")
    print(ce.head(10).to_string(index=False))
    print("Additive")
    print(additive.head(10).to_string(index=False))
    print(f"wrote {exp_dir}")


if __name__ == "__main__":
    main()
