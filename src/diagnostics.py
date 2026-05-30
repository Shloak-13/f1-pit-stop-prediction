from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from features import CAT_COLS, TARGET, build_features, load_raw
from utils import REPORT_DIR, ensure_dirs, write_json


def cramers_like_auc_signal(df: pd.DataFrame, col: str, target: str) -> float:
    stats = df.groupby(col, observed=True)[target].agg(["mean", "count"])
    weighted_var = np.average((stats["mean"] - df[target].mean()) ** 2, weights=stats["count"])
    return float(weighted_var)


def profile_columns(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    out = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "target_mean": float(train[TARGET].mean()),
        "columns": {},
        "duplicates_excluding_id_target": int(train.drop(columns=["id", TARGET]).duplicated().sum()),
        "test_duplicates_excluding_id": int(test.drop(columns=["id"]).duplicated().sum()),
    }
    both = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    for col in train.columns:
        if col == TARGET:
            continue
        entry = {
            "dtype": str(train[col].dtype),
            "train_nunique": int(train[col].nunique(dropna=False)),
            "test_nunique": int(test[col].nunique(dropna=False)),
            "missing_train": int(train[col].isna().sum()),
            "missing_test": int(test[col].isna().sum()),
        }
        if col in CAT_COLS or train[col].nunique() <= 50:
            entry["target_by_value"] = (
                train.groupby(col, observed=True)[TARGET].agg(["count", "mean"]).sort_values("mean").tail(20).reset_index().to_dict("records")
            )
            entry["categorical_signal_variance"] = cramers_like_auc_signal(train, col, TARGET)
        else:
            entry["train_mean"] = float(train[col].mean())
            entry["test_mean"] = float(test[col].mean())
            entry["train_std"] = float(train[col].std())
            entry["test_std"] = float(test[col].std())
            entry["target_corr"] = float(train[col].corr(train[TARGET]))
        entry["combined_nunique"] = int(both[col].nunique(dropna=False))
        out["columns"][col] = entry
    return out


def mutual_information_report(train: pd.DataFrame, test: pd.DataFrame, sample: int = 60_000) -> pd.DataFrame:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        return pd.DataFrame()
    if len(train) > sample:
        train = train.sample(sample, random_state=20260518).sort_index().reset_index(drop=True)
    if len(test) > sample // 2:
        test = test.sample(sample // 2, random_state=20260518).sort_index().reset_index(drop=True)
    X, y, _, cols = build_features(train, test, level="balanced")
    sample = min(sample, len(X))
    rng = np.random.default_rng(20260518)
    idx = rng.choice(len(X), sample, replace=False)
    if X.shape[1] > 220:
        variances = X.var(numeric_only=True).sort_values(ascending=False)
        keep = list(variances.head(220).index)
        X = X[keep]
        cols = keep
    mi = mutual_info_classif(X.iloc[idx], y.iloc[idx], random_state=20260518, discrete_features=False, n_neighbors=3)
    return pd.DataFrame({"feature": cols, "mutual_information": mi}).sort_values("mutual_information", ascending=False)


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.inspection import permutation_importance
    except Exception:
        return pd.DataFrame()
    X_train, _, X_test, cols = build_features(train, test, level="balanced")
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260518)
    aucs = []
    importances = np.zeros(X.shape[1], dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, l2_regularization=0.05, random_state=20260518)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], pred))
        perm = permutation_importance(model, X.iloc[va_idx].sample(min(8000, len(va_idx)), random_state=1), y[va_idx][: min(8000, len(va_idx))], n_repeats=2, random_state=1)
        importances += perm.importances_mean
    return pd.DataFrame({"feature": cols, "adv_importance": importances / skf.get_n_splits(), "adv_auc_mean": np.mean(aucs)}).sort_values(
        "adv_importance", ascending=False
    )


def pairwise_interactions(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["Driver", "Race", "Compound", "Year", "PitStop", "Stint", "Position"]
    for a, b in itertools.combinations(keys, 2):
        grp = train.groupby([a, b], observed=True)[TARGET].agg(["count", "mean"])
        grp = grp[grp["count"] >= 20]
        if len(grp):
            rows.append({"pair": f"{a}|{b}", "groups": int(len(grp)), "min": float(grp["mean"].min()), "max": float(grp["mean"].max()), "std": float(grp["mean"].std())})
    return pd.DataFrame(rows).sort_values("std", ascending=False)


def write_markdown(profile: dict, mi: pd.DataFrame, adv: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    def table_lines(df: pd.DataFrame) -> list[str]:
        try:
            return df.to_markdown(index=False).splitlines()
        except Exception:
            return df.to_string(index=False).splitlines()

    lines = [
        "# F1 Pit Stop Dataset Intelligence Report",
        "",
        f"- Train shape: {profile['train_shape']}",
        f"- Test shape: {profile['test_shape']}",
        f"- Target rate: {profile['target_mean']:.6f}",
        f"- Duplicate train rows excluding id/target: {profile['duplicates_excluding_id_target']}",
        f"- Duplicate test rows excluding id: {profile['test_duplicates_excluding_id']}",
        "",
        "## Column interpretation",
        "- Categorical: Driver, Compound, Race; low-cardinality ordinal/state: Year, PitStop, Stint, Position.",
        "- Cyclic/progression: RaceProgress is LapNumber divided by an inferred race length; lap sine/cosine and phase bins are useful.",
        "- Tyre degradation proxies: TyreLife, LapTime_Delta, Cumulative_Degradation, degradation per tyre lap, stint-lap gap.",
        "- Temporal ordering: Year/Race/Driver/LapNumber/Stint/TyreLife/id is the primary reconstructed sequence key.",
        "- Synthetic fingerprints: id order, exact rational RaceProgress, high-cardinality Driver templates, and repeated group statistics.",
        "",
        "## Top mutual information features",
    ]
    if len(mi):
        lines.extend(table_lines(mi.head(40)))
    lines.extend(["", "## Adversarial train/test drift"])
    if len(adv):
        lines.append(f"- Mean adversarial AUC: {adv['adv_auc_mean'].iloc[0]:.6f}")
        lines.extend(table_lines(adv.head(40)))
    lines.extend(["", "## Strong pairwise target interactions"])
    if len(pairwise):
        lines.extend(table_lines(pairwise.head(30)))
    (REPORT_DIR / "dataset_intelligence.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-adv", action="store_true")
    parser.add_argument("--skip-mi", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    train, test, _ = load_raw()
    profile = profile_columns(train, test)
    write_json(profile, REPORT_DIR / "profile.json")
    mi = pd.DataFrame() if args.skip_mi else mutual_information_report(train, test)
    if len(mi):
        mi.to_csv(REPORT_DIR / "mutual_information.csv", index=False)
    adv = pd.DataFrame() if args.skip_adv else adversarial_validation(train, test)
    if len(adv):
        adv.to_csv(REPORT_DIR / "adversarial_validation.csv", index=False)
    pairwise = pairwise_interactions(train)
    pairwise.to_csv(REPORT_DIR / "pairwise_interactions.csv", index=False)
    write_markdown(profile, mi, adv, pairwise)
    print("wrote reports/dataset_intelligence.md")


if __name__ == "__main__":
    main()
