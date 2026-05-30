from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_RAW, FEATURE_DIR, ensure_dirs


TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Compound", "Race"]
BASE_NUMERIC = [
    "Year",
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype="float64") / np.where(np.asarray(b, dtype="float64") == 0, np.nan, b)


def _entropy(frame: pd.DataFrame, keys: list[str], value: str) -> pd.Series:
    counts = frame.groupby(keys + [value], observed=True).size().rename("n").reset_index()
    totals = counts.groupby(keys, observed=True)["n"].transform("sum")
    probs = counts["n"] / totals
    counts["part"] = -probs * np.log1p(probs - 1e-12)
    ent = counts.groupby(keys, observed=True)["part"].sum()
    return frame.set_index(keys).index.map(ent).astype("float64")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    sub = pd.read_csv(DATA_RAW / "sample_submission.csv")
    return train, test, sub


def build_features(train: pd.DataFrame, test: pd.DataFrame, level: str = "balanced") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    y = train[TARGET].astype(int).copy()
    tr = train.drop(columns=[TARGET]).copy()
    te = test.copy()
    tr["_is_train"] = 1
    te["_is_train"] = 0
    all_df = pd.concat([tr, te], axis=0, ignore_index=True)

    all_df["row_order"] = np.arange(len(all_df))
    all_df["id_norm"] = all_df[ID_COL] / max(float(all_df[ID_COL].max()), 1.0)
    all_df["id_mod_2"] = all_df[ID_COL] % 2
    all_df["id_mod_3"] = all_df[ID_COL] % 3
    all_df["id_mod_5"] = all_df[ID_COL] % 5
    all_df["id_mod_7"] = all_df[ID_COL] % 7

    race_laps = _safe_div(all_df["LapNumber"], all_df["RaceProgress"])
    all_df["race_laps_est"] = np.rint(race_laps).clip(1, 100)
    all_df["laps_remaining_est"] = all_df["race_laps_est"] - all_df["LapNumber"]
    all_df["lap_frac_exact_error"] = np.abs(race_laps - all_df["race_laps_est"])
    all_df["race_phase_5"] = np.floor(all_df["RaceProgress"] * 5).clip(0, 4).astype(int)
    all_df["race_phase_10"] = np.floor(all_df["RaceProgress"] * 10).clip(0, 9).astype(int)
    all_df["lap_sin"] = np.sin(2 * np.pi * all_df["RaceProgress"])
    all_df["lap_cos"] = np.cos(2 * np.pi * all_df["RaceProgress"])

    all_df["tyre_life_frac"] = _safe_div(all_df["TyreLife"], all_df["LapNumber"])
    all_df["tyre_life_remaining_proxy"] = all_df["race_laps_est"] - all_df["TyreLife"]
    all_df["tyre_x_progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
    all_df["stint_x_tyre"] = all_df["Stint"] * all_df["TyreLife"]
    all_df["stint_lap_gap"] = all_df["LapNumber"] - all_df["TyreLife"]
    all_df["degradation_per_tyre_lap"] = _safe_div(all_df["Cumulative_Degradation"], all_df["TyreLife"])
    all_df["degradation_per_lap"] = _safe_div(all_df["Cumulative_Degradation"], all_df["LapNumber"])
    all_df["delta_per_tyre_lap"] = _safe_div(all_df["LapTime_Delta"], all_df["TyreLife"])
    all_df["position_pressure"] = (21 - all_df["Position"]) * (1 + all_df["RaceProgress"])
    all_df["position_change_abs"] = all_df["Position_Change"].abs()
    all_df["lap_time_rank_race"] = all_df.groupby(["Race", "Year"], observed=True)["LapTime (s)"].rank(pct=True)
    all_df["degradation_rank_race"] = all_df.groupby(["Race", "Year"], observed=True)["Cumulative_Degradation"].rank(pct=True)
    all_df["tyre_rank_race_compound"] = all_df.groupby(["Race", "Year", "Compound"], observed=True)["TyreLife"].rank(pct=True)

    for col in CAT_COLS + ["Year", "PitStop", "Stint", "Position", "race_phase_5", "race_phase_10"]:
        vc = all_df[col].value_counts(dropna=False)
        all_df[f"{col}_freq"] = all_df[col].map(vc).astype("float64")
        all_df[f"{col}_freq_norm"] = all_df[f"{col}_freq"] / len(all_df)

    group_sets = [
        ["Driver"],
        ["Race"],
        ["Compound"],
        ["Race", "Year"],
        ["Race", "Compound"],
        ["Driver", "Compound"],
        ["Driver", "Race"],
        ["Stint", "Compound"],
        ["Race", "Stint"],
        ["Race", "Year", "Compound"],
        ["Race", "Year", "Driver"],
    ]
    if level == "balanced":
        group_sets = group_sets[:7] + [["Stint", "Compound"], ["Race", "Year", "Compound"]]
    stat_cols = ["TyreLife", "LapNumber", "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "Position"]
    for keys in group_sets:
        prefix = "__".join(keys)
        sizes = all_df.groupby(keys, observed=True)[ID_COL].transform("count")
        all_df[f"{prefix}_count"] = sizes.astype("float32")
        for col in stat_cols:
            g = all_df.groupby(keys, observed=True)[col]
            mean = g.transform("mean")
            std = g.transform("std").fillna(0)
            all_df[f"{prefix}_{col}_mean"] = mean.astype("float32")
            all_df[f"{prefix}_{col}_z"] = _safe_div(all_df[col] - mean, std + 1e-6).astype("float32")

    all_df["driver_compound_entropy"] = _entropy(all_df, ["Driver"], "Compound")
    all_df["race_driver_entropy"] = _entropy(all_df, ["Race", "Year"], "Driver")
    all_df["race_compound_entropy"] = _entropy(all_df, ["Race", "Year"], "Compound")

    sort_cols = ["Year", "Race", "Driver", "LapNumber", "Stint", "TyreLife", ID_COL]
    all_df = all_df.sort_values(sort_cols).reset_index(drop=True)
    temporal_groups = [
        ["Year", "Race", "Driver"],
        ["Year", "Race", "Position"],
        ["Driver"],
        ["Race", "Year", "Compound"],
    ]
    if level == "balanced":
        temporal_groups = temporal_groups[:2]
    temporal_cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "Position", "TyreLife", "PitStop"]
    for keys in temporal_groups:
        prefix = "seq_" + "__".join(keys)
        g = all_df.groupby(keys, observed=True, sort=False)
        all_df[f"{prefix}_order"] = g.cumcount()
        for col in temporal_cols:
            lag1 = g[col].shift(1)
            all_df[f"{prefix}_{col}_lag1"] = lag1
            all_df[f"{prefix}_{col}_diff1"] = all_df[col] - lag1
            all_df[f"{prefix}_{col}_roll3_mean"] = g[col].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            all_df[f"{prefix}_{col}_ewm_a05"] = g[col].transform(lambda s: s.shift(1).ewm(alpha=0.5, adjust=False).mean())
    all_df = all_df.sort_values("row_order").reset_index(drop=True)

    for c in CAT_COLS:
        codes, _ = pd.factorize(all_df[c], sort=True)
        all_df[f"{c}_code"] = codes.astype("int32")

    for a, b in itertools.combinations(["Driver", "Race", "Compound", "Year", "Stint", "PitStop"], 2):
        key = all_df[a].astype(str) + "|" + all_df[b].astype(str)
        all_df[f"{a}_{b}_hash"] = pd.util.hash_pandas_object(key, index=False).astype("uint64") % 1_000_003

    drop_cols = CAT_COLS + ["row_order", "_is_train"]
    feature_cols = [c for c in all_df.columns if c not in drop_cols]
    X_all = all_df[feature_cols].copy()
    arr = X_all.to_numpy(dtype=np.float32, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    row_idx, col_idx = np.where(np.isnan(arr))
    if len(row_idx):
        arr[row_idx, col_idx] = med[col_idx]
    X_all = pd.DataFrame(arr, columns=feature_cols)

    X_train = X_all.iloc[: len(train)].reset_index(drop=True)
    X_test = X_all.iloc[len(train) :].reset_index(drop=True)
    return X_train, y, X_test, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FEATURE_DIR / "feature_store.parquet"))
    parser.add_argument("--level", default="balanced", choices=["balanced", "full"])
    args = parser.parse_args()
    ensure_dirs()
    train, test, _ = load_raw()
    X_train, y, X_test, cols = build_features(train, test, level=args.level)
    store = pd.concat(
        [
            X_train.assign(PitNextLap=y.values, split="train"),
            X_test.assign(PitNextLap=np.nan, split="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    out = Path(args.out)
    try:
        store.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv")
        store.to_csv(out, index=False)
    pd.Series(cols).to_csv(FEATURE_DIR / "feature_columns.csv", index=False, header=False)
    print(f"saved {out} with {len(cols)} features")


if __name__ == "__main__":
    main()
