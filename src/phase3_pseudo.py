from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from features import TARGET, load_raw
from utils import DATA_RAW, REPORT_DIR, ensure_dirs


PRED_SOURCES = {
    "cat": "experiments/20260518_165505_phase2_cat_fast_stratified/test_catboost.csv",
    "table_hgb": "experiments/20260518_162337_phase2_table_stratified_hgb/test_hgb.csv",
    "phase3": "experiments/20260519_141027_phase3_knn_euclid_phase3/test_phase3.csv",
    "specialist": "experiments/20260519_143756_phase3_specialists/test_specialist.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=float, default=0.02)
    parser.add_argument("--high", type=float, default=0.98)
    parser.add_argument("--max-std", type=float, default=0.035)
    parser.add_argument("--out", default="data/raw/train_phase3_pseudo.csv")
    args = parser.parse_args()
    ensure_dirs()
    train, test, _ = load_raw()
    pred = test[["id"]].copy()
    for name, path in PRED_SOURCES.items():
        p = pd.read_csv(path)[["id", TARGET]].rename(columns={TARGET: name})
        pred = pred.merge(p, on="id", how="left")
    cols = list(PRED_SOURCES)
    pred["mean"] = pred[cols].mean(axis=1)
    pred["std"] = pred[cols].std(axis=1)
    pred["min"] = pred[cols].min(axis=1)
    pred["max"] = pred[cols].max(axis=1)
    low_mask = (pred["max"] <= args.low) & (pred["std"] <= args.max_std)
    high_mask = (pred["min"] >= args.high) & (pred["std"] <= args.max_std)
    chosen = pred[low_mask | high_mask].copy()
    chosen[TARGET] = high_mask[low_mask | high_mask].astype(int).values
    pseudo_rows = test.merge(chosen[["id", TARGET]], on="id", how="inner")
    out = pd.concat([train, pseudo_rows[train.columns]], axis=0, ignore_index=True)
    out.to_csv(args.out, index=False)
    report = {
        "low": args.low,
        "high": args.high,
        "max_std": args.max_std,
        "selected": int(len(chosen)),
        "positive": int(chosen[TARGET].sum()) if len(chosen) else 0,
        "negative": int((1 - chosen[TARGET]).sum()) if len(chosen) else 0,
        "mean_std_selected": float(chosen["std"].mean()) if len(chosen) else None,
    }
    pd.Series(report).to_csv(REPORT_DIR / "phase3_pseudo_report.csv")
    pred.to_csv(REPORT_DIR / "phase3_pseudo_prediction_agreement.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
