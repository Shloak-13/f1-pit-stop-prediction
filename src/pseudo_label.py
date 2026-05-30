from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import TARGET
from utils import DATA_RAW, REPORT_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, help="submission or test prediction CSV")
    parser.add_argument("--low", type=float, default=0.015)
    parser.add_argument("--high", type=float, default=0.985)
    parser.add_argument("--out", default="data/raw/train_pseudo.csv")
    args = parser.parse_args()
    ensure_dirs()
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    pred = pd.read_csv(args.prediction)
    merged = test.merge(pred[["id", TARGET]], on="id", how="left")
    chosen = merged[(merged[TARGET] <= args.low) | (merged[TARGET] >= args.high)].copy()
    chosen[TARGET] = (chosen[TARGET] >= args.high).astype(int)
    pseudo_train = pd.concat([train, chosen[train.columns]], axis=0, ignore_index=True)
    pseudo_train.to_csv(args.out, index=False)
    report = {
        "prediction": args.prediction,
        "low": args.low,
        "high": args.high,
        "selected": int(len(chosen)),
        "positive_selected": int(chosen[TARGET].sum()),
        "negative_selected": int((1 - chosen[TARGET]).sum()),
    }
    pd.Series(report).to_csv(REPORT_DIR / "pseudo_label_report.csv")
    print(report)


if __name__ == "__main__":
    main()
