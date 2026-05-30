from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from features import TARGET
from utils import SUB_DIR, ensure_dirs, logit, sigmoid, timestamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--tag", default="probe")
    args = parser.parse_args()
    ensure_dirs()
    sub = pd.read_csv(args.base)
    p = np.clip(sub[TARGET].values, 1e-6, 1 - 1e-6)
    rid = f"{timestamp()}_{args.tag}"
    variants = {
        "rank": pd.Series(p).rank(pct=True).values,
        "smooth": 0.98 * p + 0.02 * p.mean(),
        "sharp_110": sigmoid(logit(p) * 1.10),
        "sharp_125": sigmoid(logit(p) * 1.25),
        "soft_090": sigmoid(logit(p) * 0.90),
    }
    for name, pred in variants.items():
        sub[TARGET] = np.clip(pred, 1e-6, 1 - 1e-6)
        sub.to_csv(SUB_DIR / f"{rid}_{name}.csv", index=False)
    print(f"wrote {len(variants)} probe submissions")


if __name__ == "__main__":
    main()
