"""Load the wt_only predictions for every sigma and every ProteinGym assay into one pickle.

Output: dict {dms_id: DataFrame(mutant, DMS_score, s0, s0.25, s0.5, s0.75, s1.0)} plus integrity checks.
"""
import os, sys, pickle
import numpy as np, pandas as pd
from multiprocessing import Pool

from common import BASE
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DIRS = {0.0: "sigma0_wt_only", 0.25: "sigma0.25_wt_only", 0.5: "sigma0.5_wt_only",
        0.75: "sigma0.75_wt_only", 1.0: "sigma1.0_wt_only"}


def load_one(dms):
    frames, issues = {}, []
    for s in SIGMAS:
        f = os.path.join(BASE, DIRS[s], dms + ".csv")
        if not os.path.exists(f):
            issues.append(f"missing {s}")
            continue
        frames[s] = pd.read_csv(f, usecols=["mutant", "DMS_score", "wt_lora_pred"])
    if not frames:
        return dms, None, issues
    ref = frames[min(frames)]
    out = ref[["mutant", "DMS_score"]].copy()
    for s, df in frames.items():
        if len(df) != len(ref) or not (df["mutant"].values == ref["mutant"].values).all():
            issues.append(f"row mismatch at sigma {s}")
        if not np.allclose(df["DMS_score"].values, ref["DMS_score"].values, equal_nan=True) if len(df) == len(ref) else True:
            issues.append(f"DMS_score mismatch at sigma {s}")
        out[f"s{s}"] = df["wt_lora_pred"].values if len(df) == len(ref) else np.nan
    return dms, out, issues


if __name__ == "__main__":
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(BASE, DIRS[0.0])) if f.endswith(".csv") and f != "summary.csv")
    with Pool(6) as p:
        res = p.map(load_one, ids)
    data = {d: df for d, df, _ in res if df is not None}
    for d, _, iss in res:
        if iss:
            print(d, iss)
    print("assays loaded:", len(data))
    pickle.dump(data, open(sys.argv[1], "wb"))
