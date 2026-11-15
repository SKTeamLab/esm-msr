"""Per-assay metrics vs sigma, for every assay found by 01_load.py.

Writes (tables/):
  per_assay_rho.csv   Spearman(all rows), Spearman(singles only), between-site rho (Spearman of per-position
                      means), mean within-site rho (per-position Spearman, >=5 substitutions), for each sigma
  boot_rho.pkl        {DMS_id: array (NB, n_sigma)} bootstrap rho (rows resampled; >CAP rows are subsampled)
Usage: 02_metrics.py data.pkl
"""
import sys, pickle, zlib
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr, rankdata
from common import *

NB, CAP = 200, 20000
cols = [SCOL[s] for s in SIGMAS]
data = pickle.load(open(sys.argv[1], "rb"))   # fork-shared with the pool workers


def within_site_rho(df, pcol):
    """Mean over positions (>=5 rows, non-constant) of the per-position Spearman(pred, score)."""
    g = df.groupby("pos")
    ok = g["pos"].transform("size") >= 5
    d = df[ok]
    if len(d) < 20:
        return np.nan
    g = d.groupby("pos")
    rx, ry = g[pcol].rank(), g["DMS_score"].rank()
    dx, dy = rx - rx.groupby(d["pos"]).transform("mean"), ry - ry.groupby(d["pos"]).transform("mean")
    num = (dx * dy).groupby(d["pos"]).sum()
    den = np.sqrt((dx ** 2).groupby(d["pos"]).sum() * (dy ** 2).groupby(d["pos"]).sum())
    r = (num / den.replace(0, np.nan)).dropna()
    return r.mean() if len(r) else np.nan


def one(d):
    df = data[d]
    out = dict(DMS_id=d, n=len(df))
    sing = ~df["mutant"].str.contains(":")
    out["n_single"] = int(sing.sum())
    for s, c in zip(SIGMAS, cols):
        out[f"rho_{s}"] = spearmanr(df[c], df["DMS_score"])[0]
    ds = df[sing].copy()
    if len(ds) >= 20 and len(ds) < len(df):
        for s, c in zip(SIGMAS, cols):
            out[f"rho1_{s}"] = spearmanr(ds[c], ds["DMS_score"])[0]
    elif len(ds) == len(df):
        for s in SIGMAS:
            out[f"rho1_{s}"] = out[f"rho_{s}"]
    if len(ds) >= 20:
        ds["pos"] = ds["mutant"].str[1:-1].astype(int)
        m = ds.groupby("pos")[cols + ["DMS_score"]].mean()
        for s, c in zip(SIGMAS, cols):
            out[f"between_{s}"] = spearmanr(m[c], m["DMS_score"])[0] if len(m) > 4 else np.nan
            out[f"within_{s}"] = within_site_rho(ds, c)
    # bootstrap of rho over rows
    b = df if len(df) <= CAP else df.sample(CAP, random_state=1)
    y, X = b["DMS_score"].values, b[cols].values
    rng = np.random.default_rng(zlib.crc32(d.encode()))
    boots = np.empty((NB, len(cols)))
    for k in range(NB):
        i = rng.integers(0, len(b), len(b))
        yb = rankdata(y[i])
        boots[k] = [np.corrcoef(rankdata(X[i, j]), yb)[0, 1] for j in range(len(cols))]
    return out, boots


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    with Pool(7) as p:
        res = p.map(one, sorted(data), chunksize=2)
    rows = pd.DataFrame([r for r, _ in res]).set_index("DMS_id")
    boots = {r["DMS_id"]: b for r, b in res}
    ref = ref_table().drop(columns=["title"]).set_index("DMS_id")
    rows = rows.join(ref, how="left")
    rows.to_csv(f"{TAB}/per_assay_rho.csv")
    pickle.dump(boots, open(f"{TAB}/boot_rho.pkl", "wb"))
    print("assays:", len(rows), "NaN rho:", int(rows[[f"rho_{s}" for s in SIGMAS]].isna().any(axis=1).sum()))
