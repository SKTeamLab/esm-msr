"""Extra rank checks: bottom-of-ranking quality, composition of the top-10% picks. Usage: 12_rank_extra.py data.pkl"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from common import *

data = pickle.load(open(sys.argv[1], "rb"))
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
AA = {"P": "Pro", "G": "Gly", **{a: "hydrophobic" for a in "AVILMFWYC"}, **{a: "polar" for a in "STNQH"}, **{a: "charged" for a in "DEKR"}}
BIG = set("WCFIM")
rows, comp = [], []
for d, df in data.items():
    if res.loc[d, "is_tsuboyama"]:
        continue
    s = df[~df.mutant.str.contains(":")].reset_index(drop=True)
    if len(s) < 200:
        continue
    pm = (rankdata(s.DMS_score) - 0.5) / len(s)
    n10 = len(s) // 10
    r = dict(DMS_id=d, category=res.loc[d, "category"])
    for sg in (0.0, 0.5, 1.0):
        order = np.argsort(-s[SCOL[sg]].values, kind="stable")
        r[f"bottom10_meanpct_s{sg}"] = pm[order[-n10:]].mean()           # worst-predicted 10%: lower = better at spotting damage
        r[f"bottom10_good_s{sg}"] = (pm[order[-n10:]] >= 0.75).mean()    # false alarms
        top = order[:n10]
        to = s.mutant.str[-1].values[top]
        r[f"top10_frac_hydrophobic_s{sg}"] = np.mean([AA.get(a) == "hydrophobic" for a in to])
        r[f"top10_frac_WCFIM_s{sg}"] = np.mean([a in BIG for a in to])
        r[f"top10_frac_polarcharged_s{sg}"] = np.mean([AA.get(a) in ("polar", "charged") for a in to])
    # measured percentile of the *swapped-in* picks: in top10 at s1 but not at s0, and vice versa
    o0 = set(np.argsort(-s[SCOL[0.0]].values, kind="stable")[:n10]); o1 = set(np.argsort(-s[SCOL[1.0]].values, kind="stable")[:n10])
    inn, out = list(o1 - o0), list(o0 - o1)
    r["overlap_top10"] = len(o0 & o1) / n10
    r["swapped_in_meanpct"] = pm[inn].mean() if inn else np.nan
    r["swapped_out_meanpct"] = pm[out].mean() if out else np.nan
    to = s.mutant.str[-1].values
    r["swapped_in_frac_WCFIM"] = np.mean([to[i] in BIG for i in inn]) if inn else np.nan
    r["swapped_out_frac_WCFIM"] = np.mean([to[i] in BIG for i in out]) if out else np.nan
    rows.append(r)
t = pd.DataFrame(rows).set_index("DMS_id")
t.to_csv(f"{TAB}/rank_extra_nonTsuboyama.csv")
m = t.drop(columns="category").mean()
print(f"n={len(t)}")
for k in ["bottom10_meanpct", "bottom10_good", "top10_frac_hydrophobic", "top10_frac_WCFIM", "top10_frac_polarcharged"]:
    print(k, [round(m[f"{k}_s{s}"], 3) for s in (0.0, 0.5, 1.0)])
print("overlap of top10 sets (s0 vs s1):", round(m.overlap_top10, 3))
print("swapped-in meanpct", round(m.swapped_in_meanpct, 3), " swapped-out meanpct", round(m.swapped_out_meanpct, 3))
print("swapped-in frac WCFIM", round(m.swapped_in_frac_WCFIM, 3), " swapped-out", round(m.swapped_out_frac_WCFIM, 3))
# correlation of bulk rho change with top10 change
ids = t.index
import pickle as pk
R = pk.load(open(f"{TAB}/rank_all_assays.pkl", "rb"))["A"]
k10 = 3
d_rho = np.array([res.loc[d, "rho_1.0"] - res.loc[d, "rho_0.0"] for d in ids])
d_top = np.array([R[d]["fwd"][1.0]["mean"][k10] - R[d]["fwd"][0.0]["mean"][k10] for d in ids])
d_inv = np.array([R[d]["inv"][1.0]["mean"][k10] - R[d]["inv"][0.0]["mean"][k10] for d in ids])
print("spearman(d_rho, d_top10_forward)=%.2f  (d_rho, d_inverse)=%.2f" % (spearmanr(d_rho, d_top)[0], spearmanr(d_rho, d_inv)[0]))
print("mean d_bottom10 (sigma1-sigma0) =", round((t["bottom10_meanpct_s1.0"] - t["bottom10_meanpct_s0.0"]).mean(), 3), " frac assays lower:", round(((t["bottom10_meanpct_s1.0"] < t["bottom10_meanpct_s0.0"])).mean(), 2))
