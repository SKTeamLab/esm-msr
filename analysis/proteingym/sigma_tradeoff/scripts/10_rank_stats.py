"""Rank-based selection statistics (singles only).

Forward: rank mutants by predicted score at sigma s, take the top K fraction, report how those mutants score in the
         measured data (own assay, and for multi-assay proteins every other assay on the shared mutants).
Inverse: rank mutants by measured score, take the top K fraction, report where the sigma-s model ranks them.
Percentiles are within-assay (within the shared set for cross-property pairs), 1 = best.

Writes tables/rank_all_assays.pkl, tables/rank_pairs.pkl and tables/rank_all_assays_top10.csv.
Usage: 10_rank_stats.py data.pkl
"""
import sys, pickle, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata
from common import *

KS = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
SEL = [0.0, 0.5, 1.0]           # sigmas used for selection / evaluation
MINN = 200
pct = lambda x: (rankdata(x) - 0.5) / len(x)

data = pickle.load(open(sys.argv[1], "rb"))
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)


def singles(df):
    return df[~df.mutant.str.contains(":")].reset_index(drop=True)


def topk_stats(sel_score, ev):
    """sel_score: score ranking the selection (high = selected); ev: dict name -> percentile array to evaluate.
    returns {name: dict(mean[K], good[K], bad[K])} with good = pct>=0.75, bad = pct<=0.25."""
    order = np.argsort(-sel_score, kind="stable")
    n = len(order)
    out = {}
    for name, p in ev.items():
        ps = p[order]
        cs = np.cumsum(ps)
        cg = np.cumsum(ps >= 0.75)
        cb = np.cumsum(ps <= 0.25)
        idx = np.maximum((KS * n).astype(int), 1) - 1
        cnt = idx + 1
        out[name] = dict(mean=cs[idx] / cnt, good=cg[idx] / cnt, bad=cb[idx] / cnt)
    return out


def decile_mean(rank_score, p):
    """mean of p within deciles of rank_score (decile 0 = top)."""
    order = np.argsort(-rank_score, kind="stable")
    return np.array([p[order][i * len(p) // 10:(i + 1) * len(p) // 10].mean() for i in range(10)])


# ------------------------------------------------------------------ all assays, own property
rows, A = [], {}
for d, df in data.items():
    s = singles(df)
    if len(s) < MINN:
        continue
    pm = pct(s["DMS_score"].values)
    pp = {sg: pct(s[SCOL[sg]].values) for sg in SIGMAS}
    r = dict(fwd={}, inv={}, fwd_dec={}, inv_dec={})
    for sg in SEL:
        r["fwd"][sg] = topk_stats(s[SCOL[sg]].values, {"own": pm})["own"]
        r["inv"][sg] = topk_stats(s["DMS_score"].values, {"own": pp[sg]})["own"]
        r["fwd_dec"][sg] = decile_mean(s[SCOL[sg]].values, pm)
        r["inv_dec"][sg] = decile_mean(s["DMS_score"].values, pp[sg])
    A[d] = r
    k10 = list(KS).index(0.10)
    rows.append(dict(DMS_id=d, n_single=len(s), **{f"top10_meanpct_s{sg}": r["fwd"][sg]["mean"][k10] for sg in SEL},
                     **{f"top10_good_s{sg}": r["fwd"][sg]["good"][k10] for sg in SEL},
                     **{f"top10_bad_s{sg}": r["fwd"][sg]["bad"][k10] for sg in SEL},
                     **{f"inv10_meanpct_s{sg}": r["inv"][sg]["mean"][k10] for sg in SEL},
                     **{f"inv10_good_s{sg}": r["inv"][sg]["good"][k10] for sg in SEL},
                     **{f"inv10_bad_s{sg}": r["inv"][sg]["bad"][k10] for sg in SEL}))
tab = pd.DataFrame(rows).set_index("DMS_id").join(res[["category", "is_tsuboyama", "protein", "group", "axis"]])
tab.to_csv(f"{TAB}/rank_all_assays_top10.csv")
pickle.dump(dict(KS=KS, SEL=SEL, A=A), open(f"{TAB}/rank_all_assays.pkl", "wb"))
print("own-property rank stats for", len(A), "assays")

# ------------------------------------------------------------------ cross-property pairs (shared singles)
multi = res[res.group.notna()]
P = []
maxdiff = 0.0
for prot, g in multi.groupby("protein"):
    ids = list(g.index)
    for a, o in itertools.permutations(ids, 2):
        if g.loc[a, "axis"] == g.loc[o, "axis"] and g.loc[a, "group"] != "C" and g.loc[a, "group"] != "B":
            continue
        sa, so = singles(data[a]), singles(data[o])
        m = sa.merge(so, on="mutant", suffixes=("_a", "_o"))
        if len(m) < MINN:
            continue
        for sg in SEL:
            maxdiff = max(maxdiff, np.abs(m[f"{SCOL[sg]}_a"] - m[f"{SCOL[sg]}_o"]).max())
        pa, po = pct(m["DMS_score_a"].values), pct(m["DMS_score_o"].values)
        pps = {sg: pct(m[f"{SCOL[sg]}_a"].values) for sg in SEL}
        rec = dict(protein=prot, group=g.loc[a, "group"], a=a, o=o, a_axis=g.loc[a, "axis"], o_axis=g.loc[o, "axis"], n=len(m),
                   label_corr=np.corrcoef(rankdata(m["DMS_score_a"]), rankdata(m["DMS_score_o"]))[0, 1], fwd={}, orc={}, inv={}, inv_orc={})
        for sg in SEL:   # select by model, evaluate in A and O
            rec["fwd"][sg] = topk_stats(m[f"{SCOL[sg]}_a"].values, {"a": pa, "o": po})
        # select by true A (oracle on A); evaluate O (labels) and model percentiles
        rec["orc"] = topk_stats(m["DMS_score_a"].values, {"a": pa, "o": po, **{f"pred{sg}": pps[sg] for sg in SEL}})
        # select by true O; evaluate A (labels) and model percentiles
        rec["inv_orc"] = topk_stats(m["DMS_score_o"].values, {"a": pa, "o": po, **{f"pred{sg}": pps[sg] for sg in SEL}})
        P.append(rec)
pickle.dump(dict(KS=KS, SEL=SEL, P=P), open(f"{TAB}/rank_pairs.pkl", "wb"))
print("pairs:", len(P), " (a,o ordered)  max |pred diff| between assays on shared mutants:", maxdiff)
