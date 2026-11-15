"""Classify each assay's rho(sigma) curve and each abundance-vs-other assay pair.

Thresholds (practical AND statistical): a change counts only if |delta rho| >= EFF and |z| >= 2,
z = delta / bootstrap SD of the paired difference (rows resampled jointly across sigmas).

Curve classes:
  RISE      best sigma* improves on sigma=0 and rho(1.0) stays within EFF of the best
  HUMP      improves at sigma*<1.0 but rho(1.0) falls >EFF below the peak, rho(1.0) still >= rho(0)-EFF
  HUMP_LOSS as HUMP, but rho(1.0) is below rho(0) by >= EFF (net loss at full sigma)
  FALL      never improves on sigma=0 and rho(1.0) <= rho(0)-EFF
  FLAT      none of the above
sigma* = smallest sigma whose rho is within 0.01 of the maximum.
"""
import pickle
import numpy as np, pandas as pd
from common import *

EFF, ZTH = 0.02, 2.0
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
boots = pickle.load(open(f"{TAB}/boot_rho.pkl", "rb"))
R = res[[f"rho_{s}" for s in SIGMAS]].values


def sig_gain(d, i, j):
    """(delta, z) of rho[j]-rho[i] using the bootstrap for assay d."""
    b = boots[d]
    dlt = b[:, j] - b[:, i]
    sd = dlt.std()
    est = res.loc[d, f"rho_{SIGMAS[j]}"] - res.loc[d, f"rho_{SIGMAS[i]}"]
    return est, (est / sd if sd > 0 else (0.0 if est == 0 else np.inf * np.sign(est)))


rows = []
for k, d in enumerate(res.index):
    r = R[k]
    sstar_i = int(np.argmax(r >= r.max() - 0.01))      # smallest sigma within 0.01 of max
    d_pk, z_pk = sig_gain(d, 0, sstar_i)
    d_1, z_1 = sig_gain(d, 0, len(SIGMAS) - 1)
    improved = sstar_i > 0 and d_pk >= EFF and z_pk >= ZTH
    drop = r[sstar_i] - r[-1]
    if improved:
        if drop <= EFF:
            cls = "RISE"
        elif d_1 >= -EFF:
            cls = "HUMP"
        else:
            cls = "HUMP_LOSS"
    else:
        cls = "FALL" if (d_1 <= -EFF and z_1 <= -ZTH) else "FLAT"
    rows.append(dict(DMS_id=d, cls=cls, sigma_star=SIGMAS[sstar_i], d_peak=d_pk, d_full=d_1, z_full=z_1,
                     best_minus_full=r.max() - r[-1], rho_0=r[0], rho_best=r.max(), rho_full=r[-1]))
cl = pd.DataFrame(rows).set_index("DMS_id")
cl = cl.join(res[["category", "is_tsuboyama", "protein", "group", "axis", "assay_note", "n"]])
cl.to_csv(f"{TAB}/assay_classes.csv")

print("=== class counts by category ===")
print(pd.crosstab(cl["category"], cl["cls"]).reindex(columns=["RISE", "HUMP", "HUMP_LOSS", "FLAT", "FALL"], fill_value=0))

# ---- pairs: abundance assay vs every other-property assay of the same protein
pairs = []
for p, g in cl[cl.group.isin(["A"])].groupby("protein"):
    ab = g[g.axis == "ABUND"]
    for a in ab.index:
        for o in g.index[g.axis != "ABUND"]:
            pairs.append(dict(protein=p, abund=a, other=o, other_axis=g.loc[o, "axis"],
                              cls_abund=g.loc[a, "cls"], cls_other=g.loc[o, "cls"],
                              dA=g.loc[a, "d_full"], dO=g.loc[o, "d_full"],
                              sstar_A=g.loc[a, "sigma_star"], sstar_O=g.loc[o, "sigma_star"]))
pr = pd.DataFrame(pairs)


def state(c, s):
    return {"RISE": "rises", "HUMP": f"rises to s={s:g}, then falls", "HUMP_LOSS": f"rises to s={s:g}, net loss at 1.0",
            "FLAT": "flat", "FALL": "falls"}[c]


def verdict(r):
    A, O = r.cls_abund, r.cls_other
    up = lambda c: c in ("RISE", "HUMP")
    if A == "RISE" and O == "RISE": return "improve both"
    if A == "RISE" and O in ("HUMP", "HUMP_LOSS"): return "other improves only up to sigma*"
    if A == "RISE" and O == "FLAT": return "abundance only"
    if A == "RISE" and O == "FALL": return "trade-off (abundance up, other down)"
    if A in ("HUMP", "HUMP_LOSS") and O in ("RISE", "HUMP", "HUMP_LOSS"): return "both improve then saturate/fall"
    if A in ("HUMP", "HUMP_LOSS") and O == "FLAT": return "abundance (peaks), other flat"
    if A in ("HUMP", "HUMP_LOSS") and O == "FALL": return "trade-off (abundance peaks, other down)"
    if A == "FLAT" and O in ("RISE",): return "other only"
    if A == "FLAT" and O in ("HUMP", "HUMP_LOSS"): return "other only (peaks)"
    if A == "FLAT" and O == "FLAT": return "neither moves"
    if A == "FLAT" and O == "FALL": return "other falls, abundance flat"
    if A == "FALL" and O in ("RISE", "HUMP", "HUMP_LOSS"): return "inverse trade-off (abundance down, other up)"
    if A == "FALL" and O == "FLAT": return "abundance falls, other flat"
    if A == "FALL" and O == "FALL": return "hurt both"
    return "?"


pr["abund_state"] = [state(c, s) for c, s in zip(pr.cls_abund, pr.sstar_A)]
pr["other_state"] = [state(c, s) for c, s in zip(pr.cls_other, pr.sstar_O)]
pr["verdict"] = pr.apply(verdict, axis=1)
pr.to_csv(f"{TAB}/abundance_vs_other_pairs.csv", index=False)
print("\n=== abundance vs other (group A) ===")
print(pr[["protein", "other_axis", "abund_state", "other_state", "dA", "dO", "verdict"]].round(3).to_string())
print("\n", pr.verdict.value_counts())

# ---- other groups
print("\n=== group A without abundance / B / C ===")
print(cl[cl.group.notna() & ~cl.index.isin(set(pr.abund) | set(pr.other))][["protein", "group", "axis", "cls", "sigma_star", "d_full", "d_peak"]].round(3).to_string())
