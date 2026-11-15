"""One 'dossier' figure per multi-assay protein: rho(sigma), between/within-site, substitution-class and position strips.

Usage: 06_dossiers.py data.pkl
"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
from matplotlib.gridspec import GridSpec
from viz import *

data = pickle.load(open(sys.argv[1], "rb"))
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
cl = pd.read_csv(f"{TAB}/assay_classes.csv", index_col=0)
boots = pickle.load(open(f"{TAB}/boot_rho.pkl", "rb"))
rc = [f"rho_{s}" for s in SIGMAS]
AA_CLASS = {"P": "Pro", "G": "Gly", **{a: "hydrophobic" for a in "AVILMFWYC"}, **{a: "polar" for a in "STNQH"}, **{a: "charged" for a in "DEKR"}}
CLASS_ORDER = ["hydrophobic", "polar", "charged", "Gly", "Pro"]
multi = res[res.group.notna()].copy()
multi["label"] = multi.index.str.replace(r"^[A-Z0-9]+_[A-Z0-9]+_", "", regex=True)
PALETTE_COND = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def singles(df):
    d = df[~df.mutant.str.contains(":")].copy()
    d["pos"] = d.mutant.str[1:-1].astype(int)
    d["to"] = d.mutant.str[-1]
    d["cls"] = d["to"].map(AA_CLASS)
    return d


def pct(x):
    return (rankdata(x) - 0.5) / len(x)


def dossier(prot):
    sub = multi[multi.protein == prot].copy()
    sub["ord"] = sub.axis.map({"ABUND": 0, "ACT": 1, "BIND": 2, "SPEC": 3, "FIT": 4, "COND": 5})
    sub = sub.sort_values("ord")
    ids = list(sub.index)
    col = {d: (AXIS_COLOR[sub.loc[d, "axis"]] if sub.loc[d, "axis"] != "COND" else PALETTE_COND[i % 4]) for i, d in enumerate(ids)}
    S = {d: singles(data[d]) for d in ids}
    fig = plt.figure(figsize=(16, 4.1 + 0.42 * (len(ids) + 4)))
    gs = GridSpec(2, 4, height_ratios=[3.2, 0.42 * (len(ids) + 4) + 0.7], hspace=0.38, wspace=0.32, figure=fig)
    # (a) rho(sigma)
    a = fig.add_subplot(gs[0, 0])
    for d in ids:
        b = boots[d]
        a.fill_between(SIGMAS, np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0), color=col[d], alpha=0.12, lw=0)
        a.plot(SIGMAS, res.loc[d, rc].values.astype(float), marker="o", color=col[d], mec=SURFACE, mew=1.2,
               label=f"{sub.loc[d, 'axis'].lower()}: {sub.loc[d, 'label'][:26]} ({cl.loc[d, 'cls'].lower()})")
    sigma_axis(a); a.set_ylabel("Spearman rho"); a.set_title("rho vs sigma (all rows)", fontsize=10)
    a.legend(fontsize=6.5, loc="best")
    # (b) between / within site change
    a = fig.add_subplot(gs[0, 1])
    x = np.arange(len(ids))
    for off, pre, hatch in [(-0.2, "between", None), (0.2, "within", "//")]:
        v = [res.loc[d, f"{pre}_1.0"] - res.loc[d, f"{pre}_0.0"] for d in ids]
        a.bar(x + off, v, 0.38, color=[col[d] for d in ids], edgecolor=SURFACE, lw=1.2, hatch=hatch, alpha=0.95 if hatch is None else 0.6)
    a.axhline(0, color=MUTED, lw=1); a.set_xticks(x); a.set_xticklabels([sub.loc[d, "axis"].lower() for d in ids], fontsize=8)
    a.set_title("change at sigma 1 vs 0\n(solid: between-site, hatched: within-site)", fontsize=9.5); a.grid(axis="x", visible=False)
    # (c) measured mean percentile by target class
    a = fig.add_subplot(gs[0, 2])
    for d in ids:
        s = S[d]
        if len(s) < 30: continue
        s["m"] = pct(s["DMS_score"].values)
        v = s.groupby("cls")["m"].mean().reindex(CLASS_ORDER)
        a.plot(range(5), v.values, marker="o", color=col[d], mec=SURFACE, mew=1.2)
    a.axhline(0.5, color=MUTED, lw=1); a.set_xticks(range(5)); a.set_xticklabels(CLASS_ORDER, fontsize=8, rotation=20)
    a.set_title("measured: mean percentile of the\nsubstituted-to amino-acid class", fontsize=9.5); a.set_ylabel("mean percentile (0=worst)")
    # (d) predicted percentile shift by target class (union of singles)
    a = fig.add_subplot(gs[0, 3])
    u = S[ids[0]]
    for d in ids[1:]:
        u = pd.concat([u, S[d]]).drop_duplicates("mutant")
    u = u.copy()
    p0, p1 = pct(u["s0.0"].values), pct(u["s1.0"].values)
    u["shift"] = p1 - p0
    v = u.groupby("cls")["shift"].mean().reindex(CLASS_ORDER)
    a.bar(range(5), v.values, color="#6250d6", edgecolor=SURFACE, lw=1.5)
    a.axhline(0, color=MUTED, lw=1); a.set_xticks(range(5)); a.set_xticklabels(CLASS_ORDER, fontsize=8, rotation=20)
    a.set_title("model: change in predicted percentile\n(sigma 1 minus 0) by substituted-to class", fontsize=9.5)
    a.grid(axis="x", visible=False)
    # (e) position strips
    a = fig.add_subplot(gs[1, :])
    rows, names = [], []
    allpos = np.arange(u.pos.min(), u.pos.max() + 1)
    def strip(vals):
        m = vals.groupby(level=0).mean().reindex(allpos)
        z = (m - m.mean()) / m.std()
        return z.values
    for d in ids:
        s = S[d]
        if len(s) < 30: continue
        rows.append(strip(s.set_index("pos")["DMS_score"])); names.append(f"MEASURED  {sub.loc[d, 'axis'].lower()}: {sub.loc[d, 'label'][:30]}")
    for sg in [0.0, 0.5, 1.0]:
        rows.append(strip(u.set_index("pos")[SCOL[sg]])); names.append(f"PREDICTED sigma = {sg:g}")
    M = np.array(rows)
    cmap = DIV.copy(); cmap.set_bad("#e4e3de")
    a.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5, extent=(allpos[0] - 0.5, allpos[-1] + 0.5, len(rows), 0), interpolation="nearest")
    a.set_yticks(np.arange(len(rows)) + 0.5); a.set_yticklabels(names, fontsize=8); a.grid(False)
    for i in range(len(ids) if len(ids) < len(rows) else 0, len(rows)):
        pass
    a.axhline(len([n for n in names if n.startswith("MEASURED")]), color=INK, lw=1.5)
    a.set_xlabel("sequence position in the assay's numbering"); a.set_title("per-position mean (singles; blue = tolerated / high, red = intolerant / low; z-scored per row)", fontsize=9.5)
    fig.suptitle(f"{prot}  -  group {sub.group.iloc[0]}  -  {', '.join(sub.axis.str.lower().unique())}", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, f"dossiers/{prot}")


if __name__ == "__main__":
    os.makedirs(f"{FIG}/dossiers", exist_ok=True)
    for p in sorted(multi.protein.unique()):
        dossier(p)
