"""Substitution-level view across all assays: which amino acids does sigma re-rank, and does it match the assay?

Usage: 07_fig_substitutions.py data.pkl
"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from viz import *

data = pickle.load(open(sys.argv[1], "rb"))
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
AAS = list("AVILMFWYCGPSTNQHDEKR")  # hydrophobic, small/polar, charged
pct = lambda x: (rankdata(x) - 0.5) / len(x)

meas, shift = {}, {}
for d, df in data.items():
    s = df[~df.mutant.str.contains(":")].copy()
    if len(s) < 100:
        continue
    s["to"] = s.mutant.str[-1]
    s["m"] = pct(s["DMS_score"].values)
    s["shift"] = pct(s["s1.0"].values) - pct(s["s0.0"].values)
    meas[d] = s.groupby("to")["m"].mean().reindex(AAS)
    shift[d] = s.groupby("to")["shift"].mean().reindex(AAS)
M, SH = pd.DataFrame(meas).T, pd.DataFrame(shift).T
M.to_csv(f"{TAB}/measured_percentile_by_target_aa.csv"); SH.to_csv(f"{TAB}/sigma_shift_by_target_aa.csv")
cat = res.loc[M.index, "category"]

# ---- F18 ----------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 3.9), gridspec_kw=dict(wspace=0.12))
for a, T, ttl, vm in [(axs[0], SH, "model: mean change in predicted percentile, sigma 1 minus 0", 0.15),
                      (axs[1], M - 0.5, "data: mean measured percentile minus 0.5 (higher = tolerated)", 0.12)]:
    H = np.array([T[cat == c].mean().values for c in CATS])
    a.imshow(H, cmap=DIV, vmin=-vm, vmax=vm, aspect="auto")
    a.set_xticks(range(20)); a.set_xticklabels(AAS); a.grid(False)
    a.set_yticks(range(5)); a.set_yticklabels([f"{c.replace('OrganismalFitness', 'Fitness')} (n={int((cat == c).sum())})" for c in CATS], fontsize=8.5)
    a.set_xlabel("substituted-to amino acid"); a.set_title(ttl, fontsize=9.5)
    if a is axs[1]: a.set_yticklabels([])
    for i in range(5):
        for j in range(20):
            a.text(j, i, f"{H[i, j] * 100:+.0f}", ha="center", va="center", fontsize=6.5, color="white" if abs(H[i, j]) > 0.65 * vm else INK)
fig.suptitle("Which substitutions does sigma re-rank (left), versus what each assay category actually penalises (right); values x100",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "F18_substitution_class_shift_vs_data")

# ---- F19: 'stability-likeness' of an assay's substitution preferences vs its gain --------------------------------
stab = M[res.loc[M.index, "is_tsuboyama"].values].mean()
like = M.apply(lambda r: spearmanr(r, stab, nan_policy="omit")[0], axis=1)
dsel = (res.loc[M.index, "rho_1.0"] - res.loc[M.index, "rho_0.0"])
tab = pd.DataFrame({"likeness": like, "d_full": dsel, "category": cat, "is_tsuboyama": res.loc[M.index, "is_tsuboyama"],
                    "axis": res.loc[M.index, "axis"], "protein": res.loc[M.index, "protein"]})
tab.to_csv(f"{TAB}/stability_likeness_vs_gain.csv")
nt = tab[~tab.is_tsuboyama]
r_, p_ = spearmanr(nt.likeness, nt.d_full, nan_policy="omit")
print(f"non-Tsuboyama n={len(nt)} spearman(likeness, delta)= {r_:+.3f} p={p_:.2g}")
fig, ax = plt.subplots(figsize=(7.4, 5.2))
for c in CATS:
    t = nt[nt.category == c]
    ax.scatter(t.likeness, t.d_full, s=24, color=CAT_COLOR[c], alpha=0.75, lw=0.5, edgecolor=SURFACE, label=CAT_LABEL[c].split(" (")[0])
for d, r in nt[nt.axis.notna()].iterrows():
    if abs(r.d_full) > 0.09 or abs(r.likeness) > 0.8:
        ax.annotate(r.protein.split("_")[0], (r.likeness, r.d_full), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("similarity of the assay's amino-acid preferences to the Tsuboyama stability pattern\n(Spearman over the 20 substituted-to residues)")
ax.set_ylabel("rho(1.0) - rho(0)")
ax.set_title(f"Assays that read like stability gain more (Spearman {r_:+.2f}, n={len(nt)}, excl. Tsuboyama)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
save(fig, "F19_stability_likeness_vs_gain")
