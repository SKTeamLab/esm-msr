"""Figures for the rank-based (top-K selection) view. Usage: 11_fig_rank.py data.pkl"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from viz import *

R = pickle.load(open(f"{TAB}/rank_all_assays.pkl", "rb"))
PP = pickle.load(open(f"{TAB}/rank_pairs.pkl", "rb"))
KS, SEL, A, P = R["KS"], R["SEL"], R["A"], PP["P"]
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
rng = np.random.default_rng(0)
SIG_COL = {0.0: "#8a8982", 0.5: "#eb6834", 1.0: "#2a78d6"}
SIG_LAB = {0.0: "sigma 0", 0.5: "sigma 0.5", 1.0: "sigma 1"}
K10 = list(KS).index(0.10)


def band(M, nb=2000):
    b = M[rng.integers(0, len(M), (nb, len(M)))].mean(1)
    return M.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def line(ax, x, M, color, label, ls="-"):
    m, lo, hi = band(M)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, ls, color=color, label=label, marker="o", ms=3.5, mec=SURFACE, mew=0.8)


ids_by_cat = {c: [d for d in A if res.loc[d, "category"] == c] for c in CATS}
ids_by_cat["nonTsuboyama (all)"] = [d for d in A if not res.loc[d, "is_tsuboyama"]]
PANELS = CATS + ["nonTsuboyama (all)"]
pname = lambda c: (CAT_LABEL.get(c, c).split(" (")[0]) + f" (n={len(ids_by_cat[c])})"


def panels(fn, key, ykey, ylabel, title, xdec=False, ylim=None, ref=0.5):
    fig, axs = plt.subplots(1, 6, figsize=(18, 3.4), sharey=True)
    for ax, c in zip(axs, PANELS):
        for sg in SEL:
            M = np.array([A[d][key][sg] if xdec else A[d][key][sg][ykey] for d in ids_by_cat[c]])
            x = np.arange(1, 11) if xdec else KS
            line(ax, x, M, SIG_COL[sg], SIG_LAB[sg])
        if ref is not None: ax.axhline(ref, color=MUTED, lw=1, ls=":")
        if not xdec: ax.plot(KS, 1 - KS / 2, color=INK, lw=1, ls="--", label="perfect ranking")
        if xdec: ax.set_xlabel("decile (1 = top)"); ax.set_xticks(range(1, 11))
        else: ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([f"{int(k*100)}%" for k in KS], fontsize=7); ax.set_xlabel("top fraction selected")
        ax.set_title(pname(c), fontsize=9.5)
        if ylim: ax.set_ylim(*ylim)
    axs[0].set_ylabel(ylabel); axs[0].legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    save(fig, fn)


# ---- F21-F24: all assays, own property ------------------------------------------------------------------------
panels("F21_forward_topK_true_percentile", "fwd", "mean", "mean measured percentile of the\nselected mutants (1 = best)",
       "Forward: select the top K% by predicted score; y = their average measured percentile (dotted = random 0.5, dashed = perfect ranking 1-K/2)")
panels("F22_forward_decile_curves", "fwd_dec", None, "mean measured percentile", "Forward: measured percentile by predicted-rank decile", xdec=True)
panels("F23_inverse_topK_pred_percentile", "inv", "mean", "mean model percentile of the\ntrue-best mutants (1 = best)",
       "Inverse: select the top K% by measured score; y = their average model percentile (dotted = random 0.5, dashed = perfect ranking 1-K/2)")
panels("F24_inverse_decile_curves", "inv_dec", None, "mean predicted percentile", "Inverse: predicted percentile by measured-rank decile", xdec=True)

# ---- F25: harm rates (bottom-quartile) forward and inverse, top 10% ---------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
x = np.arange(len(PANELS))
for ax, (key, ttl) in zip(axs, [("fwd", "top 10% by prediction that are in the worst quartile of the data"),
                                ("inv", "top 10% by measurement that the model puts in its worst quartile")]):
    for j, sg in enumerate(SEL):
        vals = [np.mean([A[d][key][sg]["bad"][K10] for d in ids_by_cat[c]]) for c in PANELS]
        ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=SIG_COL[sg], edgecolor=SURFACE, lw=1.5, label=SIG_LAB[sg])
    ax.axhline(0.25, color=MUTED, lw=1, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("OrganismalFitness", "Fitness").replace("nonTsuboyama (all)", "non-Tsub.") for c in PANELS], fontsize=8)
    ax.set_title(ttl, fontsize=9.5); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("fraction (random = 0.25)"); axs[0].legend(fontsize=8)
save(fig, "F25_harm_rates_top10")

# ---- F26: decoupling of bulk rho and top-10% quality --------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
ids = [d for d in A if not res.loc[d, "is_tsuboyama"]]
d_rho = np.array([res.loc[d, "rho_1.0"] - res.loc[d, "rho_0.0"] for d in ids])
d_top = np.array([A[d]["fwd"][1.0]["mean"][K10] - A[d]["fwd"][0.0]["mean"][K10] for d in ids])
d_inv = np.array([A[d]["inv"][1.0]["mean"][K10] - A[d]["inv"][0.0]["mean"][K10] for d in ids])
for ax, y, ttl in [(axs[0], d_top, "forward: top-10% predicted, change in mean measured percentile"),
                   (axs[1], d_inv, "inverse: top-10% measured, change in mean predicted percentile")]:
    for c in CATS:
        m = np.array([res.loc[d, "category"] == c for d in ids])
        ax.scatter(d_rho[m], y[m], s=18, color=CAT_COLOR[c], alpha=0.75, lw=0.4, edgecolor=SURFACE, label=CAT_LABEL[c].split(" (")[0])
    r, p = spearmanr(d_rho, y)
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("delta rho (sigma 1 - sigma 0), all rows"); ax.set_title(f"{ttl}\nSpearman {r:+.2f}", fontsize=9.5)
axs[0].legend(fontsize=7.5); axs[0].set_ylabel("delta (sigma 1 - sigma 0)")
save(fig, "F26_bulk_rho_vs_top10_quality")
print(f"non-Tsub n={len(ids)}: forward top10 delta mean {d_top.mean():+.3f} (frac<0 {np.mean(d_top < 0):.2f}); inverse {d_inv.mean():+.3f}")

# ---- cross-property pairs ------------------------------------------------------------------------------------------------------
ab = [r for r in P if r["group"] == "A" and r["a_axis"] == "ABUND"]              # abundance -> other (11)
rev = [r for r in P if r["group"] == "A" and r["o_axis"] == "ABUND"]             # other -> abundance (the reverse direction)
rep = [r for r in P if r["group"] == "C"]                                         # replicate control
cond = [r for r in P if r["group"] == "B"]


def pooled(ax, recs, which, key, color, label, ls="-"):
    M = np.array([r[which][key]["mean"] if which != "fwd" else None for r in recs]) if which != "fwd" else None


fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), sharey=False)
for sg in SEL:
    line(axs[0], KS, np.array([r["fwd"][sg]["a"]["mean"] for r in ab]), SIG_COL[sg], SIG_LAB[sg])
    line(axs[1], KS, np.array([r["fwd"][sg]["o"]["mean"] for r in ab]), SIG_COL[sg], SIG_LAB[sg])
    line(axs[2], KS, np.array([r["fwd"][sg]["o"]["bad"] for r in ab]), SIG_COL[sg], SIG_LAB[sg])
line(axs[1], KS, np.array([r["orc"]["o"]["mean"] for r in ab]), INK, "oracle: true-best on abundance", ls="--")
line(axs[2], KS, np.array([r["orc"]["o"]["bad"] for r in ab]), INK, "oracle: true-best on abundance", ls="--")
for sg in SEL:   # inverse: true-best on the other property, model percentile
    line(axs[3], KS, np.array([r["inv_orc"][f"pred{sg}"]["mean"] for r in ab]), SIG_COL[sg], SIG_LAB[sg])
for a_, ttl, yl in [(axs[0], "A. model picks: ABUNDANCE", "mean measured percentile"),
                    (axs[1], "B. model picks: OTHER property", "mean measured percentile"),
                    (axs[2], "C. model picks in OTHER's worst quartile", "fraction (random 0.25)"),
                    (axs[3], "D. inverse: true-best OTHER, model pct", "mean predicted percentile")]:
    a_.set_xscale("log"); a_.set_xticks(KS); a_.set_xticklabels([f"{int(k*100)}%" for k in KS], fontsize=7); a_.set_xlabel("top fraction selected")
    a_.set_title(ttl, fontsize=9); a_.set_ylabel(yl)
axs[1].axhline(0.5, color=MUTED, lw=1, ls=":"); axs[0].axhline(0.5, color=MUTED, lw=1, ls=":"); axs[3].axhline(0.5, color=MUTED, lw=1, ls=":"); axs[2].axhline(0.25, color=MUTED, lw=1, ls=":")
axs[1].legend(fontsize=7.5)
fig.suptitle(f"Cross-property selection, the {len(ab)} abundance-vs-other pairs (single mutants shared by both assays; mean and 95% CI over proteins)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "F27_cross_property_selection_pooled")

# ---- F28: per protein at top 10% -------------------------------------------------------------------------------------------------
lab = lambda r: f"{r['protein'].split('_')[0]}\n(r={r['label_corr']:.2f})"
fig, axs = plt.subplots(3, 1, figsize=(13.5, 9.5), sharex=True)
x = np.arange(len(ab))
for ax, (k, ttl, stat) in zip(axs, [("a", "measured ABUNDANCE of the top-10% selected by the model", "mean"),
                                   ("o", "measured OTHER property of the top-10% selected by the model", "mean"),
                                   ("o", "fraction of the selected that fall in the worst quartile of the OTHER property", "bad")]):
    for j, sg in enumerate(SEL):
        ax.bar(x + (j - 1) * 0.27, [r["fwd"][sg][k][stat][K10] for r in ab], 0.25, color=SIG_COL[sg], edgecolor=SURFACE, lw=1.5, label=SIG_LAB[sg])
    if k == "o":
        ax.scatter(x, [r["orc"]["o"][stat][K10] for r in ab], marker="D", s=34, color=INK, zorder=4, label="oracle: true top-10% on abundance")
    ax.axhline(0.5 if stat == "mean" else 0.25, color=MUTED, lw=1, ls=":")
    ax.set_title(ttl, fontsize=9.5); ax.grid(axis="x", visible=False)
h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0)); axs[2].set_xticks(x); axs[2].set_xticklabels([lab(r) + "\n" + r["o_axis"].lower() for r in ab], fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "F28_cross_property_top10_per_protein")

# ---- F29: inverse per protein: where does the model put the true-best? --------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14.5, 4.2), sharey=True)
for ax, (key, ttl) in zip(axs, [("orc", "true top-10% on ABUNDANCE: model percentile"), ("inv_orc", "true top-10% on the OTHER property: model percentile")]):
    for j, sg in enumerate(SEL):
        ax.bar(x + (j - 1) * 0.27, [r[key][f"pred{sg}"]["mean"][K10] for r in ab], 0.25, color=SIG_COL[sg], edgecolor=SURFACE, lw=1.5, label=SIG_LAB[sg])
    ax.axhline(0.5, color=MUTED, lw=1, ls=":"); ax.set_title(ttl, fontsize=9.5); ax.grid(axis="x", visible=False)
    ax.set_xticks(x); ax.set_xticklabels([r["protein"].split("_")[0] + "\n" + r["o_axis"].lower() for r in ab], fontsize=8)
axs[0].set_ylabel("mean predicted percentile (1 = model's best)"); axs[0].legend(fontsize=8)
save(fig, "F29_inverse_per_protein")

# ---- F30: replicate / condition controls vs property pairs ---------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
groups = [("replicate assays (same property)", rep, "#1baf7a"), ("conditions (same property)", cond, "#8a8982"),
          ("different properties, A->other", ab, "#2a78d6"), ("different properties, other->A", rev, "#eb6834")]
for name, recs, col in groups:
    if not recs: continue
    line(ax, KS, np.array([r["orc"]["o"]["mean"] for r in recs]) if name != "different properties, other->A" else np.array([r["orc"]["o"]["mean"] for r in recs]), col, f"{name} (n={len(recs)})")
ax.axhline(0.5, color=MUTED, lw=1, ls=":"); ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([f"{int(k*100)}%" for k in KS])
ax.set_xlabel("top fraction selected by the TRUE effect in assay 1"); ax.set_ylabel("mean measured percentile in assay 2")
ax.set_title("Reference: how much does the best in one assay carry over to another assay?", fontsize=10); ax.legend(fontsize=8)
save(fig, "F30_oracle_transfer_reference")

# ---- F31: exemplar scatter ------------------------------------------------------------------------------------------------------------------------
data = pickle.load(open(sys.argv[1], "rb"))
ex = [("RASK_HUMAN", "RASK_HUMAN_Weng_2022_abundance", "RASK_HUMAN_Weng_2022_binding-DARPin_K55"),
      ("KCNE1_HUMAN", "KCNE1_HUMAN_Muhammad_2023_expression", "KCNE1_HUMAN_Muhammad_2023_function"),
      ("PTEN_HUMAN", "PTEN_HUMAN_Matreyek_2021", "PTEN_HUMAN_Mighell_2018"),
      ("SPIKE_SARS2", "SPIKE_SARS2_Starr_2020_expression", "SPIKE_SARS2_Starr_2020_binding")]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
for ax, (p, a, o) in zip(axs, ex):
    sa, so = [d[~d.mutant.str.contains(":")] for d in (data[a], data[o])]
    m = sa.merge(so, on="mutant", suffixes=("_a", "_o"))
    pa, po = (rankdata(m["DMS_score_a"]) - 0.5) / len(m), (rankdata(m["DMS_score_o"]) - 0.5) / len(m)
    ax.hexbin(pa, po, gridsize=28, cmap="Greys", bins="log", mincnt=1, linewidths=0.2)
    for sg, mk, fc in [(0.0, "o", "none"), (1.0, "o", SIG_COL[1.0])]:
        top = np.argsort(-m[f"{SCOL[sg]}_a"].values)[: max(len(m) // 10, 1)]
        ax.scatter(pa[top], po[top], s=7, marker=mk, facecolor=fc, edgecolor=SIG_COL[sg] if sg else "#eb6834", lw=0.5, alpha=0.7,
                   label=f"top 10% at {SIG_LAB[sg]}")
    ax.set_title(f"{p.split('_')[0]}  (n={len(m)})", fontsize=10)
    ax.set_xlabel("abundance percentile (measured)"); ax.set_ylabel("other-property percentile (measured)")
axs[0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("Where the model's top-10% picks sit in the two measured properties (grey = all single mutants)", x=0.01, ha="left", fontsize=11, fontweight="bold")
save(fig, "F31_exemplar_selected_mutants")
