"""Figures for proteins with >1 assay: abundance-vs-other trade-offs, per-protein curve panels, heatmap.

Usage: 05_fig_multi.py data.pkl
"""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from viz import *

data = pickle.load(open(sys.argv[1], "rb"))
res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
cl = pd.read_csv(f"{TAB}/assay_classes.csv", index_col=0)
pr = pd.read_csv(f"{TAB}/abundance_vs_other_pairs.csv")
boots = pickle.load(open(f"{TAB}/boot_rho.pkl", "rb"))
rc = [f"rho_{s}" for s in SIGMAS]
rng = np.random.default_rng(1)
multi = res[res.group.notna()].copy()
multi["label"] = multi.index.str.replace(r"^[A-Z0-9]+_[A-Z0-9]+_", "", regex=True)
GROUP_TITLE = {"A": "different properties", "B": "same property, different conditions", "C": "replicates of one property"}


def curve_panel(ax, prot, show_ylabel=True):
    sub = multi[multi.protein == prot].sort_values("axis")
    ys = sub["rho_1.0"].values.astype(float)
    lo_all, hi_all = min(sub[rc].min()), max(sub[rc].max())
    gap = 0.07 * (hi_all - lo_all)
    ypos = {}
    for i in np.argsort(ys):                      # push end labels apart
        y = ys[i]
        for j in sorted(ypos.values()):
            if abs(y - j) < gap: y = j + gap
        ypos[i] = y
    for k, (d, r) in enumerate(sub.iterrows()):
        col = AXIS_COLOR[r.axis] if r.axis != "COND" else ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"][k % 4]
        b = boots[d]
        lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
        ax.fill_between(SIGMAS, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(SIGMAS, r[rc].values.astype(float), marker="o", color=col, mec=SURFACE, mew=1.2, ms=5.5,
                ls="-" if r.axis != "COND" else "--")
        ax.text(1.03, ypos[k], f"{r.axis.lower()}: {r.label[:22]}", fontsize=6.5, color=INK2, va="center")
    ax.set_title(prot.replace("_", " "), fontsize=9.5)
    sigma_axis(ax, label=False)
    ax.set_xlim(-0.05, 1.0 + 0.02)


# ---- F10: abundance delta vs other delta (scatter) ------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.9), sharex=True, sharey=True)
for a, s in zip(axs, [0.5, 0.75, 1.0]):
    dA = np.array([res.loc[x, f"rho_{s}"] - res.loc[x, "rho_0.0"] for x in pr.abund])
    dO = np.array([res.loc[x, f"rho_{s}"] - res.loc[x, "rho_0.0"] for x in pr.other])
    a.axhline(0, color=MUTED, lw=1); a.axvline(0, color=MUTED, lw=1)
    a.plot([-0.2, 0.2], [-0.2, 0.2], color=GRID, lw=1.2, ls="--")
    for ax_name in ["ACT", "BIND"]:
        m = (pr.other_axis == ax_name).values
        a.scatter(dA[m], dO[m], s=60, color=AXIS_COLOR[ax_name], edgecolor=SURFACE, lw=1.2, zorder=3,
                  label=f"other = {AXIS_NAME[ax_name]}")
    for x, y, p in zip(dA, dO, pr.protein):
        a.annotate(p.split("_")[0], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=INK2)
    a.set_title(f"sigma = {s:g} vs 0"); a.set_xlabel("delta rho, abundance / expression assay")
    a.text(0.97, 0.03, "abundance up,\nother down", transform=a.transAxes, ha="right", fontsize=7.5, color=MUTED)
    a.text(0.97, 0.97, "both improve", transform=a.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    a.text(0.03, 0.97, "abundance down,\nother up", transform=a.transAxes, va="top", fontsize=7.5, color=MUTED)
    a.text(0.03, 0.03, "both worse", transform=a.transAxes, fontsize=7.5, color=MUTED)
axs[0].set_ylabel("delta rho, the protein's other-property assay"); axs[0].legend(fontsize=8, loc="center left")
axs[0].set_xlim(-0.17, 0.17); axs[0].set_ylim(-0.13, 0.13)
fig.suptitle("Abundance vs the protein's other property: most proteins improve both; RASK (binding) and KCNE1 break the pattern",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
save(fig, "F10_abundance_vs_other_delta_scatter")

# ---- F11: heatmap of delta for all multi-assay assays ------------------------------------------------------
order = []
for g in ["A", "B", "C"]:
    for p in sorted(multi[multi.group == g].protein.unique()):
        sub = multi[multi.protein == p].copy()
        sub["ax_order"] = sub.axis.map({"ABUND": 0, "ACT": 1, "BIND": 2, "SPEC": 3, "FIT": 4, "COND": 5})
        order += list(sub.sort_values(["ax_order"]).index)
M = np.array([[multi.loc[d, f"rho_{s}"] - multi.loc[d, "rho_0.0"] for s in SIGMAS[1:]] for d in order])
fig, ax = plt.subplots(figsize=(7.6, 0.255 * len(order) + 1.6))
im = ax.imshow(M, cmap=DIV, vmin=-0.2, vmax=0.2, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(M[i, j]) > 0.13 else INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{multi.loc[d, 'protein'].split('_')[0]}  {multi.loc[d, 'label'][:30]}" for d in order], fontsize=7)
for lab, d in zip(ax.get_yticklabels(), order):
    lab.set_color(AXIS_COLOR[multi.loc[d, "axis"]] if multi.loc[d, "axis"] != "FIT" else INK2)
ax.set_xticks(range(4)); ax.set_xticklabels(["0.25", "0.5", "0.75", "1.0"]); ax.set_xlabel("sigma")
ax.xaxis.set_label_position("top"); ax.xaxis.tick_top(); ax.grid(False)
prev = None
for i, d in enumerate(order):
    g = multi.loc[d, "group"]
    if prev is not None and g != prev: ax.axhline(i - 0.5, color=INK, lw=1.5)
    prev = g
    if i and multi.loc[order[i - 1], "protein"] != multi.loc[d, "protein"]: ax.axhline(i - 0.5, color=SURFACE, lw=2)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("delta rho vs sigma 0")
ax.set_title("All 55 multi-assay measurements: change in rho relative to sigma 0\n(group A: different properties | B: conditions | C: replicates)",
             fontsize=9.5, pad=24)
save(fig, "F11_multi_assay_delta_heatmap")

# ---- F12-14: per-protein rho(sigma) small multiples ---------------------------------------------------------
for g, name, fn in [("A", "A", "F12_groupA_properties_rho_vs_sigma"), ("B", "B", "F13_groupB_conditions_rho_vs_sigma"),
                    ("C", "C", "F14_groupC_replicates_rho_vs_sigma")]:
    prots = sorted(multi[multi.group == g].protein.unique())
    nc = 4 if len(prots) > 6 else 3
    nr = int(np.ceil(len(prots) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.0 * nr), squeeze=False)
    for a, p in zip(axs.ravel(), prots):
        curve_panel(a, p)
    for a in axs.ravel()[len(prots):]: a.axis("off")
    for a in axs[:, 0]: a.set_ylabel("Spearman rho")
    for a in axs[-1]: a.set_xlabel("sigma")
    h = [matplotlib.lines.Line2D([], [], color=AXIS_COLOR[k], lw=2, marker="o", label=AXIS_NAME[k])
         for k in ["ABUND", "ACT", "BIND", "FIT", "SPEC"] if k in set(multi[multi.group == g].axis)]
    if h: fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Group {g}: {GROUP_TITLE[g]} - rho vs sigma per assay (band = 95% bootstrap over mutants)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, fn)

# ---- shared-mutant analysis for abundance vs other pairs ---------------------------------------------------------
rows, traj = [], {}
for _, r in pr.iterrows():
    dA, dO = data[r.abund], data[r.other]
    m = dA.merge(dO, on="mutant", suffixes=("_A", "_O"))
    m = m[~m.mutant.str.contains(":")] if (m.mutant.str.contains(":")).mean() > 0 and (~m.mutant.str.contains(":")).sum() > 50 else m
    n = len(m)
    rA = [spearmanr(m[f"{SCOL[s]}_A"], m["DMS_score_A"])[0] for s in SIGMAS]
    rO = [spearmanr(m[f"{SCOL[s]}_O"], m["DMS_score_O"])[0] for s in SIGMAS]
    lab = spearmanr(m["DMS_score_A"], m["DMS_score_O"])[0]
    # bootstrap delta-gap at sigma=1 on shared mutants
    yA, yO, pA, pO = m["DMS_score_A"].values, m["DMS_score_O"].values, m[[f"{SCOL[s]}_A" for s in SIGMAS]].values, m[[f"{SCOL[s]}_O" for s in SIGMAS]].values
    cap = min(n, 20000)
    gaps = []
    for _ in range(150):
        i = rng.integers(0, n, cap)
        ra, ro = rankdata(yA[i]), rankdata(yO[i])
        a0, a1 = np.corrcoef(rankdata(pA[i, 0]), ra)[0, 1], np.corrcoef(rankdata(pA[i, -1]), ra)[0, 1]
        o0, o1 = np.corrcoef(rankdata(pO[i, 0]), ro)[0, 1], np.corrcoef(rankdata(pO[i, -1]), ro)[0, 1]
        gaps.append((o1 - o0) - (a1 - a0))
    dAs, dOs = np.array(rA) - rA[0], np.array(rO) - rO[0]
    ok = [(SIGMAS[k]) for k in range(len(SIGMAS)) if dAs[k] >= -0.01 and dOs[k] >= -0.01]
    rows.append(dict(protein=r.protein, abund=r.abund, other=r.other, other_axis=r.other_axis, n_shared=n,
                     label_corr=lab, **{f"rhoA_{s}": v for s, v in zip(SIGMAS, rA)}, **{f"rhoO_{s}": v for s, v in zip(SIGMAS, rO)},
                     dA_1=dAs[-1], dO_1=dOs[-1], gap_1=dOs[-1] - dAs[-1], gap_1_se=np.std(gaps),
                     best_joint_sigma=SIGMAS[int(np.argmax(dAs + dOs))], max_sigma_no_loss=max(ok)))
    traj[r.protein] = (dAs, dOs, rA, rO)
sh = pd.DataFrame(rows)
sh.to_csv(f"{TAB}/abundance_vs_other_shared_mutants.csv", index=False)
print(sh[["protein", "n_shared", "label_corr", "dA_1", "dO_1", "gap_1", "gap_1_se", "best_joint_sigma", "max_sigma_no_loss"]].round(3).to_string())

# ---- F15: trajectories (delta on shared mutants) -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6.2))
ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
ax.plot([-0.25, 0.25], [-0.25, 0.25], color=GRID, lw=1.2, ls="--")
for _, r in sh.iterrows():
    dAs, dOs, _, _ = traj[r.protein]
    col = AXIS_COLOR[r.other_axis]
    ax.plot(dAs, dOs, "-", color=col, lw=1.6, alpha=0.8)
    ax.scatter(dAs[1:], dOs[1:], s=[14, 22, 32, 44], color=col, edgecolor=SURFACE, lw=0.8, zorder=3)
    ax.annotate(r.protein.split("_")[0], (dAs[-1], dOs[-1]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
ax.scatter([0], [0], color=INK, s=30, zorder=4)
ax.set_xlim(-0.17, 0.19); ax.set_ylim(-0.11, 0.11)
ax.set_xlabel("delta rho(abundance assay) vs sigma 0   [shared mutants]"); ax.set_ylabel("delta rho(other-property assay)")
ax.set_title("Trajectories as sigma goes 0 -> 0.25 -> 0.5 -> 0.75 -> 1 (marker size grows with sigma)")
h = [matplotlib.lines.Line2D([], [], color=AXIS_COLOR[k], lw=2, label=f"other = {AXIS_NAME[k]}") for k in ["ACT", "BIND"]]
ax.legend(handles=h, fontsize=8, loc="lower right")
save(fig, "F15_tradeoff_trajectories")

# ---- F16: why? measured label agreement vs relative gain -------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
for a, (ycol, ttl) in zip(axs, [("gap_1", "other gain minus abundance gain at sigma 1.0"), ("dO_1", "other-property gain at sigma 1.0")]):
    for _, r in sh.iterrows():
        a.errorbar(r.label_corr, r[ycol], yerr=r.gap_1_se if ycol == "gap_1" else None, fmt="o", color=AXIS_COLOR[r.other_axis],
                   ms=7, mec=SURFACE, mew=1, elinewidth=1, capsize=0, alpha=0.9)
        a.annotate(r.protein.split("_")[0], (r.label_corr, r[ycol]), xytext=(5, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    a.axhline(0, color=MUTED, lw=1); a.set_xlabel("Spearman between the two measured scores\n(shared mutants)")
    a.set_ylabel("delta rho, shared mutants"); a.set_title(ttl, fontsize=10)
rho, p = spearmanr(sh.label_corr, sh.gap_1)
axs[0].text(0.03, 0.04, f"Spearman across {len(sh)} proteins = {rho:+.2f} (p={p:.2f})", transform=axs[0].transAxes, fontsize=8, color=INK2)
save(fig, "F16_label_agreement_vs_relative_gain")

# ---- F17: between-site vs within-site change for abundance vs other assays --------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=False)
for a, pre, ttl in [(axs[0], "between", "between-site rho: change at sigma 1.0 vs 0"), (axs[1], "within", "within-site rho: change at sigma 1.0 vs 0")]:
    x = np.arange(len(pr))
    dA = [res.loc[d, f"{pre}_1.0"] - res.loc[d, f"{pre}_0.0"] for d in pr.abund]
    dO = [res.loc[d, f"{pre}_1.0"] - res.loc[d, f"{pre}_0.0"] for d in pr.other]
    a.bar(x - 0.2, dA, 0.4, color=AXIS_COLOR["ABUND"], edgecolor=SURFACE, lw=1.5, label="abundance assay")
    a.bar(x + 0.2, dO, 0.4, color=[AXIS_COLOR[o] for o in pr.other_axis], edgecolor=SURFACE, lw=1.5, label="other assay")
    a.set_xticks(x); a.set_xticklabels([p.split("_")[0] for p in pr.protein], rotation=45, ha="right", fontsize=8)
    a.axhline(0, color=MUTED, lw=1); a.set_title(ttl, fontsize=10); a.grid(axis="x", visible=False)
axs[0].set_ylabel("delta rho (singles only)")
h = [matplotlib.patches.Patch(color=AXIS_COLOR["ABUND"], label="abundance assay"), matplotlib.patches.Patch(color=AXIS_COLOR["ACT"], label="other: activity"),
     matplotlib.patches.Patch(color=AXIS_COLOR["BIND"], label="other: binding")]
axs[0].legend(handles=h, fontsize=8)
save(fig, "F17_between_within_abundance_vs_other")

# ---- F20: joint gain (mean of abundance and other delta, shared mutants) vs sigma ---------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for _, r in sh.iterrows():
    dAs, dOs, _, _ = traj[r.protein]
    j = (dAs + dOs) / 2
    ax.plot(SIGMAS, j, marker="o", color=AXIS_COLOR[r.other_axis], lw=1.5, ms=4, alpha=0.9, mec=SURFACE)
    k = int(np.argmax(j))
    ax.scatter([SIGMAS[k]], [j[k]], s=70, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
    ax.annotate(r.protein.split("_")[0], (SIGMAS[-1], j[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
ax.axhline(0, color=MUTED, lw=1); sigma_axis(ax); ax.set_xlim(-0.05, 1.18)
ax.set_ylabel("mean delta rho of the two properties\n(shared mutants)")
ax.set_title("Joint gain on both properties; ring = best sigma for that protein")
h = [matplotlib.lines.Line2D([], [], color=AXIS_COLOR[k], lw=2, label=f"other = {AXIS_NAME[k]}") for k in ["ACT", "BIND"]]
ax.legend(handles=h, fontsize=8, loc="upper left")
save(fig, "F20_joint_gain_vs_sigma")
