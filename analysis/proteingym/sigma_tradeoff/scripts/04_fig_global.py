"""Global-scale figures: all assays grouped by ProteinGym coarse selection type."""
import pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from viz import *

res = pd.read_csv(f"{TAB}/per_assay_rho.csv", index_col=0)
cl = pd.read_csv(f"{TAB}/assay_classes.csv", index_col=0)
rc = [f"rho_{s}" for s in SIGMAS]
rng = np.random.default_rng(0)


def boot_mean(M, nb=2000):
    """bootstrap over assays (rows of M) -> (mean, lo, hi) per column."""
    idx = rng.integers(0, len(M), (nb, len(M)))
    b = M[idx].mean(1)
    return M.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


# ---- F01: mean rho vs sigma by category (+ delta) ------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for c in CATS:
    M = res.loc[res.category == c, rc].values
    m, lo, hi = boot_mean(M)
    for k, (a, off) in enumerate(zip(ax, (0, 1))):
        y, l, h = (m, lo, hi) if k == 0 else (boot_mean(M - M[:, [0]]))
        a.fill_between(SIGMAS, l, h, color=CAT_COLOR[c], alpha=0.15, lw=0)
        a.plot(SIGMAS, y, "o-", color=CAT_COLOR[c], label=f"{CAT_LABEL[c]} (n={len(M)})", mec=SURFACE, mew=1.5)
Mall = res[rc].values
ax[0].plot(SIGMAS, Mall.mean(0), "k--", lw=1.5, label=f"all assays (n={len(Mall)})")
ax[1].plot(SIGMAS, (Mall - Mall[:, [0]]).mean(0), "k--", lw=1.5)
ax[1].axhline(0, color=MUTED, lw=1)
ax[0].set_title("Mean Spearman rho vs sigma"); ax[1].set_title("Change in mean rho relative to sigma = 0")
ax[0].set_ylabel("mean Spearman rho (per-assay)"); ax[1].set_ylabel("delta rho vs sigma 0")
for a in ax: sigma_axis(a)
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Stability-tuned adapters help stability and expression most; the other categories gain, then flatten or fall",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
save(fig, "F01_category_mean_rho_vs_sigma")

# ---- F02: per-assay delta distributions ------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for a, (lab, f) in zip(axs, [("sigma 0.5 minus sigma 0", lambda r: r["rho_0.5"] - r["rho_0.0"]),
                              ("sigma 1.0 minus sigma 0", lambda r: r["rho_1.0"] - r["rho_0.0"]),
                              ("best sigma minus sigma 0 (oracle)", lambda r: r[rc].max(axis=1) - r["rho_0.0"])]):
    d = f(res)
    labs = []
    for i, c in enumerate(CATS):
        v = d[res.category == c].values
        jit = rng.uniform(-0.18, 0.18, len(v))
        a.scatter(i + jit, v, s=14, color=CAT_COLOR[c], alpha=0.6, lw=0)
        a.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
        labs.append(f"{c.replace('OrganismalFitness', 'Fitness')}\n{np.mean(v >= 0.02) * 100:.0f}% >= +0.02")
    a.axhline(0, color=MUTED, lw=1)
    a.set_xticks(range(len(CATS))); a.set_xticklabels(labs, fontsize=8)
    a.set_title(lab); a.grid(axis="x", visible=False)
axs[0].set_ylabel("per-assay delta rho (black bar = median)")
fig.suptitle("Per-assay change in rho: Stability and Expression move together; Activity/Binding/Fitness are spread around zero",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
save(fig, "F02_delta_distribution_by_category")

# ---- F03: curve-class mix by category ---------------------------------------------------------
ct = pd.crosstab(cl["category"], cl["cls"]).reindex(index=CATS, columns=CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(9.5, 3.8))
left = np.zeros(len(CATS))
frac = ct.div(ct.sum(1), axis=0)
for c in CLASSES:
    ax.barh(range(len(CATS)), frac[c], left=left, color=CLASS_COLOR[c], edgecolor=SURFACE, lw=2, label=CLASS_LABEL[c])
    for i, (v, n) in enumerate(zip(frac[c], ct[c])):
        if v > 0.06:
            ax.text(left[i] + v / 2, i, str(n), ha="center", va="center", fontsize=8.5,
                    color="white" if c in ("RISE", "FALL") else INK)
    left += frac[c].values
ax.set_yticks(range(len(CATS))); ax.set_yticklabels([f"{CAT_LABEL[c]}\n(n={ct.loc[c].sum()})" for c in CATS], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False)
ax.set_xlabel("fraction of assays (counts inside bars)")
ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18))
ax.set_title("Shape of each assay's rho(sigma) curve (|delta| >= 0.02 and z >= 2)")
save(fig, "F03_curve_class_mix")

# ---- F04: heatmap category x sigma (mean delta, fraction improved) -------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
D = res[rc].values - res[["rho_0.0"]].values
rows_m, rows_f = [], []
for c in CATS:
    m = (res.category == c).values
    rows_m.append(D[m].mean(0)); rows_f.append((D[m] >= 0.02).mean(0))
for a, M, ttl, fmt, vm in [(axs[0], np.array(rows_m), "mean delta rho vs sigma 0", "{:+.3f}", 0.2),
                           (axs[1], np.array(rows_f), "fraction of assays improved by >= 0.02", "{:.0%}", 1.0)]:
    im = a.imshow(M, cmap=DIV if vm == 0.2 else matplotlib.cm.Blues, vmin=-vm if vm == 0.2 else 0, vmax=vm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5,
                   color="white" if (abs(M[i, j]) > 0.65 * vm) else INK)
    a.set_xticks(range(5)); a.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); a.set_xlabel("sigma")
    a.set_yticks(range(len(CATS))); a.set_yticklabels([c.replace("OrganismalFitness", "Fitness") for c in CATS])
    a.grid(False); a.set_title(ttl)
save(fig, "F04_category_sigma_heatmaps")

# ---- F05: best sigma distribution -----------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(14, 2.9), sharey=True)
for a, c in zip(axs, CATS):
    s = cl.loc[cl.category == c, "sigma_star"]
    cnt = s.value_counts().reindex(SIGMAS, fill_value=0)
    a.bar(range(5), cnt.values, color=CAT_COLOR[c], edgecolor=SURFACE, lw=2)
    a.set_xticks(range(5)); a.set_xticklabels(["0", ".25", ".5", ".75", "1"]); a.set_title(CAT_LABEL[c].split(" (")[0], fontsize=9.5)
    a.grid(axis="x", visible=False); a.set_xlabel("sigma*")
axs[0].set_ylabel("number of assays")
fig.suptitle("sigma* = smallest sigma within 0.01 of the assay's best rho", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
save(fig, "F05_sigma_star_hist")

# ---- F06: between-site vs within-site rho ------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=False)
for k, (pre, ttl) in enumerate([("between", "Between-site rho (position means; which sites matter)"),
                                ("within", "Within-site rho (which substitution at a site)")]):
    cols = [f"{pre}_{s}" for s in SIGMAS]
    for c in CATS:
        M = res.loc[res.category == c, cols].dropna().values
        m, lo, hi = boot_mean(M - M[:, [0]])
        axs[k].fill_between(SIGMAS, lo, hi, color=CAT_COLOR[c], alpha=0.15, lw=0)
        axs[k].plot(SIGMAS, m, "o-", color=CAT_COLOR[c], label=CAT_LABEL[c].split(" (")[0], mec=SURFACE, mew=1.5)
    axs[k].axhline(0, color=MUTED, lw=1); sigma_axis(axs[k]); axs[k].set_title(ttl, fontsize=10)
axs[0].set_ylabel("mean change vs sigma 0 (singles only)")
axs[0].legend(fontsize=8)
save(fig, "F06_between_vs_within_site")

# ---- F07: headroom - gain vs starting rho ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5))
for c in CATS:
    m = res.category == c
    ax.scatter(res.loc[m, "rho_0.0"], res.loc[m, "rho_1.0"] - res.loc[m, "rho_0.0"], s=22, color=CAT_COLOR[c],
               alpha=0.75, lw=0.5, edgecolor=SURFACE, label=CAT_LABEL[c].split(" (")[0])
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("rho at sigma = 0 (base ESM3 likelihood)"); ax.set_ylabel("rho(1.0) - rho(0)")
ax.set_title("Where the stability adapters help: gain vs the starting point"); ax.legend(fontsize=8)
save(fig, "F07_gain_vs_baseline_rho")

# ---- F08: per-category spaghetti ------------------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
for a, c in zip(axs, CATS):
    sub = res[res.category == c]
    for _, r in sub.iterrows():
        a.plot(SIGMAS, np.array([r[x] for x in rc]) - r["rho_0.0"], color=CAT_COLOR[c], alpha=0.25, lw=1)
    a.plot(SIGMAS, (sub[rc].values - sub[["rho_0.0"]].values).mean(0), color=INK, lw=2.4)
    a.axhline(0, color=MUTED, lw=1); sigma_axis(a, label=False); a.set_title(f"{CAT_LABEL[c].split(' (')[0]} (n={len(sub)})", fontsize=9.5)
    a.set_ylim(-0.45, 0.45)
axs[0].set_ylabel("delta rho vs sigma 0 (black = mean)")
fig.supxlabel("sigma", fontsize=9.5)
save(fig, "F08_all_assays_delta_curves")

# ---- F09: how far do predictions move? (needs prediction pickle) -------------------------------------------
import sys
if len(sys.argv) > 1:
    data = pickle.load(open(sys.argv[1], "rb"))
    rows = []
    for d, df in data.items():
        row = dict(DMS_id=d)
        for s in SIGMAS[1:]:
            row[f"rs_{s}"] = spearmanr(df["s0.0"], df[SCOL[s]])[0]
        rows.append(row)
    mv = pd.DataFrame(rows).set_index("DMS_id").join(res[["category", "is_tsuboyama"]])
    mv.to_csv(f"{TAB}/prediction_shift_vs_sigma0.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for c in CATS:
        M = mv.loc[mv.category == c, [f"rs_{s}" for s in SIGMAS[1:]]].values
        m, lo, hi = boot_mean(M)
        ax.fill_between(SIGMAS[1:], lo, hi, color=CAT_COLOR[c], alpha=0.15, lw=0)
        ax.plot(SIGMAS[1:], m, "o-", color=CAT_COLOR[c], label=CAT_LABEL[c].split(" (")[0], mec=SURFACE, mew=1.5)
    ax.set_xticks(SIGMAS[1:]); ax.set_xlabel("sigma"); ax.set_ylabel("Spearman(pred at sigma, pred at 0)")
    ax.set_title("How far the ranking moves away from the base model"); ax.legend(fontsize=8)
    save(fig, "F09_prediction_shift_from_sigma0")
