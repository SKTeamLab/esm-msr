"""Figures for docs/anova_epistasis_report.md. Usage: python make_figures.py <outdir with T.pkl/results.json> <figdir>"""
import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, 'analysis_notebooks/anova')
import anova_variance_partition as A

OUT, FIG = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#b9b8b3'
DIV = LinearSegmentedColormap.from_list('div', ['#184f95', '#6da7ec', '#f0efec', '#eb9a8a', '#a82c2c'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False})
T = pd.read_pickle(f'{OUT}/T.pkl'); R = json.load(open(f'{OUT}/results.json'))
try: FU = json.load(open(f'{OUT}/followup.json'))
except Exception: FU = None

# ---- Fig 1: global epistasis curve ----
iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(T.x, T.dG_AB)
xs = np.linspace(T.x.quantile(0.002), T.x.quantile(0.998), 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})
hb = ax[0].hexbin(T.x, T.dG_AB, gridsize=55, bins='log', cmap=SEQ, mincnt=1, linewidths=0)
ax[0].plot(xs, xs, color=GRAY, lw=1.5, ls='--'); ax[0].plot(xs, iso.predict(xs), color=ORANGE, lw=2.2)
ax[0].text(-5.7, 4.55, 'dashed line: no epistasis\n(measured = additive)', color=INK2, fontsize=9, ha='left', va='center')
ax[0].text(-5.7, 1.65, 'fitted curve\n(isotonic)', color=ORANGE, fontsize=9.5, fontweight='bold')
ax[0].set_xlabel('additive prediction of the double, dG_wt + ddG_A + ddG_B  (kcal/mol)')
ax[0].set_ylabel('measured dG of the double (kcal/mol)'); ax[0].set_xlim(-6, 6); ax[0].set_ylim(-1.3, 5.3)
ax[0].set_title('Measured vs additive: the assay saturates', loc='left', fontsize=11, color=INK)
cb = fig.colorbar(hb, ax=ax[0], pad=0.01, fraction=0.04); cb.set_label('doubles (log count)', color=INK2); cb.outline.set_visible(False)
bx = R['by_x']; labs = list(bx); mid = [(-6), -1.5, -0.5, 0.5, 1.5, 2.5, 3.6]
mean = [bx[k]['mean_dddG'] for k in labs]
ax[1].axhline(0, color=GRAY, lw=1)
ax[1].bar(range(len(labs)), mean, color=BLUE, width=0.62, edgecolor=SURF, linewidth=2)
for i, (k, m) in enumerate(zip(labs, mean)):
    ax[1].text(i, m + (0.07 if m >= 0 else -0.07), f'{m:+.2f}', ha='center', va='bottom' if m >= 0 else 'top', fontsize=9, color=INK)
ax[1].set_ylim(-0.3, 3.2); ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(['<-2', '-2..-1', '-1..0', '0..1', '1..2', '2..3', '>3'])
ax[1].set_xlabel('additive prediction of the double (kcal/mol)'); ax[1].set_ylabel('mean measured dddG (kcal/mol)')
ax[1].set_title('The "epistasis" is largest where the additive\nprediction falls below the assay floor', loc='left', fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_global_curve.png', dpi=170); plt.close(fig)

# ---- Fig 2: variance partition ----
cv = R['cv_increment']; g_, p_, rc_ = cv['M1 +global'], cv['M2 +pair mean'], cv['M3 +row/col']; rem = R['cv_resid_share']
nlo, nhi = R['noise_in_residual_share_opt'], R['noise_in_residual_share_pess']
fig, ax = plt.subplots(2, 1, figsize=(11, 4.4), gridspec_kw={'height_ratios': [1.5, 1]})
left = 0
segs = [('global saturation', g_, BLUE), ('position-pair mean', p_, ORANGE), ('substitution row + column effects', rc_, AQUA),
        ('left over', rem, GRAY)]
for name, v, c in segs:
    ax[0].barh(0, v * 100, left=left * 100, color=c, edgecolor=SURF, linewidth=2, height=0.55)
    ax[0].text((left + v / 2) * 100, 0, f'{v*100:.0f}%' if v > 0.05 else f'{v*100:.0f}%', ha='center', va='center',
               color='white' if c in (BLUE, ORANGE, AQUA) else INK, fontsize=11, fontweight='bold')
    below = name.startswith('substitution')
    ax[0].text((left + v / 2) * 100, -0.36 if below else 0.42, name, ha='center', va='top' if below else 'bottom', color=INK2, fontsize=9.5)
    left += v
ax[0].set_xlim(0, 100); ax[0].set_ylim(-0.75, 0.95); ax[0].set_yticks([]); ax[0].spines['left'].set_visible(False)
ax[0].set_xlabel('% of the variance in measured dddG explained out-of-sample (5-fold cross-validation)')
ax[0].set_title('Identity-independent structure explains about 90% of the variance in dddG', loc='left', fontsize=11, color=INK)
ax[1].barh(0, rem * 100, color=GRAY, edgecolor=SURF, linewidth=2, height=0.5)
ax[1].text(rem * 100 / 2, 0, f'left over {rem*100:.1f}%', ha='center', va='center', color=INK, fontsize=10, fontweight='bold')
ax[1].axvspan(nlo * 100, nhi * 100, ymin=0.08, ymax=0.92, color=MAGENTA, alpha=0.28, lw=0)
ax[1].plot([nlo * 100] * 2, [-0.3, 0.3], color=MAGENTA, lw=2); ax[1].plot([nhi * 100] * 2, [-0.3, 0.3], color=MAGENTA, lw=2)
ax[1].text((nlo + nhi) * 50, -0.42, f'expected from measurement noise alone: {nlo*100:.1f}% to {nhi*100:.1f}%', ha='center', va='top',
           color=INK, fontsize=9.5)
ax[1].set_xlim(0, 20); ax[1].set_ylim(-0.75, 0.45); ax[1].set_yticks([]); ax[1].spines['left'].set_visible(False)
ax[1].set_xlabel('zoom: % of variance (0-20%)')
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_variance_partition.png', dpi=170); plt.close(fig)

# ---- Fig 3: one pair matrix, stage by stage ----
sz = T.groupby('pair').size(); cands = sz[sz >= 340].index
var_by = T[T.pair.isin(cands)].groupby('pair').dddG.var()
pair = var_by.idxmax(); P = T[T.pair == pair].copy()
P['g'] = iso.predict(P.x) - P.x
P['r1'] = P.dddG - P.g
P['r2'] = P.r1 - P.r1.mean()
Rr, Cc = A.fit_rowcol(P.assign(pair='p', r2=P.r2), 'r2'); ra, cb = Rr['p'], Cc['p']
P['r3'] = P.r2 - P.a.map(ra) - P.b.map(cb)
rows = sorted(P.a.unique()); cols = sorted(P.b.unique())
def mat(col):
    M = np.full((len(rows), len(cols)), np.nan)
    for r in P.itertuples():
        M[rows.index(r.a), cols.index(r.b)] = getattr(r, col)
    return M
stages = [('dddG = measured double - additive', 'dddG'), ('minus the global saturation curve', 'r1'),
          ('minus the pair mean', 'r2'), ('minus row and column effects', 'r3')]
lim = float(np.nanpercentile(np.abs(mat('dddG')), 99))
fig, axs = plt.subplots(1, 4, figsize=(13, 3.9))
for a_, (t, c) in zip(axs, stages):
    M = mat(c); im = a_.imshow(M, cmap=DIV, vmin=-lim, vmax=lim, aspect='equal')
    a_.set_title(t, fontsize=9.5, loc='left', color=INK); a_.set_xticks(range(len(cols))); a_.set_xticklabels(cols, fontsize=6.5)
    a_.set_yticks(range(len(rows))); a_.set_yticklabels(rows, fontsize=6.5)
    a_.set_xlabel(f'residue at position {int(P.q.iloc[0])}', fontsize=8); a_.tick_params(length=0)
    for s in a_.spines.values(): s.set_visible(False)
    a_.text(0.5, -0.28, f'variance {np.nanvar(M):.2f}', transform=a_.transAxes, ha='center', fontsize=9, color=INK2)
axs[0].set_ylabel(f'residue at position {int(P.p.iloc[0])}', fontsize=8)
cbar = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01); cbar.set_label('kcal/mol', color=INK2); cbar.outline.set_visible(False)
fig.suptitle(f'One position-pair matrix ({P.code.iloc[0]}, positions {int(P.p.iloc[0])} and {int(P.q.iloc[0])}): what is left after each step', x=0.01, ha='left', fontsize=11)
fig.savefig(f'{FIG}/fig3_example_matrix.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- Fig 4: rank-correlation ladder ----
fig, ax = plt.subplots(figsize=(7.5, 3.2))
names = ['global saturation only', '+ position-pair mean', '+ row & column effects']
vals = [R['cv'][k]['rho'] for k in ('M1 +global', 'M2 +pair mean', 'M3 +row/col')]
ax.barh(range(3), vals, color=[BLUE, ORANGE, AQUA], height=0.55, edgecolor=SURF, linewidth=2)
for i, v in enumerate(vals): ax.text(v + 0.01, i, f'{v:.2f}', va='center', color=INK, fontsize=10, fontweight='bold')
ax.set_yticks(range(3)); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
ax.set_xlabel('Spearman correlation with measured dddG (cross-validated)')
ax.set_title('dddG correlation reaches 0.94 with no interaction term', loc='left', fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_rho_ladder.png', dpi=170); plt.close(fig)
print('pair used for fig3:', pair, 'n cells', len(P))

# ---- Fig 5: what the flip metric rewards ----
SH = json.load(open(f'{OUT}/splithalf.json')); sv = np.mean([v['validation-style'][0] for v in SH.values()])
fl = R['flip']; gv = fl['validation-style (columns pooled over partner positions)']['global only: h(additive)']['rho']
gp = fl['pair-level (one matrix per position pair)']['global only: h(additive)']['rho']
fig, ax = plt.subplots(figsize=(10.5, 4.3))
cats = ['global saturation only', 'substitution + partner position\n(no partner residue)', 'random']
vv, pp = [gv, sv, 0.0], [gp, 0.0, 0.0]
y = np.arange(3); h = 0.34
ax.barh(y - h / 2, vv, height=h, color=BLUE, edgecolor=SURF, linewidth=2, label='validation-style (pooled over partner positions)')
ax.barh(y + h / 2, pp, height=h, color=ORANGE, edgecolor=SURF, linewidth=2, label='pair-level (one matrix per position pair)')
for i in range(3):
    ax.text(vv[i] + 0.004, i - h / 2, f'{vv[i]:.3f}', va='center', fontsize=9, color=INK)
    ax.text(pp[i] + 0.004, i + h / 2, f'{pp[i]:.3f}', va='center', fontsize=9, color=INK)
ax.axvline(0.21, color=INK2, lw=1.2, ls='--'); ax.text(0.213, 0.02, 'trained models\non validation:\n~0.20-0.21', fontsize=8.5, color=INK2, va='top', transform=ax.get_xaxis_transform())
ax.set_yticks(y); ax.set_yticklabels(cats); ax.invert_yaxis(); ax.set_xlim(0, 0.3); ax.set_ylim(2.6, -0.6)
ax.set_xlabel('flip score earned by the predictor')
ax.legend(frameon=False, loc='lower left', fontsize=8.5, bbox_to_anchor=(0.12, 0.04))
ax.set_title('Much of the validation flip score needs no partner residue', loc='left', fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_flip_metric.png', dpi=170); plt.close(fig)
