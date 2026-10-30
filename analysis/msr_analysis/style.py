"""One look for every figure: the notebooks' dot-and-whisker panels with alternating grey bands and per-model colours
and markers, density scatters with marginal histograms, but fewer elements per panel."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .registry import LABEL, MODELS  # noqa: E402

BAND = '#f0f0f0'
GRID = '#e3e3e3'


def setup():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 220, 'savefig.bbox': 'tight', 'font.size': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
        'axes.labelsize': 10.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5, 'legend.frameon': False,
        'axes.grid': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'lines.linewidth': 1.2,
    })


setup()


def save(fig, path, close=True):
    """png + pdf side by side; ``path`` has no suffix."""
    path = str(path)
    fig.savefig(path + '.png')
    fig.savefig(path + '.pdf')
    if close:
        plt.close(fig)


def style_of(key):
    m = MODELS.get(key, dict(color='#999999', marker='o', fill=True, label=key))
    return m


def marker_kwargs(key, size=6.5, alpha=1.0):
    m = style_of(key)
    kw = dict(marker=m['marker'], ms=size, color=m['color'], mec=m['color'], alpha=alpha, ls='none', mew=1.2)
    if not m['fill'] and m['marker'] not in ('+', 'x'):
        kw['mfc'] = 'white'
    return kw


def band_axis(ax, n, width=1.0):
    """alternating grey bands behind groups 0..n-1 on the x axis"""
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(i - width / 2, i + width / 2, color=BAND, zorder=0, lw=0)
    ax.set_xlim(-width / 2, n - width / 2)


def dot_panel(ax, groups, models, table, ci=None, spread=0.72, highlight=None, ylabel=None, ylim=None, size=6.5,
              xlabels=None, rot=30, hline=None):
    """Dot plot, one cluster per group, one dot per model.

    table[(group, model)] -> value; ci[(group, model)] -> (lo, hi).  ``highlight``: models drawn larger and on top.
    """
    n = len(groups)
    band_axis(ax, n)
    nm = len(models)
    offs = np.linspace(-spread / 2, spread / 2, nm) if nm > 1 else np.array([0.0])
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    if hline is not None:
        ax.axhline(hline, color='#777777', lw=0.7, ls=':', zorder=1)
    for j, m in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(groups):
            v = table.get((g, m), np.nan)
            if not np.isfinite(v):
                continue
            xs.append(i + offs[j]); ys.append(v)
            c = (ci or {}).get((g, m), (np.nan, np.nan))
            lo.append(v - c[0]); hi.append(c[1] - v)
        if not xs:
            continue
        hl = highlight is not None and m in highlight
        kw = marker_kwargs(m, size=size * (1.35 if hl else 1.0))
        z = 5 if hl else 3
        if ci is not None:
            ax.errorbar(xs, ys, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt='none', ecolor=kw['color'], elinewidth=1.0,
                        capsize=1.6, capthick=0.9, alpha=0.9, zorder=z - 1)
        ax.plot(xs, ys, zorder=z, **kw)
    ax.set_xticks(range(n))
    ax.set_xticklabels(xlabels or [LABEL.get(g, g) for g in groups], rotation=rot, ha='right' if rot else 'center')
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def model_legend(ax, models, loc='best', ncol=1, title=None, **kw):
    hs = []
    for m in models:
        k = marker_kwargs(m, size=7)
        hs.append(Line2D([0], [0], label=style_of(m)['label'], **k))
    return ax.legend(handles=hs, loc=loc, ncol=ncol, title=title, **kw)


def density_scatter(ax, x, y, s=5, cmap='viridis', max_points=60000, line=True, rng=0, label_stats=True, lim=None,
                    rasterized=True):
    """Scatter coloured by local density (2-D histogram lookup), with identity line and Spearman / Pearson."""
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n > max_points:
        r = np.random.default_rng(rng)
        i = r.choice(n, max_points, replace=False)
        xs, ys = x[i], y[i]
    else:
        xs, ys = x, y
    h, xe, ye = np.histogram2d(xs, ys, bins=70)
    ix = np.clip(np.digitize(xs, xe[1:-1]), 0, h.shape[0] - 1)
    iy = np.clip(np.digitize(ys, ye[1:-1]), 0, h.shape[1] - 1)
    d = h[ix, iy]
    o = np.argsort(d)
    ax.scatter(xs[o], ys[o], c=d[o], s=s, cmap=cmap, lw=0, alpha=0.8, rasterized=rasterized)
    if lim is not None:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    if line:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color='#c0392b', lw=0.9, ls='--', zorder=4)
    if label_stats and n > 3:
        rho = spearmanr(x, y)[0]
        r = pearsonr(x, y)[0]
        ax.text(0.04, 0.96, f'ρ = {rho:.2f}\nr = {r:.2f}\nn = {n:,}', transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9))
    return n


def density_scatter_marginals(x, y, xlabel, ylabel, xname=None, yname=None, cmap='plasma', s=8, point_alpha=0.3, figsize=(7.4, 6.8), max_kde=15000, rng=0):
    """The notebooks' density scatter (``analysis_utils_msr.density_scatter``): points coloured by their kernel density, the histograms of both variables on the margins
    (x blue, y orange, each over the other), the density colour bar on the left.  The only reference line is x = y, drawn over equal axis limits (the notebooks drew a dashed
    linear fit here).  Returns the figure; the title is the caller's."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    sub = np.random.default_rng(rng).choice(n, min(n, max_kde), replace=False)
    z = gaussian_kde(np.vstack([x[sub], y[sub]]))(np.vstack([x, y]))
    o = np.argsort(z)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    lim = (lo - pad, hi + pad)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.06, wspace=0.08, width_ratios=[0.16, 0.75, 4, 1], height_ratios=[1, 4])      # column 1 is the room of the y label and tick labels
    ax = fig.add_subplot(gs[1, 2]); axt = fig.add_subplot(gs[0, 2], sharex=ax); axr = fig.add_subplot(gs[1, 3], sharey=ax); axc = fig.add_subplot(gs[1, 0])
    norm = Normalize(z.min(), z.max())
    cm = plt.get_cmap(cmap)
    col = cm(norm(z)); col[:, 3] = point_alpha
    ax.scatter(x[o], y[o], c=col[o], s=s, edgecolors='none', rasterized=True)
    ax.plot(lim, lim, color='#c0392b', ls='--', lw=1.6, zorder=4, label='x = y')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.text(0.04, 0.96, f'ρ = {spearmanr(x, y)[0]:.2f}\nr = {pearsonr(x, y)[0]:.2f}\nn = {n:,}', transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))
    b = np.linspace(lim[0], lim[1], 60)
    axt.hist(x, bins=b, alpha=0.75, color='steelblue', density=True); axt.hist(y, bins=b, alpha=0.3, color='coral', density=True)
    axr.hist(y, bins=b, alpha=0.75, color='coral', density=True, orientation='horizontal'); axr.hist(x, bins=b, alpha=0.3, color='steelblue', density=True, orientation='horizontal')
    axt.set_ylabel('density', fontsize=8); axr.set_xlabel('density', fontsize=8)
    axt.tick_params(labelbottom=False, labelsize=7.5); axr.tick_params(labelleft=False, labelsize=7.5)
    axt.legend(handles=[Patch(color='steelblue', label=xname or xlabel), Patch(color='coral', label=yname or ylabel)], loc='upper right', fontsize=7.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    cb = fig.colorbar(sm, cax=axc)
    cb.ax.yaxis.set_ticks_position('left'); cb.ax.yaxis.set_label_position('left'); cb.set_label('density'); cb.ax.tick_params(labelsize=7.5)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    return fig
