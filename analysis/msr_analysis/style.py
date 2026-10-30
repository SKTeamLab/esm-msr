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


def density_scatter_plain(x, y, xlabel, ylabel, ref='xy', ref_label=None, highlight=None, highlight_label=None, highlight_color='#1a9850', title=None, cmap='plasma', s=8,
                          point_alpha=0.3, figsize=(6.6, 5.8), max_kde=15000, rng=0, clip_pct=0.3, rmse=False):
    """The notebooks' density scatter (``analysis_utils_msr.density_scatter``) without the marginal histograms: points coloured by their kernel density, a colour bar on the right,
    Spearman / Pearson / n in the corner.  ``ref``: 'xy' draws y = x over equal axis limits, 'neg' draws y = -x, None draws no line (the notebooks drew a dashed linear fit).
    ``highlight`` (boolean, same length as x) paints those points in a flat high-contrast colour on top of the density colours, with its own legend entry.  The axis limits are the
    ``clip_pct`` .. 100 - ``clip_pct`` percentiles of x and y pooled, so a few outliers do not squash the plot (they are cut off, not dropped from the statistics)."""
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    hl = np.zeros(len(x), bool) if highlight is None else np.asarray(highlight, bool)
    x, y, hl = x[ok], y[ok], hl[ok]
    n = len(x)
    sub = np.random.default_rng(rng).choice(n, min(n, max_kde), replace=False)
    z = gaussian_kde(np.vstack([x[sub], y[sub]]))(np.vstack([x, y]))
    o = np.argsort(z)
    both = np.concatenate([x, y])
    lo, hi = np.percentile(both, [clip_pct, 100 - clip_pct])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    fig, ax = plt.subplots(figsize=figsize)
    norm = Normalize(z.min(), z.max())
    cm = plt.get_cmap(cmap)
    col = cm(norm(z)); col[:, 3] = point_alpha
    keep = o[~hl[o]]
    ax.scatter(x[keep], y[keep], c=col[keep], s=s, edgecolors='none', rasterized=True)
    handles = []
    if hl.any():
        ax.scatter(x[hl], y[hl], c=highlight_color, s=s, alpha=0.55, edgecolors='none', rasterized=True, zorder=3)
        handles.append(plt.Line2D([0], [0], marker='o', ls='', color=highlight_color, alpha=0.8, label=f'{highlight_label} (n = {int(hl.sum()):,})'))
    if ref == 'xy':
        ax.plot(lim, lim, color='#c0392b', ls='--', lw=1.6, zorder=4)
        handles.append(plt.Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label=ref_label or 'x = y'))
    elif ref == 'neg':
        ax.plot(lim, [-lim[0], -lim[1]], color='#c0392b', ls='--', lw=1.6, zorder=4)
        handles.append(plt.Line2D([0], [0], color='#c0392b', ls='--', lw=1.6, label=ref_label or 'y = −x'))
    ax.set_xlim(lim); ax.set_ylim(lim if ref == 'xy' else ax.get_ylim())
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    txt = f'ρ = {spearmanr(x, y)[0]:.2f}\nr = {pearsonr(x, y)[0]:.2f}\nn = {n:,}' + (f'\nRMSE = {np.sqrt(np.mean((x - y) ** 2)):.2f}' if rmse else '')
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))
    if handles:
        ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.92)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label('density'); cb.ax.tick_params(labelsize=7.5)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    if title:
        ax.set_title(title, fontsize=9.5, loc='left')
    fig.tight_layout()
    return fig
