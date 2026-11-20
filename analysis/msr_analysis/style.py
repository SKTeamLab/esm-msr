"""Canonical publication styling and high-level plotting primitives across all analysis figures.

Enforces one visual standard across the entire codebase:
- White ``#ffffff`` canvas, ``savefig.dpi = 220``, PDF Type 42 fonts, top/right spines off.
- Immutable style lookups via ``style_of(key_or_name, overrides=None)`` (never mutating ``registry.MODELS``).
- Shared primitives: ``dot_panel``, ``bar_panel``, ``forest_overview``, ``raincloud`` (with ``[lo, hi]`` CI whiskers),
  ``density_scatter``, ``density_scatter_plain``, ``multi_seed_kde_grid``, and ``finish_figure``.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .models import ABLATION_STYLES, BY_NAME, CATEGORY_COLOR, GROUP_TITLE, StyleSpec  # noqa: E402,F401
from .registry import LABEL, MODELS  # noqa: E402

BAND = '#f0f0f0'
GRID = '#e3e3e3'
HOLLOW_MEW = 2.0
HOLLOW_SIZE = 1.2
RING = '#c0392b'
INK_NOTE = '#444444'


def setup() -> None:
    """Apply canonical Matplotlib rcParams (`#ffffff`, `dpi=220`, Type 42 fonts)."""
    plt.rcParams.update({
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
        'savefig.facecolor': '#ffffff',
        'figure.dpi': 110,
        'savefig.dpi': 220,
        'savefig.bbox': 'tight',
        'font.size': 9.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'legend.frameon': False,
        'axes.grid': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'lines.linewidth': 1.2,
    })


setup()


def darken(hex_color: str, l_scale: float = 0.56, l_bounds: tuple[float, float] = (0.22, 0.42), s_min: float = 0.72) -> str:
    """Return a darker, richly saturated shade of ``hex_color``."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l_new = max(l_bounds[0], min(l_bounds[1], l * l_scale))
    s_new = min(1.0, max(s, s_min))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l_new, s_new))


def vivid(hex_color: str, s_min: float = 0.86, v_max: float = 0.86) -> str:
    """Raise saturation to at least ``s_min`` and cap value at ``v_max``."""
    h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(hex_color))
    return mcolors.to_hex(colorsys.hsv_to_rgb(h, max(s, s_min), min(v, v_max)))


def save(fig: plt.Figure, path: str | Path, close: bool = True) -> None:
    """Save ``<path>.png`` and ``<path>.pdf`` side by side."""
    stem = str(path)
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem + '.png')
    fig.savefig(stem + '.pdf')
    if close:
        plt.close(fig)


def style_of(key: str, overrides: dict | None = None) -> StyleSpec:
    """Look up immutable :class:`StyleSpec` of ``key`` (raw key, roster name, or ``FINAL_*`` family) with optional overrides."""
    if key in MODELS:
        spec = MODELS[key]
    elif key in ABLATION_STYLES:
        spec = ABLATION_STYLES[key]
    else:
        raise KeyError(
            f'{key!r} has no style: add it to msr_analysis.models.ROSTER / ABLATION_STYLES '
            '(and msr_analysis.registry) so every figure shares one colour, marker, and order'
        )
    if overrides:
        return spec.with_overrides(**overrides)
    return spec


def marker_kwargs(
    key: str,
    size: float = 6.5,
    alpha: float = 1.0,
    hollow: bool | None = None,
    overrides: dict | None = None,
) -> dict:
    """Build Matplotlib ``plot`` marker kwargs for ``key`` (hollow for singles version of two-version models)."""
    m = style_of(key, overrides=overrides)
    is_hollow = (not m.fill) if hollow is None else bool(hollow)
    kw = dict(marker=m.marker, ms=size, color=m.color, mec=m.color, alpha=alpha, ls='none', mew=1.2, mfc=m.color)
    if is_hollow:
        kw.update(mfc='none', mew=HOLLOW_MEW, ms=size * HOLLOW_SIZE)
    return kw


def version_handles(size: float = 6.8) -> list[Line2D]:
    grey = '#555555'
    return [
        Line2D([0], [0], marker='o', ms=size, mfc=grey, mec=grey, ls='none', label='filled: doubles version'),
        Line2D([0], [0], marker='o', ms=size * HOLLOW_SIZE, mfc='none', mec=grey, mew=HOLLOW_MEW, ls='none', label='hollow: singles version'),
    ]


def hollow_cells(keys_used: dict, names=None) -> set[tuple]:
    return {
        (g, m) for (g, m), k in keys_used.items()
        if m in BY_NAME and BY_NAME[m].two_versions and k == BY_NAME[m].single and (names is None or m in names)
    }


def band_axis(ax: plt.Axes, n: int, width: float = 1.0) -> None:
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(i - width / 2, i + width / 2, color=BAND, zorder=0, lw=0)
    ax.set_xlim(-width / 2, n - width / 2)


def dot_panel(
    ax: plt.Axes,
    groups,
    models,
    table: dict,
    ci: dict | None = None,
    spread: float = 0.72,
    highlight=None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    size: float = 6.5,
    xlabels=None,
    rot: float = 30,
    hline: float | None = None,
    hollow=(),
    style_overrides: dict[str, dict] | None = None,
) -> None:
    """Clustered dot-and-whisker panel with alternating grey bands."""
    n = len(groups)
    band_axis(ax, n)
    nm = len(models)
    offs = np.linspace(-spread / 2, spread / 2, nm) if nm > 1 else np.array([0.0])
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    if hline is not None:
        ax.axhline(hline, color='#777777', lw=0.7, ls=':', zorder=1)
    for j, m in enumerate(models):
        hl = highlight is not None and m in highlight
        z = 5 if hl else 3
        ov = (style_overrides or {}).get(m)
        for want_hollow in (False, True):
            xs, ys, lo, hi = [], [], [], []
            for i, g in enumerate(groups):
                v = table.get((g, m), np.nan)
                if not np.isfinite(v) or (((g, m) in hollow) != want_hollow):
                    continue
                xs.append(i + offs[j])
                ys.append(v)
                c = (ci or {}).get((g, m), (np.nan, np.nan))
                lo.append(v - c[0])
                hi.append(c[1] - v)
            if not xs:
                continue
            kw = marker_kwargs(
                m,
                size=size * (1.35 if hl else 1.0),
                hollow=True if want_hollow else (False if hollow else None),
                overrides=ov,
            )
            if ci is not None:
                ax.errorbar(
                    xs, ys, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)],
                    fmt='none', ecolor=kw['color'], elinewidth=1.0, capsize=1.6, capthick=0.9, alpha=0.9, zorder=z - 1,
                )
            ax.plot(xs, ys, zorder=z, **kw)
    ax.set_xticks(range(n))
    ax.set_xticklabels(xlabels or [LABEL.get(g, g) for g in groups], rotation=rot, ha='right' if rot else 'center')
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def model_legend(ax: plt.Axes, models, loc: str = 'best', ncol: int = 1, title: str | None = None,
                 style_overrides: dict[str, dict] | None = None, **kw):
    hs = []
    for m in models:
        ov = (style_overrides or {}).get(m)
        spec = style_of(m, overrides=ov)
        k = marker_kwargs(m, size=7, overrides=ov)
        hs.append(Line2D([0], [0], label=spec.label, **k))
    return ax.legend(handles=hs, loc=loc, ncol=ncol, title=title, **kw)


# ----------------------------------------------------------------------------------------------
# Shared bar_panel, forest_overview, and raincloud (with [lo, hi] CI whiskers)
# ----------------------------------------------------------------------------------------------
def layout_grouped_columns(cols: list[dict], gap: float = 0.7) -> tuple[list[float], dict[str, tuple[float, float]]]:
    """Compute x positions and category span ``(x0, x1)`` bounds for grouped bar/raincloud columns."""
    xs, spans, x, prev = [], {}, 0.0, None
    for c in cols:
        if prev is not None and c['cat'] != prev:
            x += gap
        xs.append(x)
        lo, hi = spans.get(c['cat'], (x, x))
        spans[c['cat']] = (min(lo, x), max(hi, x))
        prev = c['cat']
        x += 1.0
    return xs, spans


def bar_panel(
    ax: plt.Axes,
    cols: list[dict],
    df,
    subsets: tuple[str, ...],
    sub_labels: tuple[str, ...],
    ylabel: str,
    ylim: tuple[float, float],
    overall_df=None,
    overall_label: str = 'overall',
    tick: str = 'label',
    headers: bool = True,
    annotate: bool = True,
    zero_line: bool = False,
    tick_fs: float = 8.0,
    cats: dict[str, str] | None = None,
    brackets: list[tuple[float, float, float, str]] | None = None,
) -> tuple[list[float], dict[str, tuple[float, float]]]:
    """Draw 1-, 2-, or 3-bar grouped comparison panel with category span headers, overall dashes, and brackets."""
    cats = cats or GROUP_TITLE
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = layout_grouped_columns(cols, gap=0.7)
    n_sub = len(subsets)
    if n_sub == 1:
        offsets = (0.0,)
        bar_w = 0.52
    elif n_sub == 2:
        offsets = (-0.27, 0.05) if overall_df is not None else (-0.16, 0.16)
        bar_w = 0.30
    else:
        offsets = tuple(np.linspace(-0.28, 0.28, n_sub))
        bar_w = 0.56 / n_sub
    dash, x_err = (-0.43, 0.46), 0.38
    top = 0.012 * (ylim[1] - ylim[0])

    for c, x in zip(cols, xs):
        have = [s for s in subsets if len(df[(df['column'] == c['name']) & (df['subset'] == s)])]
        for j, sub in enumerate(subsets):
            r = df[(df['column'] == c['name']) & (df['subset'] == sub)]
            if not len(r):
                continue
            r = r.iloc[0]
            xx = x + offsets[j] if len(have) > 1 else x
            hatched = (n_sub >= 2 and j == 0)
            ax.bar(
                xx, r['value'], bar_w, color=c['color'],
                alpha=0.55 if hatched else 0.95,
                edgecolor=c['color'] if hatched else 'black',
                hatch='////' if hatched else None, lw=0.8, zorder=3,
            )
            if 'lo' in r and 'hi' in r and np.isfinite(r['lo']) and np.isfinite(r['hi']):
                ax.errorbar(
                    xx, r['value'], yerr=[[max(r['value'] - r['lo'], 0)], [max(r['hi'] - r['value'], 0)]],
                    fmt='none', ecolor=RING, elinewidth=1.3, capsize=2.5, zorder=5,
                )
            if annotate:
                hi_y = r['hi'] if ('hi' in r and np.isfinite(r['hi'])) else r['value']
                ax.text(
                    xx, max(hi_y, r['value'], 0.0) + top, f"{r['value']:.2f}",
                    ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8,
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6),
                )
        if overall_df is not None:
            o = overall_df[overall_df['column'] == c['name']]
            if len(o):
                o = o.iloc[0]
                ax.plot([x + dash[0], x + dash[1]], [o['value']] * 2, color='black', lw=1.8, zorder=2.2, solid_capstyle='butt')
                ax.errorbar(
                    x + x_err, o['value'],
                    yerr=[[max(o['value'] - o['lo'], 0)], [max(o['hi'] - o['value'], 0)]],
                    fmt='none', ecolor='black', elinewidth=1.3, capsize=2.5, zorder=5,
                )
                if annotate:
                    ax.text(
                        x + x_err, max(o['hi'], o['value'], 0.0) + top, f"{o['value']:.2f}",
                        ha='center', va='bottom', fontsize=6.8, rotation=90, zorder=8, fontweight='bold',
                        bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6),
                    )

    if brackets:
        bh = 0.018 * (ylim[1] - ylim[0])
        for xa, xb, yb, text in brackets:
            ax.plot([xa, xa, xb, xb], [yb, yb + bh, yb + bh, yb], color='#333333', lw=0.9, zorder=6)
            ax.text((xa + xb) / 2, yb + bh * 1.15, text, ha='center', va='bottom', fontsize=7.0, color='#333333', zorder=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [c.get('label_' + tick[-1], c['label']) if tick in ('label_s', 'label_d') else c['label'] for c in cols],
        fontsize=tick_fs, rotation=30, ha='right', rotation_mode='anchor',
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    if zero_line:
        ax.axhline(0, color='black', lw=0.8, zorder=2.5)
    if xs:
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.75)
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2:
            ax.axvspan(x0 - 0.5 - (0.35 if i else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers and cat in cats:
            tr = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, cats[cat], transform=tr, ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#333333')
    if n_sub >= 2:
        handles = [
            Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=sub_labels[0]),
            Patch(fc='#999999', ec='black', label=sub_labels[1]),
        ]
        if overall_df is not None:
            handles.append(Line2D([0], [0], color='black', lw=1.8, label=overall_label))
        ax.legend(handles=handles, ncol=len(handles), loc='upper right', fontsize=8.3, framealpha=0.92, borderpad=0.4, columnspacing=1.2, handlelength=1.6)
    return xs, spans


def forest_overview(
    axes,
    datasets: list[str],
    models: list[str],
    primary_ci: dict[tuple[str, str], tuple[float, float, float]],
    secondary_ci: dict[tuple[str, str], tuple[float, float, float]] | None = None,
    titles: dict[str, str] | None = None,
    hollow: set[tuple[str, str]] = frozenset(),
    leak_cells: set[tuple[str, str]] = frozenset(),
    shaded_models: set[str] = frozenset(),
    style_overrides: dict[str, dict] | None = None,
    xlabel: str = 'Spearman ρ',
) -> None:
    """Horizontal multi-panel forest overview (thick primary CI bar + thin secondary CI line + red leak ring)."""
    ax_arr = np.atleast_2d(axes)
    n_rows, n_cols = ax_arr.shape
    for idx, ds in enumerate(datasets):
        r_i, c_i = divmod(idx, n_cols)
        ax = ax_arr[r_i, c_i]
        vals = [primary_ci[(ds, m)] for m in models if (ds, m) in primary_ci]
        sec_vals = [secondary_ci[(ds, m)] for m in models if secondary_ci and (ds, m) in secondary_ci]
        if not vals:
            ax.set_visible(False)
            continue
        min_lo = min([lo for _v, lo, _hi in vals] + ([lo for _v, lo, _hi in sec_vals] if sec_vals else []))
        max_hi = max([hi for _v, _lo, hi in vals] + ([hi for _v, _lo, hi in sec_vals] if sec_vals else []))
        x_lo = float(np.floor((min_lo - 0.03) * 20) / 20)
        span = max(0.30, max_hi - x_lo)
        x_hi = max_hi + 0.24 * span

        for i, m in enumerate(models):
            if m in shaded_models:
                ax.axhspan(i - 0.5, i + 0.5, color='#bdbdbd', alpha=0.35, zorder=0)
            if (ds, m) not in primary_ci:
                continue
            v, lo, hi = primary_ci[(ds, m)]
            sec = secondary_ci.get((ds, m)) if secondary_ci else None
            ov = (style_overrides or {}).get(m)
            kw = marker_kwargs(m, size=5.6, hollow=((ds, m) in hollow), overrides=ov)
            ax.plot([lo, hi], [i, i], color=kw['color'], lw=2.3, alpha=0.45, solid_capstyle='butt')
            if sec is not None:
                ax.plot([sec[1], sec[2]], [i, i], color=kw['color'], lw=0.8)
            ax.plot(v, i, **kw)
            if (ds, m) in leak_cells:
                ax.plot(v, i, marker='o', ms=9.5, mfc='none', mec=RING, mew=0.95, ls='none')
            r_edge = max(hi, sec[2] if sec is not None else hi)
            ax.text(r_edge + 0.022 * span, i, f'{v:.3f}', va='center', fontsize=6.9, color='#444444')

        ax.set_title((titles or {}).get(ds, LABEL.get(ds, ds)), loc='left', fontsize=9.3)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.set_xlim(x_lo, x_hi)
        if r_i == n_rows - 1:
            ax.set_xlabel(xlabel, fontsize=9.0)

    group_ink = {'msr': 'black', 'retrain': '#1b6f3a', 'public': '#8a5a00', 'zeroshot': '#555555'}
    for r_i in range(n_rows):
        ax_l = ax_arr[r_i, 0]
        if not ax_l.get_visible():
            continue
        ax_l.set_yticks(range(len(models)))
        specs = [style_of(m, overrides=(style_overrides or {}).get(m)) for m in models]
        ax_l.set_yticklabels([s.label for s in specs], fontsize=8.0)
        ax_l.set_ylim(len(models) - 0.4, -0.6)
        for tick_lbl, s in zip(ax_l.get_yticklabels(), specs):
            tick_lbl.set_color(group_ink.get(s.group, '#333333'))
            if s.group == 'msr':
                tick_lbl.set_fontweight('bold')


def raincloud(
    ax: plt.Axes,
    x0: float,
    V_seeds,
    color: str,
    mean: float,
    lo: float | None = None,
    hi: float | None = None,
    width: float = 0.82,
    rng=None,
    transform=None,
    annotate: bool = True,
    fs: float = 8.2,
) -> None:
    """Draw IQR box, jittered domain dots, darkened median line, dark mean line, AND ``[lo, hi]`` 95% CI whisker.

    Fixes the ``14_property_category_gain`` bug where ``(lo, hi)`` were passed and never drawn:
    - Thick dark horizontal bar (``#333333``) = **mean**, with vertical 95% CI whisker ``[lo, hi]`` when supplied.
    - Thin darker-hued horizontal bar (``darken(color)``) = **median**.
    """
    tr = (lambda v: np.asarray(v, dtype=float)) if transform is None else transform
    rng = np.random.default_rng(0) if rng is None else rng
    arr = np.asarray(V_seeds, dtype=float)
    V2 = arr[None, :] if arr.ndim == 1 else arr
    t_dom = tr(np.nanmean(V2, axis=0))
    box_w = width * 0.88
    q1, med, q3 = np.mean(np.nanquantile(tr(V2), [0.25, 0.5, 0.75], axis=1), axis=1)
    y_mean = float(tr(mean))

    ax.add_patch(plt.Rectangle((x0 - box_w / 2, q1), box_w, q3 - q1, fc=color, alpha=0.22, ec='none', zorder=1))
    ax.plot([x0 - box_w / 2, x0 + box_w / 2], [med, med], color=darken(color), lw=1.1, zorder=4, solid_capstyle='butt')
    jit = (rng.random(len(t_dom)) - 0.5) * (box_w * 0.70) + x0
    ax.plot(jit, t_dom, 'o', ms=2.2, color=color, alpha=0.8, mew=0, zorder=3)
    ax.plot([x0 - box_w / 2, x0 + box_w / 2], [y_mean, y_mean], color='#333333', lw=1.8, zorder=5)

    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        y_lo, y_hi = float(tr(lo)), float(tr(hi))
        ax.errorbar(
            x0, y_mean,
            yerr=[[max(y_mean - y_lo, 0.0)], [max(y_hi - y_mean, 0.0)]],
            fmt='none', ecolor='#222222', elinewidth=1.1, capsize=2.4, capthick=0.9, zorder=6,
        )
    if annotate and np.isfinite(mean):
        top_anchor = max(y_mean, float(tr(hi)) if (hi is not None and np.isfinite(hi)) else y_mean)
        ax.text(
            x0, top_anchor + 0.035,
            f'{mean:.2f}' if abs(mean) < 100 else f'{mean:.0f}',
            ha='center', va='bottom', fontsize=fs, fontweight='semibold', zorder=8,
            bbox=dict(fc='white', ec='none', alpha=0.68, pad=0.10),
        )


# ----------------------------------------------------------------------------------------------
# Density scatters & 1xN multi-seed KDE scatter grids
# ----------------------------------------------------------------------------------------------
def density_scatter(
    ax: plt.Axes,
    x,
    y,
    s: float = 5,
    cmap: str = 'viridis',
    max_points: int = 60000,
    line: bool = True,
    rng: int = 0,
    label_stats: bool = True,
    lim=None,
    rasterized: bool = True,
) -> int:
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
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
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
    if line:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=RING, lw=0.9, ls='--', zorder=4)
    if label_stats and n > 3:
        rho = spearmanr(x, y)[0]
        r_val = pearsonr(x, y)[0]
        ax.text(
            0.04, 0.96, f'ρ = {rho:.2f}\nr = {r_val:.2f}\nn = {n:,}',
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9),
        )
    return n


def density_scatter_plain(
    x,
    y,
    xlabel: str,
    ylabel: str,
    ref: str | None = 'xy',
    ref_label: str | None = None,
    highlight=None,
    highlight_label: str | None = None,
    highlight_color: str = '#1a9850',
    title: str | None = None,
    foot: str | None = None,
    cmap: str = 'plasma',
    s: float = 8,
    point_alpha: float = 0.3,
    figsize: tuple[float, float] = (6.6, 5.8),
    max_kde: int = 15000,
    rng: int = 0,
    clip_pct: float = 0.3,
    rmse: bool = False,
    label_fontsize=None,
    tick_fontsize=None,
    stats_fontsize: float = 9,
    legend_fontsize: float = 8,
    title_fontsize: float = 9.5,
    foot_fontsize: float = 9.2,
    cbar_fontsize=None,
    cbar_tick_fontsize: float = 7.5,
    legend_loc: str = 'lower right',
) -> plt.Figure:
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim == 2:
        n_seeds = x.shape[0]
        ok = np.isfinite(x).all(axis=0) & np.isfinite(y)
        hl_1d = np.zeros(len(y), bool) if highlight is None else np.asarray(highlight, bool)
        x_seeds, y_1d, hl_1d = x[:, ok], y[ok], hl_1d[ok]
        n_var = len(y_1d)
        n_hl_var = int(hl_1d.sum())
        rho_val = float(np.mean([spearmanr(x_seeds[si], y_1d)[0] for si in range(n_seeds)]))
        r_val = float(np.mean([pearsonr(x_seeds[si], y_1d)[0] for si in range(n_seeds)]))
        rmse_val = float(np.mean([np.sqrt(np.mean((x_seeds[si] - y_1d) ** 2)) for si in range(n_seeds)]))
        x = x_seeds.ravel()
        y = np.tile(y_1d, n_seeds)
        hl = np.tile(hl_1d, n_seeds)
        n_txt = f'{n_var:,} (×{n_seeds} seeds)' if n_seeds > 1 else f'{n_var:,}'
    elif y.ndim == 2:
        n_seeds = y.shape[0]
        ok = np.isfinite(x) & np.isfinite(y).all(axis=0)
        hl_1d = np.zeros(len(x), bool) if highlight is None else np.asarray(highlight, bool)
        x_1d, y_seeds, hl_1d = x[ok], y[:, ok], hl_1d[ok]
        n_var = len(x_1d)
        n_hl_var = int(hl_1d.sum())
        rho_val = float(np.mean([spearmanr(x_1d, y_seeds[si])[0] for si in range(n_seeds)]))
        r_val = float(np.mean([pearsonr(x_1d, y_seeds[si])[0] for si in range(n_seeds)]))
        rmse_val = float(np.mean([np.sqrt(np.mean((x_1d - y_seeds[si]) ** 2)) for si in range(n_seeds)]))
        x = np.tile(x_1d, n_seeds)
        y = y_seeds.ravel()
        hl = np.tile(hl_1d, n_seeds)
        n_txt = f'{n_var:,} (×{n_seeds} seeds)' if n_seeds > 1 else f'{n_var:,}'
    else:
        n_seeds = 1
        ok = np.isfinite(x) & np.isfinite(y)
        hl = np.zeros(len(x), bool) if highlight is None else np.asarray(highlight, bool)
        x, y, hl = x[ok], y[ok], hl[ok]
        n_var = len(x)
        n_hl_var = int(hl.sum())
        rho_val = float(spearmanr(x, y)[0])
        r_val = float(pearsonr(x, y)[0])
        rmse_val = float(np.sqrt(np.mean((x - y) ** 2)))
        n_txt = f'{n_var:,}'

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
    col = cm(norm(z))
    col[:, 3] = point_alpha / np.sqrt(n_seeds)
    keep = o[~hl[o]]
    ax.scatter(x[keep], y[keep], c=col[keep], s=s, edgecolors='none', rasterized=True)
    handles = []
    if hl.any():
        ax.scatter(x[hl], y[hl], c=highlight_color, s=s, alpha=0.55 / np.sqrt(n_seeds), edgecolors='none', rasterized=True, zorder=3)
        handles.append(Line2D([0], [0], marker='o', ls='', color=highlight_color, alpha=0.8, label=f'{highlight_label} (n = {n_hl_var:,})'))
    if ref == 'xy':
        ax.plot(lim, lim, color=RING, ls='--', lw=1.6, zorder=4)
        handles.append(Line2D([0], [0], color=RING, ls='--', lw=1.6, label=ref_label or 'x = y'))
    elif ref == 'neg':
        ax.plot(lim, [-lim[0], -lim[1]], color=RING, ls='--', lw=1.6, zorder=4)
        handles.append(Line2D([0], [0], color=RING, ls='--', lw=1.6, label=ref_label or 'y = −x'))
    ax.set_xlim(lim)
    ax.set_ylim(lim if ref == 'xy' else ax.get_ylim())
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if tick_fontsize is not None:
        ax.tick_params(axis='both', labelsize=tick_fontsize)
    txt = f'ρ = {rho_val:.2f}\nr = {r_val:.2f}\nn = {n_txt}' + (f'\nRMSE = {rmse_val:.2f}' if rmse else '')
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=stats_fontsize, zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94))
    if handles:
        ax.legend(handles=handles, loc=legend_loc, fontsize=legend_fontsize, framealpha=0.92)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label('density', fontsize=cbar_fontsize)
    cb.ax.tick_params(labelsize=cbar_tick_fontsize)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    if title:
        ax.set_title(title, fontsize=title_fontsize, loc='left')
    fig.tight_layout()
    if foot:
        fig.text(0.015, -0.01, foot, fontsize=foot_fontsize, va='top', ha='left', color=INK_NOTE)
    return fig


def multi_seed_kde_grid(
    x_or_y_1d: np.ndarray,
    model_arrays: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    mode: str = 'pred_on_x',
    ref: str | None = 'xy',
    zero_hline: bool = False,
    cmap: str = 'plasma',
    max_kde: int = 15000,
    clip_pct: float = 0.3,
    panel_size: tuple[float, float] = (6.0, 5.8),
    foot: str | None = None,
) -> plt.Figure:
    """Draw a 1×N shared-axis multi-seed KDE scatter comparison across ``model_arrays`` (`11c` / `13` pattern).

    - ``mode='pred_on_x'``: ``model_arrays[m]`` is on x-axis and ``x_or_y_1d`` (measured y) is on y-axis (``11c``).
    - ``mode='pred_on_y'``: ``x_or_y_1d`` (e.g. ``dG_add``) is on x-axis and ``model_arrays[m]`` is on y-axis (``13``).
    """
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    vec = np.asarray(x_or_y_1d, dtype=float)
    ok = np.isfinite(vec)
    for arr in model_arrays.values():
        ok &= np.isfinite(np.atleast_2d(arr)).all(axis=0)
    vec_ok = vec[ok]
    n_var = len(vec_ok)
    names = list(model_arrays.keys())
    all_mod = np.concatenate([np.atleast_2d(model_arrays[m])[:, ok].ravel() for m in names])

    if ref == 'xy':
        both = np.concatenate([vec_ok, all_mod])
        b_lo, b_hi = np.percentile(both, [clip_pct, 100 - clip_pct])
        pad = 0.05 * (b_hi - b_lo)
        xlim = ylim = (b_lo - pad, b_hi + pad)
    else:
        xs_pool = all_mod if mode == 'pred_on_x' else vec_ok
        ys_pool = vec_ok if mode == 'pred_on_x' else all_mod
        x_lo, x_hi = np.percentile(xs_pool, [clip_pct, 100 - clip_pct])
        y_lo, y_hi = np.percentile(ys_pool, [clip_pct, 100 - clip_pct])
        px, py = 0.05 * (x_hi - x_lo), 0.05 * (y_hi - y_lo)
        xlim, ylim = (x_lo - px, x_hi + px), (y_lo - py, y_hi + py)

    cm = plt.get_cmap(cmap)
    fig, axes = plt.subplots(1, len(names), figsize=(panel_size[0] * len(names), panel_size[1]), sharex=True, sharey=True)
    ax_list = [axes] if len(names) == 1 else list(axes)
    for idx, (ax, m_name) in enumerate(zip(ax_list, names)):
        m_ok = np.atleast_2d(model_arrays[m_name])[:, ok]
        n_seeds = m_ok.shape[0]
        if mode == 'pred_on_x':
            xv, yv = m_ok.ravel(), np.tile(vec_ok, n_seeds)
            rho_v = float(np.mean([spearmanr(m_ok[si], vec_ok)[0] for si in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(m_ok[si], vec_ok)[0] for si in range(n_seeds)]))
            rmse_v = float(np.mean([np.sqrt(np.mean((m_ok[si] - vec_ok) ** 2)) for si in range(n_seeds)]))
        else:
            xv, yv = np.tile(vec_ok, n_seeds), m_ok.ravel()
            rho_v = float(np.mean([spearmanr(vec_ok, m_ok[si])[0] for si in range(n_seeds)]))
            r_v = float(np.mean([pearsonr(vec_ok, m_ok[si])[0] for si in range(n_seeds)]))
            rmse_v = np.nan

        sub = np.random.default_rng(0).choice(len(xv), min(len(xv), max_kde), replace=False)
        z = gaussian_kde(np.vstack([xv[sub], yv[sub]]))(np.vstack([xv, yv]))
        o = np.argsort(z)
        norm = Normalize(z.min(), z.max())
        col = cm(norm(z))
        col[:, 3] = 0.3 / np.sqrt(n_seeds)
        ax.scatter(xv[o], yv[o], c=col[o], s=8, edgecolors='none', rasterized=True)

        handles = []
        if ref == 'xy':
            ax.plot(xlim, xlim, color=RING, ls='--', lw=1.6, zorder=4)
            handles.append(Line2D([0], [0], color=RING, ls='--', lw=1.6, label='x = y'))
        elif ref == 'neg':
            ax.plot(xlim, [-xlim[0], -xlim[1]], color=RING, ls='--', lw=1.6, zorder=4)
            handles.append(Line2D([0], [0], color=RING, ls='--', lw=1.6, label='y = −x: dG_AB = 0'))
        if zero_hline:
            ax.axhline(0, color='#555555', lw=1.2, ls=':', zorder=4)
            handles.append(Line2D([0], [0], color='#555555', ls=':', lw=1.2, label='y = 0: no epistasis'))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=12.5)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=12.5)
        ax.tick_params(axis='both', labelsize=11.0)
        n_txt = f'{n_var:,} (×{n_seeds} seeds)' if n_seeds > 1 else f'{n_var:,}'
        stat_txt = f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {n_txt}' + (f'\nRMSE = {rmse_v:.2f}' if ref == 'xy' and np.isfinite(rmse_v) else '')
        ax.text(0.04, 0.96, stat_txt, transform=ax.transAxes, va='top', fontsize=10.5, zorder=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.94))
        if handles:
            ax.legend(handles=handles, loc='lower right' if ref == 'xy' else 'lower left', fontsize=9.5, frameon=False)
        ax.set_title(m_name, fontsize=12.5, fontweight='bold', pad=8)
    fig.tight_layout()
    if foot:
        fig.text(0.012, -0.01, foot, fontsize=9.2, va='top', ha='left', color=INK_NOTE)
    return fig


# ----------------------------------------------------------------------------------------------
# Pixel-accurate text wrapping, bounds checking, and figure finalizer
# ----------------------------------------------------------------------------------------------
def _text_width(fig: plt.Figure, text: str, **kw) -> float:
    r = fig.canvas.get_renderer()
    t = fig.text(0, 0, text, **kw)
    w = t.get_window_extent(r).width
    t.remove()
    return w


def wrap_to_px(fig: plt.Figure, text: str, max_px: float, **kw) -> str:
    out = []
    for para in str(text).split('\n'):
        stripped = para.lstrip(' ')
        indent = ' ' * (len(para) - len(stripped))
        line = ''
        for w in stripped.split(' '):
            cand = (line + ' ' + w) if line else w
            if line and _text_width(fig, indent + cand, **kw) > max_px:
                out.append(indent + line)
                line = w
            else:
                line = cand
        out.append(indent + line)
    return '\n'.join(out)


def text_outside(fig: plt.Figure, margin_px: float = 3) -> list[tuple[str, list[int]]]:
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    arts = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        arts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.get_legend() is not None:
            arts.append(ax.get_legend())
    bad = []
    for a in arts:
        if not a.get_visible() or (hasattr(a, 'get_text') and not a.get_text().strip()):
            continue
        bb = a.get_window_extent(r)
        if bb.x0 < margin_px or bb.y0 < margin_px or bb.x1 > W - margin_px or bb.y1 > H - margin_px:
            bad.append((a.get_text()[:50] if hasattr(a, 'get_text') else 'legend', [round(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]))
    return bad


def finish_figure(
    fig: plt.Figure,
    stem: str | Path | None = None,
    title: str | None = None,
    note: str | list[str] | None = None,
    ax_titles: list[tuple[plt.Axes, str]] | None = None,
    top: float = 0.98,
    pad: float = 0.02,
    note_fontsize: float = 7.6,
    close: bool = True,
) -> None:
    """Wrap footnotes/titles to pixel bounds, run iterative ``tight_layout``, check text bounds, and optionally ``save``."""
    r = fig.canvas.get_renderer()
    if title:
        fig.suptitle(title, x=0.012, ha='left', fontsize=10.5, fontweight='bold', y=0.995)
        top = min(top, 0.955)
    foot_frac = 0.0
    if note:
        raw_note = '\n'.join(note) if isinstance(note, (list, tuple)) else str(note)
        kw = dict(fontsize=note_fontsize, color='#333333')
        wrapped = wrap_to_px(fig, raw_note, 0.955 * fig.bbox.width, **kw)
        tt = fig.text(0.012, 0.01, wrapped, ha='left', va='bottom', **kw)
        foot_frac = tt.get_window_extent(r).height / fig.bbox.height + pad
    for _ in range(3):
        fig.tight_layout(rect=(0.0, foot_frac, 1.0, top))
        if ax_titles:
            for ax, raw in ax_titles:
                ax.set_title(wrap_to_px(fig, raw, 0.97 * ax.get_window_extent(r).width, fontsize=10, fontweight='bold'), loc='left', fontsize=10)
    bad = text_outside(fig)
    if bad and stem is not None:
        print('WARNING: text outside figure box in', stem, bad)
    if stem is not None:
        save(fig, stem, close=close)
