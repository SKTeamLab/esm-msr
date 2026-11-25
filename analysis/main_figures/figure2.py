#!/usr/bin/env python3
"""Build Main-Text Figure 2 (`analysis/results/00_main_text_figs/figure2.{png,pdf}`).

Layout:
  - Top row (a, b, c):
      a. Epistasis thermodynamic-cycle schematic (`_assets/epistasis_cycles_split.svg`).
      b. Label ΔG_additive vs δΔΔG density scatter with assay floor (`y = -x`, `dG_AB = 0`) and
         reporting cutoff (`y = -x - 1`, `dG_AB = -1`).
      c. Measured epistasis types as a function of label additive expectation ΔG
         (`alt_observed_additive_dG.csv`: `ΔG_wt + ΔΔG_A + ΔΔG_B`).
  - Middle row (d):
      d. Five-level epistasis ANOVA / residual decomposition schematic
         (`_assets/epistasis_levels_5panel_anova.svg`), aligned above Panel e.
  - Bottom row (e):
      e. Five-level epistasis hierarchy statistics in pooled form across all five levels
         (`naive_rho`, `beyond_rho`, `pair_rho`, `line_rho`, `cell_mag_pooled`) with `dG_add > 0` overlays.

Usage:
    PYTHONPATH=src:analysis python -m analysis.main_figures.figure2
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde, pearsonr, spearmanr

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis import models as M, style
from msr_analysis.paths import REPO_ROOT, results_dir
from pipelines._tsu_common import load_tsu_test_frame
from pipelines import p06_epistasis_hierarchy as p06
from pipelines import p07_epistasis_mechanisms as p07


def _render_svg_to_rgb(svg_path: Path, scale: float = 3.0) -> np.ndarray:
    """Render any SVG (supporting numeric width/height or viewBox with 100%) to a white-backed RGB array."""
    svg_text = svg_path.read_text(encoding='utf-8')
    root = ET.fromstring(svg_text)
    vb_attr = root.attrib.get('viewBox')
    w_attr = root.attrib.get('width', '')
    h_attr = root.attrib.get('height', '')

    if vb_attr:
        vb = [float(x) for x in vb_attr.replace(',', ' ').split()]
        w, h = int(round(vb[2])), int(round(vb[3]))
        if '%' in w_attr or not w_attr:
            svg_text = svg_text.replace(f'width="{w_attr}"', f'width="{w}"', 1) if w_attr else svg_text
        if '%' in h_attr or not h_attr:
            svg_text = svg_text.replace(f'height="{h_attr}"', f'height="{h}"', 1) if h_attr else svg_text
    else:
        w, h = int(round(float(w_attr))), int(round(float(h_attr)))

    rsvg = ctypes.CDLL('librsvg-2.so.2')
    cairo = ctypes.CDLL('libcairo.so.2')
    rsvg.rsvg_handle_new_from_data.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    rsvg.rsvg_handle_new_from_data.restype = ctypes.c_void_p
    rsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = ctypes.c_int
    cairo.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_scale.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
    cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    cairo.cairo_surface_write_to_png.restype = ctypes.c_int
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]

    svg_bytes = svg_text.encode('utf-8')
    err = ctypes.c_void_p()
    handle = rsvg.rsvg_handle_new_from_data(svg_bytes, len(svg_bytes), ctypes.byref(err))
    if not handle:
        raise RuntimeError(f'librsvg failed to parse {svg_path}')

    pw, ph = int(round(w * scale)), int(round(h * scale))
    with tempfile.TemporaryDirectory(prefix='esm_msr_fig2_') as tmpdir:
        png_path = Path(tmpdir) / f'{svg_path.stem}.png'
        surf = cairo.cairo_image_surface_create(0, pw, ph)
        cr = cairo.cairo_create(surf)
        cairo.cairo_scale(cr, ctypes.c_double(scale), ctypes.c_double(scale))
        ok = rsvg.rsvg_handle_render_cairo(handle, cr)
        if not ok:
            raise RuntimeError(f'librsvg failed to render {svg_path}')
        cairo.cairo_surface_write_to_png(surf, str(png_path).encode('utf-8'))
        cairo.cairo_destroy(cr)
        cairo.cairo_surface_destroy(surf)
        with Image.open(png_path) as im:
            rgba = im.convert('RGBA')
            bg = Image.new('RGB', rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            return np.asarray(bg)


def _draw_panel_b_scatter(ax: plt.Axes, cax: plt.Axes, split: str = 'capped') -> int:
    """Draw Panel b: ΔG_additive vs measured δΔΔG density scatter with y=-x and y=-x-1 lines."""
    fr, _ = load_tsu_test_frame(split=split)
    sp = fr.spine
    m = p07.epi_rows(fr)
    y = sp['dddG_ML'].to_numpy(float)
    x = sp['dG_wt'].to_numpy(float) + sp['ddG_additive_ML'].to_numpy(float)
    ok = m & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[ok], y[ok]

    rng = np.random.default_rng(0)
    sub = rng.choice(len(xv), min(4000, len(xv)), replace=False)
    kde = gaussian_kde(np.vstack([xv[sub], yv[sub]]))
    z = kde(np.vstack([xv, yv]))
    order = np.argsort(z)
    sc = ax.scatter(xv[order], yv[order], c=z[order], s=4.5, alpha=0.45, cmap='plasma', rasterized=True, lw=0)

    cb = plt.colorbar(sc, cax=cax)
    cb.set_label('density', fontsize=9.0)
    cb.ax.tick_params(labelsize=8.0)

    lo = min(float(np.percentile(xv, 0.2)), -3.6)
    hi = max(float(np.percentile(xv, 99.8)), 3.9)
    ylo = min(float(np.percentile(yv, 0.2)), -4.3)
    yhi = max(float(np.percentile(yv, 99.8)), 4.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)

    xx = np.array([lo, hi])
    ax.plot(xx, -xx, color='#c0392b', ls='--', lw=1.5, zorder=4, label='y = −x: ΔG_AB = 0 (unfolded floor)')
    ax.plot(xx, -xx - 1.0, color='#c0392b', ls=':', lw=1.5, zorder=4, label='y = −x − 1: ΔG_AB = −1 (assay cutoff)')

    rho_v = float(spearmanr(xv, yv)[0])
    r_v = float(pearsonr(xv, yv)[0])
    ax.text(
        0.04, 0.96,
        f'ρ = {rho_v:.2f}\nr = {r_v:.2f}\nn = {len(xv):,}',
        transform=ax.transAxes, va='top', ha='left', fontsize=8.8,
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#bbbbbb', alpha=0.92),
        zorder=6,
    )
    ax.legend(loc='lower left', fontsize=7.6, frameon=False, handlelength=2.2, labelspacing=0.35)
    ax.set_xlabel('ΔG_add = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=9.2)
    ax.set_ylabel('measured δΔΔG (kcal/mol)', fontsize=9.2)
    ax.tick_params(labelsize=8.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return len(xv)


def _draw_panel_c_types(ax_top: plt.Axes, ax_bot: plt.Axes, split: str = 'capped') -> float:
    """Draw Panel c: measured epistasis types by label additive expectation ΔG (alt_observed_additive_dG.csv)."""
    csv_path = results_dir('epistasis', split=split, create=False) / 'alt_observed_additive_dG.csv'
    df = pd.read_csv(csv_path)
    bins = list(dict.fromkeys(df['z_bin'].tolist()))
    nb = len(bins)
    xs = np.arange(nb)

    bottom = np.zeros(nb)
    tot_n = df.groupby('type')['n'].sum()
    grand_n = float(tot_n.sum())
    floor_share_total = float((df['share_floor_decides'] * df['n_bin']).sum() / max(df.groupby('z_bin')['n_bin'].first().sum(), 1))

    for t_idx, t_name in enumerate(p07.EPI_TYPES):
        sub = df[df['type'] == t_name].set_index('z_bin').reindex(bins)
        fl = sub['share_floor_decides'].to_numpy(float)
        sh = sub['share'].to_numpy(float)
        rest = np.maximum(sh - fl, 0.0)
        ax_top.bar(xs, fl, bottom=bottom, color=p07.EPI_COLOURS[t_idx], width=0.88,
                   edgecolor=p07.HATCH_EC[t_idx], lw=0, hatch='////')
        bottom += fl
        ax_top.bar(xs, rest, bottom=bottom, color=p07.EPI_COLOURS[t_idx], width=0.88,
                   edgecolor='white', lw=0.35)
        bottom += rest

    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel('fraction of doubles', fontsize=9.2)
    ax_top.tick_params(axis='y', labelsize=8.4)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    handles = [
        Patch(fc=p07.EPI_COLOURS[i], ec='#999999', lw=0.4,
              label=f'{p07.EPI_TYPES[i]} ({p07.EPI_RANGES[i]}; {100.0 * tot_n.get(p07.EPI_TYPES[i], 0) / grand_n:.0f}%)')
        for i in range(len(p07.EPI_TYPES))
    ]
    handles.append(Patch(fc='white', ec='#222222', hatch='////', lw=0.5,
                         label=f'hatched: floor decides ({100.0 * floor_share_total:.0f}%)'))
    ax_top.legend(handles=handles, loc='lower left', fontsize=6.9, frameon=True,
                  facecolor='white', framealpha=0.90, edgecolor='#cccccc',
                  handlelength=1.4, labelspacing=0.22, borderpad=0.32)

    bin_df = df.groupby('z_bin', sort=False).first().reindex(bins)
    n_neg = bin_df['n_net_ddG_additive_negative'].to_numpy(float)
    n_pos = bin_df['n_net_ddG_additive_positive'].to_numpy(float)
    ax_bot.bar(xs - 0.19, n_neg, width=0.36, color='#888888', label="singles' ΣΔΔG < 0")
    ax_bot.bar(xs + 0.19, n_pos, width=0.36, color='#222222', label="singles' ΣΔΔG > 0")
    ax_bot.set_yscale('log')
    ax_bot.set_ylabel('n', fontsize=9.0)
    ax_bot.set_xlabel('additive ΔG = ΔG_wt + ΔΔG_A + ΔΔG_B (kcal/mol)', fontsize=9.2)
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels(bins, rotation=42, ha='right', fontsize=7.6)
    ax_bot.tick_params(axis='y', labelsize=7.6)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.spines['right'].set_visible(False)
    ax_bot.legend(loc='upper right', fontsize=6.8, frameon=False, handlelength=1.2, labelspacing=0.2)
    return floor_share_total


def build_figure2(*, split: str = 'capped', cycles_scale: float = 4.0, levels_scale: float = 3.0) -> Path:
    """Build and save Main-Text Figure 2 (`00_main_text_figs/figure2.{png,pdf}`)."""
    supp_dir = results_dir('supplementary_figs', create=False)
    T_p0 = pd.read_csv(supp_dir / 'fig5_epistasis_levels_p0_bootstrap_test.csv')

    p0_entries = [
        p06._hier_entry('ProteinMPNN', 'ProteinMPNN'),
        p06._hier_entry('ESM3-small-open', 'ESM3-small-open'),
        p06._hier_entry(p06.P0_EPI_NAME, 'ESM-MSR'),
        p06._hier_entry(p06.P0_CTX_NAME, 'ESM-MSR', 'ESM-MSR ctx ctrl', M.VARIANT_COLOR['ESM-MSR ctx ctrl']),
        p06._hier_entry('ME stage 2', 'MutateEverything'),
        p06._hier_entry('ThermoMPNN-D', 'ThermoMPNN'),
        p06._hier_entry('SPURS multi', 'SPURS'),
        p06._hier_entry('GeoDDG-3D', 'GeoDDG-3D'),
        p06._hier_entry('Rosetta', 'Rosetta'),
        p06._hier_entry('ESM3ΔG', 'ESM3ΔG'),
    ]

    img_a = _render_svg_to_rgb(REPO_ROOT / '_assets' / 'epistasis_cycles_split.svg', scale=cycles_scale)
    img_d = _render_svg_to_rgb(REPO_ROOT / '_assets' / 'epistasis_levels_5panel_anova.svg', scale=levels_scale)

    fig = plt.figure(figsize=(17.6, 19.6), facecolor='white')

    # --- Top row (a, b, c): y in [0.742, 0.974] ---
    gs_a = fig.add_gridspec(1, 1, left=0.020, right=0.320, top=0.974, bottom=0.735)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a.imshow(img_a, interpolation='lanczos')
    ax_a.set_axis_off()

    gs_b = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.06,
                            left=0.376, right=0.635, top=0.968, bottom=0.748)
    ax_b = fig.add_subplot(gs_b[0, 0])
    cax_b = fig.add_subplot(gs_b[0, 1])
    n_epi = _draw_panel_b_scatter(ax_b, cax_b, split=split)

    gs_c = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.0], hspace=0.08,
                            left=0.728, right=0.986, top=0.968, bottom=0.748)
    ax_c_top = fig.add_subplot(gs_c[0, 0])
    ax_c_bot = fig.add_subplot(gs_c[1, 0], sharex=ax_c_top)
    floor_share = _draw_panel_c_types(ax_c_top, ax_c_bot, split=split)

    # --- Middle row (d): epistasis_levels_5panel_anova.svg (1560x630) ---
    # Width-constrained box so the 5 SVG cards ([15,295], [325,605], [635,915], [945,1225], [1255,1545])
    # span [0.052, 0.986] cleanly and Panel e axes lock 1:1 to their exact rendered horizontal bounds.
    gs_d = fig.add_gridspec(1, 1, left=0.043, right=0.995, top=0.708, bottom=0.358)
    ax_d = fig.add_subplot(gs_d[0, 0])
    ax_d.imshow(img_d, interpolation='lanczos')
    ax_d.set_axis_off()

    # Draw once to resolve ax_d's exact aspect-preserved bounding box in figure coordinates
    fig.canvas.draw()
    pd_box = ax_d.get_position()

    # --- Bottom row (e): 5 pooled epistasis levels locked 1:1 horizontally to Panel d's 5 SVG cards ---
    svg_w = 1560.0
    svg_cards = [(15.0, 280.0), (325.0, 280.0), (635.0, 280.0), (945.0, 280.0), (1255.0, 290.0)]
    bottom_e, height_e = 0.136, 0.176
    axes_e = np.empty(5, dtype=object)
    for i, (cx, cw) in enumerate(svg_cards):
        ax_x0 = pd_box.x0 + pd_box.width * (cx / svg_w)
        ax_w = pd_box.width * (cw / svg_w)
        axes_e[i] = fig.add_axes([ax_x0, bottom_e, ax_w, height_e])
    for i in range(1, 5):
        axes_e[i].sharey(axes_e[0])
        axes_e[i].tick_params(axis='y', labelleft=False)
    p06._draw_hierarchy_axes(axes_e, T_p0, p0_entries, p06.LEVELS_CELL_MAG, overlay_sub0=True)

    # --- Nature-style sub-figure annotations: a, b, c (top), d (middle), e (bottom) ---
    fig.canvas.draw()
    pb = ax_b.get_position()
    pc = ax_c_top.get_position()
    pe = axes_e[0].get_position()

    y_top = 0.991
    fig.text(0.008, y_top, 'a', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(pb.x0 - 0.044, y_top, 'b', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(pc.x0 - 0.044, y_top, 'c', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(0.008, pd_box.y1 + 0.006, 'd', ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(0.008, pe.y1 + 0.032, 'e', ha='left', va='top', fontsize=15, fontweight='bold')

    # --- Combined footnote ---
    nd = int(T_p0['n_domains'].iloc[0])
    B_val = int(T_p0['B'].iloc[0])
    note2 = (
        f'a. Thermodynamic-cycle schematic of epistasis types (top: weak positive A, additive B, strong negative/clash C; bottom: strong positive/compensation D, additive E, weak negative F). '
        f'b. Homology-Restricted MegaScale test doubles with both singles measured (n = {n_epi:,}, {nd} domains): unfolded-state floor (y = −x) and assay cutoff (y = −x − 1) induce a strong negative trend (ρ = −0.64). '
        f'c. Share of measured epistasis types across label additive expectation bins ΔG_add = ΔG_wt + ΔΔG_A + ΔΔG_B (hatched: floor-dominated doubles, {100.0 * floor_share:.0f}%; bottom: log-scale counts split by sign of singles\' summed ΔΔG).\n'
        f'd, e. Five-level pooled epistasis residual hierarchy (B = {B_val:,} domain × seed bootstrap; black bars: unsaturated subset ΔG_add > 0; red dotted line: −ΔG_add floor ceiling). '
        f'All five levels are evaluated in pooled form (naive / raw ρ, saturation-subtracted ρ, pair ρ across position-pair matrices ≥ 20 doubles, line ρ across pair-centred row/column lines ≥ 3 cells, and cell ρ pooled over all double-centred residual cells).'
    )
    kw_n = dict(fontsize=7.6, color='#444444')
    fig.text(0.012, 0.006, style.wrap_to_px(fig, note2, 0.975 * fig.bbox.width, **kw_n), ha='left', va='bottom', **kw_n)

    out_path = results_dir('main_text_figs') / 'figure2'
    style.save(fig, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Build Main-Text Figure 2 (epistasis cycles split, floor saturation, additive dG types, 5-level ANOVA diagram & pooled stats).')
    ap.add_argument('--split', choices=('capped', 'mega'), default='capped')
    ap.add_argument('--cycles-scale', type=float, default=4.0, help='Rasterization scale for epistasis_cycles_split.svg (default: 4.0).')
    ap.add_argument('--levels-scale', type=float, default=3.0, help='Rasterization scale for epistasis_levels_5panel_anova.svg (default: 3.0).')
    args = ap.parse_args(argv)
    out = build_figure2(split=args.split, cycles_scale=args.cycles_scale, levels_scale=args.levels_scale)
    print(f'[main_figures:figure2] Saved {out}.{{png,pdf}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
