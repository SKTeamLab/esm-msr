#!/usr/bin/env python3
"""Build Main-Text Figure 1 (`analysis/results/00_main_text_figs/figure1.{png,pdf}`).

Layout:
  - Panel a (full left side): Vertical ESM-MSR architecture schematic (`_assets/esm_msr_vertical.svg`,
    rendered without its standalone label so 'a' is placed consistently in the combined figure).
  - Panel b (top right): Stabilizing vs destabilizing MegaScale test-set Spearman's rho (`spearman_stab_destab.csv`).
  - Panel c (bottom right): Additive vs epistatic double-mutant Spearman's rho (`spearman_doubles_additive_vs_epistatic.csv`)
    with paired one-sided bootstrap p-value brackets (`epistatic_vs_additive_pvalues.csv`).

Usage:
    PYTHONPATH=src:analysis python -m analysis.main_figures.figure1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis import style
from msr_analysis.paths import REPO_ROOT, results_dir
from msr_analysis.stats import fmt_p
from pipelines._tsu_common import (
    BEST_LABEL,
    best_other,
    count_text,
    load_tsu_test_frame,
    main_columns,
    pair_columns,
)


def _load_schematic_module():
    mod_path = REPO_ROOT / '_assets' / 'generate_esm_msr_schematics.py'
    spec = importlib.util.spec_from_file_location('_esm_msr_schematics', mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load schematic generator from {mod_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_vertical_schematic_image(scale: float = 4.0) -> np.ndarray:
    """Render `_assets/esm_msr_vertical.svg` (no standalone 'a' label) to a high-DPI RGB array."""
    mod = _load_schematic_module()
    svg_str = mod.build_vertical_svg(include_label=False)
    with tempfile.TemporaryDirectory(prefix='esm_msr_fig1_') as tmpdir:
        tmp = Path(tmpdir)
        svg_path = tmp / 'esm_msr_vertical.svg'
        png_path = tmp / 'esm_msr_vertical.png'
        svg_path.write_text(svg_str, encoding='utf-8')
        mod.render_svg_to_png(svg_path, png_path, scale=scale)
        with Image.open(png_path) as im:
            return np.asarray(im.convert('RGB'))


def _draw_pair_brackets(
    ax,
    cols: list[dict],
    T: pd.DataFrame,
    p_series: pd.Series,
    subsets: tuple[str, str] = ('additive', 'epistatic'),
    p_name_map: dict[str, str] | None = None,
    B_p: int | None = None,
) -> float:
    nm = p_name_map or {}
    drawn = [c for c in cols if c.get('draw', True)]
    xs, _ = style.layout_grouped_columns(drawn, gap=0.7)
    top = 0.0
    for c, x in zip(drawn, xs):
        r = [T[(T['column'] == c['name']) & (T['subset'] == sub)] for sub in subsets]
        name = nm.get(c['name'], c['name'])
        if name not in p_series.index or not all(len(q) for q in r):
            continue
        hi = max(float(q.iloc[0]['hi']) for q in r)
        y = hi + 0.135
        x0, x1 = x - 0.16, x + 0.16
        h = 0.014
        ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color='#333333', lw=1.0, solid_capstyle='butt', clip_on=False, zorder=9)
        ax.text(
            (x0 + x1) / 2, y + 0.008, fmt_p(float(p_series[name]), B=B_p, prefix='p = '),
            ha='center', va='bottom', fontsize=8.0, color='#333333', zorder=9,
        )
        top = max(top, y + 0.075)
    return top


def build_figure1(
    *,
    split: str = 'capped',
    t2: pd.DataFrame | None = None,
    t3: pd.DataFrame | None = None,
    p_spearman_ser: pd.Series | None = None,
    B_p: int | None = None,
    schematic_scale: float = 4.0,
) -> Path:
    """Build and save Main-Text Figure 1 (`00_main_text_figs/figure1.{png,pdf}`)."""
    fr, _ = load_tsu_test_frame(split=split)
    best, _ = best_other(fr)
    name_map = {'best other ddG': BEST_LABEL[best]} if best else {}
    cols2 = main_columns(fr, best, 'unmasked')
    cols3 = pair_columns(fr, best)

    mega_dir = results_dir('megascale', split=split, create=False)
    epi_dir = results_dir('epistasis', split=split, create=False)

    if t2 is None:
        t2 = pd.read_csv(mega_dir / 'spearman_stab_destab.csv')
    if t3 is None:
        t3 = pd.read_csv(epi_dir / 'spearman_doubles_additive_vs_epistatic.csv')
    if p_spearman_ser is None:
        p_csv = epi_dir / 'epistatic_vs_additive_pvalues.csv'
        if p_csv.exists():
            p_df = pd.read_csv(p_csv)
            p_spearman_ser = p_df.set_index('model')['p_one']
            if B_p is None and 'B' in p_df.columns and len(p_df):
                B_p = int(p_df['B'].iloc[0])

    T2_st = t2[t2['subset'] != 'overall']
    ov2_st = t2[t2['subset'] == 'overall']

    # Render vertical architecture schematic without its standalone 'a' label
    img_a = _render_vertical_schematic_image(scale=schematic_scale)

    fig = plt.figure(figsize=(22.4, 12.6), facecolor='white')

    # Left panel (a): full-height vertical architecture schematic
    gs_left = fig.add_gridspec(1, 1, left=0.018, right=0.454, top=0.985, bottom=0.015)
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_a.imshow(img_a, interpolation='lanczos')
    ax_a.set_axis_off()

    # Right panels (b stacked above c): MegaScale stabilizing/destabilizing + additive/epistatic doubles
    gs_right = fig.add_gridspec(2, 1, left=0.508, right=0.992, top=0.955, bottom=0.105, hspace=0.42)
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_c = fig.add_subplot(gs_right[1, 0])

    style.bar_panel(
        ax_b, cols2, T2_st, ('stabilizing', 'destabilizing'),
        (
            count_text(T2_st, 'stabilizing', 'stabilizing, ddG > 0'),
            count_text(T2_st, 'destabilizing', 'destabilizing, ddG < 0'),
        ),
        "Spearman's ρ (mean over test domains)", (-0.2, 1.12),
        overall_df=ov2_st, overall_label=f'overall (n = {int(ov2_st.n_variants.max()):,})',
        zero_line=True, tick_fs=8.5,
    )
    style.bar_panel(
        ax_c, cols3, t3, ('additive', 'epistatic'),
        (
            count_text(t3, 'additive', 'additive: singles model summed'),
            count_text(t3, 'epistatic', 'epistatic: model scores the double'),
        ),
        "Spearman's ρ, double mutants", (0.0, 1.12),
        tick_fs=8.5,
    )
    if p_spearman_ser is not None:
        top_y = _draw_pair_brackets(ax_c, cols3, t3, p_spearman_ser, p_name_map=name_map, B_p=B_p)
        if top_y > 1.12:
            ax_c.set_ylim(0.0, top_y)

    for ax_s in (ax_b, ax_c):
        for txt in ax_s.texts:
            if txt.get_fontsize() < 8.0:
                txt.set_fontsize(8.0)

    # Nature-style bold lowercase sans-serif sub-figure annotations: a (left), b (top right), c (bottom right)
    fig.canvas.draw()
    pa = ax_a.get_position()
    pb = ax_b.get_position()
    pc = ax_c.get_position()
    y_top_lbl = max(pa.y1, pb.y1 + 0.032)
    fig.text(0.008, y_top_lbl, 'a', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(pb.x0 - 0.042, y_top_lbl, 'b', ha='left', va='top', fontsize=16, fontweight='bold')
    fig.text(pc.x0 - 0.042, pc.y1 + 0.032, 'c', ha='left', va='top', fontsize=16, fontweight='bold')

    out_path = results_dir('main_text_figs') / 'figure1'
    style.save(fig, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Build Main-Text Figure 1 (architecture schematic + MegaScale bars).')
    ap.add_argument('--split', choices=('capped', 'mega'), default='capped')
    ap.add_argument('--scale', type=float, default=4.0, help='Rasterization scale for the SVG schematic (default: 4.0).')
    args = ap.parse_args(argv)
    out = build_figure1(split=args.split, schematic_scale=args.scale)
    print(f'[main_figures:figure1] Saved {out}.{{png,pdf}}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
