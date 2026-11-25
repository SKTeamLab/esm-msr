# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Publication-quality 2D mutational landscape heatmap and ranked-pair legend card exporters.

Uses matplotlib's object-oriented Figure + FigureCanvasAgg directly so exports are
100% headless-safe and never interfere with ChimeraX's interactive Qt backend.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

try:
    from .parsing import (
        COLOR_PALETTES,
        DEFAULT_PALETTE_KEY,
        ParsedCSVData,
        PosKey,
        RankedPair,
        format_res_pos,
        get_color_palette,
        parse_esm_msr_csv,
    )
except ImportError:
    from parsing import (  # type: ignore[no-redef]
        COLOR_PALETTES,
        DEFAULT_PALETTE_KEY,
        ParsedCSVData,
        PosKey,
        RankedPair,
        format_res_pos,
        get_color_palette,
        parse_esm_msr_csv,
    )

# Physicochemical amino-acid grouping (top to bottom):
# Charged | Polar | Aliphatic | Aromatic | Special
AA_PROPERTY_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Charged", ("D", "E", "K", "R", "H")),
    ("Polar", ("N", "Q", "S", "T")),
    ("Aliphatic", ("A", "V", "I", "L", "M")),
    ("Aromatic", ("F", "Y", "W")),
    ("Special", ("C", "G", "P")),
)

AA_ORDER: Tuple[str, ...] = tuple(aa for _, group in AA_PROPERTY_GROUPS for aa in group)
_AA_TO_ROW: Dict[str, int] = {aa: idx for idx, aa in enumerate(AA_ORDER)}


def _compute_tick_step(n_pos: int) -> int:
    if n_pos <= 35:
        return 1
    if n_pos <= 80:
        return 5
    if n_pos <= 180:
        return 10
    if n_pos <= 400:
        return 20
    if n_pos <= 800:
        return 50
    return 100


def export_mutational_heatmap(
    parsed_csv: Union[ParsedCSVData, str],
    output_path: str,
    title: Optional[str] = None,
    palette_key: str = DEFAULT_PALETTE_KEY,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    chain: Optional[str] = None,
    dpi: int = 200,
) -> str:
    """Render a 20 x L single-mutant ΔΔG heatmap (PNG/PDF/SVG) with physicochemical AA grouping.

    Parameters
    ----------
    parsed_csv : ParsedCSVData or path to ESM-MSR CSV file
    output_path : destination image path (.png, .pdf, .svg)
    title : optional figure title
    palette_key : one of COLOR_PALETTES keys (default 'blue_white_red')
    vmin, vmax : optional colormap bounds (auto-symmetric around 0 for diverging palettes)
    chain : optional chain ID filter (defaults to parsed_csv.selected_chain or first chain present)
    dpi : export resolution in dots per inch (default 200)
    """
    if isinstance(parsed_csv, (str, os.PathLike)):
        parsed = parse_esm_msr_csv(str(parsed_csv), target_chain=(chain or ""))
        if not parsed.ok:
            raise ValueError(f"Failed to parse CSV for heatmap: {parsed.error}")
    else:
        parsed = parsed_csv

    if parsed is None or not parsed.all_singles_scores:
        raise ValueError("Cannot export heatmap: no single-mutant scores are loaded.")

    available_chains = sorted({c for (c, _, _) in parsed.all_singles_scores.keys()})
    target_chain = (chain or "").strip() or (parsed.selected_chain or "").strip()
    if not target_chain or target_chain not in available_chains:
        target_chain = available_chains[0]

    positions_set = {
        pos_key
        for (c, pos_key, _) in parsed.all_singles_scores.keys()
        if c == target_chain
    }
    if not positions_set:
        raise ValueError(f"No single-mutant scores found for chain '{target_chain}'.")

    positions: List[PosKey] = sorted(positions_set, key=lambda pk: (pk[0], pk[1]))
    n_pos = len(positions)
    pos_to_col = {pk: idx for idx, pk in enumerate(positions)}

    matrix = np.full((len(AA_ORDER), n_pos), np.nan, dtype=float)
    for (c, pos_key, mut_aa), score in parsed.all_singles_scores.items():
        if c != target_chain:
            continue
        row = _AA_TO_ROW.get(mut_aa.upper())
        col = pos_to_col.get(pos_key)
        if row is not None and col is not None and np.isfinite(score):
            matrix[row, col] = float(score)

    finite_vals = matrix[np.isfinite(matrix)]
    if finite_vals.size == 0:
        raise ValueError("Cannot export heatmap: all single-mutant scores are non-finite.")

    pal = get_color_palette(palette_key)
    is_diverging = bool(pal["diverging"])

    if vmin is None and vmax is None:
        if is_diverging:
            bound = float(np.percentile(np.abs(finite_vals), 98))
            if bound <= 0.05:
                bound = float(np.max(np.abs(finite_vals)))
            if bound <= 0.0:
                bound = 1.0
            eff_vmin, eff_vmax = -bound, bound
        else:
            p_lo = float(np.percentile(finite_vals, 2))
            p_hi = float(np.percentile(finite_vals, 98))
            if p_hi <= p_lo:
                p_lo = float(np.min(finite_vals))
                p_hi = float(np.max(finite_vals))
            if p_hi <= p_lo:
                p_lo, p_hi = p_lo - 0.5, p_hi + 0.5
            eff_vmin, eff_vmax = p_lo, p_hi
    else:
        eff_vmin = float(vmin) if vmin is not None else float(np.min(finite_vals))
        eff_vmax = float(vmax) if vmax is not None else float(np.max(finite_vals))
        if eff_vmax <= eff_vmin:
            eff_vmax = eff_vmin + 1.0

    # Auto-scale figure width smoothly for both compact peptides and long proteins.
    fig_w = max(8.2, min(30.0, 2.8 + n_pos * 0.145))
    fig_h = 5.9

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.075, 0.15, 0.775, 0.72])
    cax = fig.add_axes([0.895, 0.15, 0.018, 0.72])

    cmap = matplotlib.colormaps[pal["mpl"]].copy()
    cmap.set_bad(color="#e6e6e6")

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=eff_vmin,
        vmax=eff_vmax,
    )

    # Subtle group separators and physicochemical group annotations on the right edge.
    cursor = 0
    for g_idx, (group_name, group_aas) in enumerate(AA_PROPERTY_GROUPS):
        g_len = len(group_aas)
        mid_y = cursor + (g_len - 1) / 2.0
        ax.text(
            1.008,
            1.0 - (mid_y + 0.5) / len(AA_ORDER),
            group_name,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=8,
            color="#444444",
            fontweight="semibold",
        )
        if g_idx > 0:
            ax.axhline(cursor - 0.5, color="#333333", linewidth=0.9, alpha=0.65)
        cursor += g_len

    # Mark WT residues with crisp subtle diamonds.
    wt_xs: List[int] = []
    wt_ys: List[int] = []
    wt_map = getattr(parsed, "wt_residues", {}) or {}
    for col_idx, pk in enumerate(positions):
        wt_aa = wt_map.get((target_chain, pk))
        if wt_aa:
            row_idx = _AA_TO_ROW.get(wt_aa.upper())
            if row_idx is not None:
                wt_xs.append(col_idx)
                wt_ys.append(row_idx)
    if wt_xs:
        marker_sz = 26 if n_pos <= 60 else (16 if n_pos <= 160 else 9)
        ax.scatter(
            wt_xs,
            wt_ys,
            marker="D",
            s=marker_sz,
            facecolor="#222222",
            edgecolor="#ffffff",
            linewidths=0.6,
            zorder=4,
        )

    # Y-axis: 20 canonical AAs
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_yticklabels(AA_ORDER, fontsize=8.5, family="monospace")
    ax.set_ylabel("Mutant Amino Acid", fontsize=10, fontweight="semibold", color="#222222")

    # X-axis: PDB residue positions with legible auto-spaced ticks
    step = _compute_tick_step(n_pos)
    tick_indices = list(range(0, n_pos, step))
    if (n_pos - 1) not in tick_indices and (n_pos - 1 - tick_indices[-1]) >= max(1, step // 2):
        tick_indices.append(n_pos - 1)
    tick_labels = [format_res_pos(positions[i]) for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        tick_labels,
        fontsize=8,
        rotation=45 if step == 1 and n_pos > 18 else 0,
        ha="right" if step == 1 and n_pos > 18 else "center",
    )
    ax.set_xlabel(f"PDB Residue Position (Chain {target_chain})", fontsize=10, fontweight="semibold", color="#222222")

    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.8)

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Predicted ΔΔG (kcal/mol; + stabilizing)", fontsize=9.5, color="#222222", labelpad=8)
    cbar.ax.tick_params(labelsize=8.5)

    # Title & subtitle metadata
    score_col_label = parsed.singles_score_col or parsed.effective_score_col or "score"
    fig_title = title or f"ESM-MSR Single-Mutant Landscape \u2014 Chain {target_chain} ({n_pos} positions)"
    fig.suptitle(fig_title, x=0.075, y=0.955, ha="left", fontsize=12, fontweight="bold", color="#1a1a1a")

    subtitle_parts = [f"Metric: {score_col_label}", f"Palette: {pal['label']}"]
    if wt_xs:
        subtitle_parts.append("\u25c6 Wild-type")
    subtitle_parts.append("Gray: missing/WT")
    fig.text(0.075, 0.895, "  |  ".join(subtitle_parts), ha="left", fontsize=8.5, color="#555555")

    out_abs = os.path.abspath(os.path.expanduser(str(output_path)))
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_abs, dpi=dpi, facecolor="white")
    return out_abs


def _pair_attr(pair: Any, name: str, default: Any = None) -> Any:
    if isinstance(pair, dict):
        return pair.get(name, default)
    return getattr(pair, name, default)


def export_pair_legend_card(
    ranked_pairs: Sequence[Union[RankedPair, Dict[str, Any]]],
    output_path: str,
    title: Optional[str] = None,
    dpi: int = 200,
) -> str:
    """Render a clean publication legend card for Top-M categorically colored double-mutant pairs."""
    pairs = list(ranked_pairs or [])
    if not pairs:
        raise ValueError("Cannot export pair legend card: ranked_pairs is empty.")

    has_s1 = any(_pair_attr(p, "stage1_additive_pred") is not None for p in pairs)
    has_epi = any(_pair_attr(p, "epi_score") is not None for p in pairs)
    has_rc = any(_pair_attr(p, "rank_change") is not None for p in pairs)
    has_dist = any(_pair_attr(p, "dist") is not None for p in pairs)

    # Dynamic column layout: (header, width_weight, align)
    columns: List[Tuple[str, float, str]] = [
        ("", 0.48, "center"),          # Color swatch
        ("Rank", 0.65, "center"),
        ("Double Mutant", 2.25, "left"),
        ("Combined ΔΔG", 1.45, "right"),
    ]
    if has_s1:
        columns.append(("Stage-1 Add.", 1.35, "right"))
    if has_epi:
        columns.append(("Epistasis", 1.20, "right"))
    if has_rc:
        columns.append(("ΔRank", 0.90, "right"))
    if has_dist:
        columns.append(("Dist (Å)", 1.05, "right"))

    total_w = sum(w for _, w, _ in columns)
    fig_w = max(6.4, min(11.5, 0.92 * total_w + 0.8))
    row_h_in = 0.42
    header_h_in = 0.95
    footer_h_in = 0.25
    fig_h = header_h_in + row_h_in * len(pairs) + footer_h_in

    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0.0, total_w)
    ax.set_ylim(0.0, len(pairs) + 2.35)
    ax.axis("off")

    # Compute column x-intervals
    x_starts: List[float] = []
    cur_x = 0.0
    for _, w, _ in columns:
        x_starts.append(cur_x)
        cur_x += w

    # Title banner
    card_title = title or f"Top-{len(pairs)} Ranked Double-Mutant Pairs (ESM-MSR)"
    ax.text(
        0.08,
        len(pairs) + 1.85,
        card_title,
        fontsize=11.5,
        fontweight="bold",
        color="#1a1a1a",
        va="center",
        ha="left",
    )

    # Table header row
    hdr_y = len(pairs) + 0.85
    ax.add_patch(
        Rectangle(
            (0.0, hdr_y - 0.42),
            total_w,
            0.82,
            facecolor="#f1f3f5",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.plot([0.0, total_w], [hdr_y - 0.42, hdr_y - 0.42], color="#adb5bd", linewidth=1.0, zorder=2)

    for idx, (col_label, col_w, align) in enumerate(columns):
        if not col_label:
            continue
        x0 = x_starts[idx]
        if align == "left":
            tx = x0 + 0.08
        elif align == "right":
            tx = x0 + col_w - 0.12
        else:
            tx = x0 + col_w * 0.5
        ax.text(
            tx,
            hdr_y,
            col_label,
            fontsize=8.8,
            fontweight="bold",
            color="#343a40",
            va="center",
            ha=align,
            zorder=3,
        )

    # Data rows (top rank first)
    for row_i, pair in enumerate(pairs):
        y_center = len(pairs) - 1 - row_i + 0.42
        if row_i % 2 == 1:
            ax.add_patch(
                Rectangle(
                    (0.0, y_center - 0.46),
                    total_w,
                    0.92,
                    facecolor="#f8f9fa",
                    edgecolor="none",
                    zorder=1,
                )
            )
        ax.plot([0.0, total_w], [y_center - 0.46, y_center - 0.46], color="#e9ecef", linewidth=0.6, zorder=2)

        color_hex = str(_pair_attr(pair, "color_hex", "#0072B2"))
        rank = _pair_attr(pair, "rank", row_i + 1)
        m1 = _pair_attr(pair, "mut1_str", "")
        m2 = _pair_attr(pair, "mut2_str", "")
        mut_label = f"{m1} + {m2}" if (m1 and m2) else str(_pair_attr(pair, "mut_str", "")).replace(":", " + ")

        comb = _pair_attr(pair, "combined_pred")
        if comb is None:
            comb = _pair_attr(pair, "score")
        s1 = _pair_attr(pair, "stage1_additive_pred")
        epi = _pair_attr(pair, "epi_score")
        rc = _pair_attr(pair, "rank_change")
        dist = _pair_attr(pair, "dist")

        # Swatch rounded box
        sw_x = x_starts[0] + 0.09
        ax.add_patch(
            FancyBboxPatch(
                (sw_x, y_center - 0.23),
                0.28,
                0.46,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=color_hex,
                edgecolor="#222222",
                linewidth=0.6,
                zorder=3,
            )
        )

        # Populate cell text values aligned with columns[1:]
        cell_vals: List[Tuple[str, str, str]] = [
            (f"#{rank}", "#212529", "bold"),
            (mut_label, "#111111", "semibold"),
            (f"{float(comb):+.2f}" if comb is not None and np.isfinite(comb) else "\u2014", "#111111", "semibold"),
        ]
        if has_s1:
            cell_vals.append(
                (f"{float(s1):+.2f}" if s1 is not None and np.isfinite(s1) else "\u2014", "#495057", "normal")
            )
        if has_epi:
            if epi is not None and np.isfinite(epi):
                e_col = "#1b7837" if float(epi) > 0 else ("#b2182b" if float(epi) < 0 else "#495057")
                cell_vals.append((f"{float(epi):+.2f}", e_col, "semibold"))
            else:
                cell_vals.append(("\u2014", "#868e96", "normal"))
        if has_rc:
            if rc is not None and np.isfinite(rc):
                rc_i = int(rc)
                rc_col = "#1b7837" if rc_i > 0 else ("#c92a2a" if rc_i < 0 else "#495057")
                rc_txt = f"{rc_i:+d}" if rc_i != 0 else "0"
                cell_vals.append((rc_txt, rc_col, "semibold"))
            else:
                cell_vals.append(("\u2014", "#868e96", "normal"))
        if has_dist:
            cell_vals.append(
                (f"{float(dist):.1f}" if dist is not None and np.isfinite(dist) else "\u2014", "#495057", "normal")
            )

        for offset, (txt, txt_color, weight) in enumerate(cell_vals, start=1):
            _, col_w, align = columns[offset]
            x0 = x_starts[offset]
            if align == "left":
                tx = x0 + 0.08
            elif align == "right":
                tx = x0 + col_w - 0.12
            else:
                tx = x0 + col_w * 0.5
            ax.text(
                tx,
                y_center,
                txt,
                fontsize=9.0,
                fontweight=weight,
                color=txt_color,
                va="center",
                ha=align,
                family="monospace" if offset >= 3 else "sans-serif",
                zorder=3,
            )

    out_abs = os.path.abspath(os.path.expanduser(str(output_path)))
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_abs, dpi=dpi, facecolor="white")
    return out_abs
