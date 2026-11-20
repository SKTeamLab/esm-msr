"""Benchmark markdown table readers and statistical CSV / Markdown / LaTeX table formatters."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import REFERENCE, atomic_to_csv, atomic_write_text

TABLES = REFERENCE / 'benchmark_tables'
_SUP = 'ᴹᵐᴾᴴʰᴰᶜᵁ#*†‡'


def read_table(tid: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f'T{tid}.csv', dtype=str).fillna('')


def num(cell) -> tuple[float, float]:
    """Parse ``'**0.654±0.003**ʰ'`` -> ``(0.654, 0.003)``; ``'–'``, ``'q'``, ``''`` -> ``(nan, nan)``."""
    s = str(cell).replace('**', '').replace('_', '').replace('−', '-').replace('+', '').strip()
    s = ''.join(ch for ch in s if ch not in _SUP).strip()
    m = re.match(r'^(-?\d*\.?\d+(?:e-?\d+)?)\s*(?:±\s*(\d*\.?\d+))?', s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2)) if m.group(2) else np.nan


def numeric(df: pd.DataFrame, first_col: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(value_df, sd_df)`` for a benchmark table whose first column names the rows."""
    rows = df.iloc[:, 0].tolist()
    V = pd.DataFrame({c: [num(x)[0] for x in df[c]] for c in df.columns[1:]}, index=rows)
    S = pd.DataFrame({c: [num(x)[1] for x in df[c]] for c in df.columns[1:]}, index=rows)
    return V, S


def clean_label(s: str) -> str:
    s = re.sub(r'[ᴹᵐᴾᴴʰᴰᶜᵁ‡]', '', str(s))
    return s.replace('  ', ' ').replace('ESM-MSR (devel)', 'ESM-MSR').strip()


def _fmt_cell(val, fmt: str = '%.4f') -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '—'
    if isinstance(val, (int, np.integer)):
        return f'{int(val):,}'
    if isinstance(val, (float, np.floating)):
        return fmt % float(val)
    return str(val)


def _tex_escape(s: str) -> str:
    out = str(s)
    replacements = (
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
        ('Δ', r'$\Delta$'),
        ('δ', r'$\delta$'),
        ('ρ', r'$\rho$'),
        ('σ', r'$\sigma$'),
        ('±', r'$\pm$'),
        ('≤', r'$\le$'),
        ('≥', r'$\ge$'),
        ('−', '-'),
        ('→', r'$\to$'),
    )
    for a, b in replacements:
        out = out.replace(a, b)
    return out


def format_stat_table_md(
    df: pd.DataFrame,
    columns: dict[str, str] | None = None,
    bold_mask: pd.DataFrame | None = None,
    float_fmt: str = '%.4f',
) -> str:
    """Render ``df`` as a GitHub-flavored Markdown table with optional per-cell ``bold_mask``."""
    col_keys = list(columns.keys()) if columns else list(df.columns)
    headers = [columns[k] if columns else str(k) for k in col_keys]
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for idx, row in df.iterrows():
        cells = []
        for k in col_keys:
            txt = _fmt_cell(row[k], fmt=float_fmt)
            is_bold = bool(bold_mask.loc[idx, k]) if (bold_mask is not None and k in bold_mask.columns) else False
            cells.append(f'**{txt}**' if is_bold and txt != '—' else txt)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def format_stat_table_tex(
    df: pd.DataFrame,
    columns: dict[str, str] | None = None,
    bold_mask: pd.DataFrame | None = None,
    float_fmt: str = '%.4f',
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Render ``df`` as a clean LaTeX ``booktabs`` ``tabular`` environment."""
    col_keys = list(columns.keys()) if columns else list(df.columns)
    headers = [_tex_escape(columns[k] if columns else str(k)) for k in col_keys]
    align = 'l' + 'r' * max(len(col_keys) - 1, 0)
    lines = []
    if caption or label:
        lines.append(r'\begin{table}[htbp]')
        lines.append(r'\centering')
        if caption:
            lines.append(rf'\caption{{{_tex_escape(caption)}}}')
        if label:
            lines.append(rf'\label{{{label}}}')
    lines.append(rf'\begin{{tabular}}{{{align}}}')
    lines.append(r'\toprule')
    lines.append(' & '.join(headers) + r' \\')
    lines.append(r'\midrule')
    for idx, row in df.iterrows():
        cells = []
        for k in col_keys:
            txt = _tex_escape(_fmt_cell(row[k], fmt=float_fmt))
            is_bold = bool(bold_mask.loc[idx, k]) if (bold_mask is not None and k in bold_mask.columns) else False
            cells.append(rf'\textbf{{{txt}}}' if is_bold and txt != '—' else txt)
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    if caption or label:
        lines.append(r'\end{table}')
    return '\n'.join(lines) + '\n'


def write_stat_table(
    df: pd.DataFrame,
    stem: str | Path,
    columns: dict[str, str] | None = None,
    bold_mask: pd.DataFrame | None = None,
    float_fmt: str = '%.4f',
    tex: bool = False,
    caption: str | None = None,
    label: str | None = None,
) -> dict[str, Path]:
    """Atomically write ``<stem>.csv``, ``<stem>.md``, and optionally ``<stem>.tex``."""
    base = str(stem)
    for ext in ('.csv', '.md', '.tex'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    written = {}
    written['csv'] = atomic_to_csv(df, base + '.csv', index=False, float_format=float_fmt)
    written['md'] = atomic_write_text(base + '.md', format_stat_table_md(df, columns=columns, bold_mask=bold_mask, float_fmt=float_fmt))
    if tex:
        written['tex'] = atomic_write_text(
            base + '.tex',
            format_stat_table_tex(df, columns=columns, bold_mask=bold_mask, float_fmt=float_fmt, caption=caption, label=label),
        )
    return written
