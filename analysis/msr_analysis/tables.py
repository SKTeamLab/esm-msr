"""Read the tables that ``00_parse_benchmark_tables.py`` extracted from BENCHMARKS_OCT2026.md."""
import re

import numpy as np
import pandas as pd

from .registry import REFERENCE

TABLES = REFERENCE / 'benchmark_tables'
_SUP = 'ᴹᵐᴾᴴʰᴰᶜᵁ#*†‡'


def read_table(tid):
    return pd.read_csv(TABLES / f'T{tid}.csv', dtype=str).fillna('')


def num(cell):
    """'**0.654±0.003**ʰ' -> (0.654, 0.003); '–', 'q', '' -> (nan, nan)."""
    s = str(cell).replace('**', '').replace('_', '').strip()
    s = ''.join(ch for ch in s if ch not in _SUP).strip()
    m = re.match(r'^(-?\d*\.?\d+(?:e-?\d+)?)\s*(?:±\s*(\d*\.?\d+))?', s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2)) if m.group(2) else np.nan


def numeric(df, first_col=True):
    """value and sd frames of a table whose first column names the rows"""
    rows = df.iloc[:, 0].tolist()
    V = pd.DataFrame({c: [num(x)[0] for x in df[c]] for c in df.columns[1:]}, index=rows)
    S = pd.DataFrame({c: [num(x)[1] for x in df[c]] for c in df.columns[1:]}, index=rows)
    return V, S


def clean_label(s):
    s = re.sub(r'[ᴹᵐᴾᴴʰᴰᶜᵁ‡]', '', s)
    return s.replace('  ', ' ').strip()
