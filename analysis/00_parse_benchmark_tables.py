#!/usr/bin/env python
"""Extract every markdown table of ``docs/BENCHMARKS_OCT2026.md`` into ``reference/benchmark_tables/<id>.csv``.

The document is the source of the numbers for the tables whose underlying predictions are not collected (the mutant-background
ablation, the epistasis hierarchy of the third-party models, the published-paper reference, the leak codes) and the yardstick
for the ones that are.  The CSVs are committed, so the figure scripts run without the document; rerun this script when
the document changes.

    python 00_parse_benchmark_tables.py [path/to/BENCHMARKS_OCT2026.md]

Table ids are ``<section>.<k>``: the section number of the nearest ``##``/``###`` heading and the table's order inside it.
``tables_index.csv`` lists them with their header, size, and the figure script that uses them (filled by hand in README).
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'reference' / 'benchmark_tables'
DEFAULT = '/home/sareeves/playground/esm-msr-devel/repo/.claude/worktrees/devel-main-conservative-merge-5e554f/docs/BENCHMARKS_OCT2026.md'


def split_row(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return cells


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('MSR_BENCHMARKS_MD', DEFAULT))
    lines = src.read_text().splitlines()
    OUT.mkdir(parents=True, exist_ok=True)
    sec, title, count = '0', '', {}
    i, index = 0, []
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^(#{2,4})\s+(\d+(?:\.\d+)?)?\.?\s*(.*)', ln)
        if m:
            if m.group(2):
                sec = m.group(2)
            title = m.group(3).strip()
        if ln.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|?$', lines[i + 1]):
            hdr = split_row(ln)
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].startswith('|'):
                r = split_row(lines[j])
                r = (r + [''] * len(hdr))[:len(hdr)]
                rows.append(r)
                j += 1
            count[sec] = count.get(sec, 0) + 1
            tid = f'{sec}.{count[sec]}'
            hdr = [h or f'col{k}' for k, h in enumerate(hdr)]
            pd.DataFrame(rows, columns=hdr).to_csv(OUT / f'T{tid}.csv', index=False)
            index.append(dict(table=tid, section=sec, heading=title, columns=' | '.join(hdr)[:160], rows=len(rows)))
            i = j
            continue
        i += 1
    pd.DataFrame(index).to_csv(OUT / 'tables_index.csv', index=False)
    print(f'{len(index)} tables from {src.name} -> {OUT}')
    for r in index:
        print(f"  T{r['table']:7s} {r['rows']:3d} rows  {r['heading'][:50]:50s} {r['columns'][:60]}")


if __name__ == '__main__':
    main()
