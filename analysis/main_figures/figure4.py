#!/usr/bin/env python3
"""Build Main-Text Figure 4 (`analysis/results/00_main_text_figs/figure4.{png,pdf}`).

Thin CLI wrapper around `pipelines.p09_screening`'s `--only figure4` builder so all
main-text figure scripts live together under `analysis/main_figures/`.

Usage:
    PYTHONPATH=src:analysis python -m analysis.main_figures.figure4
"""
from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines import p09_screening


def main(argv: list[str] | None = None) -> int:
    extra = list(argv) if argv is not None else sys.argv[1:]
    return p09_screening.main(['--only', 'figure4', *extra])


if __name__ == '__main__':
    raise SystemExit(main())
