"""Thin CLI & compatibility wrapper over ``analysis/scoring/resources.py`` (per-run resource report)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from resources import (  # noqa: E402, F401
    cli_report as main,
    find_runs,
    run_flags,
    summarise_run as summarise,
)

if __name__ == '__main__':
    main()
