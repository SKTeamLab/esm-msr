"""Thin CLI & compatibility wrapper over ``analysis/scoring/resources.py`` (cross-model slot summary)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from resources import (  # noqa: E402, F401
    BEGIN,
    DOCS,
    DOC_ROWS,
    END,
    EXT,
    GROUPS,
    LABELS,
    OUT,
    PENDING,
    ROOT,
    cli_summary as main,
    docs_table,
    group_of,
    load_label,
    markdown,
    run_flags,
    summarise_tracked as summarise,
)

if __name__ == '__main__':
    main()
