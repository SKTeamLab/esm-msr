"""Thin CLI & test-compatibility wrapper over ``analysis/scoring/score_runs.py`` (baseline directory scoring)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from score_runs import (  # noqa: E402, F401
    PG_DROP,
    cli_baseline as main,
    load_update_stats,
    pg_summary,
    score_baseline_dir,
    score_tsuboyama,
    update_stats_df,
)

if __name__ == '__main__':
    main()
