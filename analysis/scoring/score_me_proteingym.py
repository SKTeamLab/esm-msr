"""Thin CLI & test-compatibility wrapper over ``analysis/scoring/score_runs.py`` (MutateEverything ProteinGym scoring)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from score_runs import (  # noqa: E402, F401
    CATS,
    PG_DROP,
    cli_me_pgym as main,
    rho,
    run_me_pgym,
    score_me_assay as score_assay,
    summarise_me_pgym as summarise,
)

if __name__ == '__main__':
    main()
