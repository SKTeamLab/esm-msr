"""Numbered thematic analysis pipelines (`p01` through `p12`).

Every pipeline module exposes:
    STAGE: str
    DESCRIPTION: str
    main(argv: list[str] | None = None) -> int
"""
from __future__ import annotations

import argparse

FAST_B = 100
DEFAULT_B = 1000


def base_parser(
    stage: str,
    description: str,
    *,
    default_b: int = DEFAULT_B,
    with_split: bool = False,
    split_choices: tuple[str, ...] | None = None,
) -> argparse.ArgumentParser:
    """Create a standard CLI parser supporting ``--B`` and ``--fast`` (and optionally ``--split``)."""
    ap = argparse.ArgumentParser(prog=stage, description=description)
    ap.add_argument('--B', type=int, default=default_b, help=f'Bootstrap replicates (default: {default_b})')
    ap.add_argument('--fast', action='store_true', help=f'Smoke-test mode: override B to {FAST_B}')
    if with_split or split_choices is not None:
        choices = tuple(split_choices) if split_choices is not None else ('capped', 'mega')
        ap.add_argument(
            '--split',
            choices=choices,
            default=choices[0],
            help="Tsuboyama split ('capped' homology-restricted default, or 'mega' full MegaScale)",
        )
    return ap


def resolve_b(args: argparse.Namespace) -> int:
    """Return effective bootstrap count ``B`` after honouring ``--fast``."""
    return FAST_B if getattr(args, 'fast', False) else int(args.B)
