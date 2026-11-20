#!/usr/bin/env python
"""Entrypoint for the Faure et al. (2022) & MegaScale epistasis SNR, block geometry, and ``cell_mag`` ceiling study.

Usage:
    python -m studies.faure_epistasis.run [--B 1000] [--fast] [--refresh]
"""
from __future__ import annotations

import argparse

from msr_analysis.paths import results_dir
from .steps import run_ceiling_and_significance, run_proximity, run_snr

STAGE = 'faure_epistasis'
DESCRIPTION = 'Faure et al. (DLG4/GRB2) vs MegaScale epistasis SNR, block geometry, and cell_mag noise-ceiling study.'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--B', type=int, default=1000, help='Bootstrap replicates (default: 1000)')
    ap.add_argument('--fast', action='store_true', help='Fast smoke mode (B = 100)')
    ap.add_argument('--refresh', action='store_true', help='Rebuild cached Faure double-distance pickles')
    ap.add_argument('--split', default='capped', choices=('capped', 'mega'), help='Accepted for runner uniformity')
    args = ap.parse_args(argv)
    B = 100 if args.fast else int(args.B)
    nperm = 50 if args.fast else 200

    out = results_dir('studies/faure_epistasis')
    run_proximity(out, refresh=args.refresh)
    run_snr(out, refresh=False)
    _c_df, sig_df, cmp_df = run_ceiling_and_significance(out, B=B, nperm=nperm)

    print(f'[faure_epistasis] B={B} complete -> {out}')
    print(sig_df[['set', 'stratum', 'blocks', 'cell_mag', 'block_boot_p', 'pos_boot_p', 'perm_p']].round(4).to_string(index=False))
    print(cmp_df[cmp_df['dataset'] == 'tsuboyama_capped_test'][['stratum', 'noise_model', 'blocks', 'ceiling_spearman', 'model_cell_mag']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
