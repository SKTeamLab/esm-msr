#!/usr/bin/env python
"""Entrypoint for the ``FINAL_p0`` Mechanistic ``σ`` Trade-Off Study (`studies/sigma_mechanisms/`).

Usage:
    python -m studies.sigma_mechanisms.run [--B 1000] [--fast] [--refresh]
"""
from __future__ import annotations

import argparse

from msr_analysis.paths import atomic_to_csv, results_dir
from .figures import draw_aa_category_heatmaps, draw_sigma_mechanisms_composite
from .metrics import compute_sigma_mechanisms_cache, summarize_mechanisms

STAGE = 'sigma_mechanisms'
DESCRIPTION = 'Mechanistic decomposition of LoRA scale σ ∈ {0, 0.5, 1} on FINAL_p0 across 173 held-out ProteinGym assays.'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--B', type=int, default=1000, help='Bootstrap replicates (default: 1000)')
    ap.add_argument('--fast', action='store_true', help='Fast smoke mode (B = 100)')
    ap.add_argument('--refresh', action='store_true', help='Rebuild cached per-assay mechanistic matrices')
    ap.add_argument('--split', default='capped', choices=('capped', 'mega'), help='Accepted for runner uniformity')
    args = ap.parse_args(argv)
    B = 100 if args.fast else int(args.B)

    out = results_dir('studies/sigma_mechanisms')
    payload = compute_sigma_mechanisms_cache(refresh=args.refresh)
    site_df, aa_df, tail_df, pareto_df = summarize_mechanisms(payload, B=B)

    atomic_to_csv(site_df, out / 'site_vs_sub_decomposition.csv', index=False, float_format='%.4f')
    atomic_to_csv(aa_df, out / 'aa_rank_shifts.csv', index=False, float_format='%.4f')
    atomic_to_csv(tail_df, out / 'tail_selection_vs_filtering.csv', index=False, float_format='%.4f')
    atomic_to_csv(pareto_df, out / 'pareto_multi_property.csv', index=False, float_format='%.4f')

    n_heldout = len(payload['dms_ids'])
    draw_sigma_mechanisms_composite(out, site_df, aa_df, tail_df, pareto_df, n_heldout=n_heldout, B=B)
    draw_aa_category_heatmaps(out, aa_df, site_df)

    print(f'[sigma_mechanisms] B={B} complete on {n_heldout} held-out assays -> {out}')
    print(site_df[site_df['sigma'] == 1.0][['category', 'component', 'n_assays', 'delta', 'lo', 'hi', 'p_holm']].round(4).to_string(index=False))
    print(tail_df[tail_df['sigma'] == 1.0][['category', 'n_assays', 'bot10_filtering_gain', 'top10_selection_gain', 'top10_wcfim_shift']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
