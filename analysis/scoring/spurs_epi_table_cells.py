"""The SPURS multi cells of the document's epistasis tables 2.1 / 2.2 (naive rho, beyond rho, pair, line, cell_mag, cell_rank, flipacc, global_err):
the training code's own suite (esm_msr.epi_metrics.compute) on SPURS multi(AB) minus the same seed's SPURS singles, mean over seeds.

  python analysis/scoring/spurs_epi_table_cells.py [--frames DIR]     # DIR holds frame_tsuboyama_capped_{val,test}.pkl (default: analysis/cache)
"""
import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'src'))
from esm_msr import epi_metrics as E  # noqa: E402

spec = importlib.util.spec_from_file_location('d11', os.path.join(ROOT, '11d_epistasis_levels_bootstrap.py'))
d11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d11)
COLS = ['naive_rho', 'beyond_rho', 'pair_rho', 'line_rho', 'cell_mag', 'cell_rank', 'cell_flipacc', 'global_err']


def cells(fr, direct='spursm', add='spurs'):
    sp, p = fr.spine, fr.preds
    out = []
    for s in range(p[direct].shape[0]):
        t = d11.make_table(sp, p[add][min(s, p[add].shape[0] - 1)], p[direct][s], ('epi', 'add'))
        r = E.compute(t)
        out.append([r[f'epi_{c}_epi'] for c in COLS])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=os.path.join(ROOT, 'cache'))
    a = ap.parse_args()
    print('split  ' + '  '.join(f'{c:>12s}' for c in COLS))
    for split in ('val', 'test'):
        with open(os.path.join(a.frames, f'frame_tsuboyama_capped_{split}.pkl'), 'rb') as fh:
            fr = pickle.load(fh)
        v = cells(fr)
        print(f'{split:5s}  ' + '  '.join(f'{x:12.3f}' for x in v.mean(0)) + f'   (seed SD of beyond_rho {v[:, 1].std(ddof=1):.3f}, n_seeds {len(v)})')


if __name__ == '__main__':
    main()
