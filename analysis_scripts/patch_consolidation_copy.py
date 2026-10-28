"""Bring the consolidation working copy of `benchmark_matrix.py` in line with the renamed artefacts.

`~/consolidation/benchmark_matrix.py` is the script that actually renders the doc's §1 tables. It is a fork of
`analysis_scripts/benchmark_matrix.py`, not a symlink, and the repo version is not drop-in compatible (its table
logic indexes different column names). This patch changes only what is needed:

1. `_glob_any` — accept both naming generations (`_epsilon` before commit 7a5b283, `_sigma` after) and try the
   tokens in a fixed order. The order matters: `_sigma1.0_` sorts before `_sigma1_` lexicographically and the two
   spellings hold different data (139 of 156 pairs differ on the local copy), so an unordered glob reads the wrong
   file. Callers' spelling ('1') is tried before '%g' and before the float repr ('1.0').
2. The comparison rows point at `release-baseline/s{1,2,3}`, the three-seed baseline triplicate. `REL_CK` is kept
   for the sigma=0 zero-shot control, which was run from the original released checkpoint.

Idempotent: re-running on an already-patched copy is a no-op. Operates on the file in place; keep a backup.
"""
import argparse
import pathlib
import re

HELPER = '''def _glob_any(base, eps, tail):
    """Stats/predictions file under either naming generation: _epsilon (before commit 7a5b283) or _sigma (after).
    Tokens are tried in a fixed order -- the caller's spelling first, then %g, then the float repr -- so a
    '_sigma1_' file always wins over '_sigma1.0_'. The two spellings hold different data, so the order matters."""
    try:
        v = float(eps)
        # float repr first: '_sigma1.0_' is the corrected per-mutation-bias generation, '_sigma1_' the superseded
        # per-item one, so the newer spelling must win when both are present.
        toks = [repr(v), '%g' % v, str(eps)]
    except (TypeError, ValueError):
        toks = [str(eps)]
    seen, ordered = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    for pre in ('sigma', 'epsilon'):
        for t in ordered:
            hits = sorted(glob.glob(f'{base}/*_{pre}{t}_unmasked_{tail}'))
            if hits:
                return hits
    return []


'''

GLOB_SUBS = [
    ("f = glob.glob(f'{DEV}/analysis_notebooks/predictions/{sub}/{ck}/*_epsilon{eps}_unmasked_predictions.csv')",
     "f = _glob_any(f'{DEV}/analysis_notebooks/predictions/{sub}/{ck}', eps, 'predictions.csv')"),
    ("f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}.csv')",
     "f = _glob_any(f'{stats}/{sub}/{ck}', eps, f'{head}.csv')"),
    ("f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}_avg.csv')",
     "f = _glob_any(f'{stats}/{sub}/{ck}', eps, f'{head}_avg.csv')"),
    ("f = glob.glob(f'{stats}/splits_oct06_capped-{split}/{ck}/*_epsilon1_unmasked_{head}_avg.csv')",
     "f = _glob_any(f'{stats}/splits_oct06_capped-{split}/{ck}', 1, f'{head}_avg.csv')"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='/home/sreeves/consolidation/benchmark_matrix.py')
    a = ap.parse_args()
    p = pathlib.Path(a.path)
    s = p.read_text()

    if 'def _glob_any(' not in s:
        anchor = 'def bias_convention('
        assert anchor in s, 'anchor not found'
        s = s.replace(anchor, HELPER + anchor, 1)

    for old, new in GLOB_SUBS:
        if old in s:
            s = s.replace(old, new, 1)

    if 'BL_CK' not in s:
        m = re.search(r"^REL_CK = \[.*\]$", s, re.M)
        assert m, 'REL_CK not found'
        s = s.replace(m.group(0),
                      "BL_CK = [f'release-baseline/s{i}' for i in (1, 2, 3)]\n"
                      "REL_CK = ['esm-msr-baseline-w2a4-m16a16-capped']   # sigma=0 zero-shot control only", 1)
        s = s.replace('collect_msr(rows, REL_CK, REL)', 'collect_msr(rows, BL_CK, REL)', 1)

    p.write_text(s)
    print(f'patched {p}: _glob_any={"_glob_any" in s} BL_CK={"BL_CK" in s} '
          f'triplicate_rows={"collect_msr(rows, BL_CK, REL)" in s}')


if __name__ == '__main__':
    main()
