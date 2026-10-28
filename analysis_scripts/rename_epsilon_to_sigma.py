"""Rename the LoRA scaling-factor token in ESM-MSR artefact filenames: _epsilon{eps}_ -> _sigma{sigma}_.

Commit 7a5b283 renamed the scaling factor epsilon -> sigma repo-wide and the analysis scripts now glob `_sigma`,
but every artefact produced before it is named `_epsilon`. This renames them in place.

Digits are **not** normalised. Two spellings of the unit case exist on disk (`_epsilon1` and `_epsilon1.0`) and they
are not duplicates: on the local copy of the tree, 139 of the 156 pairs that a `_epsilon1.0_` -> `_sigma1_`
normalisation would have merged differed in content. Mapping `_epsilon1.0_` to `_sigma1.0_` keeps both. Whoever
globs these files must then accept both spellings and try the exact one first -- `_sigma1.0_` sorts before
`_sigma1_` lexicographically, so an unordered glob silently reads the wrong file.

Usage:
    python analysis_scripts/rename_epsilon_to_sigma.py [--root DIR] [--apply]

Dry-run by default. Writes `<root>/../rename_sigma_to_epsilon.manifest.tsv` (or next to DIR) with one
`new<TAB>old` line per file so the rename can be undone exactly.
"""
import argparse
import os
import sys

DEFAULT_ROOT = '/home/sreeves/esm-msr-devel/analysis_notebooks'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    manifest = os.path.join(os.path.dirname(os.path.abspath(a.root)), 'rename_sigma_to_epsilon.manifest.tsv')

    rows, conflicts = [], 0
    for top in ('predictions', 'stats'):
        for dirpath, _, files in os.walk(os.path.join(a.root, top)):
            for f in files:
                if '_epsilon' not in f:
                    continue
                dst = os.path.join(dirpath, f.replace('_epsilon', '_sigma'))
                if os.path.exists(dst):
                    conflicts += 1
                    print('CONFLICT', dst)
                    continue
                rows.append((os.path.join(dirpath, f), dst))

    if a.apply:
        with open(manifest, 'w') as fh:
            for src, dst in rows:
                os.rename(src, dst)
                fh.write(f'{dst}\t{src}\n')
    print(f'{"renamed" if a.apply else "planned"}={len(rows)} conflicts={conflicts} manifest={manifest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
