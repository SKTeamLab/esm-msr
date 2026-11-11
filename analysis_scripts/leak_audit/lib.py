"""Shared helpers and path configuration for the leak audit.

Canonical conventions (used everywhere):
  protein   : an object with a unique sequence (``seq``) and a string id (``pid``).
  mutation  : ``(wt, pos, mt)`` with 1-based ``pos`` into that protein's ``seq``.
  variant   : ``(protein_seq, tuple(sorted(mutations)))``; two variants are identical
              when the protein sequences are identical and the (wt,pos,mt) tuples agree.
"""
import hashlib
import json
import os
import re
import shutil
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get('REPO_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))
LEAK_ROOT = os.environ.get('LEAK_ROOT', os.path.abspath(os.path.join(REPO_ROOT, '..')))
REPO_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
WORK_DATA_DIR = os.environ.get('LEAK_DATA_DIR', os.path.join(LEAK_ROOT, 'leak_audit', 'data'))
OUT_DIR = os.environ.get('LEAK_OUT_DIR', os.path.join(LEAK_ROOT, 'leak_audit', 'out'))
SOFTWARE_DIR = os.environ.get('SOFTWARE_DIR', os.path.expanduser('~/software'))

def _find_foldseek() -> str:
    if os.environ.get('FOLDSEEK_BIN'):
        return os.environ['FOLDSEEK_BIN']
    for cand in (
        os.path.join(WORK_DATA_DIR, 'foldseek', 'bin', 'foldseek'),
        os.path.join(WORK_DATA_DIR, 'foldseek', 'foldseek', 'bin', 'foldseek'),
        shutil.which('foldseek'),
        os.path.join(SOFTWARE_DIR, 'SaProt', 'bin', 'foldseek'),
    ):
        if cand and os.path.exists(cand):
            return cand
    return 'foldseek'


MMSEQS_BIN = os.environ.get('MMSEQS_BIN', shutil.which('mmseqs') or '/usr/bin/mmseqs')
FOLDSEEK_BIN = _find_foldseek()

MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')


def data_file(*rel_parts: str) -> str:
    """Resolve a small/raw reference input file, preferring the repo's ``analysis_scripts/leak_audit/data/``
    and falling back to ``$LEAK_DATA_DIR`` / ``$LEAK_DATA_DIR/raw`` for large unversioned corpora."""
    base = rel_parts[-1]
    candidates = [
        os.path.join(REPO_DATA_DIR, *rel_parts),
        os.path.join(REPO_DATA_DIR, base),
        os.path.join(WORK_DATA_DIR, *rel_parts),
        os.path.join(WORK_DATA_DIR, base),
        os.path.join(WORK_DATA_DIR, 'raw', base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def norm_wt_name(wt_name: str):
    """Normalize a raw MegaScale ``WT_name`` into ``(code, code_wt, bg_mut)``.

    Examples:
      ``'1UBQ.pdb'``       -> ``('1UBQ', '1UBQ', None)``
      ``'2MKX.pdb_R27A'``  -> ``('2MKX_R27A', '2MKX', 'R27A')``
      ``'EA|run2.pdb'``    -> ``('EA_run2', 'EA_run2', None)``
    """
    s = str(wt_name).strip()
    if '.pdb_' in s:
        base, bg = s.split('.pdb_', 1)
        code_wt = base.replace('|', '_')
        return f'{code_wt}_{bg}', code_wt, bg
    if s.endswith('.pdb'):
        code_wt = s[:-4].replace('|', '_')
        return code_wt, code_wt, None
    code_wt = s.replace('|', '_')
    return code_wt, code_wt, None


def hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def parse_mut_str(s: str):
    """Parse 'A11C:G12D' (also ';' or ',' separated) into [(wt,pos,mt), ...].

    Returns None if any component is not a plain substitution in the expected format.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    if s in ('', 'wt', 'WT', 'nan', 'None', '-'):
        return []  # wild type row
    parts = re.split(r'[:,;]', s)
    out = []
    for p in parts:
        p = p.strip()
        m = MUT_RE.match(p)
        if not m:
            return None
        out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def check_against_seq(seq: str, muts):
    """Return (ok, n_bad). Validates wt letter at each 1-based position."""
    bad = 0
    for wt, pos, mt in muts:
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
            bad += 1
    return bad == 0, bad


def apply_mutations(seq: str, muts) -> str:
    """Return the mutant sequence after applying validated 1-based ``(wt, pos, mt)`` substitutions."""
    arr = list(seq)
    for wt, pos, mt in muts:
        arr[pos - 1] = mt
    return ''.join(arr)


def canon_variant(seq: str, muts) -> str:
    return hash_seq(seq) + '|' + ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1]))


def seq_id(a: str, b: str) -> float:
    """Cheap positional identity for equal-length or nearly equal-length sequences (no alignment).

    Only used for quick sanity checks; the audit's identity numbers come from MMseqs2/Biopython.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
