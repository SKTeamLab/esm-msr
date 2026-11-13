"""Shared helpers and path configuration for the leak audit.

Canonical conventions (used everywhere):
  protein   : an object with a unique sequence (``seq``) and a string id (``pid``).
  mutation  : ``(wt, pos, mt)`` with 1-based ``pos`` into that protein's ``seq``.
  variant   : ``(protein_seq, tuple(sorted(mutations)))``; two variants are identical
              when the protein sequences are identical and the (wt,pos,mt) tuples agree.
"""
import hashlib
import os
import re
import shutil
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get('REPO_ROOT', os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))


def _default_leak_root() -> str:
    """The directory that holds ``leak_audit/data`` (and ``ProteinGym/``, ``remote_data/``): the parent of the repo
    checkout. Inside a git worktree (``<repo>/.claude/worktrees/<name>``) the checkout's parent is not that directory,
    so walk up until a ``leak_audit/data`` directory is found."""
    d = os.path.abspath(os.path.join(REPO_ROOT, '..'))
    while True:
        if os.path.isdir(os.path.join(d, 'leak_audit', 'data')):
            return d
        up = os.path.dirname(d)
        if up == d:
            return os.path.abspath(os.path.join(REPO_ROOT, '..'))
        d = up


LEAK_ROOT = os.environ.get('LEAK_ROOT') or _default_leak_root()
REPO_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


def repo_data(*parts: str) -> str:
    """Path under the repository's ``data/``. Untracked inputs (e.g. ``data/domainome1/``) exist only in the main
    checkout, not in a git worktree, so fall back to ``$LEAK_ROOT/repo/data`` when the path is missing here."""
    p = os.path.join(REPO_ROOT, 'data', *parts)
    if not os.path.exists(p):
        alt = os.path.join(LEAK_ROOT, 'repo', 'data', *parts)
        if os.path.exists(alt):
            return alt
    return p
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
THREADS = os.environ.get('THREADS', '8')
# Test sets that are row subsets of another test set (written by a later step); never added to the protein corpus
DERIVED_TESTSETS = {'domainome_low_id'}
FOLDSEEK_BIN = _find_foldseek()

MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')


def data_file(*rel_parts: str) -> str:
    """Resolve a small/raw reference input file, preferring the repo's ``analysis/leak_audit/data/``
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


# ---------------------------------------------------------------- shared audit rules
def expand_lib_pids(lib: pd.DataFrame, scaffolds) -> set:
    """All Megascale library pids (``lib:<code>``) of the given scaffolds, mutant-background libraries included.

    ``lib`` is ``megascale_libraries.pkl`` (index ``code``, column ``code_wt``); a scaffold matches a library whose
    ``code`` or ``code_wt`` equals it.
    """
    sc = set(scaffolds)
    out = {'lib:' + str(c) for c, cwt in lib['code_wt'].items() if c in sc or cwt in sc}
    out |= {'lib:' + str(c) for c in sc if c in lib.index}
    return out


def make_canon(maps: dict):
    """Return ``canon(pid, wt, pos, mt) -> (family, rep_pos, wt, mt) | None`` for the family maps of ``08_families.py``.

    A mutation is mapped to its family representative's coordinate only when the aligned representative residue has
    the same wild-type letter; otherwise it is unmapped (no positional search), so a row is never matched through a
    residue it does not carry.
    """
    cache = {}

    def canon(pid, wt, pos, mt):
        key = (pid, pos, wt)
        if key not in cache:
            fam, mp, rs = maps[pid]
            rp = mp.get(pos)
            if rp is not None and not (1 <= rp <= len(rs) and rs[rp - 1] == wt):
                rp = None
            cache[key] = rp
        rp = cache[key]
        return None if rp is None else (maps[pid][0], rp, wt, mt)

    return canon


# Protein-level relation of a test protein to a training inventory (one rule for every step that classifies homology).
# Same protein: same >=95%-identity family (08_families.py) or an MMseqs2 hit with fident >= 0.90 over >= 50% of the
# test protein.  Strong homolog H: MMseqs2 E <= 1e-3 over >= 50%, or Foldseek E <= 1e-3 / qTM > 0.70 over >= 50%.
# Weak h: MMseqs2 E <= 1e-2 or fident >= 0.30 over >= 30%, or Foldseek E <= 1e-2 over >= 30%.
SAMEPROT_FIDENT, SAMEPROT_COV = 0.90, 0.50
H_COV, H_EVALUE, H_QTM = 0.50, 1e-3, 0.70
WEAK_COV, WEAK_EVALUE, WEAK_FIDENT = 0.30, 1e-2, 0.30


def classify_hits(seq_hits, struct_hits=()):
    """Classify one test protein from its hits into the inventory.

    ``seq_hits``: iterable of ``(fident, cov_test, evalue)``; ``struct_hits``: iterable of ``(struct_evalue, cov_test,
    qtm)``. Every hit is evaluated (no E-value shadowing). Returns ``'sameprot' | 'H' | 'h' | 'low' | 'none'``.
    """
    has_any = is_H = is_h = False
    for f, cov, e in seq_hits:
        has_any = True
        if f >= SAMEPROT_FIDENT and cov >= SAMEPROT_COV:
            return 'sameprot'
        if cov >= H_COV and e <= H_EVALUE:
            is_H = True
        elif cov >= WEAK_COV and (e <= WEAK_EVALUE or f >= WEAK_FIDENT):
            is_h = True
    for e, cov, qtm in struct_hits:
        has_any = True
        if cov >= H_COV and (e <= H_EVALUE or qtm > H_QTM):
            is_H = True
        elif cov >= WEAK_COV and e <= WEAK_EVALUE:
            is_h = True
    if is_H:
        return 'H'
    if is_h:
        return 'h'
    return 'low' if has_any else 'none'


def load_testsets(skip_prefix=('tsu_hyperopt',)):
    """``{dataset: DataFrame}`` of the canonical test tables written by ``04_testsets.py``."""
    tdir = os.path.join(WORK_DATA_DIR, 'testsets')
    out = {}
    for f in sorted(os.listdir(tdir)):
        if f.endswith('.pkl') and not f[:-4].startswith(tuple(skip_prefix)):
            out[f[:-4]] = pd.read_pickle(os.path.join(tdir, f))
    return out
