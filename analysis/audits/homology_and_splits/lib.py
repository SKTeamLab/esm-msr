"""Shared path configuration, mutation normalization, and hit-classification helpers for homology & leakage audits.

Delegates path resolution to ``msr_analysis.paths`` and PDB / mutation parsing to ``msr_analysis.structures``.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from msr_analysis.paths import DATA_DIR, MSR_ROOT, REFERENCE, REPO_ROOT as _REPO_ROOT_PATH
from msr_analysis.structures import AA3_TO_1, load_pdb_residues, parse_mutations, pdb_sequence

HERE = Path(__file__).resolve().parent
SCRIPT_DIR = str(HERE)
REPO_ROOT = os.environ.get('REPO_ROOT', str(_REPO_ROOT_PATH))
LEAK_ROOT = os.environ.get('LEAK_ROOT', str(MSR_ROOT))
REPO_DATA_DIR = str(HERE / 'data')
WORK_DATA_DIR = os.environ.get('LEAK_DATA_DIR', os.path.join(LEAK_ROOT, 'leak_audit', 'data'))
OUT_DIR = os.environ.get('LEAK_OUT_DIR', os.path.join(LEAK_ROOT, 'leak_audit', 'out'))
SOFTWARE_DIR = os.environ.get('SOFTWARE_DIR', os.path.expanduser('~/software'))
REFERENCE_DIR = str(REFERENCE)
PROVENANCE_DIR = HERE / 'provenance'


def repo_data(*parts: str) -> str:
    """Resolve a path under ``DATA_DIR`` (with fallback to ``$LEAK_ROOT/repo/data`` for untracked raw inputs)."""
    p = DATA_DIR.joinpath(*parts)
    if p.exists():
        return str(p)
    alt = Path(LEAK_ROOT).joinpath('repo', 'data', *parts)
    return str(alt) if alt.exists() else str(p)


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
THREADS = os.environ.get('THREADS', '8')
DERIVED_TESTSETS = {'domainome_low_id'}


def data_file(*rel_parts: str) -> str:
    """Resolve a small/raw reference input file in ``audits/homology_and_splits/data/`` or ``$LEAK_DATA_DIR``."""
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


def pdb_ca_sequence(pdb_path: str | Path, chain: str | None = None) -> str:
    """Extract 1-letter C-alpha amino acid sequence from ``pdb_path`` via ``msr_analysis.structures``."""
    seq, _ = pdb_sequence(load_pdb_residues(pdb_path, chain=chain))
    return seq


def norm_wt_name(wt_name: str) -> tuple[str, str, str | None]:
    """Normalize a raw MegaScale ``WT_name`` into ``(code, code_wt, bg_mut)``."""
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


def parse_mut_str(s: str) -> list[tuple[str, int, str]] | None:
    """Parse ``'A11C:G12D'`` (also ``';'`` or ``','`` separated) into ``[(wt, pos, mt), ...]`` using ``msr_analysis.structures``."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    txt = str(s).strip()
    if txt in ('', 'wt', 'WT', 'nan', 'None', '-'):
        return []
    toks = parse_mutations(txt.replace(';', ':'))
    if toks is None:
        return None
    # Oct-2026 audit restricts to canonical positive 1-based single-letter amino-acid substitutions
    for wt, pos, mt in toks:
        if pos < 1 or wt == '*' or mt == '*':
            return None
    return toks


def check_against_seq(seq: str, muts) -> tuple[bool, int]:
    bad = 0
    for wt, pos, _mt in muts:
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt:
            bad += 1
    return bad == 0, bad


def apply_mutations(seq: str, muts) -> str:
    arr = list(seq)
    for _wt, pos, mt in muts:
        arr[pos - 1] = mt
    return ''.join(arr)


def expand_lib_pids(lib: pd.DataFrame, scaffolds) -> set[str]:
    sc = set(scaffolds)
    out = {'lib:' + str(c) for c, cwt in lib['code_wt'].items() if c in sc or cwt in sc}
    out |= {'lib:' + str(c) for c in sc if c in lib.index}
    return out


def make_canon(maps: dict):
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


# Oct-2026 coverage-gated classification rule (used ONLY by oct2026_leak_audit/ steps 01-20;
# active holdout filtering uses the 4-part Tsuboyama split rule in active_builders/tsuboyama_rule_flags.py).
SAMEPROT_FIDENT, SAMEPROT_COV = 0.90, 0.50
H_COV, H_EVALUE, H_QTM = 0.50, 1e-3, 0.70
WEAK_COV, WEAK_EVALUE, WEAK_FIDENT = 0.30, 1e-2, 0.30


def classify_hits(seq_hits, struct_hits=()) -> str:
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


def load_testsets(skip_prefix=('tsu_hyperopt',)) -> dict[str, pd.DataFrame]:
    tdir = os.path.join(WORK_DATA_DIR, 'testsets')
    out = {}
    for f in sorted(os.listdir(tdir)):
        if f.endswith('.pkl') and not f[:-4].startswith(tuple(skip_prefix)):
            out[f[:-4]] = pd.read_pickle(os.path.join(tdir, f))
    return out


__all__ = [
    'AA3_TO_1', 'DERIVED_TESTSETS', 'FOLDSEEK_BIN', 'HERE', 'H_COV', 'H_EVALUE', 'H_QTM',
    'LEAK_ROOT', 'MMSEQS_BIN', 'OUT_DIR', 'PROVENANCE_DIR', 'REFERENCE_DIR', 'REPO_DATA_DIR',
    'REPO_ROOT', 'SAMEPROT_COV', 'SAMEPROT_FIDENT', 'SCRIPT_DIR', 'SOFTWARE_DIR', 'THREADS',
    'WEAK_COV', 'WEAK_EVALUE', 'WEAK_FIDENT', 'WORK_DATA_DIR', 'apply_mutations', 'check_against_seq',
    'classify_hits', 'data_file', 'expand_lib_pids', 'hash_seq', 'load_testsets', 'make_canon',
    'norm_wt_name', 'parse_mut_str', 'pdb_ca_sequence', 'repo_data',
]
