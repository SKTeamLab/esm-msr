"""Shared helpers for the leak audit.

Canonical conventions (used everywhere):
  protein   : an object with a unique sequence (``seq``) and a string id (``pid``).
  mutation  : ``(wt, pos, mt)`` with 1-based ``pos`` into that protein's ``seq``.
  variant   : ``(protein_seq, tuple(sorted(mutations)))``; two variants are identical
              when the protein sequences are identical and the (wt,pos,mt) tuples agree.
"""
import re, hashlib, json, os
import pandas as pd, numpy as np

MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')

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
            # allow forms like 'p.Ala11Cys'? not used in this repo
            return None
        out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out

def check_against_seq(seq: str, muts):
    """Return (ok, n_bad). Validates wt letter at each 1-based position."""
    bad = 0
    for wt, pos, mt in muts:
        if pos < 1 or pos > len(seq) or seq[pos-1] != wt:
            bad += 1
    return bad == 0, bad

def canon_variant(seq: str, muts) -> str:
    return hash_seq(seq) + '|' + ';'.join(f'{w}{p}{m}' for w, p, m in sorted(muts, key=lambda x: x[1]))

def seq_id(a: str, b: str) -> float:
    """Cheap identity for equal-length or nearly equal-length sequences (no alignment).

    Only used for quick checks; the audit's identity numbers come from MMseqs2/Biopython.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(1 for x, y in zip(a[:n], b[:n]) if x == y) / n

def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
