"""Shared PDB coordinate/distance/sequence parsers and mutation string parsers.

Consolidates the 5 duplicated PDB parsers and 4 duplicated colon/comma mutation parsers across
``faure_epistasis``, ``11g_link_figure``, ``12_pair_epistasis_heatmaps``, ``98_final_ablation_table``,
``tsuboyama/mega_splits``, and ``leak_audit``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

AA3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y',
    'VAL': 'V',
}

_MUT_TOKEN_RE = re.compile(r'^([A-Za-z\*])(-?\d+)([A-Za-z\*])$')


def parse_mutation_token(token: str) -> tuple[str, int, str] | None:
    """Parse a single substitution token such as ``'A12C'`` or ``'M-3V'`` into ``(wt_aa, pos, mut_aa)``."""
    m = _MUT_TOKEN_RE.match(str(token).strip())
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def parse_mutations(mut_str: str, sep: str | None = None) -> list[tuple[str, int, str]] | None:
    """Parse a colon- or comma-delimited mutation string into ``[(wt_aa, pos, mut_aa), ...]``.

    Returns ``[]`` for wild-type markers (``''``, ``'WT'``, ``'wt'``, ``'-'``) and ``None`` if any token is malformed.
    """
    s = str(mut_str).strip()
    if not s or s in ('WT', 'wt', '-'):
        return []
    parts = s.split(sep) if sep is not None else re.split(r'[:,]', s)
    out = []
    for part in parts:
        tok = parse_mutation_token(part)
        if tok is None:
            return None
        out.append(tok)
    return out


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Reconstruct ideal tetrahedral C_beta coordinates from backbone ``(N, CA, C)`` (for Glycine)."""
    b = ca - n
    c_vec = c - ca
    return -0.58273431 * np.cross(b, c_vec) + 0.56802827 * b - 0.54067466 * c_vec + ca


def load_pdb_residues(pdb_path: str | Path, chain: str | None = None) -> dict[int, dict]:
    """Parse ``ATOM`` records of ``pdb_path`` into ``{resseq: {'aa', 'ca', 'cb', 'heavy'}}``.

    - Filters to ``chain`` when given (otherwise uses the first chain encountered if multiple chains share residue numbers).
    - Keeps alternate conformer ``' '`` or ``'A'`` only.
    - Reconstructs virtual ``cb`` for Glycine from ``(N, CA, C)`` when available, falling back to ``ca``.
    """
    raw: dict[int, dict] = {}
    chosen_chain = chain
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            if not line.startswith('ATOM  '):
                continue
            alt = line[16]
            if alt not in (' ', 'A', '1'):
                continue
            ch = line[21].strip()
            if chosen_chain is None:
                chosen_chain = ch
            if ch != chosen_chain:
                continue
            icode = line[26].strip()
            if icode:
                continue
            resseq = int(line[22:26])
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ''
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            rec = raw.setdefault(resseq, {'aa': AA3_TO_1.get(resname, 'X'), 'atoms': {}, 'heavy': []})
            rec['atoms'][atom_name] = xyz
            if not atom_name.startswith('H') and element != 'H':
                rec['heavy'].append(xyz)

    out: dict[int, dict] = {}
    for k in sorted(raw):
        r = raw[k]
        atoms = r['atoms']
        ca = atoms.get('CA')
        cb = atoms.get('CB')
        if cb is None:
            if 'N' in atoms and 'CA' in atoms and 'C' in atoms:
                cb = _virtual_cb(atoms['N'], atoms['CA'], atoms['C'])
            elif ca is not None:
                cb = ca.copy()
        heavy = np.asarray(r['heavy'], dtype=float) if r['heavy'] else np.empty((0, 3), dtype=float)
        if ca is None and len(heavy):
            ca = heavy[0]
        if cb is None and ca is not None:
            cb = ca.copy()
        if ca is not None and cb is not None:
            out[k] = {'aa': r['aa'], 'ca': ca, 'cb': cb, 'heavy': heavy}
    return out


def pdb_sequence(residues: dict[int, dict]) -> tuple[str, list[int]]:
    """Return ``(one_letter_sequence, sorted_residue_numbers)`` from ``load_pdb_residues`` output."""
    ks = sorted(residues)
    return ''.join(residues[k]['aa'] for k in ks), ks


def align_domain_residues(pdb_path: str | Path, domain_seq: str, chain: str | None = None) -> tuple[dict[int, dict], int]:
    """Align ``domain_seq`` onto ``pdb_path`` and return ``({1..L: residue_dict}, pdb_start_resseq)``."""
    residues = load_pdb_residues(pdb_path, chain=chain)
    full_seq, ks = pdb_sequence(residues)
    clean_dom = domain_seq.rstrip('*')
    off = full_seq.find(clean_dom)
    if off < 0:
        raise ValueError(f'Domain sequence ({len(clean_dom)} aa) not found as substring in {pdb_path}')
    mapped = {p: residues[ks[off + p - 1]] for p in range(1, len(clean_dom) + 1)}
    return mapped, ks[off]


def cb_distance_matrix(residues: dict[int, dict], one_based_size: int | None = None) -> np.ndarray:
    """Compute pairwise C_beta-C_beta distance matrix.

    When ``one_based_size`` is given (or keys are ``1..L``), returns a ``(L+1, L+1)`` array indexed directly by
    1-based residue number (row/col 0 is NaN); otherwise returns a ``(N, N)`` matrix in sorted key order.
    """
    ks = sorted(residues)
    coords = np.vstack([residues[k]['cb'] for k in ks]) if ks else np.empty((0, 3))
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1) if len(ks) else np.empty((0, 0))
    if one_based_size is not None or (ks and ks == list(range(1, len(ks) + 1))):
        size = int(one_based_size or len(ks))
        out = np.full((size + 1, size + 1), np.nan, dtype=float)
        for ia, ka in enumerate(ks):
            if 1 <= ka <= size:
                for ib, kb in enumerate(ks):
                    if 1 <= kb <= size:
                        out[ka, kb] = dmat[ia, ib]
        return out
    return dmat


def heavy_atom_distance_matrix(residues: dict[int, dict], one_based_size: int | None = None) -> np.ndarray:
    """Compute pairwise minimum heavy-atom distance matrix."""
    ks = sorted(residues)
    n = len(ks)
    dmat = np.full((n, n), np.nan, dtype=float)
    for ia in range(n):
        dmat[ia, ia] = 0.0
        ha = residues[ks[ia]]['heavy']
        if not len(ha):
            continue
        for ib in range(ia + 1, n):
            hb = residues[ks[ib]]['heavy']
            if not len(hb):
                continue
            d = float(np.sqrt(((ha[:, None, :] - hb[None, :, :]) ** 2).sum(-1).min()))
            dmat[ia, ib] = dmat[ib, ia] = d
    if one_based_size is not None or (ks and ks == list(range(1, len(ks) + 1))):
        size = int(one_based_size or len(ks))
        out = np.full((size + 1, size + 1), np.nan, dtype=float)
        for ia, ka in enumerate(ks):
            if 1 <= ka <= size:
                for ib, kb in enumerate(ks):
                    if 1 <= kb <= size:
                        out[ka, kb] = dmat[ia, ib]
        return out
    return dmat


def pair_cb_distance(residues: dict[int, dict], pos_a: int, pos_b: int) -> float:
    """Return C_beta-C_beta distance (in Angstroms) between ``pos_a`` and ``pos_b``, or ``NaN`` if missing."""
    ra = residues.get(int(pos_a))
    rb = residues.get(int(pos_b))
    if ra is None or rb is None:
        return np.nan
    return float(np.linalg.norm(ra['cb'] - rb['cb']))
