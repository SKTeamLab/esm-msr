"""Sequence (MMseqs2) and structure (Foldseek) homology detection for dataset splitting.

Sequence identity misses remote homologs: two domains can share a fold (and therefore much
of their mutational stability landscape) at <25% identity. This module runs an all-vs-all
Foldseek search over a set of single-chain structures and turns the hits into homology edges
and single-linkage (connected-component) clusters that the split generator can use alongside,
or instead of, MMseqs2 sequence clusters.

Foldseek is an external binary (https://github.com/steineggerlab/foldseek). Pass its path via
`foldseek_bin` or put it on PATH; a static build is available from
https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz.
"""
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from esm.utils.structure.protein_chain import ProteinChain


FOLDSEEK_COLUMNS = [
    'query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits', 'prob',
    'alntmscore', 'qtmscore', 'ttmscore', 'lddt',
]


MMSEQS_COLUMNS = ['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'evalue', 'bits']


def _run(cmd: List[str], tool: str) -> None:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{tool} failed ({result.returncode}):\n{result.stderr[-3000:]}")


def _check_binary(binary: str) -> None:
    if shutil.which(binary) is None and not os.path.exists(binary):
        raise FileNotFoundError(f"Binary '{binary}' not found. Install it or pass its path.")


def mmseqs_all_vs_all(fasta: str, work_dir: str, mmseqs_bin: str = 'mmseqs', threads: int = 4,
                      sensitivity: float = 7.5, evalue: float = 10.0) -> pd.DataFrame:
    """All-vs-all `mmseqs easy-search` at high sensitivity; returns directed hits without self hits.

    E-values are used rather than percent identity because identity over a local alignment is
    not length-aware: a 45-residue design reaches 30% identity against some 300-residue
    protein by chance far more often than a 45-residue alignment reaches a significant
    E-value.
    """
    _check_binary(mmseqs_bin)
    os.makedirs(work_dir, exist_ok=True)
    out_m8 = os.path.join(work_dir, 'mmseqs_all_vs_all.m8')
    _run([mmseqs_bin, 'easy-search', fasta, fasta, out_m8, os.path.join(work_dir, 'mmseqs_tmp'),
          '-s', str(sensitivity), '-e', str(evalue), '--max-seqs', '10000',
          '--format-output', ','.join(MMSEQS_COLUMNS), '--threads', str(threads)], 'MMseqs2')
    hits = pd.read_csv(out_m8, sep='\t', header=None, names=MMSEQS_COLUMNS)
    return hits.loc[hits['query'] != hits['target']].reset_index(drop=True)


def write_single_chain_pdbs(entries: Dict[str, Tuple[str, str]], out_dir: str, overwrite: bool = False) -> Dict[str, str]:
    """Extract one chain per entry into `out_dir/<name>.pdb` so Foldseek sees exactly one chain per id.

    entries: {name: (pdb_path, chain_id)}. Names become Foldseek ids, so they must not contain
    whitespace. Returns {name: written_path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, (pdb_path, chain) in entries.items():
        if any(c.isspace() for c in name):
            raise ValueError(f"Structure name '{name}' contains whitespace; Foldseek ids cannot.")
        out_path = os.path.join(out_dir, f'{name}.pdb')
        if overwrite or not os.path.exists(out_path):
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"Structure for '{name}' not found: {pdb_path}")
            ProteinChain.from_pdb(pdb_path, chain).to_pdb(out_path)
        written[name] = out_path
    return written


def foldseek_all_vs_all(pdb_dir: str, work_dir: str, foldseek_bin: str = 'foldseek', threads: int = 4,
                        evalue: float = 10.0, exhaustive: bool = True) -> pd.DataFrame:
    """Run `foldseek easy-search` of every structure in `pdb_dir` against every other one.

    Exhaustive search skips the k-mer prefilter, which matters for the many 40-70 residue
    domains in the mega-scale set: short queries produce few prefilter hits. The permissive
    E-value keeps weak hits so thresholds can be chosen afterwards from the returned table.
    Self hits are dropped. Returns one row per directed (query, target) hit.
    """
    _check_binary(foldseek_bin)
    os.makedirs(work_dir, exist_ok=True)
    out_m8 = os.path.join(work_dir, 'foldseek_all_vs_all.m8')
    tmp_dir = os.path.join(work_dir, 'foldseek_tmp')
    cmd = [
        foldseek_bin, 'easy-search', pdb_dir, pdb_dir, out_m8, tmp_dir,
        '--exhaustive-search', '1' if exhaustive else '0',
        '-e', str(evalue),
        '--format-output', ','.join(FOLDSEEK_COLUMNS),
        '--threads', str(threads),
    ]
    _run(cmd, 'Foldseek')

    hits = pd.read_csv(out_m8, sep='\t', header=None, names=FOLDSEEK_COLUMNS)
    # Foldseek ids are file names minus the extension
    for col in ['query', 'target']:
        hits[col] = hits[col].astype(str).str.replace(r'\.pdb$', '', regex=True)
    return hits.loc[hits['query'] != hits['target']].reset_index(drop=True)


def symmetrize_sequence_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Collapse directed MMseqs2 hits to one row per unordered pair (best E-value / identity)."""
    h = hits.copy()
    h['a'] = h[['query', 'target']].min(axis=1)
    h['b'] = h[['query', 'target']].max(axis=1)
    return h.groupby(['a', 'b'], as_index=False).agg(
        evalue=('evalue', 'min'), fident=('fident', 'max'), alnlen=('alnlen', 'max'), bits=('bits', 'max'))


def symmetrize_structure_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Collapse directed Foldseek hits to one row per unordered pair, keeping the strongest evidence.

    `tm_max` is the TM-score normalised by the shorter chain (max of the two normalisations),
    which flags a small domain embedded in a larger protein; `tm_min` is normalised by the
    longer chain and only rewards whole-chain similarity.
    """
    h = hits.copy()
    h['a'] = h[['query', 'target']].min(axis=1)
    h['b'] = h[['query', 'target']].max(axis=1)
    h['tm_max_dir'] = h[['qtmscore', 'ttmscore']].max(axis=1)
    h['tm_min_dir'] = h[['qtmscore', 'ttmscore']].min(axis=1)
    return h.groupby(['a', 'b'], as_index=False).agg(
        evalue=('evalue', 'min'), prob=('prob', 'max'), bits=('bits', 'max'), fident=('fident', 'max'),
        alnlen=('alnlen', 'max'), lddt=('lddt', 'max'),
        tm_max=('tm_max_dir', 'max'), tm_min=('tm_min_dir', 'max'),
    )


def homology_edges(pairs: pd.DataFrame, tm_threshold: Optional[float] = None, tm_norm: str = 'max',
                   evalue_threshold: Optional[float] = 1e-3, prob_threshold: Optional[float] = None,
                   require_all: bool = False) -> List[Tuple[str, str]]:
    """Select homologous pairs from `symmetrize_structure_hits` output.

    Each non-None criterion (TM-score >= tm_threshold, E-value <= evalue_threshold,
    prob >= prob_threshold) is evaluated; a pair is an edge if any criterion holds, or all of
    them when `require_all`. Using "any" is the conservative choice for leakage control.

    The default (Foldseek E-value <= 1e-3, no TM-score cut) is deliberate: the mega-scale
    domains are 40-72 residues and fall into a handful of folds, so TM-score >= 0.5 links
    most of them (fold-level similarity, not homology), while E-value <= 1e-3 produced no
    designed-vs-natural hits at all on that set.
    """
    if tm_norm not in ('max', 'min'):
        raise ValueError(f"tm_norm must be 'max' or 'min', got {tm_norm}")
    masks = []
    if tm_threshold is not None:
        masks.append(pairs[f'tm_{tm_norm}'] >= tm_threshold)
    if evalue_threshold is not None:
        masks.append(pairs['evalue'] <= evalue_threshold)
    if prob_threshold is not None:
        masks.append(pairs['prob'] >= prob_threshold)
    if not masks:
        raise ValueError("At least one homology criterion must be set.")
    mask = masks[0]
    for m in masks[1:]:
        mask = (mask & m) if require_all else (mask | m)
    sel = pairs.loc[mask]
    return list(zip(sel['a'], sel['b']))


def connected_component_clusters(ids: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Single-linkage clusters: {representative: sorted members}. Every id appears exactly once.

    Unlike MMseqs2/Foldseek greedy set-cover clustering, connected components guarantee that
    no edge crosses two clusters, so assigning whole clusters to splits leaves no detected
    homolog pair straddling a split boundary.
    """
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(e for e in edges if e[0] in g and e[1] in g)
    clusters = {}
    for comp in nx.connected_components(g):
        members = sorted(comp)
        clusters[members[0]] = members
    return clusters


def cross_set_hits(pairs: pd.DataFrame, membership: Dict[str, str]) -> pd.DataFrame:
    """Annotate symmetrized pairs with the set of each member and keep only cross-set pairs."""
    p = pairs.copy()
    p['set_a'] = p['a'].map(membership)
    p['set_b'] = p['b'].map(membership)
    p = p.dropna(subset=['set_a', 'set_b'])
    return p.loc[p['set_a'] != p['set_b']].reset_index(drop=True)
