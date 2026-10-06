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
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from Bio import Align

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


_ID_SEQS: Dict[str, str] = {}
_ID_ALIGNER = None


def _init_identity_worker(sequences: Dict[str, str]) -> None:
    global _ID_SEQS, _ID_ALIGNER
    _ID_SEQS = sequences
    _ID_ALIGNER = Align.PairwiseAligner()
    _ID_ALIGNER.mode = 'local'
    _ID_ALIGNER.open_gap_score = -10
    _ID_ALIGNER.extend_gap_score = -1
    _ID_ALIGNER.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')


def _local_identity(pair: Tuple[str, str]) -> Tuple[str, str, float]:
    a, b = pair
    s1, s2 = _ID_SEQS[a], _ID_SEQS[b]
    if not s1 or not s2:
        return a, b, 0.0
    try:
        aln = next(iter(_ID_ALIGNER.align(s1, s2)))
    except StopIteration:
        return a, b, 0.0
    matches = sum(1 for x, y in zip(aln[0], aln[1]) if x == y and x != '-')
    return a, b, matches / min(len(s1), len(s2))


def pairwise_identity(sequences: Dict[str, str], pairs: Iterable[Tuple[str, str]], threads: int = 4) -> pd.DataFrame:
    """Identity of each pair as BLOSUM62 local-alignment matches / length of the shorter sequence.

    This is the measure split_tsuboyama.py reports in its overlap figures
    (`calculate_rigorous_identity`), so a cap on it bounds exactly the number a reader sees.
    It is computed for every pair rather than only for search hits: MMseqs2's prefilter and
    composition-bias correction drop some highly similar designed sequences entirely.
    Returns columns a, b (a < b) and identity.
    """
    pairs = sorted({(min(a, b), max(a, b)) for a, b in pairs if a != b})
    with Pool(threads, initializer=_init_identity_worker, initargs=(sequences,)) as pool:
        rows = pool.map(_local_identity, pairs, chunksize=500)
    return pd.DataFrame(rows, columns=['a', 'b', 'identity'])


def prune_bridges(nodes: Iterable[str], edges: Iterable[Tuple[str, str]], max_component_size: int,
                  min_family_size: int = 5, max_cut_frac: float = 0.25) -> Set[str]:
    """Nodes to drop so that oversized components split into their constituent families.

    Single linkage chains distinct families (e.g. designed topologies) into one component
    through a few borderline members. For each component larger than `max_component_size`,
    families are found by greedy modularity; a family of at least `min_family_size` members is
    separated from the rest of its component by a minimum node cut when that cut is at most
    `max_cut_frac` of the family's size. Repeats until no admissible cut remains. Dropped
    nodes leave the dataset; no remaining edge crosses the new components.
    """
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(e for e in edges if e[0] in g and e[1] in g)
    removed: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for comp in sorted(nx.connected_components(g), key=len, reverse=True):
            if len(comp) <= max_component_size:
                continue
            sub = g.subgraph(comp)
            # Greedy modularity is deterministic for a given graph
            families = nx.community.greedy_modularity_communities(sub)
            best = None
            for fam in sorted(families, key=lambda f: sorted(f)):
                rest = comp - fam
                if len(fam) < min_family_size or not rest:
                    continue
                k = nx.Graph(sub)
                k.add_edges_from(('__src__', n) for n in fam)
                k.add_edges_from(('__snk__', n) for n in rest)
                cut = nx.minimum_node_cut(k, '__src__', '__snk__')
                if len(cut) <= max_cut_frac * len(fam) and (best is None or len(cut) < len(best)):
                    best = cut
            if best:
                removed |= best
                g.remove_nodes_from(best)
                changed = True
                break
    return removed


def homology_edge_mask(pairs: pd.DataFrame, seq_evalue: float = 1e-3, struct_evalue: float = 1e-3,
                       max_identity: Optional[float] = 0.40, max_tm: Optional[float] = None,
                       tm_norm: str = 'max') -> pd.Series:
    """Pairs that may not be split apart: either search is significant, or an absolute cap is exceeded.

    A pair is an edge when MMseqs2 E <= seq_evalue, Foldseek E <= struct_evalue, identity >
    max_identity, or Foldseek TM-score > max_tm (normalised by the shorter chain for 'max',
    the longer for 'min'). Criteria whose columns are absent (search not run) are skipped.
    """
    col = lambda c: pairs[c] if c in pairs else pd.Series(np.nan, index=pairs.index)
    mask = (col('seq_evalue') <= seq_evalue) | (col('struct_evalue') <= struct_evalue)
    if max_identity is not None:
        mask |= col('identity') > max_identity
    if max_tm is not None:
        mask |= col(f'struct_tm_{tm_norm}') > max_tm
    return mask.fillna(False)
