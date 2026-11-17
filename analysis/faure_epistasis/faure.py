"""Shared loaders for the epistasis analysis of the four Faure et al. (2022) datasets: PSD95-PDZ3 (DLG4) and GRB2-SH3 (GRB2), abundance (AbundancePCA) and binding (BindingPCA).

Labels are the library's ``fitness`` (``truth``) and its measurement error ``sigma`` (the merged triplicate, DiMSum's count-based error model) from ``baselines/public/_tasks``; the epistasis of a double is
eps = f_AB - f_A - f_B on that scale (the benchmark's ``dddG_ML``), with the error sigma_eps^2 = sigma_AB^2 + sigma_A^2 + sigma_B^2.  Structures: the ProteinGym AlphaFold2 models of the full proteins
(``ProteinGym/ProteinGym_AF2_structures``); the domain's residue p is residue ``offset + p`` of the model (found by aligning the domain sequence).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS))
from msr_analysis import style  # noqa: E402,F401
from msr_analysis.registry import ROOT, TASKS  # noqa: E402

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
CACHE = ANALYSIS / 'cache'                    # git-ignored: the doubles tables of 01_proximity.py
CACHE.mkdir(exist_ok=True)
SETS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain']
LABEL = {'DLG4_HUMAN_Faure_2021_abundance_domain': 'PDZ3 (DLG4) abundance', 'DLG4_HUMAN_Faure_2021_binding_domain': 'PDZ3 (DLG4) binding',
         'GRB2_HUMAN_Faure_2021_abundance_domain': 'SH3 (GRB2) abundance', 'GRB2_HUMAN_Faure_2021_binding_domain': 'SH3 (GRB2) binding'}
SHORT = {'DLG4_HUMAN_Faure_2021_abundance_domain': 'DLG4 abund.', 'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 bind.', 'GRB2_HUMAN_Faure_2021_abundance_domain': 'GRB2 abund.',
         'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 bind.'}
COLOR = {'DLG4_HUMAN_Faure_2021_abundance_domain': '#2a78d6', 'DLG4_HUMAN_Faure_2021_binding_domain': '#1baf7a', 'GRB2_HUMAN_Faure_2021_abundance_domain': '#6250d6',
         'GRB2_HUMAN_Faure_2021_binding_domain': '#eb6834'}
STRUCT = {'DLG4': ROOT / 'ProteinGym' / 'ProteinGym_AF2_structures' / 'DLG4_HUMAN.pdb', 'GRB2': ROOT / 'ProteinGym' / 'ProteinGym_AF2_structures' / 'GRB2_HUMAN.pdb'}
AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
RC = {'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb', 'axes.edgecolor': '#e4e3de', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e',
      'ytick.color': '#52514e', 'text.color': '#0b0b0b', 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': '#e4e3de',
      'grid.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True, 'font.size': 9.5, 'legend.frameon': False, 'lines.linewidth': 1.4,
      'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'}


def protein(ds):
    return ds.split('_')[0]


def load(ds):
    """(singles, doubles) of one set.  singles: mutation -> (fitness, sigma); doubles: DataFrame with the two mutations, positions (domain numbering), fitness, sigma, and the two singles' fitness and
    sigma, restricted to the doubles with both singles measured; eps = f_AB - f_A - f_B, sig_eps, x = f_A + f_B."""
    d = pd.read_csv(TASKS / f'{ds}.csv.gz', low_memory=False).drop_duplicates('id')
    s = d[d.NMUTS == 1]
    single = {m: (float(f), float(e)) for m, f, e in zip(s.mutation1, s.truth, s.sigma)}
    t = d[d.NMUTS == 2].copy()
    t['fA'] = [single.get(m, (np.nan, np.nan))[0] for m in t.mutation1]
    t['sA'] = [single.get(m, (np.nan, np.nan))[1] for m in t.mutation1]
    t['fB'] = [single.get(m, (np.nan, np.nan))[0] for m in t.mutation2]
    t['sB'] = [single.get(m, (np.nan, np.nan))[1] for m in t.mutation2]
    out = pd.DataFrame(dict(id=t.id.to_numpy(), mut1=t.mutation1.to_numpy(), mut2=t.mutation2.to_numpy(), pos1=t.pos1.astype(int).to_numpy(), pos2=t.pos2.astype(int).to_numpy(), aa1=t.to1.to_numpy(), aa2=t.to2.to_numpy(),
                            wt1=t.fr1.to_numpy(), wt2=t.fr2.to_numpy(), f=t.truth.to_numpy(float), s=t.sigma.to_numpy(float), fA=t.fA.to_numpy(float), sA=t.sA.to_numpy(float), fB=t.fB.to_numpy(float),
                            sB=t.sB.to_numpy(float)))
    out['n_all_doubles'] = len(out)
    out = out[np.isfinite(out[['f', 's', 'fA', 'sA', 'fB', 'sB']]).all(axis=1)].reset_index(drop=True)
    out['x'] = out.fA + out.fB
    out['eps'] = out.f - out.x
    out['sig_eps'] = np.sqrt(out.s ** 2 + out.sA ** 2 + out.sB ** 2)
    return single, out, d


def structure(ds, ref_seq):
    """Residue records of the AF2 model aligned to the domain: {domain position: dict(cb=coord, atoms=heavy-atom array, aa=letter)}."""
    res = {}
    for l in open(STRUCT[protein(ds)]):
        if l.startswith('ATOM'):
            k = int(l[22:26])
            r = res.setdefault(k, dict(aa=AA3.get(l[17:20], 'X'), atoms=[], cb=None, ca=None))
            xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
            if l[12:16].strip() != 'H' and l[76:78].strip() != 'H':
                r['atoms'].append(xyz)
            if l[12:16] == ' CB ':
                r['cb'] = xyz
            if l[12:16] == ' CA ':
                r['ca'] = xyz
    seq = ''.join(res[k]['aa'] for k in sorted(res))
    off = seq.find(ref_seq.rstrip('*'))
    assert off >= 0, 'domain sequence not found in the model'
    ks = sorted(res)
    dom = {}
    for p in range(1, len(ref_seq.rstrip('*')) + 1):
        r = res[ks[off + p - 1]]
        r['atoms'] = np.array(r['atoms'])
        r['cb'] = np.array(r['cb'] if r['cb'] is not None else r['ca'])
        dom[p] = r
    return dom, ks[off]


def distances(dom):
    """Matrices (n+1 x n+1, 1-based): Cbeta-Cbeta (Calpha for Gly) and minimal heavy-atom distance between two residues."""
    n = max(dom)
    cb = np.full((n + 1, n + 1), np.nan)
    mn = np.full((n + 1, n + 1), np.nan)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cb[i, j] = np.linalg.norm(dom[i]['cb'] - dom[j]['cb'])
            a, b = dom[i]['atoms'], dom[j]['atoms']
            mn[i, j] = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).min())
    return cb, mn


def finish(fig, note, stem, pad=0.02):
    """Wrap ``note`` into a footer, lay the figure out above it, check that no text leaves the figure box and save png + pdf into ``results/``."""
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=7.6, color='#333333')
    tt = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = tt.get_window_extent(r).height / fig.bbox.height
    for _ in range(3):
        fig.tight_layout(rect=(0, foot + pad, 1, 0.99))
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, OUT / stem)
