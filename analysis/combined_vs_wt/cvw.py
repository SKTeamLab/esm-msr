"""Shared definitions of the combined-versus-WT analysis: the families compared, the benchmark cells, the look of the figures.

The question: when does the combined prediction (1/2 (MT + WT), ``combined_pred``) beat the WT head alone (``wt_lora_pred``) at sigma = 1?  Both are read from the same
prediction file of one checkpoint, so every comparison is paired (same rows, same checkpoint, same seed).

Families (``msr_analysis.final_runs``):  FINAL_p0 = the ESM-MSR recipe (unmasked WT and MT passes), FINAL_p1 = independent single-site masking of both passes,
FINAL_p0_pw0 = FINAL_p0 without the pair loss (block_pair_weight 0).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(ANALYSIS / 'proteingym' / 'p0_sigma'))
from msr_analysis import style  # noqa: E402,F401
from msr_analysis.final_runs import load_final_frame  # noqa: E402,F401

OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)

FAMS = ['FINAL_p0', 'FINAL_p1', 'FINAL_p0_pw0']
FAM_LABEL = {'FINAL_p0': 'ESM-MSR (unmasked)', 'FINAL_p1': 'independent masking', 'FINAL_p0_pw0': 'no pair loss'}
FAM_SHORT = {'FINAL_p0': 'ESM-MSR', 'FINAL_p1': 'masked', 'FINAL_p0_pw0': 'no pair loss'}
FAM_COLOR = {'FINAL_p0': '#000000', 'FINAL_p1': '#c0392b', 'FINAL_p0_pw0': '#2f7fc1'}
FAM_MARKER = {'FINAL_p0': 'o', 'FINAL_p1': 'D', 'FINAL_p0_pw0': 's'}
HEADS = ['wt', 'mt', 'comb']

EXT_SINGLE = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369']
EXT_MULTI = ['ptmul', 'ptmuld']
DMS = ['DLG4_HUMAN_Faure_2021_binding_domain', 'GRB2_HUMAN_Faure_2021_binding_domain', 'MYO_HUMAN_Kung_2025_display']     # the DMS benchmark: the sets that are not ProteinGym assays
TSU = ['tsuboyama_capped_val', 'tsuboyama_capped_test']
LABEL = {'s571': 'S571', 's783': 'S783', 's2648': 'S2648', 's8754': 'S8754', 's669': 'S669', 's461': 'S461', 'ssym': 'Ssym', 'q3421': 'Q3421', 'k3822': 'K3822', 'k2369': 'K2369',
         'ptmul': 'PTMUL', 'ptmuld': 'PTMUL-D', 'domainome': 'Domainome',
         'DLG4_HUMAN_Faure_2021_binding_domain': 'DLG4 binding',
         'GRB2_HUMAN_Faure_2021_binding_domain': 'GRB2 binding', 'MYO_HUMAN_Kung_2025_display': 'MYO display',
         'tsuboyama_capped_val': 'Homology-Restricted MegaScale val', 'tsuboyama_capped_test': 'Homology-Restricted MegaScale test'}

# ProteinGym categories (colours of the p0_sigma figures)
CATS = ['Stability', 'Expression', 'Activity', 'Binding', 'OrganismalFitness']
CAT_LABEL = {'Stability': 'Stability', 'Expression': 'Expression / abundance', 'Activity': 'Activity', 'Binding': 'Binding', 'OrganismalFitness': 'Organismal fitness'}
from msr_analysis.models import CATEGORY_COLOR as CAT_COLOR  # noqa: E402   one colour per ProteinGym category in every figure

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8982', '#e4e3de'
RC = {'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
      'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': GRID,
      'grid.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True, 'font.size': 9.5, 'legend.frameon': False, 'lines.linewidth': 1.4,
      'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'}

B_DEFAULT = 1000                 # default bootstrap size (paper: 2000)


def finish(fig, note, stem, ax_titles=(), out=None, pad=0.02, tight=True):
    """Wrap ``note`` into a footer, re-lay out the figure above it, wrap the axis titles to their axes, check that no text leaves the figure box and save png + pdf."""
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=7.6, color='#333333')
    tt = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = tt.get_window_extent(r).height / fig.bbox.height
    for _ in range(3 if tight else 0):
        fig.tight_layout(rect=(0, foot + pad, 1, 0.99))
        for ax, raw in ax_titles:
            ax.set_title(style.wrap_to_px(fig, raw, 0.97 * ax.get_window_extent(r).width, fontsize=10, fontweight='bold'), loc='left', fontsize=10)
    bad = style.text_outside(fig)
    if bad:
        print('WARNING: text outside the figure box in', stem, bad)
    style.save(fig, (out or OUT) / stem)


def holm(p):
    """Holm-adjusted p-values of one family of raw p-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[o])
    out = np.empty(len(p))
    out[o] = np.maximum.accumulate(adj)
    return out


def fmt_p(p):
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}' if p < 0.1 else f'p={p:.2f}'
