#!/usr/bin/env python3
"""Generate copy-pastable LaTeX SI tables matching the manuscript format:
  - Tables S1a–S1b: Concise summary of ALL ESM-MSR ablations split into In-Distribution
                    (MegaScale test + held-out MGnify & Rocklin) and OOD (7+1 Thermostability,
                    held-out Domainome, held-out ProteinGym).
  - Tables S2–S6: Comprehensive main-model comparison tables including ALL models with
                  predictions, Spearman rho, Pearson r, raw RMSE, one-sided paired bootstrap
                  p-values comparing ESM-MSR, and per-model/per-dataset homology overlap notes
                  grounded in leak_audit_matrix.csv.
  - Table S4: Single vertical multi-page longtable stacking all 12 external thermostability
              datasets one on top of the other with all 19 models and per-(dataset, model)
              homology notes.

Usage:
    /home/sareeves/miniconda3/envs/msr_venv/bin/python analysis/scripts/build_si_latex_tables.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path('/home/sareeves/playground/esm-msr-devel')
WT_DDDG = ROOT / 'wt-dddg-scatters'
WT_TOBIT = ROOT / 'wt-tobit-floor-ablation'
WT_SI = Path(__file__).resolve().parents[2]

MAIN_PVAL = WT_DDDG / 'analysis/results/00_main_text_figs/pvalues'
MAIN_SRC = WT_DDDG / 'analysis/results/00_main_text_figs/source_data'
SUPP_PVAL = WT_DDDG / 'analysis/results/00_supplementary_figs/pvalues'
REF_DIR = WT_DDDG / 'analysis/reference'
ABL_DIR = WT_TOBIT / 'analysis/results/09_ablation'

OUT_DIR = WT_SI / 'analysis/results/si_latex_tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_CSV = OUT_DIR / '_extra_models_pvalues.csv'
TOBIT_CSV = OUT_DIR / '_tobit_linked_si_overrides.csv'

for _p in (str(WT_DDDG / 'src'), str(WT_DDDG / 'analysis')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm_msr import epi_metrics as E  # noqa: E402
from esm_msr.val_stats import tobit_link  # noqa: E402
from msr_analysis import epistasis as EPI_LIB  # noqa: E402
from msr_analysis.heldout import excluded as heldout_excluded  # noqa: E402
from msr_analysis.tobit import TOBIT  # noqa: E402


# --------------------------------------------------------------------------------------
# LaTeX formatting utilities
# --------------------------------------------------------------------------------------
def tex_escape(s: str) -> str:
    s = str(s)
    repl = {
        'Δ': r'$\Delta$',
        'σ': r'$\sigma$',
        'ρ': r'$\rho$',
        'δ': r'$\delta$',
        '≤': r'$\le$',
        '≥': r'$\ge$',
        '−': '$-$',
        '–': '--',
        '—': '---',
        '·': r'$\cdot$',
        '×': r'$\times$',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    parts = s.split('$')
    for i in range(0, len(parts), 2):
        parts[i] = (
            parts[i]
            .replace('_', r'\_')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('#', r'\#')
            .replace('<', r'$<$')
            .replace('>', r'$>$')
        )
    return '$'.join(parts)


def fmt_num(val: float | None, decimals: int = 3, bold: bool = False, suffix: str = '') -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return '---'
    v = float(val)
    if abs(v) < 0.5 * (10.0 ** (-decimals)):
        v = 0.0
    s = f'{v:.{decimals}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    s = f'{s}{suffix}'
    return f'\\textbf{{{s}}}' if bold else s


def fmt_mean_sd(mean: float | None, sd: float | None, decimals: int = 3, bold: bool = False, suffix: str = '') -> str:
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return '---'
    mv = float(mean)
    if abs(mv) < 0.5 * (10.0 ** (-decimals)):
        mv = 0.0
    m_s = f'{mv:.{decimals}f}'
    if m_s.startswith('-'):
        m_s = '$-$' + m_s[1:]
    if sd is None or (isinstance(sd, float) and not math.isfinite(sd)):
        s = f'{m_s}{suffix}'
    else:
        s = f'{m_s} $\\pm$ {float(sd):.{decimals}f}{suffix}'
    return f'\\textbf{{{s}}}' if bold else s


def fmt_p(p: float | None, B: int | float | None = None) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return '---'
    p = float(p)
    if B is not None and math.isfinite(float(B)):
        floor = 1.0 / (float(B) + 1.0)
        if p <= floor * 1.05:
            if float(B) >= 9999:
                return r'$<10^{-4}$'
            if float(B) >= 999:
                return r'$<10^{-3}$'
            return f'$<{floor:.4f}$'
    if p < 1e-4:
        return r'$<10^{-4}$'
    if p < 1e-3:
        return f'{p:.4f}'
    if p >= 0.9995:
        return '1.000'
    return f'{p:.3f}'


def render_table(
    *,
    caption_title: str,
    caption_body: str,
    label: str,
    col_spec: str,
    header_lines: list[str],
    body_rows: list[str],
) -> str:
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        f'\\caption{{\\textbf{{{caption_title}}}. {caption_body}}}',
        f'\\label{{{label}}}',
        r'\resizebox{\textwidth}{!}{%',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        *header_lines,
        r'\midrule',
        *body_rows,
        r'\bottomrule',
        r'\end{tabular}',
        r'}',
        r'\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def render_longtable(
    *,
    caption_title: str,
    caption_body: str,
    label: str,
    col_spec: str,
    header_lines: list[str],
    body_rows: list[str],
    ncols: int,
    fontsize: str = r'\scriptsize',
) -> str:
    """Render a multi-page LaTeX longtable with repeating headers."""
    lines = [
        r'{',
        fontsize,
        r'\setlength{\tabcolsep}{4pt}',
        f'\\begin{{longtable}}{{{col_spec}}}',
        f'\\caption{{\\textbf{{{caption_title}}}. {caption_body}}}\\label{{{label}}} \\\\',
        r'\toprule',
        *header_lines,
        r'\midrule',
        r'\endfirsthead',
        f'\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Continued from previous page}}}} \\\\',
        r'\toprule',
        *header_lines,
        r'\midrule',
        r'\endhead',
        r'\midrule',
        f'\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\',
        r'\endfoot',
        r'\bottomrule',
        r'\endlastfoot',
        *body_rows,
        r'\end{longtable}',
        r'}',
    ]
    return '\n'.join(lines) + '\n'


def write_tex(filename: str, content: str) -> Path:
    p = OUT_DIR / filename
    p.write_text(content, encoding='utf-8')
    return p


# --------------------------------------------------------------------------------------
# Homology / overlap lookup (from leak_audit_matrix.csv)
# --------------------------------------------------------------------------------------
class HomologyLookup:
    def __init__(self) -> None:
        lam = pd.read_csv(REF_DIR / 'leak_audit_matrix.csv')
        self.inv_map: dict[tuple[str, str], str] = {}
        for _, r in lam.iterrows():
            inv = str(r['inventory'])
            ds = str(r['dataset'])
            code = '' if pd.isna(r['code']) else str(r['code']).strip()
            if code in ('–', '-', 'nan', ''):
                code = ''
            self.inv_map[(inv, ds)] = code

    def inventory_for(self, model: str, multi: bool = False) -> str:
        m = model.strip()
        if m in (
            'ESM-MSR', 'ESM-MSR (Dual-view)', 'ESM-MSR (σ=1)', 'ESM-MSR WT-view',
            'ESM-MSR (WT-view)', 'ESM-MSR σ=0.5', 'ESM-MSR ctx ctrl',
            'SPURS', 'SPURS-multi', 'MutateEverything', 'MutateEverything Stage 2',
            'MutateEverything Stage 1', 'ThermoMPNN', 'ThermoMPNN-D',
        ):
            return 'esm_msr_capped'
        if m == 'GeoDDG-3D':
            return 'geoddg_s8754'
        if m in ('ESM3ΔG', 'ESM3dG'):
            return 'esm3dg_mgnify'
        if m == 'Rosetta':
            return 'rosetta_beta_nov16'
        if m == 'DDGemb':
            return 'ddgemb_s2450'
        if m == 'DDMut':
            return 'ddmut'
        if m == 'ESMtherm':
            return 'esmtherm'
        if m in ('IFUM', 'IFUM (WT backbone)', 'IFUM (ESMFold)'):
            return 'ifum'
        if m in ('SPURS published', 'SPURS (published)'):
            return 'spurs_pub_megasplits'
        if m in ('ThermoMPNN published*', 'ThermoMPNN (published)'):
            return 'thermompnn_pub'
        if m in (
            'ProSST', 'ProSST published', 'ProteinMPNN', 'ESM3-small-open',
            'ESM3-small-open WT-view', 'ESM3-small', 'ESM3-medium', 'ESM3-large',
        ):
            return 'corpus'
        return 'other'

    def note_and_overlap(self, model: str, dataset_key: str, multi: bool = False) -> tuple[str, bool]:
        """Return (latex_note, has_training_overlap_for_bolding).

        Training overlap (excludes model from clean bolding): exact mutation (M), constituent (m),
        same protein (P), close homolog (H), test=train (#), or direct training on benchmark source.
        Distant homologs (h) and Rosetta calibration (V) are reported in the note without disqualifying.
        """
        inv = self.inventory_for(model, multi=multi)
        # Special cases not tabulated in leak_audit_matrix
        if dataset_key in ('mgnify', 'rocklin_denovo'):
            if inv == 'esm3dg_mgnify':
                return ('Trained on MGnify', True)
            if inv == 'esmtherm':
                return ('Trained on MegaScale (unfiltered)', True)
            if inv == 'ifum':
                return ('Trained on MegaScale (unfiltered)', True)
            if inv == 'geoddg_s8754':
                return ('---', False)
            return ('---', False)
        if dataset_key == 'tsuboyama_mega_test':
            if inv == 'spurs_pub_megasplits':
                return (r'Same \texttt{mega\_splits} (pub.)', False)
            if inv == 'thermompnn_pub':
                return (r'Same \texttt{mega\_splits} (pub.\ ens.)', False)
            if inv == 'rosetta_beta_nov16':
                return ('V (Rocklin cal.)', False)
            if inv == 'esm_msr_capped':
                return (r'Trained on \texttt{mega\_splits}', False)
            return ('---', False)

        # Map canonical benchmark keys to leak_audit_matrix dataset names
        ds_map = {
            'tsuboyama_capped_test': 'tsu_capped_test',
            'MegaScale_capped_test': 'tsu_capped_test',
            'MegaScale_capped_test_doubles': 'tsu_capped_test',
            'MegaScale_epistasis_5level_hierarchy': 'tsu_capped_test',
            'domainome_heldout': 'domainome_low_id',
            'Domainome_heldout_436_domains': 'domainome_low_id',
            'proteingym_heldout': 'PG avg',
            'ProteinGym_heldout_173_assays': 'PG avg',
        }
        ds_l = ds_map.get(dataset_key, dataset_key.lower())
        code = self.inv_map.get((inv, ds_l), '')

        # On held-out Domainome & held-out ProteinGym, esm_msr_capped homologs were removed by the 4-part filter
        if inv == 'esm_msr_capped' and ds_l in ('PG avg', 'domainome_low_id'):
            if ds_l == 'PG avg':
                return ('--- (held-out)', False)
            return ('h.03 (held-out)' if code else '--- (held-out)', False)
        if inv == 'geoddg_s8754' and ds_l == 'PG avg':
            return ('GeoFitness pretrain (74 DMS)', True)

        if not code:
            return ('---', False)

        # Clean up LaTeX rendering of overlap codes
        has_ov = bool(re.search(r'(^|\s)(M\??|m|P|H|#)', code))
        note = tex_escape(code)
        if ds_l == 's8754' and inv == 'geoddg_s8754':
            note = r'M1 \# (train set)'
        elif ds_l == 'tsu_capped_test' and inv == 'rosetta_beta_nov16':
            note = 'V.12 (Rocklin cal.)'
        return (note, has_ov)


HOM = HomologyLookup()

HOMOLOGY_LEGEND = (
    r"Homology / overlap codes (fraction of benchmark rows affected, e.g.\ M.50 = 50\%, M1 = 100\%): "
    r"M = exact mutation in training set (\#: test set equals training set; M?: protein-level ceiling); "
    r"m = constituent single mutation of a multi-mutant in training set; "
    r"P = same protein ($\ge 95\%$ identity); H = close homolog ($E\le 10^{-3}$ or TM $>0.70$); "
    r"h = distant homolog ($E\le 10^{-2}$ or $\ge 30\%$ identity); V = validation/calibration set; "
    r"D = training set includes reverse mutations; --- = no detected overlap. "
    r"Models with M/m/P/H/\# training overlap (or trained on the unfiltered benchmark source) are marked $^*$ and excluded from clean bolding."
)


# Canonical model display ordering for main-model tables
CANONICAL_MODEL_ORDER = [
    'ESM-MSR',
    'ESM-MSR (Dual-view)',
    'ESM-MSR (σ=1)',
    'ESM-MSR WT-view',
    'ESM-MSR (WT-view)',
    'ESM-MSR (unlinked)',
    'ESM-MSR σ=0.5',
    'ESM-MSR ctx ctrl',
    'SPURS',
    'SPURS-multi',
    'SPURS published',
    'MutateEverything',
    'MutateEverything Stage 2',
    'MutateEverything Stage 1',
    'ThermoMPNN',
    'ThermoMPNN-D',
    'ThermoMPNN published*',
    'GeoDDG-3D',
    'ESM3ΔG',
    'Rosetta',
    'DDGemb',
    'DDMut',
    'ESMtherm',
    'IFUM',
    'IFUM (WT backbone)',
    'IFUM (ESMFold)',
    'ESM3-small-open',
    'ESM3-small-open WT-view',
    'ESM3-small',
    'ESM3-medium',
    'ESM3-large',
    'ProteinMPNN',
    'ProSST',
    'ProSST published',
    'Top 96 per Domain',
    '−dG_add (floor ceiling)',
]


def sort_models(models: list[str], msr_first: str | None = None) -> list[str]:
    rank = {m: i for i, m in enumerate(CANONICAL_MODEL_ORDER)}
    def key(m: str):
        if msr_first and m == msr_first:
            return (-1, m)
        return (rank.get(m, 999), m)
    return sorted(models, key=key)


def clean_model_display(name: str, overlap: bool = False) -> str:
    mapping = {
        'ESM-MSR (σ=1)': 'ESM-MSR',
        'ESM-MSR σ=0.5': r'ESM-MSR ($\sigma$=0.5)',
        'ESM-MSR WT-view': 'ESM-MSR (WT-view)',
        'ESM-MSR ctx ctrl': 'ESM-MSR (context control)',
        'ESM3ΔG': r'ESM3$\Delta G$',
        'MutateEverything': 'Mutate Everything',
        'MutateEverything Stage 2': 'Mutate Everything (Stage 2)',
        'MutateEverything Stage 1': 'Mutate Everything (Stage 1)',
        'ESM3-small-open WT-view': 'ESM3-small-open',
        'SPURS published': 'SPURS (published)',
        'ThermoMPNN published*': 'ThermoMPNN (published)',
        'ProSST published': 'ProSST (published)',
        '−dG_add (floor ceiling)': r'$-\Delta G_{\mathrm{add}}$ (floor ceiling)',
        'Top 96 per Domain': 'Oracle (top 96 per domain)',
    }
    disp = mapping.get(name, tex_escape(name))
    if overlap:
        disp += r'$^*$'
    return disp


def _is_internal_model(name: str) -> bool:
    return name in ('ESM-MSR (unlinked)', 'ESM-MSR σ=0.5', 'ESM-MSR ctx ctrl', 'Top 96 per Domain', '−dG_add (floor ceiling)')


# --------------------------------------------------------------------------------------
# PART 1: TABLES S1a & S1b — ABLATIONS SPLIT INTO IN-DISTRIBUTION AND OOD
# --------------------------------------------------------------------------------------
EXT_MAIN_SINGLES_7 = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754']


def _load_heldout_mgnify_spine(run: str, src_name: str) -> pd.DataFrame:
    excl = set(heldout_excluded(src_name))
    pkl_path = ROOT / 'repo/analysis/cache' / f'frame_mgnify_FINAL_p0_{run}_s123.pkl'
    with open(pkl_path, 'rb') as fh:
        fr = pickle.load(fh)
    sp = fr.spine
    m = sp['use'].to_numpy() & (~sp['code_wt'].astype(str).isin(excl).to_numpy())
    return pd.DataFrame({
        'uid': sp.loc[m, 'id'].astype(str).to_numpy(),
        'dG_wt': sp.loc[m, 'dG_wt'].to_numpy(float),
        'y': sp.loc[m, 'y'].to_numpy(float),
    }).set_index('uid')


def _score_mgnify_task(args: tuple[str, str, str, str, tuple[str, ...], str]) -> tuple[str, str, dict[str, tuple[float, float]]]:
    base_dir_s, fam, run, colkey, heads, src_name = args
    base_dir = Path(base_dir_s)
    spine = _load_heldout_mgnify_spine(run, src_name)
    h_link = tobit_link(**TOBIT)
    usecols = ['uid'] + list(heads)
    by_head: dict[str, list[float]] = {h: [] for h in heads}
    dg_ok = spine['dG_wt'].to_numpy(float)
    y_ok = spine['y'].to_numpy(float)
    target_uids = spine.index
    for s in (1, 2, 3):
        p = base_dir / fam / f's{s}' / run / 'predictions.csv.gz'
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df['uid'] = df['uid'].astype(str)
        df = df.set_index('uid').reindex(target_uids)
        for h in heads:
            pred = df[h].to_numpy(float)
            ok = np.isfinite(pred) & np.isfinite(y_ok) & np.isfinite(dg_ok)
            linked = h_link(dg_ok[ok] + pred[ok]) - dg_ok[ok]
            by_head[h].append(float(spearmanr(linked, y_ok[ok]).statistic))
    out = {}
    for h, vals in by_head.items():
        if len(vals) == 3:
            out[h] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    tag = 'FINAL_p0_s05' if 'sigma0.5' in base_dir_s else ('FINAL_p1@unmasked' if fam == 'FINAL_p1_unmasked' else fam)
    return tag, colkey, out


def _load_mgnify_rocklin_ablations() -> dict[str, dict[str, tuple[float, float]]]:
    fams = [
        'FINAL_p0', 'FINAL_p1', 'FINAL_p1_unmasked', 'FINAL_p1_marg',
        'FINAL_p0_pw0', 'FINAL_p0_norank', 'FINAL_p0_rcthr0',
        'FINAL_p0_norange', 'FINAL_p0_mse', 'FINAL_p0_deepfloor', 'FINAL_p0_floor0',
        'FINAL_p0_noseqhwm', 'FINAL_p0_r2a4', 'FINAL_p0_mega',
    ]
    tasks = []
    for run, src_name, colkey in (('all', 'mgnify', 'mgnify'), ('rdenovo', 'rocklin_denovo', 'rocklin')):
        for fam in fams:
            heads = ('combined_pred', 'wt_lora_pred', 'mt_lora_pred') if fam == 'FINAL_p0' else ('combined_pred',)
            tasks.append((str(ROOT / 'final_mgnify'), fam, run, colkey, heads, src_name))
        tasks.append((str(ROOT / 'final_mgnify_sigma0.5'), 'FINAL_p0', run, colkey, ('combined_pred',), src_name))

    res: dict[str, dict[str, tuple[float, float]]] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for tag, colkey, sc in ex.map(_score_mgnify_task, tasks):
            if 'combined_pred' in sc:
                res.setdefault(tag, {})[colkey] = sc['combined_pred']
            if tag == 'FINAL_p0':
                if 'wt_lora_pred' in sc:
                    res.setdefault('FINAL_p0_WT', {})[colkey] = sc['wt_lora_pred']
                if 'mt_lora_pred' in sc:
                    res.setdefault('FINAL_p0_MT', {})[colkey] = sc['mt_lora_pred']
    return res


def _compute_epi_beyond_cellmag(split_dir: str, fam: str, tag: str, seeds: tuple[int, ...] = (1, 2, 3)) -> tuple[tuple[float, float], tuple[float, float]]:
    fpred = ROOT / 'final_bench/predictions' / split_dir / 'final_checkpoints' / fam
    beys, cmags = [], []
    for seed in seeds:
        p_pred = fpred / f's{seed}' / f'best.ckpt_{tag}_predictions.csv'
        if not p_pred.exists():
            continue
        df = pd.read_csv(p_pred, low_memory=False)
        code = df['code_wt'].astype(str).to_numpy()
        mut = df['mut_type'].astype(str).to_numpy()
        keys = [EPI_LIB.parse_mut_key(m, c) for m, c in zip(mut, code)]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        dg_wt = df['dG_ML'].to_numpy(float) - df['ddG_ML'].to_numpy(float)
        items = dict(
            code=code, mut_key=keys, subset_type=st,
            ddG=df['ddG_ML'].to_numpy(float), dddG=df['dddG_ML'].to_numpy(float),
            dG_wt=dg_wt, wt=df['wt_lora_pred'].to_numpy(float),
            mt=df['combined_pred'].to_numpy(float), comb=df['combined_pred'].to_numpy(float),
        )
        tbl = E.Table(items, None, heads=('comb', 'add'))
        res_e = E.compute(tbl)
        b_v = float(res_e.get('epi_beyond_rho_comb', np.nan))
        c_v = float(res_e.get('epi_cell_mag_comb', np.nan))
        if math.isfinite(b_v):
            beys.append(b_v)
        if math.isfinite(c_v):
            cmags.append(c_v)

    def agg(lst):
        if not lst:
            return (np.nan, np.nan)
        return (float(np.mean(lst)), float(np.std(lst, ddof=1)) if len(lst) > 1 else np.nan)

    return agg(beys), agg(cmags)


def _compute_extra_ablation_row_stats(row_key: str) -> dict[str, tuple[float, float]]:
    if row_key != 'FINAL_p0_s05':
        return {}
    fb = ROOT / 'final_bench/stats'
    dom_excl = heldout_excluded('domainome')
    f3c = pd.read_csv(MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv')
    pg_held = f3c[(~f3c['leaked']) & (~f3c['dropped_large'])][['assay', 'category']].copy()

    fam, tag, seeds = 'FINAL_p0', 'sigma0.5_unmasked', (1, 2, 3)
    tsu_all, tsu_sng, tsu_dbl, ext_s, ext_m, dom_l, pg_m = [], [], [], [], [], [], []
    bey_agg, cmag_agg = _compute_epi_beyond_cellmag('splits_oct06_capped-test', fam, tag, seeds)
    for s in seeds:
        p_t = fb / 'splits_oct06_capped-test/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_Combined_avg.csv'
        if p_t.exists():
            d = pd.read_csv(p_t).set_index('Unnamed: 0')['0']
            tsu_all.append(float(d['spearman_all']))
            tsu_sng.append(float(d['spearman_singles']))
            tsu_dbl.append(float(d['spearman_doubles']))
        p_ew = fb / 'external/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_WT_LoRA.csv'
        p_ec = fb / 'external/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_Combined.csv'
        if p_ew.exists():
            d = pd.read_csv(p_ew).set_index('Unnamed: 0')['spearman_all']
            ext_s.append(float(d.loc[EXT_MAIN_SINGLES_7].mean()))
        if p_ec.exists():
            d = pd.read_csv(p_ec).set_index('Unnamed: 0')['spearman_all']
            ext_m.append(float(d.loc['ptmul']))
        p_d = fb / 'domainome/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag.replace("unmasked", "skip_reverse_unmasked")}_WT_LoRA.csv'
        if p_d.exists():
            d = pd.read_csv(p_d)
            keep = ~d['Unnamed: 0'].astype(str).isin(dom_excl)
            dom_l.append(float(d.loc[keep, 'spearman_all'].mean()))
        pg_dir = ROOT / 'final_pgym_wt_sigma0.5/FINAL_p0' / f's{s}'
        if pg_dir.exists():
            cat_vals: dict[str, list[float]] = {}
            for _, row in pg_held.iterrows():
                assay, cat = row['assay'], row['category']
                f_a = pg_dir / f'{assay}.csv'
                if not f_a.exists():
                    matches = list(pg_dir.glob(f'{assay}*.csv'))
                    f_a = matches[0] if matches else None
                if f_a is not None and f_a.exists():
                    da = pd.read_csv(f_a)
                    ycol = 'DMS_score' if 'DMS_score' in da.columns else ('y' if 'y' in da.columns else None)
                    pcol = 'wt_lora_pred' if 'wt_lora_pred' in da.columns else ('pred' if 'pred' in da.columns else None)
                    if ycol and pcol:
                        ok = np.isfinite(da[ycol]) & np.isfinite(da[pcol])
                        if ok.sum() >= 3:
                            cat_vals.setdefault(cat, []).append(float(spearmanr(da.loc[ok, pcol], da.loc[ok, ycol]).statistic))
            if len(cat_vals) == 5:
                pg_m.append(float(np.mean([np.mean(v) for v in cat_vals.values()])))

    def agg(lst):
        if not lst:
            return (np.nan, np.nan)
        return (float(np.mean(lst)), float(np.std(lst, ddof=1)) if len(lst) > 1 else np.nan)

    return {
        'tsu_overall': agg(tsu_all), 'tsu_singles': agg(tsu_sng), 'tsu_doubles': agg(tsu_dbl),
        'tsu_beyond': bey_agg, 'tsu_cellmag': cmag_agg,
        'ext_singles': agg(ext_s), 'ext_multi': agg(ext_m),
        'domainome': agg(dom_l), 'pgym': agg(pg_m),
    }


def build_table_s1_ablations() -> list[Path]:
    cl = pd.read_csv(ABL_DIR / 'cells_long.csv')
    cl_mega = pd.read_csv(ABL_DIR / 'cells_long_mega_own_split.csv')
    mg_map = _load_mgnify_rocklin_ablations()
    extra_s05 = _compute_extra_ablation_row_stats('FINAL_p0_s05')
    mega_beyond, mega_cellmag = _compute_epi_beyond_cellmag('mega_splits-test', 'FINAL_p0_mega', 'sigma1.0_unmasked', (1, 2, 3))

    def get_cl(fam: str, col: str, head: str = 'primary', df_src: pd.DataFrame = cl) -> tuple[float, float]:
        sub = df_src[(df_src['family'] == fam) & (df_src['column'] == col) & (df_src['head'] == head)]
        if len(sub) == 0:
            return (np.nan, np.nan)
        vals = sub['rho'].to_numpy(float)
        return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)

    def get_cl_ext7(fam: str, head: str = 'primary') -> tuple[float, float]:
        sub = cl[(cl['family'] == fam) & (cl['head'] == head) & (cl['column'].isin(EXT_MAIN_SINGLES_7))]
        if len(sub) == 0:
            return (np.nan, np.nan)
        per_seed = sub.groupby('seed')['rho'].mean().to_numpy(float)
        return (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan)

    row_specs = [
        ('FINAL_p0', 'Dual-View (Reference)', 'cl_primary'),
        ('FINAL_p0_WT', 'WT-View only', 'cl_wt'),
        ('FINAL_p0_MT', 'MT-View only', 'cl_mt'),
        ('FINAL_p0_s05', r'$\sigma=0.5$', 'extra_s05'),
        ('FINAL_p0@independent', 'Indep. mask. (eval only)', 'cl_primary'),
        ('FINAL_p1', 'Indep. mask. (Ret.)', 'cl_primary'),
        ('FINAL_p1@unmasked', 'Indep. mask., eval unmask. (Ret.)', 'cl_primary'),
        ('FINAL_p1_marg', 'Masked marg. MT (Ret.)', 'cl_primary'),
        ('FINAL_p0_pw0', 'No pair loss (Ret.)', 'cl_primary'),
        ('FINAL_p0_norank', 'No rank loss (Ret.)', 'cl_primary'),
        ('FINAL_p0_rcthr0', r'Rank censor $\tau=0$ (Ret.)', 'cl_primary'),
        ('FINAL_p0_norange', 'No range cells (Ret.)', 'cl_primary'),
        ('FINAL_p0_mse', 'MSE, no range (Ret.)', 'cl_primary'),
        ('FINAL_p0_deepfloor', r'Tobit floor ($-1, 1$) (Ret.)', 'cl_primary'),
        ('FINAL_p0_floor0', r'Tobit floor ($0, 0.5$) (Ret.)', 'cl_primary'),
        ('FINAL_p0_noseqhwm', 'No seq. head (Ret.)', 'cl_primary'),
        ('FINAL_p0_r2a4', 'MT LoRA rank 2 (Ret.)', 'cl_primary'),
        ('FINAL_p0_mega', r'Full MegaScale$^\dagger$ (Ret.)', 'cl_mega'),
    ]

    id_cols = ['tsu_overall', 'tsu_singles', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag', 'mgnify', 'rocklin']
    ood_cols = ['ext_singles', 'ext_multi', 'domainome', 'pgym']
    all_cols = id_cols + ood_cols

    records = []
    for key, label, mode in row_specs:
        rec = {'key': key, 'label': label}
        if mode == 'cl_primary':
            rec['tsu_overall'] = get_cl(key, 'tsuboyama_capped_test', 'primary')
            rec['tsu_singles'] = get_cl(key, 'tsuboyama_capped_test|singles', 'comb')
            rec['tsu_doubles'] = get_cl(key, 'tsuboyama_capped_test|multi', 'primary')
            rec['tsu_beyond'] = get_cl(key, 'tsuboyama_capped_test|beyond_rho', 'primary')
            rec['tsu_cellmag'] = get_cl(key, 'tsuboyama_capped_test|cell_mag', 'primary')
            rec['ext_singles'] = get_cl_ext7(key, 'primary')
            rec['ext_multi'] = get_cl(key, 'ptmul', 'primary')
            rec['domainome'] = get_cl(key, 'domainome|lowid', 'primary')
            rec['pgym'] = get_cl(key, 'pg|avg_lowid', 'wt')
        elif mode == 'cl_wt':
            rec['tsu_overall'] = get_cl('FINAL_p0', 'tsuboyama_capped_test', 'wt')
            rec['tsu_singles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|singles', 'wt')
            rec['tsu_doubles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|multi', 'wt')
            rec['tsu_beyond'] = (np.nan, np.nan)
            rec['tsu_cellmag'] = (np.nan, np.nan)
            rec['ext_singles'] = get_cl_ext7('FINAL_p0', 'wt')
            rec['ext_multi'] = get_cl('FINAL_p0', 'ptmul', 'wt')
            rec['domainome'] = get_cl('FINAL_p0', 'domainome|lowid', 'wt')
            rec['pgym'] = get_cl('FINAL_p0', 'pg|avg_lowid', 'wt')
        elif mode == 'cl_mt':
            rec['tsu_overall'] = get_cl('FINAL_p0', 'tsuboyama_capped_test', 'mt')
            rec['tsu_singles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|singles', 'mt')
            rec['tsu_doubles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|multi', 'mt')
            rec['tsu_beyond'] = (np.nan, np.nan)
            rec['tsu_cellmag'] = (np.nan, np.nan)
            rec['ext_singles'] = get_cl_ext7('FINAL_p0', 'mt')
            rec['ext_multi'] = get_cl('FINAL_p0', 'ptmul', 'mt')
            rec['domainome'] = get_cl('FINAL_p0', 'domainome|lowid', 'mt')
            rec['pgym'] = (np.nan, np.nan)
        elif mode == 'extra_s05':
            for ck in ['tsu_overall', 'tsu_singles', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag', 'ext_singles', 'ext_multi', 'domainome', 'pgym']:
                rec[ck] = extra_s05.get(ck, (np.nan, np.nan))
        elif mode == 'cl_mega':
            rec['tsu_overall'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test', 'comb', cl_mega)
            rec['tsu_singles'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test|singles', 'comb', cl_mega)
            rec['tsu_doubles'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test|multi', 'comb', cl_mega)
            rec['tsu_beyond'] = mega_beyond
            rec['tsu_cellmag'] = mega_cellmag
            rec['ext_singles'] = get_cl_ext7('FINAL_p0_mega', 'primary')
            rec['ext_multi'] = get_cl('FINAL_p0_mega', 'ptmul', 'primary')
            rec['domainome'] = get_cl('FINAL_p0_mega', 'domainome|lowid', 'primary')
            rec['pgym'] = get_cl('FINAL_p0_mega', 'pg|avg_lowid', 'wt')

        rec['mgnify'] = mg_map.get(key, {}).get('mgnify', (np.nan, np.nan))
        rec['rocklin'] = mg_map.get(key, {}).get('rocklin', (np.nan, np.nan))
        records.append(rec)

    best_by_col: dict[str, tuple[float, set[str]]] = {}
    for ck in all_cols:
        cands = []
        for rec in records:
            if rec['key'] == 'FINAL_p0_mega':
                continue
            m, sd = rec[ck]
            if math.isfinite(m):
                cands.append((m, rec['key']))
        if not cands:
            best_by_col[ck] = (np.nan, set())
            continue
        max_v = max(v for v, _ in cands)
        winners = {k for v, k in cands if round(v, 3) >= round(max_v, 3)}
        ref_m, ref_sd = records[0][ck]
        if math.isfinite(ref_m) and math.isfinite(ref_sd) and (ref_m + ref_sd >= max_v - 1e-12):
            winners.add('FINAL_p0')
        best_by_col[ck] = (max_v, winners)

    rows_id = []
    for rec in records:
        cells = [rec['label']]
        for ck in id_cols:
            m, sd = rec[ck]
            sfx = r'$^\dagger$' if (rec['key'] == 'FINAL_p0_mega' and math.isfinite(m)) else ''
            cells.append(fmt_mean_sd(m, sd, decimals=3, bold=(rec['key'] in best_by_col[ck][1]), suffix=sfx))
        rows_id.append(' & '.join(cells) + r' \\')

    tex_id = render_table(
        caption_title='Summary of ESM-MSR Ablations and Variations: In-Distribution Proteolysis Stability Benchmarks',
        caption_body=(
            r"Spearman's $\rho$ (mean $\pm$ SD across 3 independent training seeds) on proteolysis stability benchmarks "
            r"under the Dual-View readout (except WT-View only and MT-View only). Models marked (Ret.) are retrained with modifications on the Homology-Restricted "
            r"MegaScale split, while those without use the exact same weights as Dual-View (Reference). "
            r"MegaScale test columns report per-domain mean $\rho$ (Overall and Singles: 33 domains; Doubles: 13 domains) and epistasis metrics "
            r"(Beyond $\rho$: Spearman above global saturation curve; Cell mag.: double-centred complete-block residual correlation). "
            r"MGnify and Rocklin columns report pooled Tobit-linked $\rho$ strictly on the 4-part homology-held-out sets defined against Homology-Restricted MegaScale train "
            r"(MGnify: 612,406 variants across 140,694 scaffolds; Rocklin de novo designs: 109,581 variants across 623 scaffolds). "
            r"Highest performing Homology-Restricted models in each column are indicated in bold; Dual-View (Reference) is also bolded if its mean $\pm$ SD range overlaps the highest average performance. "
            r"$^\dagger$Trained on the full MegaScale split (\texttt{mega\_splits}): evaluated on its own 28-domain test split (22,213 doubles) for MegaScale columns, "
            r"and on the exact same Homology-Restricted held-out slice for MGnify and Rocklin (where additional training domains may introduce homology overlap); excluded from bolding in all columns. "
            r"--- indicates not applicable or not evaluated."
        ),
        label='tab:si_ablations_in_distribution',
        col_spec='l' + 'c' * len(id_cols),
        header_lines=[
            r'{} & \multicolumn{5}{c}{MegaScale test} & \multicolumn{2}{c}{Held-out Proteolysis} \\',
            r'\cmidrule(lr){2-6} \cmidrule(lr){7-8}',
            r'ID & Overall & Singles & Doubles & Beyond $\rho$ & Cell mag. & MGnify & Rocklin \\',
        ],
        body_rows=rows_id,
    )
    p1a = write_tex('table_S1a_ablations_in_distribution.tex', tex_id)

    rows_ood = []
    for rec in records:
        cells = [rec['label']]
        for ck in ood_cols:
            m, sd = rec[ck]
            cells.append(fmt_mean_sd(m, sd, decimals=3, bold=(rec['key'] in best_by_col[ck][1])))
        rows_ood.append(' & '.join(cells) + r' \\')

    tex_ood = render_table(
        caption_title='Summary of ESM-MSR Ablations and Variations: Out-of-Distribution Benchmarks',
        caption_body=(
            r"Spearman's $\rho$ (mean $\pm$ SD across 3 independent training seeds) on out-of-distribution benchmarks "
            r"under the deployed readout (WT-View on single-mutant thermostability sets, Domainome, and ProteinGym; Dual-View on PTMUL). "
            r"Models marked (Ret.) are retrained with modifications on the Homology-Restricted MegaScale split, "
            r"while those without use the exact same weights as Dual-View (Reference). "
            r"Thermostability columns match the 8 main-text benchmarks: Singles (7) averages pooled $\rho$ across Ssym, S461, K2369, S783, S2648, Q3421, and S8754; "
            r"PTMUL reports pooled $\rho$ on the multi-mutant thermostability set ($n=914$). "
            r"Held-out DMS columns strictly use the 4-part homology-held-out sets defined against Homology-Restricted MegaScale train: Domainome reports mean per-domain $\rho$ across 436 held-out domains ($n=451{,}569$) "
            r"and ProteinGym reports equal-category-weighted macro-average $\rho$ across 173 held-out assays (WT-View; Dual-View for reference and p1 differs by $<0.001$). "
            r"Highest performing Homology-Restricted models in each column are indicated in bold; Dual-View (Reference) is also bolded if its mean $\pm$ SD range overlaps the highest average performance. "
            r"$^\dagger$Trained on the full MegaScale split (\texttt{mega\_splits}) and evaluated on the identical OOD sets; excluded from bolding in all columns. "
            r"--- indicates not applicable or not evaluated."
        ),
        label='tab:si_ablations_ood',
        col_spec='l' + 'c' * len(ood_cols),
        header_lines=[
            r'{} & \multicolumn{2}{c}{Thermostability} & \multicolumn{2}{c}{Held-out DMS} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
            r'ID & Singles (7) & PTMUL & Domainome & ProteinGym \\',
        ],
        body_rows=rows_ood,
    )
    p1b = write_tex('table_S1b_ablations_ood.tex', tex_ood)
    return [p1a, p1b]


# --------------------------------------------------------------------------------------
# PART 2: MAIN-MODEL COMPREHENSIVE TABLES (wt-dddg-scatters CSVs + _extra_models_pvalues.csv)
# --------------------------------------------------------------------------------------
def _load_extra_df() -> pd.DataFrame:
    if EXTRA_CSV.exists():
        return pd.read_csv(EXTRA_CSV)
    return pd.DataFrame()


def _load_tobit_df() -> pd.DataFrame:
    if TOBIT_CSV.exists():
        return pd.read_csv(TOBIT_CSV)
    return pd.DataFrame()


def _apply_tobit_overrides(
    panel: dict[str, dict],
    msr_name: str,
    tobit_df: pd.DataFrame | None,
    section: str,
    benchmark: str,
    subset: str,
    primary_val_is_rmse: bool = False,
) -> None:
    """Apply Tobit-linked ESM-MSR stats, updated p-values, and ESM-MSR (unlinked) RMSE-only row."""
    if tobit_df is None or len(tobit_df) == 0:
        return
    sub = tobit_df[(tobit_df['section'] == section) & (tobit_df['benchmark'] == benchmark) & (tobit_df['subset'] == subset)]
    if len(sub) == 0:
        return
    for _, r in sub.iterrows():
        m = str(r['model'])
        is_un = bool(r.get('is_unlinked', False))
        v_val = float(r['value']) if pd.notna(r.get('value')) else np.nan
        v_r = float(r['pearson_r']) if pd.notna(r.get('pearson_r')) else np.nan
        v_rm = float(r['rmse']) if pd.notna(r.get('rmse')) else np.nan
        v_p = float(r['p_one_sided']) if pd.notna(r.get('p_one_sided')) else np.nan
        v_B = int(r['B']) if pd.notna(r.get('B')) else 10000

        if is_un or m == 'ESM-MSR (unlinked)':
            panel['ESM-MSR (unlinked)'] = {
                'val': np.nan, 'spearman': np.nan, 'pearson': np.nan,
                'rmse': v_rm, 'p': np.nan, 'p_prot': np.nan,
                'B': v_B, 'overlap': False, 'internal': True, 'hom_note': '---',
            }
            continue

        target_key = msr_name if m == 'ESM-MSR' else m
        if target_key not in panel:
            continue
        d = panel[target_key]
        if target_key in (msr_name, 'ESM-MSR σ=0.5'):
            if math.isfinite(v_val):
                d['val'] = v_val
                if not primary_val_is_rmse:
                    d['spearman'] = v_val
            if math.isfinite(v_r):
                d['pearson'] = v_r
            if math.isfinite(v_rm):
                d['rmse'] = v_rm
            if target_key != msr_name and math.isfinite(v_p):
                d['p'] = v_p
                d['B'] = v_B
        else:
            if math.isfinite(v_p):
                d['p'] = v_p
                d['B'] = v_B


def _extract_model_panel(
    df: pd.DataFrame,
    benchmark: str,
    metric: str,
    resampling_unit: str | None = None,
    primary_val_is_rmse: bool = False,
    dataset_key: str | None = None,
    multi: bool = False,
    extra_df: pd.DataFrame | None = None,
    extra_section: str | None = None,
) -> tuple[dict[str, dict], str, int, int, int]:
    """Extract {model: {val, spearman, pearson, rmse, p, p_prot, B, overlap, internal, hom_note}}."""
    sub = df[(df['test_type'] == 'pairwise') & (df['benchmark'] == benchmark) & (df['subset_or_metric'] == metric)]
    if resampling_unit is not None:
        sub = sub[sub['resampling_unit'] == resampling_unit]
    if len(sub) == 0:
        raise ValueError(f'No rows for {benchmark} / {metric} / {resampling_unit}')
    r0 = sub.iloc[0]
    msr_name = str(r0['msr_model'])
    B = int(r0['B']) if pd.notna(r0.get('B')) else 10000
    n_units = int(r0['n_units']) if pd.notna(r0.get('n_units')) else 0
    n_var = int(round(float(r0['n_variants']))) if pd.notna(r0.get('n_variants')) else 0
    ds_key = dataset_key or benchmark

    out: dict[str, dict] = {}
    msr_note, msr_ov = HOM.note_and_overlap(msr_name, ds_key, multi=multi)
    out[msr_name] = {
        'val': float(r0['value_msr']) if pd.notna(r0.get('value_msr')) else np.nan,
        'spearman': float(r0['spearman_rho_msr']) if 'spearman_rho_msr' in sub.columns and pd.notna(r0.get('spearman_rho_msr')) else (
            np.nan if primary_val_is_rmse else (float(r0['value_msr']) if pd.notna(r0.get('value_msr')) else np.nan)
        ),
        'pearson': float(r0['pearson_r_msr']) if 'pearson_r_msr' in sub.columns and pd.notna(r0.get('pearson_r_msr')) else np.nan,
        'rmse': float(r0['value_msr']) if primary_val_is_rmse else (
            float(r0['rmse_msr']) if 'rmse_msr' in sub.columns and pd.notna(r0.get('rmse_msr')) else np.nan
        ),
        'p': np.nan,
        'p_prot': np.nan,
        'B': B,
        'overlap': msr_ov,
        'internal': False,
        'hom_note': msr_note,
    }
    for _, row in sub.iterrows():
        comp = str(row['comparator'])
        if ds_key != 'tsuboyama_mega_test' and 'spurs' in comp.lower() and 'pub' in comp.lower():
            continue
        note, ov_hom = HOM.note_and_overlap(comp, ds_key, multi=multi)
        ov_csv = bool(row['excluded_training_overlap']) if 'excluded_training_overlap' in sub.columns and pd.notna(row.get('excluded_training_overlap')) else False
        iv = bool(row['internal_control_or_variant']) if 'internal_control_or_variant' in sub.columns and pd.notna(row.get('internal_control_or_variant')) else _is_internal_model(comp)
        out[comp] = {
            'val': float(row['value_comparator']) if pd.notna(row.get('value_comparator')) else np.nan,
            'spearman': float(row['spearman_rho_comparator']) if 'spearman_rho_comparator' in sub.columns and pd.notna(row.get('spearman_rho_comparator')) else (
                np.nan if primary_val_is_rmse else (float(row['value_comparator']) if pd.notna(row.get('value_comparator')) else np.nan)
            ),
            'pearson': float(row['pearson_r_comparator']) if 'pearson_r_comparator' in sub.columns and pd.notna(row.get('pearson_r_comparator')) else np.nan,
            'rmse': float(row['value_comparator']) if primary_val_is_rmse else (
                float(row['rmse_comparator']) if 'rmse_comparator' in sub.columns and pd.notna(row.get('rmse_comparator')) else np.nan
            ),
            'p': float(row['p_one_sided_uncorrected']) if pd.notna(row.get('p_one_sided_uncorrected')) else np.nan,
            'p_prot': np.nan,
            'B': int(row['B']) if pd.notna(row.get('B')) else B,
            'overlap': ov_csv or ov_hom,
            'internal': iv,
            'hom_note': note,
        }

    # Merge additional models from _extra_models_pvalues.csv (never overwrite existing wt-dddg-scatters rows!)
    if extra_df is not None and len(extra_df) > 0 and extra_section is not None:
        esub = extra_df[(extra_df['section'] == extra_section) & (extra_df['benchmark'] == benchmark) & (extra_df['subset'] == metric)]
        for _, er in esub.iterrows():
            m = str(er['model'])
            if ds_key != 'tsuboyama_mega_test' and 'spurs' in m.lower() and 'pub' in m.lower():
                continue
            if m in out:
                continue
            note, ov_hom = HOM.note_and_overlap(m, ds_key, multi=multi)
            out[m] = {
                'val': float(er['value']) if pd.notna(er.get('value')) else np.nan,
                'spearman': np.nan if primary_val_is_rmse else (float(er['value']) if pd.notna(er.get('value')) else np.nan),
                'pearson': float(er['pearson_r']) if pd.notna(er.get('pearson_r')) else np.nan,
                'rmse': float(er['rmse']) if pd.notna(er.get('rmse')) else np.nan,
                'p': float(er['p_one_sided']) if pd.notna(er.get('p_one_sided')) else np.nan,
                'p_prot': float(er['p_one_sided_prot']) if pd.notna(er.get('p_one_sided_prot')) else np.nan,
                'B': int(er['B']) if pd.notna(er.get('B')) else B,
                'overlap': ov_hom,
                'internal': _is_internal_model(m),
                'hom_note': note,
            }
    return out, msr_name, n_units, n_var, B


def _best_clean_set(panel: dict[str, dict], field: str, higher_is_better: bool = True, exclude_internal: bool = True) -> set[str]:
    vals = []
    for m, d in panel.items():
        if d.get('overlap'):
            continue
        if exclude_internal and d.get('internal'):
            continue
        v = d.get(field, np.nan)
        if v is not None and math.isfinite(v):
            vals.append((v, m))
    if not vals:
        return set()
    best = max(v for v, _ in vals) if higher_is_better else min(v for v, _ in vals)
    return {m for v, m in vals if round(v, 3) == round(best, 3)}


def build_table_s2a_megascale_overall(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> Path:
    df = pd.read_csv(MAIN_PVAL / 'figure1b_megascale_stab_destab_pvalues.csv')
    specs = [
        ('MegaScale_capped_test', 'per_domain_mean_spearman_overall', r'Overall ($n$=48,740)'),
        ('MegaScale_capped_test', 'per_domain_mean_spearman_stabilizing', r'Stabilizing ($\Delta\Delta G>0$, $n$=7,417)'),
        ('MegaScale_capped_test', 'per_domain_mean_spearman_destabilizing', r'Destabilizing ($\Delta\Delta G\le 0$, $n$=41,323)'),
    ]
    panels, all_models, msr_name = [], set(), None
    for bm, met, title in specs:
        pan, msr, _, _, B = _extract_model_panel(
            df, bm, met, dataset_key='tsu_capped_test', extra_df=extra_df, extra_section='megascale_s2a'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'megascale_s2a', bm, met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    header_lines = [
        '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' & {} \\',
        r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}',
        'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * len(panels)) + r' & Homology / Overlap \\',
    ]
    bests = [
        (_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False))
        for _, p, _ in panels
    ]
    body_rows = []
    for m in models:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsu_capped_test', multi=False)
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---', '---', '---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_rho)))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append('---' if m in (msr_name, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B))
        cells.append(note)
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='All-Model Comparison on Homology-Restricted MegaScale Test Set (Overall, Stabilizing, and Destabilizing Variants)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 33 held-out MegaScale test domains "
            r"for all models with available predictions (3-seed mean for retrained models). "
            r"ESM-MSR (and ESM-MSR $\sigma=0.5$) uses the Tobit observation link $h(\Delta G_{\mathrm{wt}} + \widehat{\Delta\Delta G}) - \Delta G_{\mathrm{wt}}$; "
            r"ESM-MSR (unlinked) reports raw latent RMSE only (excluded from bolding). "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_megascale_overall',
        col_spec='l' + 'c' * (4 * len(panels)) + 'l',
        header_lines=header_lines,
        body_rows=body_rows,
    )
    return write_tex('table_S2a_megascale_overall_stab_destab.tex', tex)


def build_table_s2b_megascale_doubles(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> Path:
    df = pd.read_csv(MAIN_PVAL / 'figure1c_doubles_add_vs_epi_pvalues.csv')
    specs = [
        ('MegaScale_capped_test_doubles', 'per_domain_mean_spearman_epistatic', r'Epistatic Readout ($n$=13,319)'),
        ('MegaScale_capped_test_doubles', 'per_domain_mean_spearman_additive', r'Additive Readout ($n$=13,319)'),
    ]
    panels, all_models, msr_name = [], set(), None
    for bm, met, title in specs:
        pan, msr, _, _, B = _extract_model_panel(
            df, bm, met, dataset_key='tsu_capped_test', multi=True, extra_df=extra_df, extra_section='megascale_s2b'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'megascale_s2b', bm, met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    bests = [
        (_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False))
        for _, p, _ in panels
    ]
    body_rows = []
    for m in models:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsu_capped_test', multi=True)
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---'] * 4
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_rho)))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append('---' if m in (msr_name, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B))
        cells.append(note)
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='All-Model Comparison on Homology-Restricted MegaScale Test Double Mutants (Epistatic vs. Additive Readout)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 13 held-out MegaScale double-mutant domains "
            r"($n=13{,}319$ double mutants; 3-seed mean for retrained models). "
            r"ESM-MSR (and ESM-MSR $\sigma=0.5$) uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ within each readout via paired domain-cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_megascale_doubles',
        col_spec='l' + 'c' * (4 * len(panels)) + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' & {} \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * len(panels)) + r' & Homology / Overlap \\',
        ],
        body_rows=body_rows,
    )
    return write_tex('table_S2b_megascale_doubles.tex', tex)


def build_table_s2c_megasplits(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> Path:
    df_a = pd.read_csv(SUPP_PVAL / '_mega_splits_fig1a_stab_destab_pvalues.csv')
    df_b = pd.read_csv(SUPP_PVAL / '_mega_splits_fig1b_doubles_pvalues.csv')

    specs_a = [
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_overall', r'Overall ($n$=49,048)'),
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_stabilizing', r'Stabilizing ($n$=8,791)'),
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_destabilizing', r'Destabilizing ($n$=40,257)'),
    ]
    panels_a, models_a, msr_a = [], set(), None
    for bm, met, title in specs_a:
        pan, msr, _, _, B = _extract_model_panel(
            df_a, bm, met, dataset_key='tsuboyama_mega_test', extra_df=extra_df, extra_section='megascale_s2c'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'megascale_s2c', bm, met)
        msr_a = msr
        panels_a.append((title, pan, B))
        models_a.update(pan.keys())
    models_a = sort_models(list(models_a), msr_first=msr_a)
    bests_a = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels_a]
    rows_a = []
    for m in models_a:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsuboyama_mega_test')
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels_a, bests_a):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m in (msr_a, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B),
            ]
        cells.append(note)
        rows_a.append(' & '.join(cells) + r' \\')

    tex_a = render_table(
        caption_title='All-Model Comparison on Full MegaScale Split Test Set: Overall, Stabilizing, and Destabilizing Variants',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 28 test domains of the full MegaScale split (\texttt{mega\_splits}; "
            r"16/28 test domains have sequence/structural homology to train/val). "
            r"ESM-MSR uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            r"Published SPURS and ThermoMPNN checkpoints (both trained on the identical \texttt{mega\_splits} split, verified by MD5) appear consecutively after their retrained counterparts."
        ),
        label='tab:si_megasplits_overall',
        col_spec='l' + 'c' * 12 + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels_a) + r' & {} \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 3) + r' & Homology / Overlap \\',
        ],
        body_rows=rows_a,
    )

    specs_b = [
        ('MegaScale_mega_splits_test_doubles', 'per_domain_mean_spearman_epistatic', r'Epistatic Readout ($n$=18,574)'),
        ('MegaScale_mega_splits_test_doubles', 'per_domain_mean_spearman_additive', r'Additive Readout ($n$=18,574)'),
    ]
    panels_b, models_b, msr_b = [], set(), None
    for bm, met, title in specs_b:
        pan, msr, _, _, B = _extract_model_panel(
            df_b, bm, met, dataset_key='tsuboyama_mega_test', multi=True, extra_df=extra_df, extra_section='megascale_s2c'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'megascale_s2c', bm, met)
        msr_b = msr
        panels_b.append((title, pan, B))
        models_b.update(pan.keys())
    models_b = sort_models(list(models_b), msr_first=msr_b)
    bests_b = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels_b]
    rows_b = []
    for m in models_b:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsuboyama_mega_test', multi=True)
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels_b, bests_b):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m in (msr_b, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B),
            ]
        cells.append(note)
        rows_b.append(' & '.join(cells) + r' \\')

    tex_b = render_table(
        caption_title='All-Model Comparison on Full MegaScale Split Test Set: Double Mutants (Epistatic vs. Additive Readout)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 20 double-mutant test domains of \texttt{mega\_splits} ($n=18{,}574$ doubles). "
            r"ESM-MSR uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected)."
        ),
        label='tab:si_megasplits_doubles',
        col_spec='l' + 'c' * 8 + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels_b) + r' & {} \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 2) + r' & Homology / Overlap \\',
        ],
        body_rows=rows_b,
    )
    return write_tex('table_S2c_megascale_megasplits.tex', tex_a + '\n' + tex_b)


def build_table_s2d_backbone_origin(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> Path:
    df = pd.read_csv(SUPP_PVAL / '_backbone_and_origin_pvalues.csv')
    specs = [
        ('MegaScale_capped_test_backbone', 'per_code_mean_spearman_native', 'Native Backbone'),
        ('MegaScale_capped_test_backbone', 'per_code_mean_spearman_destabilized', 'Destabilized Backbone'),
        ('MegaScale_capped_test_origin', 'per_group_mean_spearman_natural', 'Natural Domains'),
        ('MegaScale_capped_test_origin', 'per_group_mean_spearman_synthetic', 'Synthetic Domains'),
    ]
    panels, all_models, msr_name = [], set(), None
    for bm, met, title in specs:
        pan, msr, _, n_v, B = _extract_model_panel(
            df, bm, met, dataset_key='tsu_capped_test', extra_df=extra_df, extra_section='megascale_s2d'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'megascale_s2d', bm, met)
        msr_name = msr
        panels.append((f'{title} ($n$={n_v:,})', pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)
    bests = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels]

    body_rows = []
    for m in models:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsu_capped_test')
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m in (msr_name, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B),
            ]
        cells.append(note)
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='All-Model Comparison on Homology-Restricted MegaScale Test Set Stratified by Backbone Stability and Domain Origin',
        caption_body=(
            r"Mean per-library Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) stratified by wild-type backbone type "
            r"(native vs.\ pre-destabilized) and protein origin (natural vs.\ de novo synthetic domains). "
            r"ESM-MSR (and ESM-MSR $\sigma=0.5$) uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_megascale_backbone_origin',
        col_spec='l' + 'c' * 16 + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' & {} \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 4) + r' & Homology / Overlap \\',
        ],
        body_rows=body_rows,
    )
    return write_tex('table_S2d_megascale_backbone_origin.tex', tex)


def build_table_s3_epistasis(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> tuple[Path, Path]:
    df = pd.read_csv(MAIN_PVAL / 'figure2e_epistasis_hierarchy_pvalues.csv')
    specs = [
        ('naive_rho (all_doubles)', r'Naive $\rho$'),
        ('beyond_rho (all_doubles)', r'Beyond $\rho$'),
        ('pair_rho (all_doubles)', r'Pair $\rho$'),
        ('line_rho (all_doubles)', r'Line $\rho$'),
        ('cell_mag_pooled (all_doubles)', r'Cell mag. (all)'),
        ('cell_mag_gt0 (dG_add_gt0)', r'Cell mag. ($\Delta G_{\mathrm{add}}>0$)'),
    ]
    panels, all_models, msr_name = [], set(), None
    for met, title in specs:
        pan, msr, _, _, B = _extract_model_panel(
            df, 'MegaScale_epistasis_5level_hierarchy', met,
            dataset_key='tsu_capped_test', multi=True, extra_df=extra_df, extra_section='epistasis_s3a'
        )
        _apply_tobit_overrides(pan, msr, tobit_df, 'epistasis_s3a', 'MegaScale_epistasis_5level_hierarchy', met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    pan_naive = panels[0][1]
    b_r_naive = _best_clean_set(pan_naive, 'pearson', True)
    b_rm_naive = _best_clean_set(pan_naive, 'rmse', False)
    bests = [_best_clean_set(p, 'val', True) for _, p, _ in panels]

    rows_3a = []
    for m in models:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsu_capped_test', multi=True)
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), b_set in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_set)))
                cells.append('---' if m in (msr_name, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B))
        dn = pan_naive.get(m, {})
        cells.append(fmt_num(dn.get('pearson'), 3, bold=(m in b_r_naive)))
        cells.append(fmt_num(dn.get('rmse'), 3, bold=(m in b_rm_naive)))
        cells.append(note)
        rows_3a.append(' & '.join(cells) + r' \\')

    tex_3a = render_table(
        caption_title='All-Model Comparison Across the 5-Level Epistasis Hierarchy on Homology-Restricted MegaScale Test Doubles',
        caption_body=(
            r"Evaluation of predicted epistasis ($\widehat{\Delta\Delta\Delta G} = \hat{y}_{AB} - \hat{y}_A - \hat{y}_B$) on the 12,455 "
            r"MegaScale test doubles with both constituent singles measured across 13 domains (7,405 doubles with $\Delta G_{\mathrm{add}}>0$). "
            r"ESM-MSR (and ESM-MSR $\sigma=0.5$) uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent $\Delta\Delta\Delta G$ RMSE only. "
            r"One-sided $p$-values test $H_1\colon \text{ESM-MSR} > \text{model}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected; "
            r"point metrics shown for supplementary external comparators without paired bootstrap runs). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_epistasis_hierarchy',
        col_spec='l' + 'c' * (2 * len(panels) + 2) + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels) + r' & \multicolumn{2}{c}{All Doubles $\Delta\Delta\Delta G$} & {} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15}',
            'ID & ' + ' & '.join([r'Score & $p$'] * len(panels)) + r' & $r$ & RMSE & Homology / Overlap \\',
        ],
        body_rows=rows_3a,
    )
    p3a = write_tex('table_S3a_epistasis_hierarchy.tex', tex_3a)

    df_rm = pd.read_csv(SUPP_PVAL / '_epi_rmse_pvalues.csv')
    df_mc = pd.read_csv(SUPP_PVAL / '_epi_mcc_pvalues.csv')
    p_rm_all, msr_rm, _, _, B_rm = _extract_model_panel(
        df_rm, 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles', primary_val_is_rmse=True,
        dataset_key='tsu_capped_test', multi=True, extra_df=extra_df, extra_section='epistasis_s3b_rmse'
    )
    _apply_tobit_overrides(p_rm_all, msr_rm, tobit_df, 'epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles', primary_val_is_rmse=True)
    p_rm_gt0, _, _, _, _ = _extract_model_panel(
        df_rm, 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0', primary_val_is_rmse=True,
        dataset_key='tsu_capped_test', multi=True, extra_df=extra_df, extra_section='epistasis_s3b_rmse'
    )
    _apply_tobit_overrides(p_rm_gt0, msr_rm, tobit_df, 'epistasis_s3b_rmse', 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0', primary_val_is_rmse=True)
    p_mc_raw, msr_mc, _, _, B_mc = _extract_model_panel(
        df_mc, 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_raw_dddG_gt0', dataset_key='tsu_capped_test', multi=True
    )
    p_mc_sat, _, _, _, _ = _extract_model_panel(
        df_mc, 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_saturation_corrected_r_gt0', dataset_key='tsu_capped_test', multi=True
    )
    # Harmonize multi-mutant comparator names between RMSE and MCC CSVs so each model appears on one row
    _mc_alias = {'SPURS': 'SPURS-multi', 'MutateEverything': 'MutateEverything Stage 2', 'ThermoMPNN': 'ThermoMPNN-D'}
    for _pan in (p_mc_raw, p_mc_sat):
        for _old, _new in _mc_alias.items():
            if _old in _pan and _new not in _pan:
                _pan[_new] = _pan.pop(_old)
    _apply_tobit_overrides(p_mc_raw, msr_mc, tobit_df, 'epistasis_s3b_mcc', 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_raw_dddG_gt0')

    all_m3b = sort_models(list(set(p_rm_all) | set(p_rm_gt0) | set(p_mc_raw) | set(p_mc_sat)), msr_first=msr_rm)
    b_rm1, b_r1 = _best_clean_set(p_rm_all, 'rmse', False), _best_clean_set(p_rm_all, 'pearson', True)
    b_rm2, b_r2 = _best_clean_set(p_rm_gt0, 'rmse', False), _best_clean_set(p_rm_gt0, 'pearson', True)
    b_m1, b_m2 = _best_clean_set(p_mc_raw, 'val', True), _best_clean_set(p_mc_sat, 'val', True)

    rows_3b = []
    for m in all_m3b:
        note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, 'tsu_capped_test', multi=True)
        cells = [clean_model_display(m, overlap=ov)]
        for pan, b_r, b_rm in ((p_rm_all, b_r1, b_rm1), (p_rm_gt0, b_r2, b_rm2)):
            d = pan.get(m)
            cells += ['---'] * 3 if not d else [
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                '---' if m in (msr_rm, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B_rm),
            ]
        for pan, b_mc in ((p_mc_raw, b_m1), (p_mc_sat, b_m2)):
            d = pan.get(m)
            cells += ['---'] * 2 if (not d or m == 'ESM-MSR (unlinked)') else [
                fmt_num(d['val'], 3, bold=(m in b_mc)),
                '---' if m == msr_mc else fmt_p(d['p'], B_mc),
            ]
        cells.append(note)
        rows_3b.append(' & '.join(cells) + r' \\')

    tex_3b = render_table(
        caption_title=r'Epistasis $\Delta\Delta\Delta G$ Calibration (RMSE, Pearson $r$) and Sign Classification (MCC) on MegaScale Test Doubles',
        caption_body=(
            r"Epistasis calibration RMSE (kcal/mol, lower is better) and Pearson's $r$ on all 12,455 test doubles and 7,405 unsaturated doubles ($\Delta G_{\mathrm{add}}>0$), "
            r"alongside Matthews Correlation Coefficient (MCC) for classifying epistasis sign on raw $\Delta\Delta\Delta G>0$ and saturation-corrected residuals $r>0$. "
            r"ESM-MSR (and ESM-MSR $\sigma=0.5$) uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test ESM-MSR superiority via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_epistasis_calibration_mcc',
        col_spec='l' + 'c' * 10 + 'l',
        header_lines=[
            r'{} & \multicolumn{3}{c}{All Doubles ($n$=12,455)} & \multicolumn{3}{c}{$\Delta G_{\mathrm{add}}>0$ ($n$=7,405)} & \multicolumn{2}{c}{Sign MCC (Raw)} & \multicolumn{2}{c}{Sign MCC (Sat.-Corr.)} & {} \\',
            r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}',
            r'ID & RMSE & $r$ & $p_{\mathrm{RMSE}}$ & RMSE & $r$ & $p_{\mathrm{RMSE}}$ & MCC & $p_{\mathrm{MCC}}$ & MCC & $p_{\mathrm{MCC}}$ & Homology / Overlap \\',
        ],
        body_rows=rows_3b,
    )
    p3b = write_tex('table_S3b_epistasis_calibration_mcc.tex', tex_3b)
    return p3a, p3b


def build_table_s4_thermostability_longtable(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> tuple[Path, Path]:
    """Build ONE massive vertical multi-page longtable stacking ALL 12 external thermostability sets."""
    df_3a = pd.read_csv(MAIN_PVAL / 'figure3a_thermostability_pvalues.csv')

    # Natural alphabetical order across all 12 external thermostability benchmarks,
    # with explicit subset/superset relation notes verified at sequence/variant level.
    dataset_specs = [
        ('K2369', 'k2369', False, True, r'$\subset$ K3822; 91\% in S8754'),
        ('K3822', 'k3822', False, False, r'$\supset$ K2369; 89\% in S8754'),
        ('PTMUL', 'ptmul', True, True, r'$\supset$ PTMUL-D'),
        ('PTMUL-D', 'ptmuld', True, False, r'doubles $\subset$ PTMUL'),
        ('Q3421', 'q3421', False, True, r'98\% in S8754'),
        ('S461', 's461', False, True, r'$\subset$ S669'),
        ('S571', 's571', False, False, ''),
        ('S669', 's669', False, False, r'$\supset$ S461'),
        ('S783', 's783', False, True, r'88\% in S8754'),
        ('S2648', 's2648', False, True, r'$\subset$ S8754'),
        ('S8754', 's8754', False, True, r'$\supset$ S2648, Ssym direct; 88--98\% of K2369, K3822, Q3421, S783'),
        ('Ssym', 'ssym', False, True, r'342 direct/rev.\ pairs; direct $\subset$ S8754'),
    ]

    body_rows: list[str] = []
    for idx_ds, (bm_label, ds_key, is_multi, in_3a, rel_note) in enumerate(dataset_specs):
        if in_3a:
            pm, msr_name, _, n_var, B = _extract_model_panel(
                df_3a, bm_label, 'pooled_spearman_rho', 'mutation_x_seed',
                dataset_key=ds_key, multi=is_multi, extra_df=extra_df, extra_section='thermostability',
            )
            pp, _, n_prot, _, _ = _extract_model_panel(
                df_3a, bm_label, 'pooled_spearman_rho', 'protein_cluster_x_seed',
                dataset_key=ds_key, multi=is_multi,
            )
            # Attach p_prot from figure3a for main models
            for m_k, d_p in pp.items():
                if m_k in pm and not math.isfinite(pm[m_k].get('p_prot', np.nan)):
                    pm[m_k]['p_prot'] = d_p['p']
        else:
            # All 19 models come from _extra_models_pvalues.csv
            esub = extra_df[(extra_df['section'] == 'thermostability') & (extra_df['dataset_key'] == ds_key)]
            if len(esub) == 0:
                continue
            r0 = esub.iloc[0]
            msr_name = 'ESM-MSR'
            B = int(r0['B'])
            n_prot = int(r0['n_units'])
            n_var = int(r0['n_variants'])
            msr_note, msr_ov = HOM.note_and_overlap(msr_name, ds_key, multi=is_multi)
            pm = {
                msr_name: {
                    'val': float(r0['value_msr']),
                    'pearson': float(r0['pearson_r_msr']),
                    'rmse': float(r0['rmse_msr']),
                    'p': np.nan, 'p_prot': np.nan, 'B': B,
                    'overlap': msr_ov, 'internal': False, 'hom_note': msr_note,
                }
            }
            for _, er in esub.iterrows():
                m_k = str(er['model'])
                note, ov_h = HOM.note_and_overlap(m_k, ds_key, multi=is_multi)
                pm[m_k] = {
                    'val': float(er['value']),
                    'pearson': float(er['pearson_r']),
                    'rmse': float(er['rmse']),
                    'p': float(er['p_one_sided']),
                    'p_prot': float(er['p_one_sided_prot']),
                    'B': int(er['B']),
                    'overlap': ov_h,
                    'internal': _is_internal_model(m_k),
                    'hom_note': note,
                }

        models = sort_models(list(pm.keys()), msr_first=msr_name)
        b_rho = _best_clean_set(pm, 'val', True)
        b_r = _best_clean_set(pm, 'pearson', True)
        b_rm = _best_clean_set(pm, 'rmse', False)

        if idx_ds > 0:
            body_rows.append(r'\midrule')
        tag_type = r'single-mutant $\Delta T_m$ ($^\circ$C)' if ds_key == 's571' else ('multi-mutant' if is_multi else 'single-mutant')
        sec_tag = 'Main-text' if in_3a else 'Additional'
        rel_part = f'; {rel_note}' if rel_note else ''
        body_rows.append(
            f'\\multicolumn{{7}}{{l}}{{\\textbf{{{bm_label}}} ({sec_tag}, {tag_type}{rel_part}; $n={n_var:,}$ variants, {n_prot} protein clusters)}} \\\\*'
        )
        body_rows.append(r'\midrule')
        for m in models:
            d = pm[m]
            cells = [
                clean_model_display(m, overlap=d['overlap']),
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m == msr_name else fmt_p(d['p'], d['B']),
                '---' if m == msr_name else fmt_p(d['p_prot'], d['B']),
                d['hom_note'],
            ]
            body_rows.append(' & '.join(cells) + r' \\')

    tex_s4 = render_longtable(
        caption_title='Comprehensive All-Model Comparison Across All 12 External Thermostability Benchmarks',
        caption_body=(
            r"Vertical breakdown stacking all 12 external thermostability benchmarks in alphabetical order "
            r"(K2369, K3822, PTMUL, PTMUL-D, Q3421, S461, S571, S669, S783, S2648, S8754, Ssym; 8 main-text and 4 additional sets), "
            r"with section headers indicating exact subset/superset relationships ($\subset$/$\supset$) and major compilation overlaps "
            r"(strict curated pairs: K2369 $\subset$ K3822, PTMUL-D $\subset$ PTMUL, S461 $\subset$ S669, S2648 $\subset$ S8754, and Ssym direct $\subset$ S8754). "
            r"Every model with available predictions is shown "
            r"(singles-only heads on single-mutant datasets; multi-mutant heads on PTMUL and PTMUL-D; 3-seed mean for retrained models). "
            r"Columns report pooled Spearman's $\rho$, Pearson's $r$, raw (unrecalibrated) RMSE (kcal/mol, except S571 which measures $\Delta T_m$ in $^\circ$C), one-sided paired bootstrap $p$-values testing "
            r"$H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ under mutation $\times$ seed ($p_{\mathrm{mut}}$) and protein-cluster $\times$ seed ($p_{\mathrm{prot}}$) "
            r"resampling ($B=10{,}000$; uncorrected), and per-(dataset, model) homology / training-set overlap notes. "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_thermostability_all',
        col_spec='lcccccl',
        header_lines=[
            r'Model & $\rho$ & $r$ & RMSE & $p_{\mathrm{mut}}$ & $p_{\mathrm{prot}}$ & Homology / Overlap \\',
        ],
        body_rows=body_rows,
        ncols=7,
        fontsize=r'\scriptsize',
    )
    p4_all = write_tex('table_S4_thermostability_all.tex', tex_s4)

    # Keep Table S4c: Calibration RMSE across MegaScale + 8 External & Ssym Direct-Reverse Antisymmetry
    df_rmse = pd.read_csv(SUPP_PVAL / '_thermostability_rmse_pvalues.csv')
    bms_r = [
        ('tsuboyama_capped_test', 'MegaScale', 'tsu_capped_test', False),
        ('ssym', 'Ssym', 'ssym', False),
        ('s461', 'S461', 's461', False),
        ('k2369', 'K2369', 'k2369', False),
        ('s783', 'S783', 's783', False),
        ('s2648', 'S2648', 's2648', False),
        ('q3421', 'Q3421', 'q3421', False),
        ('s8754', 'S8754', 's8754', False),
        ('ptmul', 'PTMUL', 'ptmul', True),
    ]
    panels_raw, all_mr, msr_r = [], set(), None
    for bk, label_b, ds_k, is_m in bms_r:
        prw, msr, _, _, B = _extract_model_panel(df_rmse, bk, 'rmse_raw', primary_val_is_rmse=True, dataset_key=ds_k, multi=is_m)
        if bk == 'tsuboyama_capped_test':
            _apply_tobit_overrides(prw, msr, tobit_df, 'thermo_rmse_s4c', bk, 'rmse_raw', primary_val_is_rmse=True)
        msr_r = msr
        panels_raw.append((label_b, prw, B))
        all_mr.update(prw.keys())
    models_r = sort_models(list(all_mr), msr_first=msr_r)
    bests_l = [_best_clean_set(p, 'rmse', False) for _, p, _ in panels_raw]
    rows_r = []
    for m in models_r:
        cells = [clean_model_display(m)]
        for (_, pan, B), b_set in zip(panels_raw, bests_l):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                sfx = r'$^*$' if d.get('overlap') else ''
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_set), suffix=sfx))
                cells.append('---' if m in (msr_r, 'ESM-MSR (unlinked)') else fmt_p(d['p'], B))
        rows_r.append(' & '.join(cells) + r' \\')
    cmids = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(len(panels_raw)))
    tex_raw = render_table(
        caption_title='Raw Calibration RMSE Comparison Across MegaScale and 8 External Thermostability Benchmarks',
        caption_body=(
            r"Raw (unrecalibrated) RMSE (kcal/mol, lower is better) across Homology-Restricted MegaScale test (33 domains) and 8 external thermostability sets. "
            r"On MegaScale, ESM-MSR uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE on MegaScale only (external thermostability sets have no cDNA proteolysis $\Delta G_{\mathrm{wt}}$). "
            r"One-sided $p$-values test $H_1\colon \text{RMSE}_{\text{ESM-MSR}} < \text{RMSE}_{\text{model}}$ via paired bootstrap "
            r"(domain-cluster $\times$ seed on MegaScale; mutation $\times$ seed on external sets; $B=10{,}000$; uncorrected). "
            r"Lowest clean (non-overlapping) RMSE per column is bolded. $^*$Documented training overlap (see Table~\ref{tab:si_thermostability_all} for full overlap codes)."
        ),
        label='tab:si_thermo_rmse_raw',
        col_spec='l' + 'c' * (2 * len(panels_raw)),
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_raw) + r' \\',
            cmids,
            'ID & ' + ' & '.join([r'RMSE & $p$'] * len(panels_raw)) + r' \\',
        ],
        body_rows=rows_r,
    )

    df_as = pd.read_csv(SUPP_PVAL / '_ssym_antisymmetry_bias_pvalues.csv')
    pan_r_dr, msr_as, _, _, B_as = _extract_model_panel(df_as, 'Ssym_342_pairs', 'antisymmetry_r_dir_rev', dataset_key='ssym')
    pan_bias, _, _, _, _ = _extract_model_panel(df_as, 'Ssym_342_pairs', 'abs_bias_delta', dataset_key='ssym')
    models_as = sort_models(list(set(pan_r_dr) | set(pan_bias)), msr_first=msr_as)
    b_rdr = _best_clean_set(pan_r_dr, 'val', higher_is_better=False)
    b_bias = _best_clean_set(pan_bias, 'val', higher_is_better=False)
    b_pr = _best_clean_set(pan_r_dr, 'pearson', True)
    b_rm = _best_clean_set(pan_r_dr, 'rmse', False)
    rows_as = []
    for m in models_as:
        d1 = pan_r_dr.get(m, {})
        d2 = pan_bias.get(m, {})
        note, ov = HOM.note_and_overlap(m, 'ssym')
        cells = [
            clean_model_display(m, overlap=ov),
            fmt_num(d1.get('val'), 3, bold=(m in b_rdr)),
            '---' if m == msr_as else fmt_p(d1.get('p'), B_as),
            fmt_num(d2.get('val'), 3, bold=(m in b_bias)),
            '---' if m == msr_as else fmt_p(d2.get('p'), B_as),
            fmt_num(d1.get('pearson'), 3, bold=(m in b_pr)),
            fmt_num(d1.get('rmse'), 3, bold=(m in b_rm)),
            note,
        ]
        rows_as.append(' & '.join(cells) + r' \\')
    tex_as = render_table(
        caption_title='Ssym Direct--Reverse Antisymmetry and Bias Across Models (342 Mutation Pairs)',
        caption_body=(
            r"Direct--reverse Pearson correlation $r_{\mathrm{dir,rev}}$ (ideal $-1.000$), absolute mean antisymmetry bias $|\langle\delta\rangle|$ (kcal/mol, ideal $0.000$), "
            r"pooled Ssym Pearson's $r$, and raw RMSE (kcal/mol) across 342 direct/reverse mutation pairs ($n=684$). "
            r"One-sided $p$-values test ESM-MSR (Dual-view) superiority via paired mutation-pair $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_ssym_antisymmetry',
        col_spec='lccccccl',
        header_lines=[
            r'{} & \multicolumn{2}{c}{Antisymmetry ($r_{\mathrm{dir,rev}}$)} & \multicolumn{2}{c}{Abs. Bias ($|\langle\delta\rangle|$)} & \multicolumn{2}{c}{Pooled Ssym} & {} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}',
            r'ID & $r_{\mathrm{dir,rev}}$ & $p$ & $|\langle\delta\rangle|$ & $p$ & $r$ & RMSE & Homology / Overlap \\',
        ],
        body_rows=rows_as,
    )
    p4c = write_tex('table_S4c_thermostability_rmse_and_ssym_antisym.tex', tex_raw + '\n' + tex_as)
    return p4_all, p4c


def build_table_s5_heldout_ood(extra_df: pd.DataFrame, tobit_df: pd.DataFrame) -> tuple[Path, Path]:
    """Vertical multi-section longtable for all 4 Held-Out OOD benchmarks with ALL models & per-dataset homology notes."""
    df = pd.read_csv(MAIN_PVAL / 'figure3b_heldout_benchmarks_pvalues.csv')
    specs = [
        ('ProteinGym_heldout_173_assays', 'category_weighted_mean_spearman_rho', 'proteingym_heldout', r'ProteinGym (173 held-out DMS assays; $n=2{,}351{,}490$; category-weighted macro average)', False),
        ('Domainome_heldout_436_domains', 'per_domain_mean_spearman_rho', 'domainome_heldout', r'Domainome (436 held-out domains; $n=451{,}569$; per-domain mean)', True),
        ('MGnify_heldout_pooled', 'pooled_spearman_rho', 'mgnify', r'MGnify Stability (140,694 held-out scaffolds; $n=612{,}406$; pooled)', True),
        ('Rocklin_designs_heldout_pooled', 'pooled_spearman_rho', 'rocklin_denovo', r'Rocklin De Novo Designs (623 held-out scaffolds; $n=109{,}581$; pooled)', True),
    ]

    body_rows: list[str] = []
    for idx_s, (bm, met, ds_key, sec_title, has_rmse) in enumerate(specs):
        pan, msr, _, _, B = _extract_model_panel(
            df, bm, met, dataset_key=ds_key, extra_df=extra_df, extra_section='ood_s5a'
        )
        if msr != 'ESM-MSR':
            pan['ESM-MSR'] = pan.pop(msr)
            msr = 'ESM-MSR'
        if bm in ('MGnify_heldout_pooled', 'Rocklin_designs_heldout_pooled'):
            _apply_tobit_overrides(pan, msr, tobit_df, 'ood_s5a_unlinked', bm, met)
        models = sort_models(list(pan.keys()), msr_first=msr)
        b_rho = _best_clean_set(pan, 'val', True)
        b_r = _best_clean_set(pan, 'pearson', True)
        b_rm = _best_clean_set(pan, 'rmse', False) if has_rmse else set()

        if idx_s > 0:
            body_rows.append(r'\midrule')
        body_rows.append(f'\\multicolumn{{6}}{{l}}{{\\textbf{{{sec_title}}}}} \\\\*')
        body_rows.append(r'\midrule')
        for m in models:
            d = pan[m]
            cells = [
                clean_model_display(m, overlap=d['overlap']),
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)) if has_rmse else '---',
                '---' if m in (msr, 'ESM-MSR (unlinked)') else fmt_p(d['p'], d['B']),
                d['hom_note'],
            ]
            body_rows.append(' & '.join(cells) + r' \\')

    tex_5a = render_longtable(
        caption_title='Comprehensive All-Model Comparison on Large-Scale Held-Out Out-of-Distribution Benchmarks',
        caption_body=(
            r"Comparison of all models with available predictions across the four 4-part homology-held-out benchmarks "
            r"(ProteinGym 173 assays, Domainome 436 domains, MGnify pooled, and Rocklin de novo designs pooled). "
            r"RMSE is omitted on ProteinGym because assays span heterogeneous fitness scales. "
            r"On MGnify and Rocklin, ESM-MSR uses the Tobit observation link; ESM-MSR (unlinked) reports raw latent RMSE only. "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired assay/domain/scaffold $\times$ seed bootstrap "
            r"($B=10{,}000$ on ProteinGym and Domainome; $B=1{,}000$ on MGnify and Rocklin; uncorrected). "
            r"Internal variant ESM-MSR ($\sigma=0.5$) is excluded from bolding of external comparators. "
            r"ProteinGym overlap codes for external models reflect the full ProteinGym audit (GeoDDG-3D's encoder was pretrained on GeoFitness, 74 DMS assays); "
            r"Domainome overlap codes reflect the held-out Domainome audit. "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_heldout_ood',
        col_spec='lccccl',
        header_lines=[
            r'Model & $\rho$ & $r$ & RMSE & $p$ & Homology / Overlap \\',
        ],
        body_rows=body_rows,
        ncols=6,
        fontsize=r'\scriptsize',
    )
    p5a = write_tex('table_S5a_heldout_ood_benchmarks.tex', tex_5a)

    # Table S5b: ProteinGym LoRA scale sigma sweep across 5 categories
    df_3c = pd.read_csv(MAIN_PVAL / 'figure3c_proteingym_sigma_pvalues.csv')
    src_3c = pd.read_csv(MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv')
    held = src_3c[(~src_3c['leaked']) & (~src_3c['dropped_large'])]
    cats = [
        ('Stability', 'Stability'), ('Expression', 'Expression'), ('Activity', 'Activity'),
        ('Binding', 'Binding'), ('OrganismalFitness', 'Org. Fitness'),
    ]
    rows_5b = []
    for cat_key, cat_disp in cats:
        g = held[held['category'] == cat_key]
        n_a = len(g)
        r0, r05, r10 = float(g['rho_dep_0.0'].mean()), float(g['rho_dep_0.5'].mean()), float(g['rho_dep_1.0'].mean())
        pr0, pr05, pr10 = float(g['pearson_r_dep_0.0'].mean()), float(g['pearson_r_dep_0.5'].mean()), float(g['pearson_r_dep_1.0'].mean())
        sub_p = df_3c[df_3c['subset_or_metric'] == f'{cat_key}: mean_delta_rho']
        p_05_vs_0 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=0.5)') & (sub_p['comparator'] == 'ESM3-small-open (σ=0)')]['p_one_sided_uncorrected'].iloc[0]
        p_10_vs_0 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=1.0)') & (sub_p['comparator'] == 'ESM3-small-open (σ=0)')]['p_one_sided_uncorrected'].iloc[0]
        p_10_vs_05 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=1.0)') & (sub_p['comparator'] == 'ESM-MSR (σ=0.5)')]['p_one_sided_uncorrected'].iloc[0]
        B_3c = int(sub_p['B'].iloc[0])
        best_rho, best_pr = max(r0, r05, r10), max(pr0, pr05, pr10)
        rows_5b.append(' & '.join([
            f'{cat_disp} ($n$={n_a})',
            fmt_num(r0, 3, bold=(round(r0, 3) == round(best_rho, 3))),
            fmt_num(pr0, 3, bold=(round(pr0, 3) == round(best_pr, 3))),
            fmt_num(r05, 3, bold=(round(r05, 3) == round(best_rho, 3))),
            fmt_num(pr05, 3, bold=(round(pr05, 3) == round(best_pr, 3))),
            fmt_p(p_05_vs_0, B_3c),
            fmt_num(r10, 3, bold=(round(r10, 3) == round(best_rho, 3))),
            fmt_num(pr10, 3, bold=(round(pr10, 3) == round(best_pr, 3))),
            fmt_p(p_10_vs_0, B_3c),
            fmt_p(p_10_vs_05, B_3c),
        ]) + r' \\')
    cat_r0, cat_r05, cat_r10 = (held.groupby('category')[f'rho_dep_{s}'].mean().mean() for s in ('0.0', '0.5', '1.0'))
    cat_p0, cat_p05, cat_p10 = (held.groupby('category')[f'pearson_r_dep_{s}'].mean().mean() for s in ('0.0', '0.5', '1.0'))
    pg_sub = df[(df['benchmark'] == 'ProteinGym_heldout_173_assays') & (df['test_type'] == 'pairwise')]
    p_mac_10_0 = pg_sub[pg_sub['comparator'] == 'ESM3-small-open']['p_one_sided_uncorrected'].iloc[0]
    p_mac_10_05 = pg_sub[pg_sub['comparator'] == 'ESM-MSR σ=0.5']['p_one_sided_uncorrected'].iloc[0]
    best_mr, best_mp = max(cat_r0, cat_r05, cat_r10), max(cat_p0, cat_p05, cat_p10)
    rows_5b.append(r'\midrule')
    rows_5b.append(' & '.join([
        r'Macro average (173 assays)',
        fmt_num(cat_r0, 3, bold=(round(cat_r0, 3) == round(best_mr, 3))),
        fmt_num(cat_p0, 3, bold=(round(cat_p0, 3) == round(best_mp, 3))),
        fmt_num(cat_r05, 3, bold=(round(cat_r05, 3) == round(best_mr, 3))),
        fmt_num(cat_p05, 3, bold=(round(cat_p05, 3) == round(best_mp, 3))),
        '---',
        fmt_num(cat_r10, 3, bold=(round(cat_r10, 3) == round(best_mr, 3))),
        fmt_num(cat_p10, 3, bold=(round(cat_p10, 3) == round(best_mp, 3))),
        fmt_p(p_mac_10_0, 10000),
        fmt_p(p_mac_10_05, 10000),
    ]) + r' \\')
    tex_5b = render_table(
        caption_title=r'ESM-MSR LoRA Scale ($\sigma$) Breakdown Across the 5 Held-Out ProteinGym Functional Categories',
        caption_body=(
            r"Per-category mean Spearman's $\rho$ and Pearson's $r$ across the 173 held-out ProteinGym DMS assays under the deployed readout for "
            r"zero-shot ESM3-small-open ($\sigma=0$), ESM-MSR ($\sigma=0.5$), and ESM-MSR ($\sigma=1.0$). "
            r"One-sided paired assay $\times$ seed bootstrap $p$-values ($B=10{,}000$; uncorrected). Best $\rho$ and $r$ per row are indicated in bold."
        ),
        label='tab:si_proteingym_sigma',
        col_spec='lccccccccc',
        header_lines=[
            r'{} & \multicolumn{2}{c}{$\sigma=0$ (ESM3)} & \multicolumn{3}{c}{ESM-MSR ($\sigma=0.5$)} & \multicolumn{4}{c}{ESM-MSR ($\sigma=1.0$)} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-10}',
            r'Category & $\rho$ & $r$ & $\rho$ & $r$ & $p_{0.5\text{ vs }0}$ & $\rho$ & $r$ & $p_{1.0\text{ vs }0}$ & $p_{1.0\text{ vs }0.5}$ \\',
        ],
        body_rows=rows_5b,
    )
    p5b = write_tex('table_S5b_proteingym_sigma_categories.tex', tex_5b)
    return p5a, p5b


def build_table_s6_screening_and_ndcg(tobit_df: pd.DataFrame) -> tuple[Path, Path]:
    df_n = pd.read_csv(SUPP_PVAL / '_ndcg96_singles_doubles_pvalues.csv')
    specs_n = [
        ('ndcg96_shifted_overall', 'Shifted Overall'), ('ndcg96_shifted_singles', 'Shifted Singles'),
        ('ndcg96_shifted_doubles', 'Shifted Doubles'), ('ndcg96_rectified_overall', 'Rectified Overall'),
        ('ndcg96_rectified_singles', 'Rectified Singles'), ('ndcg96_rectified_doubles', 'Rectified Doubles'),
    ]
    panels_n, all_mn, msr_n = [], set(), None
    for met, title in specs_n:
        pan, msr, _, _, B = _extract_model_panel(df_n, 'MegaScale_capped_test', met, dataset_key='tsu_capped_test')
        msr_n = msr
        panels_n.append((title, pan, B))
        all_mn.update(pan.keys())
    models_n = sort_models(list(all_mn), msr_first=msr_n)
    bests_n = [_best_clean_set(p, 'val', True) for _, p, _ in panels_n]
    rows_n = []
    for m in models_n:
        note, ov = HOM.note_and_overlap(m, 'tsu_capped_test')
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), b_set in zip(panels_n, bests_n):
            d = pan.get(m)
            cells += ['---', '---'] if not d else [fmt_num(d['val'], 3, bold=(m in b_set)), '---' if m == msr_n else fmt_p(d['p'], B)]
        cells.append(note)
        rows_n.append(' & '.join(cells) + r' \\')
    tex_ndcg = render_table(
        caption_title='Main Model Comparison on MegaScale Test NDCG@96 (Shifted and Rectified Variants)',
        caption_body=(
            r"Mean per-domain shifted and rectified NDCG@96 across the 33 Homology-Restricted MegaScale test domains (13 domains for doubles). "
            r"One-sided $p$-values test $H_1\colon \text{NDCG@96}_{\text{ESM-MSR}} > \text{NDCG@96}_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_megascale_ndcg96',
        col_spec='l' + 'c' * 12 + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_n) + r' & {} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}',
            'ID & ' + ' & '.join([r'NDCG & $p$'] * 6) + r' & Homology / Overlap \\',
        ],
        body_rows=rows_n,
    )

    df_4a = pd.read_csv(MAIN_PVAL / 'figure4a_property_gain_pvalues.csv')
    specs_4a = [
        ('Total rectified gain @ k=96 (sum over 4 quadrants)', r'Total Gain ($k$=96)'),
        ('Δ hydrophobicity ≤ 0 & relative ASA ≤ 0.5', r'$\Delta H\le 0$, Buried'),
        ('Δ hydrophobicity ≤ 0 & relative ASA > 0.5', r'$\Delta H\le 0$, Surface'),
        ('Δ hydrophobicity > 0 & relative ASA ≤ 0.5', r'$\Delta H>0$, Buried'),
        ('Δ hydrophobicity > 0 & relative ASA > 0.5', r'$\Delta H>0$, Surface'),
    ]
    panels_4a, all_m4a, msr_4a = [], set(), None
    for met, title in specs_4a:
        pan, msr, _, _, B = _extract_model_panel(df_4a, 'MegaScale_capped_test_singles_k96', met, dataset_key='tsu_capped_test')
        msr_4a = msr
        panels_4a.append((title, pan, B))
        all_m4a.update(pan.keys())
    models_4a = sort_models(list(all_m4a), msr_first=msr_4a)
    bests_4a = [_best_clean_set(p, 'val', True) for _, p, _ in panels_4a]
    rows_4a = []
    for m in models_4a:
        note, ov = HOM.note_and_overlap(m, 'tsu_capped_test')
        cells = [clean_model_display(m, overlap=ov)]
        for (_, pan, B), b_set in zip(panels_4a, bests_4a):
            d = pan.get(m)
            cells += ['---', '---'] if not d else [fmt_num(d['val'], 2, bold=(m in b_set)), '---' if m == msr_4a else fmt_p(d['p'], B)]
        cells.append(note)
        rows_4a.append(' & '.join(cells) + r' \\')
    tex_4a = render_table(
        caption_title=r'Rectified Stabilizing Gain Across Hydrophobicity $\times$ Solvent-Accessibility Quadrants ($k=96$ Picks per Domain)',
        caption_body=(
            r"Mean per-domain total rectified stabilizing gain ($\sum \max(0, \Delta\Delta G)$, kcal/mol) among the top $k=96$ predicted single mutants across 33 MegaScale test domains. "
            r"One-sided $p$-values test $H_1\colon \text{Gain}_{\text{ESM-MSR}} > \text{Gain}_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            + HOMOLOGY_LEGEND
        ),
        label='tab:si_property_gain',
        col_spec='l' + 'c' * 10 + 'l',
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_4a) + r' & {} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}',
            'ID & ' + ' & '.join([r'Gain & $p$'] * 5) + r' & Homology / Overlap \\',
        ],
        body_rows=rows_4a,
    )
    p6a = write_tex('table_S6a_ndcg96_and_property_gain.tex', tex_ndcg + '\n' + tex_4a)

    df_4b = pd.read_csv(MAIN_PVAL / 'figure4bcd_screening_pvalues.csv')
    bm_cfgs = [
        ('Figure4b_MegaScale_singles_nocys', 'tsu_capped_test', 'MegaScale Singles (33 domains, no-Cys)',
         [('hit_rate_gt_0.5@K=1', r'Hit$>0.5$ @$K$=1', True), ('hit_rate_gt_0.5@K=10', r'Hit$>0.5$ @$K$=10', True),
          ('hit_rate_gt_0.5@K=50', r'Hit$>0.5$ @$K$=50', True), ('P_at_least_one_hit@K=10', r'$P(\ge 1)$ @$K$=10', True),
          ('P_at_least_one_hit@K=50', r'$P(\ge 1)$ @$K$=50', True)],
         True, r'Top-$K$ stabilizing hit rate ($\Delta\Delta G>0.5$ kcal/mol) and probability of at least one hit $P(\ge 1)$'),
        ('Figure4c_Domainome_heldout_singles', 'domainome_heldout', 'Domainome Held-Out (436 domains)',
         [('hit_rate_gt_WT@K=1', r'Hit$>$WT @$K$=1', True), ('hit_rate_gt_WT@K=10', r'Hit$>$WT @$K$=10', True),
          ('hit_rate_gt_WT@K=50', r'Hit$>$WT @$K$=50', True), ('P_at_least_one_hit@K=10', r'$P(\ge 1)$ @$K$=10', True),
          ('P_at_least_one_hit@K=50', r'$P(\ge 1)$ @$K$=50', True)],
         True, r'Top-$K$ stabilizing hit rate ($\Delta\Delta G>0$) and probability of at least one hit $P(\ge 1)$'),
        ('Figure4d_PGym_Activity_all_38', 'proteingym_heldout', 'ProteinGym Activity (38 assays)',
         [('harm_rate_lt_WT_minus_0.5SD@K=1', r'Harm @$K$=1', False), ('harm_rate_lt_WT_minus_0.5SD@K=10', r'Harm @$K$=10', False),
          ('harm_rate_lt_WT_minus_0.5SD@K=20', r'Harm @$K$=20', False), ('harm_rate_lt_WT_minus_0.5SD@K=50', r'Harm @$K$=50', False),
          ('harm_rate_lt_WT_minus_0.5SD@K=100', r'Harm @$K$=100', False)],
         False, r'Top-$K$ deleterious harm rate ($\text{activity}<\text{WT}-0.5\text{ SD}$, lower is better)'),
        ('Figure4d_PGym_Activity_enzymes_16', 'proteingym_heldout', 'ProteinGym Enzymes (16 assays)',
         [('harm_rate_lt_WT_minus_0.5SD@K=1', r'Harm @$K$=1', False), ('harm_rate_lt_WT_minus_0.5SD@K=10', r'Harm @$K$=10', False),
          ('harm_rate_lt_WT_minus_0.5SD@K=20', r'Harm @$K$=20', False), ('harm_rate_lt_WT_minus_0.5SD@K=50', r'Harm @$K$=50', False),
          ('harm_rate_lt_WT_minus_0.5SD@K=100', r'Harm @$K$=100', False)],
         False, r'Top-$K$ deleterious harm rate ($\text{activity}<\text{WT}-0.5\text{ SD}$, lower is better)'),
    ]

    subtables_6b = []
    for bm_key, ds_k, bm_title, met_list, has_rmse, metric_desc in bm_cfgs:
        panels_s, all_ms, msr_s = [], set(), None
        for met, label_m, hib in met_list:
            pan, msr, _, _, B = _extract_model_panel(df_4b, bm_key, met, dataset_key=ds_k)
            msr_s = msr
            panels_s.append((label_m, pan, B, hib))
            all_ms.update(pan.keys())
        pan0 = panels_s[0][1]
        if bm_key == 'Figure4b_MegaScale_singles_nocys':
            _apply_tobit_overrides(pan0, msr_s, tobit_df, 'screening_s6b', bm_key, 'pool')
            all_ms.update(pan0.keys())
        models_s = sort_models(list(all_ms), msr_first=msr_s)
        bests_s = [_best_clean_set(p, 'val', hib) for _, p, _, hib in panels_s]
        b_pr = _best_clean_set(pan0, 'pearson', True)
        b_rm = _best_clean_set(pan0, 'rmse', False) if has_rmse else set()

        rows_s = []
        for m in models_s:
            d0 = pan0.get(m, {})
            note, ov = ('---', False) if m == 'ESM-MSR (unlinked)' else HOM.note_and_overlap(m, ds_k)
            cells = [clean_model_display(m, overlap=ov)]
            for (_, pan, B, _), b_set in zip(panels_s, bests_s):
                d = pan.get(m)
                cells += ['---', '---'] if (not d or m == 'ESM-MSR (unlinked)') else [fmt_num(d['val'], 3, bold=(m in b_set)), '---' if m == msr_s else fmt_p(d['p'], B)]
            cells.append(fmt_num(d0.get('pearson'), 3, bold=(m in b_pr)))
            if has_rmse:
                cells.append(fmt_num(d0.get('rmse'), 3, bold=(m in b_rm)))
            cells.append(note)
            rows_s.append(' & '.join(cells) + r' \\')

        extra_hdr = r' & \multicolumn{2}{c}{Pool} & {}' if has_rmse else r' & Pool & {}'
        extra_cmid = r' \cmidrule(lr){12-13}' if has_rmse else ''
        extra_sub = r' & $r$ & RMSE & Homology / Overlap' if has_rmse else r' & $r$ & Homology / Overlap'
        ncols_total = 10 + (2 if has_rmse else 1)
        cmids = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(5)) + extra_cmid
        tobit_note = (
            r" ESM-MSR uses the Tobit observation link for full-pool $r$ and RMSE; ESM-MSR (unlinked) reports raw latent RMSE only."
            if bm_key == 'Figure4b_MegaScale_singles_nocys' else ""
        )
        subtables_6b.append(
            render_table(
                caption_title=f'Practical Library Screening Performance on {bm_title}',
                caption_body=(
                    f"{metric_desc} across library budgets $K$, alongside full-pool Pearson's $r$"
                    + (r" and raw RMSE (kcal/mol)" if has_rmse else "")
                    + "." + tobit_note
                    + r" One-sided $p$-values test ESM-MSR (WT-view) superiority via paired cluster/assay $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
                    + HOMOLOGY_LEGEND
                ),
                label=f'tab:si_screening_{bm_key.lower()}',
                col_spec='l' + 'c' * ncols_total + 'l',
                header_lines=[
                    '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _, _ in panels_s) + extra_hdr + r' \\',
                    cmids,
                    'ID & ' + ' & '.join([r'Rate & $p$'] * 5) + extra_sub + r' \\',
                ],
                body_rows=rows_s,
            )
        )
    p6b = write_tex('table_S6b_screening_enrichment.tex', '\n'.join(subtables_6b))
    return p6a, p6b


def main() -> None:
    extra_df = _load_extra_df()
    tobit_df = _load_tobit_df()
    print(f'Loaded {len(extra_df)} extra-model rows from {EXTRA_CSV.name} and {len(tobit_df)} Tobit-linked override rows from {TOBIT_CSV.name}')
    generated: list[Path] = []

    print('[1/6] Building Tables S1a–S1b (Ablations: In-Distribution & OOD)...')
    generated.extend(build_table_s1_ablations())

    print('[2/6] Building Tables S2a–S2d (MegaScale All Models + Tobit Link + Homology Notes)...')
    generated.append(build_table_s2a_megascale_overall(extra_df, tobit_df))
    generated.append(build_table_s2b_megascale_doubles(extra_df, tobit_df))
    generated.append(build_table_s2c_megasplits(extra_df, tobit_df))
    generated.append(build_table_s2d_backbone_origin(extra_df, tobit_df))

    print('[3/6] Building Tables S3a–S3b (Epistasis Hierarchy & Calibration All Models + Tobit Link + Homology Notes)...')
    generated.extend(build_table_s3_epistasis(extra_df, tobit_df))

    print('[4/6] Building Table S4 (Vertical Multi-Page Thermostability Longtable: All 12 Sets x All 19 Models) & S4c...')
    generated.extend(build_table_s4_thermostability_longtable(extra_df, tobit_df))

    print('[5/6] Building Tables S5a–S5b (Held-Out OOD All Models Longtable & ProteinGym Sigma)...')
    generated.extend(build_table_s5_heldout_ood(extra_df, tobit_df))

    print('[6/6] Building Tables S6a–S6b (NDCG@96, Property Gain & Practical Screening + Tobit Link + Homology Notes)...')
    generated.extend(build_table_s6_screening_and_ndcg(tobit_df))

    # Remove superseded horizontal thermostability files if present
    for old_f in ('table_S4a_thermostability_part1.tex', 'table_S4b_thermostability_part2.tex', 'table_S1_ablations_summary.tex'):
        old_p = OUT_DIR / old_f
        if old_p.exists():
            old_p.unlink()

    combined_parts = [
        '% Copy-pastable SI LaTeX tables (requires \\usepackage{booktabs,longtable,float,graphicx,amsmath})\n'
    ]
    for p in generated:
        combined_parts.append(f'% ==================== {p.name} ====================\n')
        combined_parts.append(p.read_text(encoding='utf-8'))
        combined_parts.append('\n')
    all_tex = OUT_DIR / 'all_si_tables.tex'
    all_tex.write_text('\n'.join(combined_parts), encoding='utf-8')

    print(f'\nDone! Wrote {len(generated)} individual .tex files + {all_tex.name} to {OUT_DIR}')
    for p in generated:
        print(' ', p)


if __name__ == '__main__':
    main()
