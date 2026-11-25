#!/usr/bin/env python3
"""Generate copy-pastable LaTeX SI tables matching the manuscript format:
  - Table S1: Concise summary of ALL ESM-MSR ablations and variations across ALL datasets
              (MegaScale test, Epistasis, Thermostability, Domainome, ProteinGym, DMS, MGnify, Rocklin).
  - Tables S2–S6: Comprehensive main-model comparison tables (Spearman rho, Pearson r, RMSE,
                  and one-sided paired bootstrap p-value comparing ESM-MSR), read directly from
                  already-computed CSVs on the wt-dddg-scatters worktree (zero recomputation).

Usage:
    /home/sareeves/miniconda3/envs/msr_venv/bin/python analysis/scripts/build_si_latex_tables.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Paths to existing precomputed outputs (read-only inputs)
ROOT = Path('/home/sareeves/playground/esm-msr-devel')
WT_DDDG = ROOT / 'wt-dddg-scatters'
WT_TOBIT = ROOT / 'wt-tobit-floor-ablation'
WT_SI = Path(__file__).resolve().parents[2]

MAIN_PVAL = WT_DDDG / 'analysis/results/00_main_text_figs/pvalues'
MAIN_SRC = WT_DDDG / 'analysis/results/00_main_text_figs/source_data'
SUPP_PVAL = WT_DDDG / 'analysis/results/00_supplementary_figs/pvalues'
SUPP_SRC = WT_DDDG / 'analysis/results/00_supplementary_figs/source_data'
ABL_DIR = WT_TOBIT / 'analysis/results/09_ablation'

OUT_DIR = WT_SI / 'analysis/results/si_latex_tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add wt-tobit-floor-ablation / wt-dddg-scatters analysis modules for held-out masks and MGnify loader
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
    """Escape plain-text model names while preserving intentional LaTeX math ($...$)."""
    s = str(s)
    # Handle common unicode symbols in comparator names
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
    # Escape underscores & ampersands outside math mode
    parts = s.split('$')
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')
    return '$'.join(parts)


def fmt_num(val: float | None, decimals: int = 3, bold: bool = False, suffix: str = '') -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return '---'
    s = f'{float(val):.{decimals}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    s = f'{s}{suffix}'
    return f'\\textbf{{{s}}}' if bold else s


def fmt_mean_sd(mean: float | None, sd: float | None, decimals: int = 3, bold: bool = False, suffix: str = '') -> str:
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return '---'
    m_s = f'{float(mean):.{decimals}f}'
    if m_s.startswith('-'):
        m_s = '$-$' + m_s[1:]
    if sd is None or (isinstance(sd, float) and not math.isfinite(sd)):
        s = f'{m_s}{suffix}'
    else:
        s = f'{m_s} $\\pm$ {float(sd):.{decimals}f}{suffix}'
    return f'\\textbf{{{s}}}' if bold else s


def fmt_p(p: float | None, B: int | float | None = None) -> str:
    """Format one-sided Monte Carlo p-value cleanly."""
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


def write_tex(filename: str, content: str) -> Path:
    p = OUT_DIR / filename
    p.write_text(content, encoding='utf-8')
    return p


# Canonical model display ordering for main-model tables
CANONICAL_MODEL_ORDER = [
    'ESM-MSR',
    'ESM-MSR (Dual-view)',
    'ESM-MSR (σ=1)',
    'ESM-MSR WT-view',
    'ESM-MSR (WT-view)',
    'ESM-MSR σ=0.5',
    'ESM-MSR ctx ctrl',
    'SPURS',
    'SPURS-multi',
    'MutateEverything',
    'MutateEverything Stage 2',
    'MutateEverything Stage 1',
    'ThermoMPNN',
    'ThermoMPNN-D',
    'GeoDDG-3D',
    'ESM3ΔG',
    'Rosetta',
    'ESM3-small-open',
    'ESM3-small-open WT-view',
    'ProteinMPNN',
    'ProSST',
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
    """Format model display name cleanly in LaTeX, appending * for training overlap."""
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
        '−dG_add (floor ceiling)': r'$-\Delta G_{\mathrm{add}}$ (floor ceiling)',
        'Top 96 per Domain': 'Oracle (top 96 per domain)',
    }
    disp = mapping.get(name, tex_escape(name))
    if overlap:
        disp += r'$^*$'
    return disp


# --------------------------------------------------------------------------------------
# PART 1: TABLES S1a & S1b — ABLATIONS SPLIT INTO IN-DISTRIBUTION AND OOD
# --------------------------------------------------------------------------------------
# The 7 single-mutant sets matching the 8 main thermostability benchmarks (+ PTMUL = 8)
EXT_MAIN_SINGLES_7 = ['ssym', 's461', 'k2369', 's783', 's2648', 'q3421', 's8754']


def _load_heldout_mgnify_spine(run: str, src_name: str) -> pd.DataFrame:
    """Load exact held-out uid, dG_wt, ddG_ML spine matching Figure 3b (612,406 MGnify / 109,581 Rocklin)."""
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
        if vals:
            out[h] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)
    tag = 'FINAL_p0_s05' if 'sigma0.5' in base_dir_s else fam
    return tag, colkey, out


def _load_mgnify_rocklin_ablations() -> dict[str, dict[str, tuple[float, float]]]:
    """Compute held-out MGnify (612,406) & Rocklin (109,581) pooled Spearman (comb~ matching Figure 3b) in parallel."""
    fams = [
        'FINAL_p0', 'FINAL_p1', 'FINAL_p1_marg',
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


def _compute_extra_ablation_row_stats(row_key: str) -> dict[str, tuple[float, float]]:
    """Compute non-cells_long stats for FINAL_p0_s05 (seeds 1..3) and FINAL_p0_learntobit_all (seed 1)."""
    fb = ROOT / 'final_bench/stats'
    fpred = ROOT / 'final_bench/predictions'
    dom_excl = heldout_excluded('domainome')

    # Held-out ProteinGym assay list (exact 173 assays from Figure 3c source_data)
    f3c = pd.read_csv(MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv')
    pg_held = f3c[(~f3c['leaked']) & (~f3c['dropped_large'])][['assay', 'category']].copy()

    def load_epi_beyond_cellmag(fam: str, seed: int, tag: str) -> tuple[float, float]:
        """Compute beyond_rho and cell_mag from predictions CSV on capped test doubles via exact E.Table / E.compute."""
        p_pred = fpred / 'splits_oct06_capped-test/final_checkpoints' / fam / f's{seed}' / f'best.ckpt_{tag}_predictions.csv'
        if not p_pred.exists():
            return (np.nan, np.nan)
        df = pd.read_csv(p_pred, low_memory=False)
        code = df['code_wt'].astype(str).to_numpy()
        mut = df['mut_type'].astype(str).to_numpy()
        keys = [EPI_LIB.parse_mut_key(m, c) for m, c in zip(mut, code)]
        st = np.array(['single' if k and len(k) == 1 else ('double' if k and len(k) == 2 else 'other') for k in keys], dtype=object)
        # dG_wt = dG_ML - ddG_ML
        dg_wt = df['dG_ML'].to_numpy(float) - df['ddG_ML'].to_numpy(float)
        items = dict(
            code=code,
            mut_key=keys,
            subset_type=st,
            ddG=df['ddG_ML'].to_numpy(float),
            dddG=df['dddG_ML'].to_numpy(float),
            dG_wt=dg_wt,
            wt=df['wt_lora_pred'].to_numpy(float),
            mt=df['combined_pred'].to_numpy(float),
            comb=df['combined_pred'].to_numpy(float),
        )
        try:
            tbl = E.Table(items, None, heads=('comb', 'add'))
            res_e = E.compute(tbl)
            return (float(res_e.get('epi_beyond_rho_comb', np.nan)), float(res_e.get('epi_cell_mag_comb', np.nan)))
        except Exception as exc:
            print(f'  [warn] epistasis compute failed for {fam} s{seed}: {exc}')
            return (np.nan, np.nan)

    if row_key == 'FINAL_p0_s05':
        fam, tag, seeds = 'FINAL_p0', 'sigma0.5_unmasked', (1, 2, 3)
    elif row_key == 'FINAL_p0_learntobit_all':
        fam, tag, seeds = 'FINAL_p0_learntobit_all', 'sigma1.0_unmasked', (1,)
    else:
        return {}

    tsu_all, tsu_dbl, ext_s, ext_m, dom_l, pg_m = [], [], [], [], [], []
    bey_l, cmag_l = [], []
    for s in seeds:
        # MegaScale test
        p_t = fb / 'splits_oct06_capped-test/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_Combined_avg.csv'
        if p_t.exists():
            d = pd.read_csv(p_t).set_index('Unnamed: 0')['0']
            tsu_all.append(float(d['spearman_all']))
            tsu_dbl.append(float(d['spearman_doubles']))
        b_v, c_v = load_epi_beyond_cellmag(fam, s, tag)
        if math.isfinite(b_v):
            bey_l.append(b_v)
        if math.isfinite(c_v):
            cmag_l.append(c_v)
        # External singles (7 main sets, WT) & PTMUL (Combined)
        p_ew = fb / 'external/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_WT_LoRA.csv'
        p_ec = fb / 'external/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag}_Combined.csv'
        if p_ew.exists():
            d = pd.read_csv(p_ew).set_index('Unnamed: 0')['spearman_all']
            ext_s.append(float(d.loc[EXT_MAIN_SINGLES_7].mean()))
        if p_ec.exists():
            d = pd.read_csv(p_ec).set_index('Unnamed: 0')['spearman_all']
            ext_m.append(float(d.loc['ptmul']))
        # Domainome low-id (436 held-out domains)
        p_d = fb / 'domainome/final_checkpoints' / fam / f's{s}' / f'best.ckpt_{tag.replace("unmasked", "skip_reverse_unmasked")}_WT_LoRA.csv'
        if p_d.exists():
            d = pd.read_csv(p_d)
            keep = ~d['Unnamed: 0'].astype(str).isin(dom_excl)
            dom_l.append(float(d.loc[keep, 'spearman_all'].mean()))
        # ProteinGym held-out 173 assays (WT-view macro avg)
        if row_key == 'FINAL_p0_s05':
            pg_dir = ROOT / 'final_pgym_wt_sigma0.5/FINAL_p0' / f's{s}'
        else:
            pg_dir = ROOT / 'final_pgym_wt' / fam / f's{s}'
        if pg_dir.exists():
            # Read per-assay Spearman from CSVs
            cat_vals: dict[str, list[float]] = {}
            for _, row in pg_held.iterrows():
                assay = row['assay']
                cat = row['category']
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
                            r_a = float(spearmanr(da.loc[ok, pcol], da.loc[ok, ycol]).statistic)
                            cat_vals.setdefault(cat, []).append(r_a)
            if len(cat_vals) == 5:
                pg_m.append(float(np.mean([np.mean(v) for v in cat_vals.values()])))

    def agg(lst):
        if not lst:
            return (np.nan, np.nan)
        return (float(np.mean(lst)), float(np.std(lst, ddof=1)) if len(lst) > 1 else np.nan)

    return {
        'tsu_overall': agg(tsu_all),
        'tsu_doubles': agg(tsu_dbl),
        'tsu_beyond': agg(bey_l),
        'tsu_cellmag': agg(cmag_l),
        'ext_singles': agg(ext_s),
        'ext_multi': agg(ext_m),
        'domainome': agg(dom_l),
        'pgym': agg(pg_m),
    }


def build_table_s1_ablations() -> tuple[list[Path], pd.DataFrame]:
    cl = pd.read_csv(ABL_DIR / 'cells_long.csv')
    cl_mega = pd.read_csv(ABL_DIR / 'cells_long_mega_own_split.csv')
    mg_map = _load_mgnify_rocklin_ablations()
    extra_s05 = _compute_extra_ablation_row_stats('FINAL_p0_s05')
    extra_lt = _compute_extra_ablation_row_stats('FINAL_p0_learntobit_all')

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
        # Average across the 7 main single-mutant thermostability sets per seed, then mean +/- SD across seeds
        per_seed = sub.groupby('seed')['rho'].mean().to_numpy(float)
        return (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan)

    # Row specs: (key, display_label, source_mode)
    row_specs = [
        ('FINAL_p0', 'ESM-MSR (reference)', 'cl_primary'),
        ('FINAL_p0_WT', 'ESM-MSR, WT-view only', 'cl_wt'),
        ('FINAL_p0_MT', 'ESM-MSR, MT-view only', 'cl_mt'),
        ('FINAL_p0_s05', r'ESM-MSR, $\sigma$=0.5', 'extra_s05'),
        ('FINAL_p0@independent', 'ESM-MSR, indep. masking (eval only)', 'cl_primary'),
        ('FINAL_p1', 'ESM-MSR*, indep. masking', 'cl_primary'),
        ('FINAL_p1@unmasked', 'ESM-MSR*, indep. masking (eval unmasked)', 'cl_primary'),
        ('FINAL_p1_marg', 'ESM-MSR*, masked marginal MT', 'cl_primary'),
        ('FINAL_p0_pw0', 'ESM-MSR*, no pair loss', 'cl_primary'),
        ('FINAL_p0_norank', 'ESM-MSR*, no rank loss', 'cl_primary'),
        ('FINAL_p0_rcthr0', r'ESM-MSR*, rank censor $\tau$=0', 'cl_primary'),
        ('FINAL_p0_norange', 'ESM-MSR*, no range-tagged cells', 'cl_primary'),
        ('FINAL_p0_mse', 'ESM-MSR*, MSE loss (no range-tagged)', 'cl_primary'),
        ('FINAL_p0_deepfloor', r'ESM-MSR*, Tobit floor ($-1.0$, $1.0$)', 'cl_primary'),
        ('FINAL_p0_floor0', r'ESM-MSR*, Tobit floor ($0.0$, $0.5$)', 'cl_primary'),
        ('FINAL_p0_learntobit_all', r'ESM-MSR*, learnable Tobit floor$^\ddagger$', 'extra_lt'),
        ('FINAL_p0_noseqhwm', 'ESM-MSR*, exclude seq. head', 'cl_primary'),
        ('FINAL_p0_r2a4', 'ESM-MSR*, MT LoRA rank 2', 'cl_primary'),
        ('FINAL_p0_mega', r'ESM-MSR*, full MegaScale split$^\dagger$', 'cl_mega'),
    ]

    id_cols = ['tsu_overall', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag', 'mgnify', 'rocklin']
    ood_cols = ['ext_singles', 'ext_multi', 'domainome', 'pgym']
    all_cols = id_cols + ood_cols

    records = []
    for key, label, mode in row_specs:
        rec = {'key': key, 'label': label}
        if mode == 'cl_primary':
            rec['tsu_overall'] = get_cl(key, 'tsuboyama_capped_test', 'primary')
            rec['tsu_doubles'] = get_cl(key, 'tsuboyama_capped_test|multi', 'primary')
            rec['tsu_beyond'] = get_cl(key, 'tsuboyama_capped_test|beyond_rho', 'primary')
            rec['tsu_cellmag'] = get_cl(key, 'tsuboyama_capped_test|cell_mag', 'primary')
            rec['ext_singles'] = get_cl_ext7(key, 'primary')
            rec['ext_multi'] = get_cl(key, 'ptmul', 'primary')
            rec['domainome'] = get_cl(key, 'domainome|lowid', 'primary')
            # Strictly 173 held-out assays (pg|avg_lowid == pg|avg in cells_long)
            rec['pgym'] = get_cl(key, 'pg|avg_lowid', 'wt')
        elif mode == 'cl_wt':
            rec['tsu_overall'] = get_cl('FINAL_p0', 'tsuboyama_capped_test', 'wt')
            rec['tsu_doubles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|multi', 'wt')
            rec['tsu_beyond'] = (np.nan, np.nan)
            rec['tsu_cellmag'] = (np.nan, np.nan)
            rec['ext_singles'] = get_cl_ext7('FINAL_p0', 'wt')
            rec['ext_multi'] = get_cl('FINAL_p0', 'ptmul', 'wt')
            rec['domainome'] = get_cl('FINAL_p0', 'domainome|lowid', 'wt')
            rec['pgym'] = get_cl('FINAL_p0', 'pg|avg_lowid', 'wt')
        elif mode == 'cl_mt':
            rec['tsu_overall'] = get_cl('FINAL_p0', 'tsuboyama_capped_test', 'mt')
            rec['tsu_doubles'] = get_cl('FINAL_p0', 'tsuboyama_capped_test|multi', 'mt')
            rec['tsu_beyond'] = (np.nan, np.nan)
            rec['tsu_cellmag'] = (np.nan, np.nan)
            rec['ext_singles'] = get_cl_ext7('FINAL_p0', 'mt')
            rec['ext_multi'] = get_cl('FINAL_p0', 'ptmul', 'mt')
            rec['domainome'] = get_cl('FINAL_p0', 'domainome|lowid', 'mt')
            rec['pgym'] = (np.nan, np.nan)
        elif mode == 'extra_s05':
            for ck in ['tsu_overall', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag', 'ext_singles', 'ext_multi', 'domainome', 'pgym']:
                rec[ck] = extra_s05.get(ck, (np.nan, np.nan))
        elif mode == 'extra_lt':
            for ck in ['tsu_overall', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag', 'ext_singles', 'ext_multi', 'domainome', 'pgym']:
                rec[ck] = extra_lt.get(ck, (np.nan, np.nan))
        elif mode == 'cl_mega':
            rec['tsu_overall'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test', 'comb', cl_mega)
            rec['tsu_doubles'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test|multi', 'comb', cl_mega)
            rec['tsu_beyond'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test|beyond_rho', 'epi', cl_mega)
            rec['tsu_cellmag'] = get_cl('FINAL_p0_mega', 'tsuboyama_capped_test|cell_mag', 'epi', cl_mega)
            rec['ext_singles'] = get_cl_ext7('FINAL_p0_mega', 'primary')
            rec['ext_multi'] = get_cl('FINAL_p0_mega', 'ptmul', 'primary')
            rec['domainome'] = get_cl('FINAL_p0_mega', 'domainome|lowid', 'primary')
            rec['pgym'] = get_cl('FINAL_p0_mega', 'pg|avg_lowid', 'wt')

        rec['mgnify'] = mg_map.get(key, {}).get('mgnify', (np.nan, np.nan))
        rec['rocklin'] = mg_map.get(key, {}).get('rocklin', (np.nan, np.nan))
        records.append(rec)

    # Determine bolding per column (matching user SI table caption)
    best_by_col: dict[str, tuple[float, set[str]]] = {}
    tsu_cols = {'tsu_overall', 'tsu_doubles', 'tsu_beyond', 'tsu_cellmag'}
    for ck in all_cols:
        cands = []
        for rec in records:
            if ck in tsu_cols and rec['key'] == 'FINAL_p0_mega':
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

    # --- Table S1a: In-Distribution Proteolysis Stability (MegaScale Test + Held-Out MGnify & Rocklin) ---
    rows_id = []
    for rec in records:
        cells = [rec['label']]
        for ck in id_cols:
            m, sd = rec[ck]
            sfx = r'$^\dagger$' if (rec['key'] == 'FINAL_p0_mega' and ck in tsu_cols and math.isfinite(m)) else ''
            is_bold = rec['key'] in best_by_col[ck][1]
            cells.append(fmt_mean_sd(m, sd, decimals=3, bold=is_bold, suffix=sfx))
        rows_id.append(' & '.join(cells) + r' \\')

    tex_id = render_table(
        caption_title='Summary of ESM-MSR Ablations and Variations: In-Distribution Proteolysis Stability Benchmarks',
        caption_body=(
            r"Spearman's $\rho$ (mean $\pm$ SD across 3 independent training seeds unless noted) on proteolysis stability benchmarks "
            r"under the deployed Dual-view readout. Models with an asterisk (ESM-MSR*) are retrained with modifications on the Homology-Restricted "
            r"MegaScale split, while those without use the exact same weights as ESM-MSR (reference). "
            r"MegaScale test columns report per-domain mean $\rho$ (Overall: 33 domains; Doubles: 13 domains) and epistasis metrics "
            r"(Beyond $\rho$: Spearman above global saturation curve; Cell mag.: double-centred complete-block residual correlation). "
            r"MGnify and Rocklin columns report pooled Tobit-linked $\rho$ strictly on the 4-part homology-held-out sets "
            r"(MGnify: 612,406 variants across 140,694 scaffolds; Rocklin de novo designs: 109,581 variants across 623 scaffolds). "
            r"Highest performing models in each column are indicated in bold; ESM-MSR (reference) is also bolded if its mean $\pm$ SD range overlaps the highest average performance. "
            r"$^\dagger$Trained on the full MegaScale split (\texttt{mega\_splits}) and evaluated on its own 28-domain test split for MegaScale columns (excluded from MegaScale bolding), "
            r"and on the identical held-out MGnify and Rocklin sets. "
            r"$^\ddagger$Single training seed (seed 1; MGnify/Rocklin not run). --- indicates not applicable or not evaluated."
        ),
        label='tab:si_ablations_in_distribution',
        col_spec='l' + 'c' * len(id_cols),
        header_lines=[
            r'{} & \multicolumn{4}{c}{MegaScale test} & \multicolumn{2}{c}{Held-out Proteolysis} \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-7}',
            r'ID & Overall & Doubles & Beyond $\rho$ & Cell mag. & MGnify & Rocklin \\',
        ],
        body_rows=rows_id,
    )
    p1a = write_tex('table_S1a_ablations_in_distribution.tex', tex_id)

    # --- Table S1b: Out-of-Distribution Benchmarks (Thermostability 7+1, Held-Out Domainome, Held-Out ProteinGym) ---
    rows_ood = []
    for rec in records:
        cells = [rec['label']]
        for ck in ood_cols:
            m, sd = rec[ck]
            is_bold = rec['key'] in best_by_col[ck][1]
            cells.append(fmt_mean_sd(m, sd, decimals=3, bold=is_bold))
        rows_ood.append(' & '.join(cells) + r' \\')

    tex_ood = render_table(
        caption_title='Summary of ESM-MSR Ablations and Variations: Out-of-Distribution Benchmarks',
        caption_body=(
            r"Spearman's $\rho$ (mean $\pm$ SD across 3 independent training seeds unless noted) on out-of-distribution benchmarks "
            r"under the deployed readout (WT-view on single-mutant thermostability sets, Domainome, and ProteinGym; Dual-view on PTMUL). "
            r"Models with an asterisk (ESM-MSR*) are retrained with modifications on the Homology-Restricted MegaScale split, "
            r"while those without use the exact same weights as ESM-MSR (reference). "
            r"Thermostability columns match the 8 main-text benchmarks: Singles (7) averages pooled $\rho$ across Ssym, S461, K2369, S783, S2648, Q3421, and S8754; "
            r"PTMUL reports pooled $\rho$ on the multi-mutant thermostability set ($n=914$). "
            r"Held-out DMS columns strictly use the 4-part homology-held-out sets: Domainome reports mean per-domain $\rho$ across 436 held-out domains ($n=451{,}569$) "
            r"and ProteinGym reports equal-category-weighted macro-average $\rho$ across 173 held-out assays (WT-view; Dual-view for reference and p1 differs by $<0.001$). "
            r"Highest performing models in each column are indicated in bold; ESM-MSR (reference) is also bolded if its mean $\pm$ SD range overlaps the highest average performance. "
            r"$^\dagger$Trained on the full MegaScale split (\texttt{mega\_splits}) and evaluated on the identical OOD sets. "
            r"$^\ddagger$Single training seed (seed 1). --- indicates not applicable or not evaluated."
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

    return [p1a, p1b], pd.DataFrame([{**{'key': r['key'], 'label': r['label']}, **{ck: r[ck][0] for ck in all_cols}} for r in records])


# --------------------------------------------------------------------------------------
# PART 2: MAIN-MODEL COMPREHENSIVE TABLES (DIRECTLY FROM wt-dddg-scatters CSVs)
# --------------------------------------------------------------------------------------
def _extract_model_panel(
    df: pd.DataFrame,
    benchmark: str,
    metric: str,
    resampling_unit: str | None = None,
    primary_val_is_rmse: bool = False,
) -> tuple[dict[str, dict], str, int, int, int]:
    """Extract {model_name: {val, pearson, rmse, p, B, overlap, internal}} from a pairwise pvalues CSV."""
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

    out: dict[str, dict] = {}
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
        'p_bonf': np.nan,
        'B': B,
        'overlap': False,
        'internal': False,
    }
    for _, row in sub.iterrows():
        comp = str(row['comparator'])
        ov = bool(row['excluded_training_overlap']) if 'excluded_training_overlap' in sub.columns and pd.notna(row.get('excluded_training_overlap')) else False
        iv = bool(row['internal_control_or_variant']) if 'internal_control_or_variant' in sub.columns and pd.notna(row.get('internal_control_or_variant')) else False
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
            'p_bonf': float(row['p_one_sided_bonferroni']) if 'p_one_sided_bonferroni' in sub.columns and pd.notna(row.get('p_one_sided_bonferroni')) else np.nan,
            'B': int(row['B']) if pd.notna(row.get('B')) else B,
            'overlap': ov,
            'internal': iv,
        }
    return out, msr_name, n_units, n_var, B


def _best_clean_set(panel: dict[str, dict], field: str, higher_is_better: bool = True, exclude_internal: bool = False) -> set[str]:
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


def build_table_s2a_megascale_overall() -> Path:
    df = pd.read_csv(MAIN_PVAL / 'figure1b_megascale_stab_destab_pvalues.csv')
    specs = [
        ('MegaScale_capped_test', 'per_domain_mean_spearman_overall', r'Overall ($n$=48,740)'),
        ('MegaScale_capped_test', 'per_domain_mean_spearman_stabilizing', r'Stabilizing ($\Delta\Delta G>0$, $n$=7,417)'),
        ('MegaScale_capped_test', 'per_domain_mean_spearman_destabilizing', r'Destabilizing ($\Delta\Delta G\le 0$, $n$=41,323)'),
    ]
    panels = []
    all_models = set()
    msr_name = None
    for bm, met, title in specs:
        pan, msr, n_u, n_v, B = _extract_model_panel(df, bm, met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    header_lines = [
        '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' \\',
        r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}',
        'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * len(panels)) + r' \\',
    ]
    bests = [
        (_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False))
        for _, p, _ in panels
    ]
    body_rows = []
    for m in models:
        cells = [clean_model_display(m)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---', '---', '---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_rho)))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append('---' if m == msr_name else fmt_p(d['p'], B))
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='Main Model Comparison on Homology-Restricted MegaScale Test Set (Overall, Stabilizing, and Destabilizing Variants)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 33 held-out MegaScale test domains "
            r"(3-seed mean for retrained models; no comparator has training homology overlap). "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ training-seed bootstrap "
            r"($B=10{,}000$; uncorrected; Bonferroni adjustment uses $m=8$ comparators). "
            r"Best values in each metric column are indicated in bold (highest $\rho$, highest $r$, lowest RMSE)."
        ),
        label='tab:si_megascale_overall',
        col_spec='l' + 'c' * (4 * len(panels)),
        header_lines=header_lines,
        body_rows=body_rows,
    )
    return write_tex('table_S2a_megascale_overall_stab_destab.tex', tex)


def build_table_s2b_megascale_doubles() -> Path:
    df = pd.read_csv(MAIN_PVAL / 'figure1c_doubles_add_vs_epi_pvalues.csv')
    specs = [
        ('MegaScale_capped_test_doubles', 'per_domain_mean_spearman_epistatic', r'Epistatic Readout ($n$=13,319)'),
        ('MegaScale_capped_test_doubles', 'per_domain_mean_spearman_additive', r'Additive Readout ($n$=13,319)'),
    ]
    panels = []
    all_models = set()
    msr_name = None
    for bm, met, title in specs:
        pan, msr, _, _, B = _extract_model_panel(df, bm, met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    header_lines = [
        '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' \\',
        r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
        'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * len(panels)) + r' \\',
    ]
    bests = [
        (_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False))
        for _, p, _ in panels
    ]
    body_rows = []
    for m in models:
        cells = [clean_model_display(m)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---', '---', '---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_rho)))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append('---' if m == msr_name else fmt_p(d['p'], B))
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='Main Model Comparison on Homology-Restricted MegaScale Test Double Mutants (Epistatic vs. Additive Readout)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 13 held-out MegaScale double-mutant domains "
            r"($n=13{,}319$ double mutants; 3-seed mean for retrained models). "
            r"Epistatic readout evaluates each model's multi-mutant prediction; additive readout sums single-mutant predictions ($\hat{y}_A + \hat{y}_B$). "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ within each readout via paired domain-cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected; Bonferroni adjustment uses $m=7$ comparators). Best values in each metric column are indicated in bold."
        ),
        label='tab:si_megascale_doubles',
        col_spec='l' + 'c' * (4 * len(panels)),
        header_lines=header_lines,
        body_rows=body_rows,
    )
    return write_tex('table_S2b_megascale_doubles.tex', tex)


def build_table_s2c_megasplits() -> Path:
    df_a = pd.read_csv(SUPP_PVAL / '_mega_splits_fig1a_stab_destab_pvalues.csv')
    df_b = pd.read_csv(SUPP_PVAL / '_mega_splits_fig1b_doubles_pvalues.csv')

    # Subtable 1: Overall, Stabilizing, Destabilizing
    specs_a = [
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_overall', r'Overall ($n$=49,048)'),
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_stabilizing', r'Stabilizing ($n$=8,791)'),
        ('MegaScale_mega_splits_test', 'per_domain_mean_spearman_destabilizing', r'Destabilizing ($n$=40,257)'),
    ]
    panels_a, models_a, msr_a = [], set(), None
    for bm, met, title in specs_a:
        pan, msr, _, _, B = _extract_model_panel(df_a, bm, met)
        msr_a = msr
        panels_a.append((title, pan, B))
        models_a.update(pan.keys())
    models_a = sort_models(list(models_a), msr_first=msr_a)

    bests_a = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels_a]
    rows_a = []
    for m in models_a:
        cells = [clean_model_display(m)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels_a, bests_a):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m == msr_a else fmt_p(d['p'], B),
            ]
        rows_a.append(' & '.join(cells) + r' \\')

    tex_a = render_table(
        caption_title='Main Model Comparison on Full MegaScale Split Test Set: Overall, Stabilizing, and Destabilizing Variants',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 28 test domains of the full MegaScale split "
            r"(\texttt{mega\_splits}). One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected). Best values in each metric column are indicated in bold."
        ),
        label='tab:si_megasplits_overall',
        col_spec='l' + 'c' * 12,
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels_a) + r' \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 3) + r' \\',
        ],
        body_rows=rows_a,
    )

    # Subtable 2: Doubles
    specs_b = [
        ('MegaScale_mega_splits_test_doubles', 'per_domain_mean_spearman_epistatic', r'Epistatic Readout ($n$=18,574)'),
        ('MegaScale_mega_splits_test_doubles', 'per_domain_mean_spearman_additive', r'Additive Readout ($n$=18,574)'),
    ]
    panels_b, models_b, msr_b = [], set(), None
    for bm, met, title in specs_b:
        pan, msr, _, _, B = _extract_model_panel(df_b, bm, met)
        msr_b = msr
        panels_b.append((title, pan, B))
        models_b.update(pan.keys())
    models_b = sort_models(list(models_b), msr_first=msr_b)
    bests_b = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels_b]
    rows_b = []
    for m in models_b:
        cells = [clean_model_display(m)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels_b, bests_b):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m == msr_b else fmt_p(d['p'], B),
            ]
        rows_b.append(' & '.join(cells) + r' \\')

    tex_b = render_table(
        caption_title='Main Model Comparison on Full MegaScale Split Test Set: Double Mutants (Epistatic vs. Additive Readout)',
        caption_body=(
            r"Mean per-domain Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) across the 20 double-mutant test domains of \texttt{mega\_splits} "
            r"($n=18{,}574$ doubles). One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected). Best values in each metric column are indicated in bold."
        ),
        label='tab:si_megasplits_doubles',
        col_spec='l' + 'c' * 8,
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels_b) + r' \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 2) + r' \\',
        ],
        body_rows=rows_b,
    )
    return write_tex('table_S2c_megascale_megasplits.tex', tex_a + '\n' + tex_b)


def build_table_s2d_backbone_origin() -> Path:
    df = pd.read_csv(SUPP_PVAL / '_backbone_and_origin_pvalues.csv')
    specs = [
        ('MegaScale_capped_test_backbone', 'per_code_mean_spearman_native', r'Native Backbone ($n$=40,866)'),
        ('MegaScale_capped_test_backbone', 'per_code_mean_spearman_destabilized', r'Destabilized Backbone ($n$=7,874)'),
        ('MegaScale_capped_test_origin', 'per_group_mean_spearman_natural', r'Natural Domains ($n$=31,910)'),
        ('MegaScale_capped_test_origin', 'per_group_mean_spearman_synthetic', r'Synthetic Domains ($n$=16,830)'),
    ]
    panels, all_models, msr_name = [], set(), None
    for bm, met, title in specs:
        pan, msr, n_u, n_v, B = _extract_model_panel(df, bm, met)
        msr_name = msr
        # Use exact variant count from CSV in title
        title_clean = title.split('(')[0].strip() + f' ($n$={n_v:,})'
        panels.append((title_clean, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)
    bests = [(_best_clean_set(p, 'val', True), _best_clean_set(p, 'pearson', True), _best_clean_set(p, 'rmse', False)) for _, p, _ in panels]

    body_rows = []
    for m in models:
        cells = [clean_model_display(m)]
        for (_, pan, B), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            cells += ['---'] * 4 if not d else [
                fmt_num(d['val'], 3, bold=(m in b_rho)),
                fmt_num(d['pearson'], 3, bold=(m in b_r)),
                fmt_num(d['rmse'], 3, bold=(m in b_rm)),
                '---' if m == msr_name else fmt_p(d['p'], B),
            ]
        body_rows.append(' & '.join(cells) + r' \\')

    tex = render_table(
        caption_title='Main Model Comparison on Homology-Restricted MegaScale Test Set Stratified by Backbone Stability and Domain Origin',
        caption_body=(
            r"Mean per-library Spearman's $\rho$, Pearson's $r$, and raw RMSE (kcal/mol) stratified by wild-type backbone type "
            r"(native vs.\ pre-destabilized mutant backgrounds) and protein origin (natural vs.\ de novo synthetic domains). "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected). Best values in each metric column are indicated in bold."
        ),
        label='tab:si_megascale_backbone_origin',
        col_spec='l' + 'c' * 16,
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for t, _, _ in panels) + r' \\',
            r'\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}',
            'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p$'] * 4) + r' \\',
        ],
        body_rows=body_rows,
    )
    return write_tex('table_S2d_megascale_backbone_origin.tex', tex)


def build_table_s3_epistasis() -> tuple[Path, Path]:
    # Table S3a: 5-level epistasis hierarchy (figure2e)
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
        pan, msr, _, _, B = _extract_model_panel(df, 'MegaScale_epistasis_5level_hierarchy', met)
        msr_name = msr
        panels.append((title, pan, B))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first=msr_name)

    # Also grab companion Pearson r & RMSE from naive_rho (all doubles)
    pan_naive = panels[0][1]
    b_r_naive = _best_clean_set(pan_naive, 'pearson', True, exclude_internal=True)
    b_rm_naive = _best_clean_set(pan_naive, 'rmse', False, exclude_internal=True)
    bests = [_best_clean_set(p, 'val', True, exclude_internal=True) for _, p, _ in panels]

    rows_3a = []
    for m in models:
        cells = [clean_model_display(m)]
        for (_, pan, B), b_set in zip(panels, bests):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_set)))
                cells.append('---' if m == msr_name else fmt_p(d['p'], B))
        dn = pan_naive.get(m, {})
        cells.append(fmt_num(dn.get('pearson'), 3, bold=(m in b_r_naive)))
        cells.append(fmt_num(dn.get('rmse'), 3, bold=(m in b_rm_naive)))
        rows_3a.append(' & '.join(cells) + r' \\')

    tex_3a = render_table(
        caption_title='Main Model Comparison Across the 5-Level Epistasis Hierarchy on Homology-Restricted MegaScale Test Doubles',
        caption_body=(
            r"Evaluation of predicted epistasis ($\widehat{\Delta\Delta\Delta G} = \hat{y}_{AB} - \hat{y}_A - \hat{y}_B$) on the 12,455 "
            r"MegaScale test doubles with both constituent singles measured across 13 domains (7,405 doubles with $\Delta G_{\mathrm{add}}>0$). "
            r"Columns report all 5 hierarchy levels (Naive $\rho$, Beyond-global $\rho$, Pair $\rho$, Line $\rho$, and double-centred Cell magnitude "
            r"on all doubles and $\Delta G_{\mathrm{add}}>0$) together with one-sided paired domain-cluster $\times$ seed bootstrap $p$-values "
            r"($B=10{,}000$; uncorrected; testing $H_1\colon \text{ESM-MSR} > \text{model}$), plus pooled $\Delta\Delta\Delta G$ Pearson's $r$ and RMSE (kcal/mol). "
            r"Internal controls (context control and $-\Delta G_{\mathrm{add}}$ floor ceiling) are shown for reference and excluded from bolding of best external models."
        ),
        label='tab:si_epistasis_hierarchy',
        col_spec='l' + 'c' * (2 * len(panels) + 2),
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels) + r' & \multicolumn{2}{c}{All Doubles $\Delta\Delta\Delta G$} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15}',
            'ID & ' + ' & '.join([r'Score & $p$'] * len(panels)) + r' & $r$ & RMSE \\',
        ],
        body_rows=rows_3a,
    )
    p3a = write_tex('table_S3a_epistasis_hierarchy.tex', tex_3a)

    # Table S3b: Epistasis RMSE & Sign MCC
    df_rm = pd.read_csv(SUPP_PVAL / '_epi_rmse_pvalues.csv')
    df_mc = pd.read_csv(SUPP_PVAL / '_epi_mcc_pvalues.csv')
    p_rm_all, msr_rm, _, _, B_rm = _extract_model_panel(df_rm, 'MegaScale_capped_test_epistasis_RMSE', 'all_doubles', primary_val_is_rmse=True)
    p_rm_gt0, _, _, _, _ = _extract_model_panel(df_rm, 'MegaScale_capped_test_epistasis_RMSE', 'dG_add_gt0', primary_val_is_rmse=True)
    p_mc_raw, msr_mc, _, _, B_mc = _extract_model_panel(df_mc, 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_raw_dddG_gt0')
    p_mc_sat, _, _, _, _ = _extract_model_panel(df_mc, 'MegaScale_capped_test_epistasis', 'epistasis_sign_MCC_saturation_corrected_r_gt0')

    # Note: _epi_rmse has pearson_r & rmse; also attach spearman from naive_rho / figure2e if available
    all_m3b = sort_models(list(set(p_rm_all) | set(p_rm_gt0) | set(p_mc_raw) | set(p_mc_sat)), msr_first=msr_rm)
    b_rm1 = _best_clean_set(p_rm_all, 'rmse', False)
    b_r1 = _best_clean_set(p_rm_all, 'pearson', True)
    b_rm2 = _best_clean_set(p_rm_gt0, 'rmse', False)
    b_r2 = _best_clean_set(p_rm_gt0, 'pearson', True)
    b_m1 = _best_clean_set(p_mc_raw, 'val', True)
    b_m2 = _best_clean_set(p_mc_sat, 'val', True)

    rows_3b = []
    for m in all_m3b:
        cells = [clean_model_display(m)]
        for pan, b_r, b_rm in ((p_rm_all, b_r1, b_rm1), (p_rm_gt0, b_r2, b_rm2)):
            d = pan.get(m)
            if not d:
                cells += ['---', '---', '---']
            else:
                cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                cells.append('---' if m == msr_rm else fmt_p(d['p'], B_rm))
        for pan, b_mc in ((p_mc_raw, b_m1), (p_mc_sat, b_m2)):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_mc)))
                cells.append('---' if m == msr_mc else fmt_p(d['p'], B_mc))
        rows_3b.append(' & '.join(cells) + r' \\')

    tex_3b = render_table(
        caption_title=r'Epistasis $\Delta\Delta\Delta G$ Calibration (RMSE, Pearson $r$) and Sign Classification (MCC) on MegaScale Test Doubles',
        caption_body=(
            r"Epistasis calibration RMSE (kcal/mol, lower is better) and Pearson's $r$ on all 12,455 test doubles and on the 7,405 non-floor doubles "
            r"($\Delta G_{\mathrm{add}}>0$), alongside Matthews Correlation Coefficient (MCC) for classifying epistasis sign on raw $\Delta\Delta\Delta G>0$ "
            r"and saturation-corrected residuals $r>0$. One-sided $p$-values test ESM-MSR superiority (lower RMSE or higher MCC) via paired domain-cluster $\times$ seed "
            r"bootstrap ($B=10{,}000$; uncorrected). Best values in each metric column are indicated in bold."
        ),
        label='tab:si_epistasis_calibration_mcc',
        col_spec='l' + 'c' * 10,
        header_lines=[
            r'{} & \multicolumn{3}{c}{All Doubles ($n$=12,455)} & \multicolumn{3}{c}{$\Delta G_{\mathrm{add}}>0$ ($n$=7,405)} & \multicolumn{2}{c}{Sign MCC (Raw)} & \multicolumn{2}{c}{Sign MCC (Sat.-Corr.)} \\',
            r'\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}',
            r'ID & RMSE & $r$ & $p_{\mathrm{RMSE}}$ & RMSE & $r$ & $p_{\mathrm{RMSE}}$ & MCC & $p_{\mathrm{MCC}}$ & MCC & $p_{\mathrm{MCC}}$ \\',
        ],
        body_rows=rows_3b,
    )
    p3b = write_tex('table_S3b_epistasis_calibration_mcc.tex', tex_3b)
    return p3a, p3b


def build_table_s4_thermostability() -> tuple[Path, Path, Path]:
    df = pd.read_csv(MAIN_PVAL / 'figure3a_thermostability_pvalues.csv')

    def build_part(bms: list[str], fname: str, part_title: str, label: str) -> Path:
        panels_mut, panels_prot, titles = [], [], []
        all_models, msr_name = set(), None
        for bm in bms:
            pm, msr, _, n_var, B = _extract_model_panel(df, bm, 'pooled_spearman_rho', 'mutation_x_seed')
            pp, _, n_prot, _, _ = _extract_model_panel(df, bm, 'pooled_spearman_rho', 'protein_cluster_x_seed')
            msr_name = msr
            panels_mut.append((pm, B))
            panels_prot.append((pp, B))
            titles.append(f'{bm} ($n$={n_var:,}; {n_prot} prot.)')
            all_models.update(pm.keys())
        models = sort_models(list(all_models), msr_first=msr_name)

        # Bolding strictly among clean (non-overlapping) models per benchmark
        bests = [
            (_best_clean_set(pm, 'val', True), _best_clean_set(pm, 'pearson', True), _best_clean_set(pm, 'rmse', False))
            for pm, _ in panels_mut
        ]
        # Track if a model has overlap on ANY benchmark in this table (still annotate cell-level * if benchmark-specific!)
        body_rows = []
        for m in models:
            cells = [clean_model_display(m)]
            for (pm, B), (pp, _), (b_rho, b_r, b_rm) in zip(panels_mut, panels_prot, bests):
                dm = pm.get(m)
                dp = pp.get(m)
                if not dm:
                    cells += ['---'] * 5
                else:
                    sfx = r'$^*$' if dm.get('overlap') else ''
                    cells.append(fmt_num(dm['val'], 3, bold=(m in b_rho), suffix=sfx))
                    cells.append(fmt_num(dm['pearson'], 3, bold=(m in b_r)))
                    cells.append(fmt_num(dm['rmse'], 3, bold=(m in b_rm)))
                    cells.append('---' if m == msr_name else fmt_p(dm['p'], B))
                    cells.append('---' if m == msr_name else fmt_p(dp['p'] if dp else np.nan, B))
            body_rows.append(' & '.join(cells) + r' \\')

        cmids = ' '.join(f'\\cmidrule(lr){{{2 + 5*i}-{6 + 5*i}}}' for i in range(len(bms)))
        tex = render_table(
            caption_title=f'Main Model Comparison on External Thermostability Benchmarks ({part_title})',
            caption_body=(
                r"Pooled Spearman's $\rho$, Pearson's $r$, and raw (unrecalibrated) RMSE (kcal/mol) across external thermostability benchmarks "
                r"(singles-only heads on single-mutant datasets; multi-mutant heads on PTMUL; 3-seed mean for retrained models). "
                r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ under both mutation $\times$ seed resampling ($p_{\mathrm{mut}}$, primary) "
                r"and protein-cluster $\times$ seed resampling ($p_{\mathrm{prot}}$, conservative sensitivity check; $B=10{,}000$; uncorrected). "
                r"Best values among clean (non-overlapping) models in each metric column are indicated in bold. "
                r"$^*$Documented training-set homology overlap with the benchmark; excluded from clean bolding and Bonferroni families."
            ),
            label=label,
            col_spec='l' + 'c' * (5 * len(bms)),
            header_lines=[
                '{} & ' + ' & '.join(f'\\multicolumn{{5}}{{c}}{{{t}}}' for t in titles) + r' \\',
                cmids,
                'ID & ' + ' & '.join([r'$\rho$ & $r$ & RMSE & $p_{\mathrm{mut}}$ & $p_{\mathrm{prot}}$'] * len(bms)) + r' \\',
            ],
            body_rows=body_rows,
        )
        return write_tex(fname, tex)

    p4a = build_part(['Ssym', 'S461', 'K2369', 'S783'], 'table_S4a_thermostability_part1.tex', 'Part 1: Ssym, S461, K2369, S783', 'tab:si_thermostability_p1')
    p4b = build_part(['S2648', 'Q3421', 'S8754', 'PTMUL'], 'table_S4b_thermostability_part2.tex', 'Part 2: S2648, Q3421, S8754, PTMUL', 'tab:si_thermostability_p2')

    # Table S4c: Raw RMSE (MegaScale + 8 External) + Ssym Antisymmetry
    df_rmse = pd.read_csv(SUPP_PVAL / '_thermostability_rmse_pvalues.csv')
    bms_r = [
        ('tsuboyama_capped_test', 'MegaScale'),
        ('ssym', 'Ssym'),
        ('s461', 'S461'),
        ('k2369', 'K2369'),
        ('s783', 'S783'),
        ('s2648', 'S2648'),
        ('q3421', 'Q3421'),
        ('s8754', 'S8754'),
        ('ptmul', 'PTMUL'),
    ]
    panels_raw = []
    all_mr, msr_r = set(), None
    for bk, label_b in bms_r:
        prw, msr, _, _, B = _extract_model_panel(df_rmse, bk, 'rmse_raw', primary_val_is_rmse=True)
        msr_r = msr
        panels_raw.append((label_b, prw, B))
        all_mr.update(prw.keys())
    models_r = sort_models(list(all_mr), msr_first=msr_r)

    def make_rmse_subtable(panels_list, title_sfx, lbl):
        bests_l = [_best_clean_set(p, 'rmse', False) for _, p, _ in panels_list]
        rows = []
        for m in models_r:
            cells = [clean_model_display(m)]
            for (_, pan, B), b_set in zip(panels_list, bests_l):
                d = pan.get(m)
                if not d:
                    cells += ['---', '---']
                else:
                    sfx = r'$^*$' if d.get('overlap') else ''
                    cells.append(fmt_num(d['rmse'], 3, bold=(m in b_set), suffix=sfx))
                    cells.append('---' if m == msr_r else fmt_p(d['p'], B))
            rows.append(' & '.join(cells) + r' \\')
        cmids = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(len(panels_list)))
        return render_table(
            caption_title=f'Calibration RMSE Comparison Across MegaScale and 8 External Thermostability Benchmarks ({title_sfx})',
            caption_body=(
                f"{title_sfx} RMSE (kcal/mol, lower is better) across Homology-Restricted MegaScale test (33 domains) and 8 external thermostability sets. "
                r"One-sided $p$-values test $H_1\colon \text{RMSE}_{\text{ESM-MSR}} < \text{RMSE}_{\text{model}}$ via paired bootstrap "
                r"(domain-cluster $\times$ seed on MegaScale; mutation $\times$ seed on external sets; $B=10{,}000$; uncorrected). "
                r"Lowest clean (non-overlapping) RMSE per column is bolded. $^*$Documented training overlap."
            ),
            label=lbl,
            col_spec='l' + 'c' * (2 * len(panels_list)),
            header_lines=[
                '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_list) + r' \\',
                cmids,
                'ID & ' + ' & '.join([r'RMSE & $p$'] * len(panels_list)) + r' \\',
            ],
            body_rows=rows,
        )

    tex_raw = make_rmse_subtable(panels_raw, 'Raw (Unrecalibrated)', 'tab:si_thermo_rmse_raw')

    # Ssym Antisymmetry & Bias subtable
    df_as = pd.read_csv(SUPP_PVAL / '_ssym_antisymmetry_bias_pvalues.csv')
    pan_r_dr, msr_as, _, _, B_as = _extract_model_panel(df_as, 'Ssym_342_pairs', 'antisymmetry_r_dir_rev')
    pan_bias, _, _, _, _ = _extract_model_panel(df_as, 'Ssym_342_pairs', 'abs_bias_delta')
    models_as = sort_models(list(set(pan_r_dr) | set(pan_bias)), msr_first=msr_as)
    b_rdr = _best_clean_set(pan_r_dr, 'val', higher_is_better=False)  # more negative is better (ideal -1)
    b_bias = _best_clean_set(pan_bias, 'val', higher_is_better=False)  # smaller |bias| is better (ideal 0)
    b_pr = _best_clean_set(pan_r_dr, 'pearson', True)
    b_rm = _best_clean_set(pan_r_dr, 'rmse', False)

    rows_as = []
    for m in models_as:
        d1 = pan_r_dr.get(m, {})
        d2 = pan_bias.get(m, {})
        sfx = r'$^*$' if d1.get('overlap') else ''
        cells = [
            clean_model_display(m) + sfx,
            fmt_num(d1.get('val'), 3, bold=(m in b_rdr)),
            '---' if m == msr_as else fmt_p(d1.get('p'), B_as),
            fmt_num(d2.get('val'), 3, bold=(m in b_bias)),
            '---' if m == msr_as else fmt_p(d2.get('p'), B_as),
            fmt_num(d1.get('pearson'), 3, bold=(m in b_pr)),
            fmt_num(d1.get('rmse'), 3, bold=(m in b_rm)),
        ]
        rows_as.append(' & '.join(cells) + r' \\')

    tex_as = render_table(
        caption_title='Ssym Direct--Reverse Antisymmetry and Bias Across Models (342 Mutation Pairs)',
        caption_body=(
            r"Direct--reverse Pearson correlation $r_{\mathrm{dir,rev}}$ (ideal $-1.000$), absolute mean antisymmetry bias $|\langle\delta\rangle|$ (kcal/mol, ideal $0.000$), "
            r"pooled Ssym Pearson's $r$, and raw RMSE (kcal/mol) across the 342 direct/reverse mutation pairs ($n=684$). "
            r"One-sided $p$-values test ESM-MSR (Dual-view) superiority (more negative $r_{\mathrm{dir,rev}}$ or smaller $|\langle\delta\rangle|$) "
            r"via paired mutation-pair $\times$ seed bootstrap ($B=10{,}000$; uncorrected). "
            r"Best clean (non-overlapping) values are bolded. $^*$Documented Ssym training overlap."
        ),
        label='tab:si_ssym_antisymmetry',
        col_spec='lcccccc',
        header_lines=[
            r'{} & \multicolumn{2}{c}{Antisymmetry ($r_{\mathrm{dir,rev}}$)} & \multicolumn{2}{c}{Abs. Bias ($|\langle\delta\rangle|$)} & \multicolumn{2}{c}{Pooled Ssym} \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}',
            r'ID & $r_{\mathrm{dir,rev}}$ & $p$ & $|\langle\delta\rangle|$ & $p$ & $r$ & RMSE \\',
        ],
        body_rows=rows_as,
    )
    p4c = write_tex('table_S4c_thermostability_rmse_and_ssym_antisym.tex', tex_raw + '\n' + tex_as)
    return p4a, p4b, p4c


def build_table_s5_heldout_ood() -> tuple[Path, Path]:
    df = pd.read_csv(MAIN_PVAL / 'figure3b_heldout_benchmarks_pvalues.csv')
    specs = [
        ('ProteinGym_heldout_173_assays', 'category_weighted_mean_spearman_rho', r'ProteinGym (173 assays)', False),
        ('Domainome_heldout_436_domains', 'per_domain_mean_spearman_rho', r'Domainome (436 domains)', True),
        ('MGnify_heldout_pooled', 'pooled_spearman_rho', r'MGnify ($n$=612,406)', True),
        ('Rocklin_designs_heldout_pooled', 'pooled_spearman_rho', r'Rocklin Designs ($n$=109,581)', True),
    ]
    panels = []
    all_models = set()
    for bm, met, title, has_rmse in specs:
        pan, msr, _, _, B = _extract_model_panel(df, bm, met)
        # Normalize 'ESM-MSR (σ=1)' -> 'ESM-MSR' so ProteinGym aligns on the same ESM-MSR row!
        if msr != 'ESM-MSR':
            pan['ESM-MSR'] = pan.pop(msr)
        panels.append((title, pan, B, has_rmse))
        all_models.update(pan.keys())
    models = sort_models(list(all_models), msr_first='ESM-MSR')

    bests = [
        (
            _best_clean_set(p, 'val', True, exclude_internal=True),
            _best_clean_set(p, 'pearson', True, exclude_internal=True),
            _best_clean_set(p, 'rmse', False, exclude_internal=True) if has_rm else set(),
        )
        for _, p, _, has_rm in panels
    ]

    body_rows = []
    for m in models:
        cells = [clean_model_display(m)]
        for (_, pan, B, has_rm), (b_rho, b_r, b_rm) in zip(panels, bests):
            d = pan.get(m)
            ncols = 4 if has_rm else 3
            if not d:
                cells += ['---'] * ncols
            else:
                sfx = r'$^*$' if d.get('overlap') else ''
                cells.append(fmt_num(d['val'], 3, bold=(m in b_rho), suffix=sfx))
                cells.append(fmt_num(d['pearson'], 3, bold=(m in b_r)))
                if has_rm:
                    cells.append(fmt_num(d['rmse'], 3, bold=(m in b_rm)))
                cells.append('---' if m == 'ESM-MSR' else fmt_p(d['p'], B))
        body_rows.append(' & '.join(cells) + r' \\')

    tex_5a = render_table(
        caption_title='Main Model Comparison on Large-Scale Held-Out Out-of-Distribution Benchmarks',
        caption_body=(
            r"Comparison across four homology-held-out benchmarks: ProteinGym (173 held-out DMS assays, equal-category-weighted macro-average Spearman's $\rho$ "
            r"and Pearson's $r$; RMSE omitted because assays span heterogeneous fitness units), Domainome (436 held-out domains, $n=451{,}569$), "
            r"MGnify stability ($n=612{,}406$ variants across 140,694 held-out scaffolds), and Rocklin de novo designs ($n=109{,}581$ variants across 623 held-out scaffolds). "
            r"One-sided $p$-values test $H_1\colon \rho_{\text{ESM-MSR}} > \rho_{\text{model}}$ via paired cluster/assay $\times$ seed bootstrap "
            r"($B=10{,}000$ on ProteinGym/Domainome; $B=1{,}000$ on MGnify/Rocklin; uncorrected). "
            r"Best values among clean external models (excluding internal $\sigma=0.5$ variant and overlapping models) are indicated in bold. "
            r"$^*$Documented training overlap (ESM3$\Delta G$: H 2\% on ProteinGym; trained directly on MGnify stability data). "
            r"--- indicates not evaluated or not applicable."
        ),
        label='tab:si_heldout_ood',
        col_spec='l' + 'c' * 15,
        header_lines=[
            r'{} & \multicolumn{3}{c}{ProteinGym (173 assays)} & \multicolumn{4}{c}{Domainome (436 domains)} & \multicolumn{4}{c}{MGnify ($n$=612,406)} & \multicolumn{4}{c}{Rocklin Designs ($n$=109,581)} \\',
            r'\cmidrule(lr){2-4} \cmidrule(lr){5-8} \cmidrule(lr){9-12} \cmidrule(lr){13-16}',
            r'ID & $\rho$ & $r$ & $p$ & $\rho$ & $r$ & RMSE & $p$ & $\rho$ & $r$ & RMSE & $p$ & $\rho$ & $r$ & RMSE & $p$ \\',
        ],
        body_rows=body_rows,
    )
    p5a = write_tex('table_S5a_heldout_ood_benchmarks.tex', tex_5a)

    # Table S5b: ProteinGym LoRA scale sigma sweep across 5 categories
    df_3c = pd.read_csv(MAIN_PVAL / 'figure3c_proteingym_sigma_pvalues.csv')
    src_3c = pd.read_csv(MAIN_SRC / 'figure3_panel_c_proteingym_sigma_sourcedata.csv')
    held = src_3c[(~src_3c['leaked']) & (~src_3c['dropped_large'])]
    cats = [
        ('Stability', 'Stability'),
        ('Expression', 'Expression'),
        ('Activity', 'Activity'),
        ('Binding', 'Binding'),
        ('OrganismalFitness', 'Org. Fitness'),
    ]
    rows_5b = []
    for cat_key, cat_disp in cats:
        g = held[held['category'] == cat_key]
        n_a = len(g)
        r0 = float(g['rho_dep_0.0'].mean())
        r05 = float(g['rho_dep_0.5'].mean())
        r10 = float(g['rho_dep_1.0'].mean())
        pr0 = float(g['pearson_r_dep_0.0'].mean())
        pr05 = float(g['pearson_r_dep_0.5'].mean())
        pr10 = float(g['pearson_r_dep_1.0'].mean())
        sub_p = df_3c[df_3c['subset_or_metric'] == f'{cat_key}: mean_delta_rho']
        p_05_vs_0 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=0.5)') & (sub_p['comparator'] == 'ESM3-small-open (σ=0)')]['p_one_sided_uncorrected'].iloc[0]
        p_10_vs_0 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=1.0)') & (sub_p['comparator'] == 'ESM3-small-open (σ=0)')]['p_one_sided_uncorrected'].iloc[0]
        p_10_vs_05 = sub_p[(sub_p['msr_model'] == 'ESM-MSR (σ=1.0)') & (sub_p['comparator'] == 'ESM-MSR (σ=0.5)')]['p_one_sided_uncorrected'].iloc[0]
        B_3c = int(sub_p['B'].iloc[0])

        best_rho = max(r0, r05, r10)
        best_pr = max(pr0, pr05, pr10)
        cells = [
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
        ]
        rows_5b.append(' & '.join(cells) + r' \\')

    # Macro average row
    cat_r0 = held.groupby('category')['rho_dep_0.0'].mean().mean()
    cat_r05 = held.groupby('category')['rho_dep_0.5'].mean().mean()
    cat_r10 = held.groupby('category')['rho_dep_1.0'].mean().mean()
    cat_p0 = held.groupby('category')['pearson_r_dep_0.0'].mean().mean()
    cat_p05 = held.groupby('category')['pearson_r_dep_0.5'].mean().mean()
    cat_p10 = held.groupby('category')['pearson_r_dep_1.0'].mean().mean()
    # Pull macro p-values from figure3b
    pg_sub = df[(df['benchmark'] == 'ProteinGym_heldout_173_assays') & (df['test_type'] == 'pairwise')]
    p_mac_10_0 = pg_sub[pg_sub['comparator'] == 'ESM3-small-open']['p_one_sided_uncorrected'].iloc[0]
    p_mac_10_05 = pg_sub[pg_sub['comparator'] == 'ESM-MSR σ=0.5']['p_one_sided_uncorrected'].iloc[0]
    best_mr = max(cat_r0, cat_r05, cat_r10)
    best_mp = max(cat_p0, cat_p05, cat_p10)
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
            r"One-sided paired assay $\times$ seed bootstrap $p$-values ($B=10{,}000$; uncorrected) test $H_1\colon \rho_{\sigma=0.5} > \rho_{\sigma=0}$ "
            r"($p_{0.5\text{ vs }0}$), $H_1\colon \rho_{\sigma=1.0} > \rho_{\sigma=0}$ ($p_{1.0\text{ vs }0}$), and $H_1\colon \rho_{\sigma=1.0} > \rho_{\sigma=0.5}$ "
            r"($p_{1.0\text{ vs }0.5}$). Best $\rho$ and $r$ per row are indicated in bold."
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


def build_table_s6_screening_and_ndcg() -> tuple[Path, Path]:
    # Table S6a: NDCG@96 + Property Quadrant Gain
    df_n = pd.read_csv(SUPP_PVAL / '_ndcg96_singles_doubles_pvalues.csv')
    specs_n = [
        ('ndcg96_shifted_overall', 'Shifted Overall'),
        ('ndcg96_shifted_singles', 'Shifted Singles'),
        ('ndcg96_shifted_doubles', 'Shifted Doubles'),
        ('ndcg96_rectified_overall', 'Rectified Overall'),
        ('ndcg96_rectified_singles', 'Rectified Singles'),
        ('ndcg96_rectified_doubles', 'Rectified Doubles'),
    ]
    panels_n, all_mn, msr_n = [], set(), None
    for met, title in specs_n:
        pan, msr, _, _, B = _extract_model_panel(df_n, 'MegaScale_capped_test', met)
        msr_n = msr
        panels_n.append((title, pan, B))
        all_mn.update(pan.keys())
    models_n = sort_models(list(all_mn), msr_first=msr_n)
    bests_n = [_best_clean_set(p, 'val', True) for _, p, _ in panels_n]

    rows_n = []
    for m in models_n:
        cells = [clean_model_display(m)]
        for (_, pan, B), b_set in zip(panels_n, bests_n):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                cells.append(fmt_num(d['val'], 3, bold=(m in b_set)))
                cells.append('---' if m == msr_n else fmt_p(d['p'], B))
        rows_n.append(' & '.join(cells) + r' \\')

    tex_ndcg = render_table(
        caption_title='Main Model Comparison on MegaScale Test NDCG@96 (Shifted and Rectified Variants)',
        caption_body=(
            r"Mean per-domain shifted and rectified NDCG@96 across the 33 Homology-Restricted MegaScale test domains "
            r"(Overall, Singles, and Doubles; 13 domains for doubles). "
            r"One-sided $p$-values test $H_1\colon \text{NDCG@96}_{\text{ESM-MSR}} > \text{NDCG@96}_{\text{model}}$ via paired domain-cluster $\times$ seed "
            r"bootstrap ($B=10{,}000$; uncorrected). Best values per column are indicated in bold."
        ),
        label='tab:si_megascale_ndcg96',
        col_spec='l' + 'c' * 12,
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_n) + r' \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}',
            'ID & ' + ' & '.join([r'NDCG & $p$'] * 6) + r' \\',
        ],
        body_rows=rows_n,
    )

    # Property quadrant gain (figure4a)
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
        pan, msr, _, _, B = _extract_model_panel(df_4a, 'MegaScale_capped_test_singles_k96', met)
        msr_4a = msr
        panels_4a.append((title, pan, B))
        all_m4a.update(pan.keys())
    models_4a = sort_models(list(all_m4a), msr_first=msr_4a)
    # Exclude Oracle ('Top 96 per Domain') from bolding
    bests_4a = [_best_clean_set(p, 'val', True, exclude_internal=True) for _, p, _ in panels_4a]

    rows_4a = []
    for m in models_4a:
        cells = [clean_model_display(m)]
        for (_, pan, B), b_set in zip(panels_4a, bests_4a):
            d = pan.get(m)
            if not d:
                cells += ['---', '---']
            else:
                cells.append(fmt_num(d['val'], 2, bold=(m in b_set)))
                cells.append('---' if m == msr_4a else fmt_p(d['p'], B))
        rows_4a.append(' & '.join(cells) + r' \\')

    tex_4a = render_table(
        caption_title=r'Rectified Stabilizing Gain Across Hydrophobicity $\times$ Solvent-Accessibility Quadrants ($k=96$ Picks per Domain)',
        caption_body=(
            r"Mean per-domain total rectified stabilizing gain ($\sum \max(0, \Delta\Delta G)$, kcal/mol) among the top $k=96$ predicted single mutants "
            r"across the 33 Homology-Restricted MegaScale test domains, decomposed by hydrophobicity change ($\Delta H$) and relative solvent accessibility "
            r"(Buried: $\mathrm{rASA}\le 0.5$; Surface: $\mathrm{rASA}>0.5$). "
            r"One-sided $p$-values test $H_1\colon \text{Gain}_{\text{ESM-MSR}} > \text{Gain}_{\text{model}}$ via paired domain-cluster $\times$ seed bootstrap "
            r"($B=10{,}000$; uncorrected). Oracle (top 96 per domain) is shown as an upper bound and excluded from bolding."
        ),
        label='tab:si_property_gain',
        col_spec='l' + 'c' * 10,
        header_lines=[
            '{} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for t, _, _ in panels_4a) + r' \\',
            r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}',
            'ID & ' + ' & '.join([r'Gain & $p$'] * 5) + r' \\',
        ],
        body_rows=rows_4a,
    )
    p6a = write_tex('table_S6a_ndcg96_and_property_gain.tex', tex_ndcg + '\n' + tex_4a)

    # Table S6b: Practical Screening Enrichment (figure4bcd)
    df_4b = pd.read_csv(MAIN_PVAL / 'figure4bcd_screening_pvalues.csv')
    bm_cfgs = [
        (
            'Figure4b_MegaScale_singles_nocys',
            'MegaScale Singles (33 domains, no-Cys)',
            [
                ('hit_rate_gt_0.5@K=1', r'Hit$>0.5$ @$K$=1', True),
                ('hit_rate_gt_0.5@K=10', r'Hit$>0.5$ @$K$=10', True),
                ('hit_rate_gt_0.5@K=50', r'Hit$>0.5$ @$K$=50', True),
                ('P_at_least_one_hit@K=10', r'$P(\ge 1)$ @$K$=10', True),
                ('P_at_least_one_hit@K=50', r'$P(\ge 1)$ @$K$=50', True),
            ],
            True,
            r'Top-$K$ stabilizing hit rate ($\Delta\Delta G>0.5$ kcal/mol) and probability of at least one hit $P(\ge 1)$',
        ),
        (
            'Figure4c_Domainome_heldout_singles',
            'Domainome Held-Out (436 domains)',
            [
                ('hit_rate_gt_WT@K=1', r'Hit$>$WT @$K$=1', True),
                ('hit_rate_gt_WT@K=10', r'Hit$>$WT @$K$=10', True),
                ('hit_rate_gt_WT@K=50', r'Hit$>$WT @$K$=50', True),
                ('P_at_least_one_hit@K=10', r'$P(\ge 1)$ @$K$=10', True),
                ('P_at_least_one_hit@K=50', r'$P(\ge 1)$ @$K$=50', True),
            ],
            True,
            r'Top-$K$ stabilizing hit rate ($\Delta\Delta G>0$) and probability of at least one hit $P(\ge 1)$',
        ),
        (
            'Figure4d_PGym_Activity_all_38',
            'ProteinGym Activity (38 assays)',
            [
                ('harm_rate_lt_WT_minus_0.5SD@K=1', r'Harm @$K$=1', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=10', r'Harm @$K$=10', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=20', r'Harm @$K$=20', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=50', r'Harm @$K$=50', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=100', r'Harm @$K$=100', False),
            ],
            False,
            r'Top-$K$ deleterious harm rate ($\text{activity}<\text{WT}-0.5\text{ SD}$, lower is better)',
        ),
        (
            'Figure4d_PGym_Activity_enzymes_16',
            'ProteinGym Enzymes (16 assays)',
            [
                ('harm_rate_lt_WT_minus_0.5SD@K=1', r'Harm @$K$=1', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=10', r'Harm @$K$=10', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=20', r'Harm @$K$=20', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=50', r'Harm @$K$=50', False),
                ('harm_rate_lt_WT_minus_0.5SD@K=100', r'Harm @$K$=100', False),
            ],
            False,
            r'Top-$K$ deleterious harm rate ($\text{activity}<\text{WT}-0.5\text{ SD}$, lower is better)',
        ),
    ]

    subtables_6b = []
    for bm_key, bm_title, met_list, has_rmse, metric_desc in bm_cfgs:
        panels_s, all_ms, msr_s = [], set(), None
        for met, label_m, hib in met_list:
            pan, msr, _, _, B = _extract_model_panel(df_4b, bm_key, met)
            msr_s = msr
            panels_s.append((label_m, pan, B, hib))
            all_ms.update(pan.keys())
        models_s = sort_models(list(all_ms), msr_first=msr_s)
        bests_s = [_best_clean_set(p, 'val', hib) for _, p, _, hib in panels_s]
        pan0 = panels_s[0][1]
        b_pr = _best_clean_set(pan0, 'pearson', True)
        b_rm = _best_clean_set(pan0, 'rmse', False) if has_rmse else set()

        rows_s = []
        for m in models_s:
            d0 = pan0.get(m, {})
            sfx = r'$^*$' if d0.get('overlap') else ''
            cells = [clean_model_display(m) + sfx]
            for (_, pan, B, _), b_set in zip(panels_s, bests_s):
                d = pan.get(m)
                if not d:
                    cells += ['---', '---']
                else:
                    cells.append(fmt_num(d['val'], 3, bold=(m in b_set)))
                    cells.append('---' if m == msr_s else fmt_p(d['p'], B))
            cells.append(fmt_num(d0.get('pearson'), 3, bold=(m in b_pr)))
            if has_rmse:
                cells.append(fmt_num(d0.get('rmse'), 3, bold=(m in b_rm)))
            rows_s.append(' & '.join(cells) + r' \\')

        extra_hdr = r' & \multicolumn{2}{c}{Pool}' if has_rmse else r' & Pool'
        extra_cmid = r' \cmidrule(lr){12-13}' if has_rmse else ''
        extra_sub = r' & $r$ & RMSE' if has_rmse else r' & $r$'
        ncols_total = 10 + (2 if has_rmse else 1)
        cmids = ' '.join(f'\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}' for i in range(5)) + extra_cmid
        subtables_6b.append(
            render_table(
                caption_title=f'Practical Library Screening Performance on {bm_title}',
                caption_body=(
                    f"{metric_desc} across library budgets $K$, alongside full-pool Pearson's $r$"
                    + (r" and raw RMSE (kcal/mol)" if has_rmse else "")
                    + r". One-sided $p$-values test ESM-MSR (WT-view) superiority via paired cluster/assay $\times$ seed bootstrap "
                    r"($B=10{,}000$; uncorrected). Best clean values per column are indicated in bold."
                ),
                label=f'tab:si_screening_{bm_key.lower()}',
                col_spec='l' + 'c' * ncols_total,
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
    generated: list[Path] = []
    print('[1/6] Building Tables S1a–S1b (Ablations: In-Distribution & OOD)...')
    p1_list, df_s1 = build_table_s1_ablations()
    generated.extend(p1_list)
    print(df_s1.to_string(index=False))

    print('\n[2/6] Building Tables S2a–S2d (MegaScale Main, Doubles, MegaSplits, Backbone/Origin)...')
    generated.append(build_table_s2a_megascale_overall())
    generated.append(build_table_s2b_megascale_doubles())
    generated.append(build_table_s2c_megasplits())
    generated.append(build_table_s2d_backbone_origin())

    print('[3/6] Building Tables S3a–S3b (Epistasis 5-Level Hierarchy, Calibration RMSE & Sign MCC)...')
    generated.extend(build_table_s3_epistasis())

    print('[4/6] Building Tables S4a–S4c (External Thermostability Benchmarks, Recalibrated RMSE, Ssym Antisymmetry)...')
    generated.extend(build_table_s4_thermostability())

    print('[5/6] Building Tables S5a–S5b (Held-Out OOD Benchmarks & ProteinGym LoRA-Scale Breakdown)...')
    generated.extend(build_table_s5_heldout_ood())

    print('[6/6] Building Tables S6a–S6b (NDCG@96, Property-Quadrant Gain & Practical Screening Enrichment)...')
    generated.extend(build_table_s6_screening_and_ndcg())

    # Write combined master copy-paste file (requires booktabs, float, graphicx, amsmath)
    combined_parts = [
        '% Copy-pastable SI LaTeX tables (requires \\usepackage{booktabs,float,graphicx,amsmath})\n'
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
