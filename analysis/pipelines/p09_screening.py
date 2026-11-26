"""Stage 09 — Virtual screening, property-category gain, Activity preservation, and the ln(K) gain law.

Consolidates:
  - ``08_screening.py``: pooled top-fraction external stability screens + per-protein top-K screens on MegaScale & Domainome
  - ``14_property_category_gain.py``: rectified top-K gain stratified by amino acid, ASA × hydrophobicity, volume, conservation
  - ``15_tsuboyama_single_screening.py``: thresholded precision/recall/counts, top-6 hypergeometric recovery, best-ΔΔG curves
  - ``16_screening_overview.py``: 2×3 multi-benchmark screening overview + rectified gain per variant
  - ``17_gain_per_variant_fit.py``: logarithmic diminishing-returns law ``g(K) = a - b ln K`` + alternative functional forms
  - ``pg_activity/{01_wt_reference,02_activity_screening}.py``: ProteinGym Activity wild-type drop ("hurt rate") screening

Bug fixes & scientific improvements:
  1. ``14_property_category_gain``: uses ``style.raincloud`` so the ``[lo, hi]`` 95% domain+seed bootstrap CI whiskers are
     actually drawn on the dark mean bar alongside the darker-hued median bar, with matching figure caption.
  2. ``17_gain_per_variant_fit``: resamples the 3 training seeds jointly with proteins in ``boot_line`` (eliminating pre-bootstrap
     seed averaging) and fixes the ``multi`` figure footnote (``singles + doubles, Cys excluded`` instead of ``single mutants``).
  3. ``pg_activity``: reads curated reference CSVs from ``analysis/reference/pg_activity/``, writes outputs under
     ``results/03_proteingym/screening/``, and reports the dynamic held-out enzyme count (``16 held-out assays`` rather than
     hardcoded ``18 assays``).
  4. Zero ``importlib`` cross-script imports, zero ``MSR_TSU_SPLIT`` environment mutation, zero ``MODELS`` registry mutation.
  5. Builds **New High-Impact Composite Figure #3**: ``results/00_supplementary_figs/screening_gain_law_composite.{png,pdf}``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.special import gammaln
from scipy.stats import gaussian_kde

from msr_analysis import models as M
from msr_analysis import screening as scr_core
from msr_analysis import style
from msr_analysis.boot import ci
from msr_analysis.common import has_multi
from msr_analysis.data import Frame, load_frame
from msr_analysis.external import register_ext_styles
from msr_analysis.heldout import excluded
from msr_analysis.paths import (
    PGYM_DIR,
    PGYM_REF_CSV,
    PUBLIC,
    REFERENCE,
    RETRAINED,
    ROOT,
    TSUBOYAMA_FEATS_CSV,
    atomic_to_csv,
    results_dir,
)
from msr_analysis.registry import LABEL, MODELS, tsu_datasets
from pipelines import base_parser, resolve_b
from pipelines._tsu_common import (
    BEST_LABEL,
    add_holm_by_group,
    best_other,
    boot_mean,
    compare_grouped,
    compare_pooled,
    fig_width,
    finish_bar_fig,
    load_tsu_test_frame,
    make_domain_counts,
    split_title,
)

STAGE = 'screening'
DESCRIPTION = 'Pooled & per-protein virtual screening, property-category gain, Activity preservation, and g(K) = a - b ln K.'

POOLED_SETS = ['s783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
KS_CURVE = [1, 3, 5, 10, 20, 50, 100]
KS_TICKS = [1, 2, 3, 6, 12, 24, 48, 96]
TAU_DS = {'domainome': 0.0}

FAM_NAME_BASE = {
    'prosst': 'ProSST',
    'pmpnn': 'ProteinMPNN',
    'esm3': 'ESM3-small-open',
    'msr': 'ESM-MSR',
    'me': 'MutateEverything',
    'tm': 'ThermoMPNN',
    'spurs': 'SPURS',
    'geoddg': 'GeoDDG-3D',
    'rosetta': 'Rosetta',
    'esm3dg': 'ESM3ΔG',
}

AICDA = 'AICDA_HUMAN_Gajula_2014_3cycles'
PG_ENZYME = {
    AICDA, 'AMIE_PSEAE_Wrenbeck_2017', 'ANCSZ_Hobbs_2022', 'CAS9_STRP1_Spencer_2017_positive',
    'CASP3_HUMAN_Roychowdhury_2020', 'CASP7_HUMAN_Roychowdhury_2020', 'ENVZ_ECOLI_Ghose_2023',
    'HEM3_HUMAN_Loggerenberg_2023', 'LGK_LIPST_Klesmith_2015', 'MET_HUMAN_Estevam_2023',
    'OTC_HUMAN_Lo_2023', 'OXDA_RHOTO_Vanella_2023_activity', 'PTEN_HUMAN_Mighell_2018',
    'Q59976_STRSQ_Romero_2015', 'RNC_ECOLI_Weeks_2023', 'SRC_HUMAN_Ahler_2019',
    'UBE4B_MOUSE_Starita_2013', 'VKOR1_HUMAN_Chiasson_2020_activity',
}
PG_FLUOR = {
    'D7PM05_CLYGR_Somermeyer_2022', 'GFP_AEQVI_Sarkisyan_2016', 'Q6WV12_9MAXI_Somermeyer_2022',
    'Q8WTC7_9CNID_Somermeyer_2022', 'RPC1_LAMBD_Li_2019_high-expression',
    'RPC1_LAMBD_Li_2019_low-expression', 'PHOT_CHLRE_Chen_2023',
}

PALETTE_CAT = ['#2c7bb6', '#fdae61', '#abd9e9', '#d7191c']
AA_GROUPS = {
    'Hydrophobic': (['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P', 'G'], plt.cm.Blues),
    'Polar': (['S', 'T', 'Y', 'N', 'Q'], plt.cm.Greens),
    'Positive': (['R', 'H', 'K'], plt.cm.Reds),
    'Negative': (['D', 'E'], plt.cm.Purples),
}
AA_ORDER = ['C'] + [a for g, _ in AA_GROUPS.values() for a in g]
AA_COLOR = {'C': '#FFD700'}
for _aas, _cmap in AA_GROUPS.values():
    for _aa, _c in zip(_aas, _cmap(np.linspace(0.35, 0.9, len(_aas)))):
        AA_COLOR[_aa] = _c

SHAPES = {
    'a - b ln K': (lambda k, a, b: a - b * np.log(k), [0.8, 0.1]),
    'a K^-b': (lambda k, a, b: a * k ** -b, [0.8, 0.2]),
    'a K^-b + c': (lambda k, a, b, c: a * k ** -b + c, [0.8, 0.5, 0.0]),
    'a exp(-K/t) + c': (lambda k, a, t, c: a * np.exp(-k / t) + c, [0.5, 30, 0.2]),
    'a - b ln(K + k0)': (lambda k, a, b, k0: a - b * np.log(k + k0), [0.8, 0.1, 0.5]),
}


# ----------------------------------------------------------------------------------------------
# Family roster helpers (pure; zero mutation of msr_analysis.registry.MODELS)
# ----------------------------------------------------------------------------------------------
def build_family_specs(split: str = 'capped') -> dict:
    fam_name = dict(FAM_NAME_BASE)
    fam_all = sorted(fam_name, key=lambda f: M.ORDER_INDEX[fam_name[f]])
    fam_transfer = [f for f in fam_all if M.BY_NAME[fam_name[f]].group == 'transfer']
    fam_label = {f: M.BY_NAME[n].label_all.replace('\n', ' ') for f, n in fam_name.items()}
    fam_label_singles = {f: (M.version_label(n, 0) if M.BY_NAME[n].two_versions else n) for f, n in fam_name.items()}
    fam_label_composite = dict(fam_label)
    single_key = {f: M.BY_NAME[n].single for f, n in fam_name.items()}
    multi_key = {f: M.BY_NAME[n].multi for f, n in fam_name.items()}
    fam_style = dict(fam_name)
    fam_color_all = {f: M.line_style(n)[:3] + (8 if n == 'ESM-MSR' else 6,) for f, n in fam_name.items()}

    if split == 'mega':
        extra = {
            'spursu': ('SPURS, trained adapter', 'SPURS multi, trained adapter', 'spursu', 'spursum', 'spursu', (MODELS['spursu']['color'], '-', 2.0, 6)),
            'tmpub': ('ThermoMPNN, published', 'ThermoMPNN-D, published', 'tmpnnpub', 'tmpnndpub', 'tmpnnpub', (MODELS['tmpnnpub']['color'], '--', 2.0, 6)),
        }
        for f, (ls, lc, sk, mk, st, col) in extra.items():
            fam_label[f] = ls
            fam_label_singles[f] = ls
            fam_label_composite[f] = f"{ls} | {lc.split(',')[0]}"
            single_key[f], multi_key[f], fam_style[f], fam_color_all[f] = sk, mk, st, col
            fam_name[f] = {'spursu': 'SPURS', 'tmpub': 'ThermoMPNN'}[f]
        fam_all = fam_all[:7] + list(extra) + fam_all[7:]

    return dict(
        fam_name=fam_name,
        fam_all=fam_all,
        fam_transfer=fam_transfer,
        fam_label=fam_label,
        fam_label_singles=fam_label_singles,
        fam_label_composite=fam_label_composite,
        single_key=single_key,
        multi_key=multi_key,
        fam_style=fam_style,
        fam_color_all=fam_color_all,
    )


def fam_key(fam: str, fr: Frame, specs: dict, mode: str = 'default') -> str | None:
    ks, kd = specs['single_key'][fam], specs['multi_key'][fam]
    if ks == kd or mode == 'singles' or not has_multi(fr):
        return ks if ks in fr.preds else None
    if ks not in fr.preds or kd not in fr.preds:
        return None
    ck = f'{ks}|{kd}'
    if ck not in fr.preds:
        n1 = (fr.spine['n_mut'] == 1).to_numpy()
        fr.preds[ck] = np.where(n1[None, :], fr.preds[ks], fr.preds[kd])
    return ck


# ----------------------------------------------------------------------------------------------
# Part 1: 08_screening (pooled external + per-protein MegaScale & Domainome)
# ----------------------------------------------------------------------------------------------
def run_pooled_external(specs: dict, tau: float, B: int, frac: float, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows, reps_store = [], {}
    for ds in POOLED_SETS:
        fr = load_frame(ds)
        fr = Frame(fr.name, fr.spine, dict(fr.preds), dict(fr.notes))
        sp = fr.spine
        for fam in specs['fam_transfer']:
            k = fam_key(fam, fr, specs)
            if k is None:
                continue
            ok = sp['use'].to_numpy() & np.isfinite(fr.preds[k]).all(axis=0)
            y = sp['y'].to_numpy()[ok]
            cl = sp['cluster'].to_numpy()[ok]
            outs = [scr_core.pooled_screen(p[ok], y, cl, frac, tau, B, seed=5) for p in fr.preds[k]]
            n_seeds = len(outs)
            if n_seeds == 1:
                reps = outs[0]['reps']
            else:
                sw = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=B).astype(float) / n_seeds
                reps = np.einsum('sbm,bs->bm', np.stack([o['reps'] for o in outs], axis=0), sw)
            base = np.mean([o['base'] for o in outs], axis=0)
            reps_store[(ds, fam)] = reps
            # Keep 'gain' as mean_y for compatibility while also recording rectified_gain & rectified_gain_mean
            metric_cols = ('hit_rate', 'best', 'base_rate', 'gain', 'rectified_gain', 'rectified_gain_mean')
            for j, m in enumerate(metric_cols[: reps.shape[1]]):
                lo, hi = np.nanquantile(reps[:, j], [0.025, 0.975])
                rows.append(dict(
                    dataset=ds, family=fam, key=k, metric=m, value=float(base[j]),
                    lo=float(lo), hi=float(hi), n=int(ok.sum()), oracle_best=float(y.max()),
                ))
    P = pd.DataFrame(rows)
    atomic_to_csv(P, out_dir / 'screening_pooled_external.csv', index=False, float_format='%.4f')

    # Paired differences ESM-MSR - baseline in hit rate
    diff_rows = []
    for ds in POOLED_SETS:
        for f in specs['fam_transfer']:
            if f == 'msr':
                continue
            if (ds, 'msr') in reps_store and (ds, f) in reps_store:
                d = reps_store[(ds, 'msr')][:, 0] - reps_store[(ds, f)][:, 0]
                diff_rows.append(dict(
                    dataset=ds, baseline=f, d_hit_rate=float(np.nanmean(d)),
                    lo=float(np.nanquantile(d, 0.025)), hi=float(np.nanquantile(d, 0.975)),
                ))
    atomic_to_csv(pd.DataFrame(diff_rows), out_dir / 'screening_pooled_paired_diff.csv', index=False, float_format='%.4f')

    # Pooled external figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.4), sharex=True)
    fam_name = specs['fam_name']
    fam_transfer = specs['fam_transfer']
    for ax, metric, yl in (
        (axes[0], 'hit_rate', f'hit rate (ddG > {tau})'),
        (axes[1], 'best', 'best ddG in selection (kcal/mol)'),
        (axes[2], 'gain', 'mean ddG of selection (kcal/mol)'),
    ):
        d = P[P.metric == metric]
        table = {(r.dataset, fam_name[r.family]): r.value for r in d.itertuples() if r.family in fam_name}
        ci_map = {(r.dataset, fam_name[r.family]): (r.lo, r.hi) for r in d.itertuples() if r.family in fam_name}
        style.dot_panel(ax, POOLED_SETS, [fam_name[f] for f in fam_transfer], table, ci_map, highlight=['ESM-MSR'], ylabel=yl, size=6.5)
        if metric == 'hit_rate':
            br = P[P.metric == 'base_rate'].groupby('dataset').value.first()
            for i, ds in enumerate(POOLED_SETS):
                if ds in br:
                    ax.plot([i - 0.42, i + 0.42], [br[ds]] * 2, color='#c0392b', lw=1.2, ls='--', zorder=2)
            ax.plot([], [], color='#c0392b', ls='--', label='hit rate of the whole set (random selection)')
        elif metric == 'gain':
            ax.axhline(0, color='#999999', lw=0.8, ls=':')
        else:
            orc = P.groupby('dataset').oracle_best.first()
            for i, ds in enumerate(POOLED_SETS):
                if ds in orc:
                    ax.plot([i - 0.42, i + 0.42], [orc[ds]] * 2, color='#2c3e50', lw=1.2, ls=':', zorder=2)
            ax.plot([], [], color='#2c3e50', ls=':', label='best variant in the set (oracle)')
        if metric != 'gain':
            ax.legend(loc='upper right', ncol=2, fontsize=8)
    hs = [Line2D([0], [0], label=specs['fam_label'][f], **style.marker_kwargs(fam_name[f], 7, hollow=False)) for f in fam_transfer]
    axes[0].legend(handles=hs + axes[0].get_legend_handles_labels()[0][-1:], loc='upper left', ncol=3, fontsize=8)
    axes[0].set_title(f'Pooled screen: the top {int(frac * 100)}% of each set by predicted ddG; whiskers 95% protein-level bootstrap (B = {B})')
    axes[2].set_xticklabels([LABEL[d] for d in POOLED_SETS], rotation=30, ha='right')
    fig.tight_layout()
    style.save(fig, out_dir / 'screening_pooled_external')
    return P, reps_store


def per_protein_curves(
    ds: str,
    ks: list[int],
    tau: float,
    B: int,
    specs: dict,
    exclude_groups: set[str] | None = None,
    mode: str = 'default',
    exclude_cys: bool = False,
    fams: list[str] | None = None,
    fr: Frame | None = None,
) -> pd.DataFrame:
    if fr is None:
        raw = load_frame(ds)
        fr = Frame(raw.name, raw.spine, dict(raw.preds), dict(raw.notes))
    else:
        fr = Frame(fr.name, fr.spine, dict(fr.preds), dict(fr.notes))
    sp = fr.spine
    keep = ~sp['group'].astype(str).isin(exclude_groups).to_numpy() if exclude_groups else np.ones(len(sp), dtype=bool)
    if exclude_cys:
        to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
        keep = keep & ~to_cys
    if mode == 'singles':
        keep = keep & (sp['n_mut'].to_numpy() == 1)
    fams = specs['fam_transfer'] if fams is None else fams
    rows = []
    for fam in fams:
        k = fam_key(fam, fr, specs, mode=mode)
        if k is None:
            continue
        ok = sp['use'].to_numpy() & keep & np.isfinite(fr.preds[k]).all(axis=0)
        if not ok.any():
            continue
        y = sp['y'].to_numpy()[ok]
        g = sp['group'].to_numpy()[ok]
        for K in ks:
            tabs = [scr_core.per_protein_table(g, p[ok], y, K, tau) for p in fr.preds[k]]
            res = scr_core.boot_summary(tabs, B=B, seed=K)
            for m, (v, lo, hi) in res.items():
                rows.append(dict(dataset=ds, family=fam, key=k, K=K, metric=m, value=v, lo=lo, hi=hi, n_proteins=len(tabs[0])))
    return pd.DataFrame(rows)


def render_domainome_screening_curves(dom_df_all: pd.DataFrame, specs: dict, B: int, out_dom: Path) -> None:
    """Render standalone 3-panel Domainome virtual screening curves (`domainome_screening_curves.{pdf,png}`)."""
    fams = [f for f in specs['fam_all'] if f in dom_df_all['family'].values]
    flab = specs['fam_label_singles']
    n_dom = int(dom_df_all['n_proteins'].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4))
    _draw_curve_panel(
        axes[0], dom_df_all, fams, flab, specs, False,
        'hit_rate', 'rand_hit', 'oracle_hit', 'Hit rate (abundance > wild type)',
    )
    _draw_curve_panel(
        axes[1], dom_df_all, fams, flab, specs, False,
        'p_recover', 'rand_rec', 'oracle_rec', 'P(at least one hit in top K)',
    )
    g = dom_df_all[dom_df_all.metric.isin(['gain', 'rand_gain'])].copy()
    for col_c in ('value', 'lo', 'hi'):
        g[col_c] = g[col_c] / g['K']
    _draw_curve_panel(
        axes[2], g, fams, flab, specs, False,
        'gain', 'rand_gain', None, 'Rectified gain per variant (abundance)',
    )
    axes[2].set_ylim(0, None)

    titles = [
        f'a   Stabilizing hit rate across {n_dom} held-out domains',
        f'b   Probability of ≥ 1 hit in top K ({n_dom} domains)',
        'c   Rectified abundance gain per screened variant',
    ]
    for ax, ttl in zip(axes, titles):
        ax.set_xlabel('Variants screened per domain (K)')
        ax.set_title(ttl, loc='left', fontsize=9.8, fontweight='bold')

    _overview_legend(fig, dom_df_all, fams, flab, specs, markers=False, oracle=True)
    note = (
        f'Domainome 1.0 held-out single-mutant libraries ({n_dom} human protein domains after 4-part sequence/structure holdout filter). '
        f'Single-mutant model readouts evaluated across top-K budgets K ∈ {{1, 3, 5, 10, 20, 50, 100}}; hit = abundance above wild type (> 0). '
        f'Shaded band: 95% joint domain × triplicate-seed bootstrap interval (B = {B}) for ESM-MSR.'
    )
    _finish_overview(fig, [(ax, ttl) for ax, ttl in zip(axes, titles)], note, out_dom / 'domainome_screening_curves')


def run_08_screening(split: str, tau: float, frac: float, B: int, only_domainome: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    specs = build_family_specs(split)
    out_mega = results_dir('megascale_screening', split)
    tsu_test = tsu_datasets(split)[1]
    tables_out: dict[str, pd.DataFrame] = {}

    if split == 'mega':
        for mode, stem in (('singles', 'screening_per_protein_singles'), ('composite', 'screening_per_protein_composite')):
            for fams, sfx in ((specs['fam_transfer'], ''), (specs['fam_all'], '_all_models')):
                D = per_protein_curves(tsu_test, KS_CURVE, tau, B, specs, mode=mode, exclude_cys=True, fams=fams)
                atomic_to_csv(D, out_mega / f'{stem}{sfx}.csv', index=False, float_format='%.4f')
                tables_out[f'{stem}{sfx}'] = D
        return tables_out, specs

    if not only_domainome:
        run_pooled_external(specs, tau, B, frac, results_dir('thermostability_screening'))

    out_dom = results_dir('domainome_screening')
    dom_exclude = excluded('domainome')
    dom_df = per_protein_curves('domainome', KS_CURVE, TAU_DS['domainome'], B, specs, exclude_groups=dom_exclude, mode='default', fams=specs['fam_transfer'])
    dom_df_all = per_protein_curves('domainome', KS_CURVE, TAU_DS['domainome'], B, specs, exclude_groups=dom_exclude, mode='default', fams=specs['fam_all'])
    atomic_to_csv(dom_df, out_dom / 'screening_per_protein.csv', index=False, float_format='%.4f')
    atomic_to_csv(dom_df_all, out_dom / 'screening_per_protein_all_models.csv', index=False, float_format='%.4f')
    render_domainome_screening_curves(dom_df_all, specs, B, out_dom)

    for mode, stem in (('singles', 'screening_per_protein_singles'), ('composite', 'screening_per_protein_composite')):
        if only_domainome and (out_mega / f'{stem}.csv').exists() and (out_mega / f'{stem}_all_models.csv').exists():
            tsu_df = pd.read_csv(out_mega / f'{stem}.csv')
            tsu_df_all = pd.read_csv(out_mega / f'{stem}_all_models.csv')
        else:
            tsu_df = per_protein_curves(tsu_test, KS_CURVE, tau, B, specs, mode=mode, exclude_cys=True, fams=specs['fam_transfer'])
            tsu_df_all = per_protein_curves(tsu_test, KS_CURVE, tau, B, specs, mode=mode, exclude_cys=True, fams=specs['fam_all'])
            atomic_to_csv(tsu_df, out_mega / f'{stem}.csv', index=False, float_format='%.4f')
            atomic_to_csv(tsu_df_all, out_mega / f'{stem}_all_models.csv', index=False, float_format='%.4f')
        tables_out[stem] = pd.concat([tsu_df, dom_df], ignore_index=True)
        tables_out[f'{stem}_all_models'] = pd.concat([tsu_df_all, dom_df_all], ignore_index=True)

    return tables_out, specs


# ----------------------------------------------------------------------------------------------
# Part 2: 15_tsuboyama_single_screening (suggested hit rate, top-6 recovery, best-ddG vs K)
# ----------------------------------------------------------------------------------------------
def _logc(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


def _hypergeom_p_atleast1(n: int, n_hits: int, k: int) -> float:
    k = min(k, n)
    if n_hits <= 0 or k <= 0:
        return 0.0
    if n - n_hits < k:
        return 1.0
    return float(1.0 - np.exp(_logc(n - n_hits, k) - _logc(n, k)))


def _exact_random_best(y_sorted: np.ndarray, k: int) -> float:
    n = len(y_sorted)
    k = min(k, n)
    j = np.arange(1, n + 1)
    mask = j >= k
    logp = np.full(n, -np.inf)
    logp[mask] = _logc(j[mask] - 1, k - 1) - _logc(n, k)
    return float(np.sum(np.exp(logp) * y_sorted))


def _single_columns(fr: Frame, best: str | None) -> list[dict]:
    cols = []
    for name in M.ZEROSHOT + M.TRANSFER + M.SUPERVISED:
        if name == 'GeoDDG-3D':
            if not best:
                continue
            name = BEST_LABEL[best]
        m = M.BY_NAME[name]
        cols.append(dict(
            name=name,
            cat='supervised' if m.group == 'megascale' else m.group,
            color=m.color,
            ls=m.ls,
            label=M.tick_label(name, 0) if m.two_versions else name,
            key=m.single,
            draw=True,
        ))
    return [c for c in cols if c['key'] in fr.preds]


def _draw_cat_spans(ax: plt.Axes, xs: list[float], spans: dict, headers: bool = True, cats: dict | None = None,
                    xpad: tuple[float, float] = (0.7, 0.75), gap: float = 0.7, header_fs: float = 8.8) -> None:
    cats = cats or style.GROUP_TITLE
    if not xs:
        return
    ax.set_xlim(xs[0] - xpad[0], xs[-1] + xpad[1])
    for i, (cat, (x0, x1)) in enumerate(spans.items()):
        if i % 2 or cat == 'oracle':
            ax.axvspan(x0 - 0.5 - (0.35 if i else 0), x1 + 0.5 + 0.35, color='#000000', alpha=0.045, lw=0, zorder=0)
        if headers and cat in cats:
            tr_ax = ax.get_xaxis_transform()
            ax.plot([x0 - 0.45, x1 + 0.45], [1.012, 1.012], transform=tr_ax, clip_on=False, color='#444444', lw=1.1)
            ax.text((x0 + x1) / 2, 1.02, cats[cat], transform=tr_ax, ha='center', va='bottom', fontsize=header_fs, fontweight='bold', color='#333333')


def _bars_single(ax: plt.Axes, cols: list[dict], T: pd.DataFrame, subset: str, bar_label: str, ylabel: str,
                 ylim: tuple[float, float], ref_lines=(), headers: bool = False, fmt: str = '{:.1f}') -> None:
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = style.layout_grouped_columns(cols, gap=0.7)
    w = 0.42
    top = 0.012 * (ylim[1] - ylim[0])
    for c, x in zip(cols, xs):
        r = T[(T.column == c['name']) & (T.subset == subset)]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.bar(x, r.value, w, color=c['color'], alpha=0.90, edgecolor='black', lw=0.8, zorder=3)
        ax.errorbar(x, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]],
                    fmt='none', ecolor=style.RING, elinewidth=1.3, capsize=2.5, zorder=5)
        ax.text(x, max(r.hi, r.value, 0) + top, fmt.format(r.value), ha='center', va='bottom',
                fontsize=6.8, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.6))
    ax.set_xticks(xs)
    ax.set_xticklabels([c['label'] for c in cols], fontsize=8.0, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    _draw_cat_spans(ax, xs, spans, headers=headers)
    handles = [Patch(fc='#999999', ec='black', alpha=0.90, label=bar_label)] + list(ref_lines)
    ax.legend(handles=handles, ncol=len(handles), loc='upper right', fontsize=8.3, framealpha=0.92)


def _bars_triple(ax: plt.Axes, cols: list[dict], T: pd.DataFrame, subsets: list[str], sub_labels: list[str],
                 ylabel: str, ylim: tuple[float, float], ref_lines=(), headers: bool = True) -> None:
    cols = [c for c in cols if c.get('draw', True)]
    xs, spans = style.layout_grouped_columns(cols, gap=0.7)
    w = 0.22
    centres = (-0.24, 0.0, 0.24)
    shades = (0.38, 0.68, 1.0)
    hatches = ('////', None, None)
    top = 0.012 * (ylim[1] - ylim[0])
    for c, x in zip(cols, xs):
        for j, (sub, off, sh, ht) in enumerate(zip(subsets, centres, shades, hatches)):
            r = T[(T.column == c['name']) & (T.subset == sub)]
            if not len(r):
                continue
            r = r.iloc[0]
            xx = x + off
            ax.bar(xx, r.value, w, color=c['color'], alpha=sh, edgecolor=c['color'] if j == 0 else 'black',
                   hatch=ht, lw=0.8, zorder=3)
            ax.errorbar(xx, r.value, yerr=[[max(r.value - r.lo, 0)], [max(r.hi - r.value, 0)]],
                        fmt='none', ecolor=style.RING, elinewidth=1.2, capsize=2.2, zorder=5)
            ax.text(xx, max(r.hi, r.value, 0) + top, f'{r.value:.2f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=90, zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([c['label'] for c in cols], fontsize=8.0, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in np.arange(0.0, 1.0001, 0.2)])
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    _draw_cat_spans(ax, xs, spans, headers=headers)
    handles = [
        Patch(fc='#888888', ec='#666666' if j == 0 else 'black', alpha=sh, hatch=ht, label=lab)
        for j, (sh, ht, lab) in enumerate(zip(shades, hatches, sub_labels))
    ] + list(ref_lines)
    ax.legend(handles=handles, ncol=3, loc='upper right', fontsize=8.3, framealpha=0.92)


def _compute_suggested_threshold(fr: Frame, cols: list[dict], pool_mask: np.ndarray, libs: list[str], counts: np.ndarray,
                                 tau_pred: float = 0.5, tau_hit: float = 0.5):
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()
    n_libs = len(libs)
    dom_rows, pool_rows = [], []
    dom_vals, pool_reps = {}, {}

    base_hit_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > tau_hit).mean()) for lib in libs})
    base_stab_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > 0.0).mean()) for lib in libs})
    true_hits_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > tau_hit).sum()) for lib in libs})
    true_stab_dom = pd.Series({lib: float((y[pool_mask & (g == lib)] > 0.0).sum()) for lib in libs})

    for c in cols:
        arr = fr.preds[c['key']]
        n_seeds = arr.shape[0]
        n_sug_s = np.zeros((n_seeds, n_libs))
        n_hit_s = np.zeros((n_seeds, n_libs))
        n_stab_s = np.zeros((n_seeds, n_libs))
        n_true_hit_s = np.zeros((n_seeds, n_libs))
        n_true_stab_s = np.zeros((n_seeds, n_libs))

        for s in range(n_seeds):
            p = arr[s]
            for j, lib in enumerate(libs):
                m = pool_mask & (g == lib) & np.isfinite(p) & np.isfinite(y)
                sug = m & (p > tau_pred)
                n_sug_s[s, j] = sug.sum()
                n_hit_s[s, j] = (sug & (y > tau_hit)).sum()
                n_stab_s[s, j] = (sug & (y > 0.0)).sum()
                n_true_hit_s[s, j] = (m & (y > tau_hit)).sum()
                n_true_stab_s[s, j] = (m & (y > 0.0)).sum()

        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            prec_hit_dom = pd.DataFrame(np.where(n_sug_s > 0, n_hit_s / n_sug_s, np.nan).T, index=libs)
            prec_stab_dom = pd.DataFrame(np.where(n_sug_s > 0, n_stab_s / n_sug_s, np.nan).T, index=libs)
            rec_hit_dom = pd.DataFrame(np.where(n_true_hit_s > 0, n_hit_s / n_true_hit_s, np.nan).T, index=libs)
            rec_stab_dom = pd.DataFrame(np.where(n_true_stab_s > 0, n_stab_s / n_true_stab_s, np.nan).T, index=libs)

        sug_count_dom = pd.DataFrame(n_sug_s.T, index=libs)
        hit_count_dom = pd.DataFrame(n_hit_s.T, index=libs)
        sug_total = float(sug_count_dom.mean(axis=1).sum())
        hit_total = float(hit_count_dom.mean(axis=1).sum())

        for sub, ser in (
            (f'precision_ddG_gt_{tau_hit:g}', prec_hit_dom),
            ('precision_ddG_gt_0', prec_stab_dom),
            (f'recall_ddG_gt_{tau_hit:g}', rec_hit_dom),
            ('recall_ddG_gt_0', rec_stab_dom),
            ('n_suggested_per_domain', sug_count_dom),
            ('n_hits_recovered_per_domain', hit_count_dom),
        ):
            pt, lo, hi, n_dom = boot_mean(ser, libs, counts)
            dom_vals[(c['name'], sub)] = (c['key'], ser, pool_mask, y)
            dom_rows.append(dict(
                column=c['name'], key=c['key'], subset=sub, value=pt, lo=lo, hi=hi,
                n_libraries=n_dom, n_suggested_total=sug_total, n_hits_total=hit_total,
            ))

        sug_w = counts @ n_sug_s.T
        hit_w = counts @ n_hit_s.T
        stab_w = counts @ n_stab_s.T
        thit_w = counts @ n_true_hit_s.T
        tstab_w = counts @ n_true_stab_s.T
        seed_w = (np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
                  if n_seeds > 1 else np.ones((len(counts), 1), float))
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter('ignore', RuntimeWarning)
            rep_prec_hit = np.nansum(np.where(sug_w > 0, hit_w / sug_w, np.nan) * seed_w, axis=1)
            rep_prec_stab = np.nansum(np.where(sug_w > 0, stab_w / sug_w, np.nan) * seed_w, axis=1)
            rep_rec_hit = np.nansum(np.where(thit_w > 0, hit_w / thit_w, np.nan) * seed_w, axis=1)
            rep_rec_stab = np.nansum(np.where(tstab_w > 0, stab_w / tstab_w, np.nan) * seed_w, axis=1)

        pt_prec_hit = float(np.nanmean(n_hit_s.sum(axis=1) / np.maximum(n_sug_s.sum(axis=1), 1)))
        pt_prec_stab = float(np.nanmean(n_stab_s.sum(axis=1) / np.maximum(n_sug_s.sum(axis=1), 1)))
        pt_rec_hit = float(np.nanmean(n_hit_s.sum(axis=1) / np.maximum(n_true_hit_s.sum(axis=1), 1)))
        pt_rec_stab = float(np.nanmean(n_stab_s.sum(axis=1) / np.maximum(n_true_stab_s.sum(axis=1), 1)))

        for sub, pt, rep in (
            (f'precision_ddG_gt_{tau_hit:g}', pt_prec_hit, rep_prec_hit),
            ('precision_ddG_gt_0', pt_prec_stab, rep_prec_stab),
            (f'recall_ddG_gt_{tau_hit:g}', pt_rec_hit, rep_rec_hit),
            ('recall_ddG_gt_0', pt_rec_stab, rep_rec_stab),
        ):
            lo, hi = ci(rep)
            pool_reps[(sub, c['name'])] = rep
            pool_rows.append(dict(
                column=c['name'], key=c['key'], subset=sub, value=pt, lo=lo, hi=hi,
                n_libraries=int((sug_count_dom.mean(axis=1) > 0).sum() if sub.startswith('precision') else n_libs),
                n_suggested_total=sug_total, n_hits_total=hit_total,
            ))

    return (pd.DataFrame(dom_rows), pd.DataFrame(pool_rows), dom_vals, pool_reps,
            base_hit_dom, base_stab_dom, true_hits_dom, true_stab_dom)


def _compute_topn_recovery(fr: Frame, cols: list[dict], pool_mask: np.ndarray, libs: list[str], counts: np.ndarray,
                           top_n: int = 6, budgets: tuple[int, ...] = (6, 24, 96)):
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()

    dom_info = {}
    for lib in libs:
        idx = np.flatnonzero(pool_mask & (g == lib) & np.isfinite(y))
        yd = y[idx]
        topn_local = set(np.argsort(-yd, kind='stable')[:top_n])
        dom_info[lib] = (idx, yd, topn_local, len(yd))

    frac_rows, any1_rows = [], []
    frac_vals, any1_vals = {}, {}
    rand_frac, rand_any1 = {}, {}

    for K in budgets:
        b_label = f'K = {K}'
        rf = pd.Series({lib: min(K, dom_info[lib][3]) / float(dom_info[lib][3]) for lib in libs})
        ra = pd.Series({lib: _hypergeom_p_atleast1(dom_info[lib][3], top_n, K) for lib in libs})
        rand_frac[b_label] = float(rf.mean())
        rand_any1[b_label] = float(ra.mean())

        for c in cols:
            arr = fr.preds[c['key']]
            f_dom, a_dom = {}, {}
            for lib in libs:
                idx, _yd, topn_local, n_d = dom_info[lib]
                k_eff = min(K, n_d)
                f_seeds, a_seeds = [], []
                for p in arr:
                    pd_ = p[idx]
                    if not np.isfinite(pd_).all():
                        continue
                    picks = set(np.argsort(-pd_, kind='stable')[:k_eff])
                    ov = len(picks & topn_local)
                    f_seeds.append(ov / float(top_n))
                    a_seeds.append(float(ov >= 1))
                if f_seeds:
                    f_dom[lib] = f_seeds
                    a_dom[lib] = a_seeds

            ser_f = pd.DataFrame.from_dict(f_dom, orient='index', dtype=float)
            ser_a = pd.DataFrame.from_dict(a_dom, orient='index', dtype=float)
            pt_f, lo_f, hi_f, n_f = boot_mean(ser_f, libs, counts)
            pt_a, lo_a, hi_a, n_a = boot_mean(ser_a, libs, counts)

            frac_vals[(c['name'], b_label)] = (c['key'], ser_f, pool_mask, y)
            any1_vals[(c['name'], b_label)] = (c['key'], ser_a, pool_mask, y)

            frac_rows.append(dict(
                column=c['name'], key=c['key'], subset=b_label, budget=K, top_n=top_n,
                metric='fraction_of_topN_recovered', value=pt_f, lo=lo_f, hi=hi_f,
                random_expectation=rand_frac[b_label], n_libraries=n_f, n_variants=int(pool_mask.sum()),
            ))
            any1_rows.append(dict(
                column=c['name'], key=c['key'], subset=b_label, budget=K, top_n=top_n,
                metric='p_recover_at_least_1_of_topN', value=pt_a, lo=lo_a, hi=hi_a,
                random_expectation=rand_any1[b_label], n_libraries=n_a, n_variants=int(pool_mask.sum()),
            ))

    nth_vals = [np.sort(dom_info[lib][1])[-min(top_n, len(dom_info[lib][1]))] for lib in libs]
    return pd.DataFrame(frac_rows), pd.DataFrame(any1_rows), frac_vals, any1_vals, rand_frac, rand_any1, nth_vals


def _compute_best_ddg_curve(fr: Frame, cols: list[dict], pool_mask: np.ndarray, libs: list[str], counts: np.ndarray, ks: list[int]):
    sp = fr.spine
    y = sp['y'].to_numpy()
    g = sp['group'].to_numpy()

    dom_y = {}
    for lib in libs:
        idx = np.flatnonzero(pool_mask & (g == lib) & np.isfinite(y))
        dom_y[lib] = (idx, y[idx], np.sort(y[idx]))

    rows, vals = [], {}
    for K in ks:
        orc_ser = pd.Series({lib: float(dom_y[lib][2][-1]) for lib in libs})
        rnd_ser = pd.Series({lib: _exact_random_best(dom_y[lib][2], K) for lib in libs})
        for name, key, ser in (('Oracle (best in domain)', 'oracle', orc_ser), ('Random picks', 'random', rnd_ser)):
            pt, lo, hi, n_d = boot_mean(ser, libs, counts)
            rows.append(dict(column=name, key=key, K=K, value=pt, lo=lo, hi=hi, n_libraries=n_d))

    for c in cols:
        arr = fr.preds[c['key']]
        seed_sorted_y = {}
        for lib in libs:
            idx, yd, _ = dom_y[lib]
            s_list = []
            for p in arr:
                pd_ = p[idx]
                if np.isfinite(pd_).all():
                    order = np.argsort(-pd_, kind='stable')
                    s_list.append(np.maximum.accumulate(yd[order]))
            seed_sorted_y[lib] = np.array(s_list, dtype=float) if s_list else None

        for K in ks:
            b_dom = {}
            for lib in libs:
                curves = seed_sorted_y[lib]
                if curves is not None:
                    b_dom[lib] = curves[:, min(K, curves.shape[1]) - 1]
            ser = pd.DataFrame.from_dict(b_dom, orient='index', dtype=float)
            pt, lo, hi, n_d = boot_mean(ser, libs, counts)
            vals[(c['name'], f'K = {K}')] = (c['key'], ser, pool_mask, y)
            rows.append(dict(column=c['name'], key=c['key'], K=K, value=pt, lo=lo, hi=hi, n_libraries=n_d))

    return pd.DataFrame(rows), vals


def _plot_best_ddg_ax(ax: plt.Axes, D: pd.DataFrame, models_to_draw: list[dict], ks_ticks: list[int], title: str | None = None) -> None:
    ks = sorted(D['K'].unique())
    orc = D[D.column == 'Oracle (best in domain)'].sort_values('K')
    rnd = D[D.column == 'Random picks'].sort_values('K')
    ax.plot(orc.K, orc.value, color='#2c3e50', ls=':', lw=1.6, label='Oracle (best in domain)', zorder=2)
    ax.plot(rnd.K, rnd.value, color='#c0392b', ls='--', lw=1.4, label='Random picks (exact expectation)', zorder=2)
    ax.axhline(0.0, color='#888888', lw=0.7, ls=':', zorder=1)
    for b_k in (6, 24, 96):
        if b_k in ks:
            ax.axvline(b_k, color='#cccccc', lw=0.8, ls='--', zorder=1)
    for c in models_to_draw:
        sub = D[D.column == c['name']].sort_values('K')
        is_m = c['name'].startswith('ESM-MSR')
        lw = 2.2 if is_m else 1.6
        z = 6 if is_m else 4
        ax.plot(sub.K, sub.value, color=c['color'], ls=c.get('ls', '-'), lw=lw, label=c['name'], zorder=z)
        if is_m:
            ax.fill_between(sub.K, sub.lo, sub.hi, color=c['color'], alpha=0.16, lw=0, zorder=z - 1)
    ax.set_xscale('log')
    ax.set_xticks(ks_ticks)
    ax.set_xticklabels([str(k) for k in ks_ticks])
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Variants screened per protein (K)')
    ax.set_ylabel('Best measured ddG in top K (kcal/mol)')
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    if title:
        ax.set_title(title, loc='left', fontsize=9.8)


def run_15_single_screening(split: str, tau: float, B: int, top_n: int = 6, budgets: tuple[int, ...] = (6, 24, 96)) -> None:
    out = results_dir('megascale_screening', split)
    fr, _ = load_tsu_test_frame(split)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    to_cys = (sp['mut'].astype(str).str[-1] == 'C').to_numpy()
    single_nocys = single & ~to_cys
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    counts = make_domain_counts(libs, B=B, seed=5)
    best, _ = best_other(fr)
    cols = _single_columns(fr, best)
    drawn = [c for c in cols if c.get('draw', True)]
    s_title = split_title(split)
    n_libs = len(libs)

    for pool_mask, sfx, pool_title in (
        (single, '', 'single mutants'),
        (single_nocys, '_nocys', 'single mutants (excluding mutations to Cys)'),
    ):
        n_pool_total = int(pool_mask.sum())
        T_dom, T_pool, dom_vals, pool_reps, bh_dom, bs_dom, th_dom, ts_dom = _compute_suggested_threshold(
            fr, cols, pool_mask, libs, counts, tau_pred=tau, tau_hit=tau,
        )
        atomic_to_csv(T_dom, out / f'suggested_hit_rate{sfx}.csv', index=False, float_format='%.4f')
        atomic_to_csv(T_pool, out / f'suggested_hit_rate_pooled{sfx}.csv', index=False, float_format='%.4f')
        p_subs = [f'precision_ddG_gt_{tau:g}', 'precision_ddG_gt_0', f'recall_ddG_gt_{tau:g}', 'recall_ddG_gt_0']
        p_dom = compare_grouped(fr, dom_vals, p_subs + ['n_suggested_per_domain'], counts, libs, B_mut=B)
        atomic_to_csv(add_holm_by_group(pd.DataFrame(p_dom)), out / f'suggested_hit_rate{sfx}_pvalues.csv', index=False, float_format='%.4g')
        p_pool = compare_pooled(T_pool, pool_reps, p_subs)
        atomic_to_csv(add_holm_by_group(pd.DataFrame(p_pool)), out / f'suggested_hit_rate_pooled{sfx}_pvalues.csv', index=False, float_format='%.4g')

        # Draw suggested figures
        p_hit, p_stab = f'precision_ddG_gt_{tau:g}', 'precision_ddG_gt_0'
        r_hit, r_stab = f'recall_ddG_gt_{tau:g}', 'recall_ddG_gt_0'
        bh_d, bs_d = float(bh_dom.mean()), float(bs_dom.mean())
        tot_hits, tot_stab = int(th_dom.sum()), int(ts_dom.sum())
        n_dom_hits = int((th_dom > 0).sum())
        bh_p, bs_p = float(tot_hits / n_pool_total), float(tot_stab / n_pool_total)
        foot = [
            f'Suggested = {pool_title} with predicted ddG > {tau:g} kcal/mol (n = {n_pool_total:,} across {n_libs} test domains; '
            f'{tot_hits:,} have measured ddG > {tau:g} kcal/mol in {n_dom_hits} domains, {tot_stab:,} have ddG > 0 in {n_libs} domains).'
        ]
        for T_cur, tag, yl, bh, bs, desc in (
            (T_dom[T_dom.subset.isin([p_stab, p_hit])], '', 'Precision among suggested\n(mean over domains)', bh_d, bs_d, 'mean over domains with ≥ 1 suggested variant'),
            (T_pool[T_pool.subset.isin([p_stab, p_hit])], '_pooled', 'Precision among suggested\n(pooled over domains)', bh_p, bs_p, 'pooled over domains'),
        ):
            fig, ax = plt.subplots(figsize=(fig_width(drawn), 4.9))
            sub_labs = ('stabilizing: measured ddG > 0', f'exceeds threshold: measured ddG > {tau:g} kcal/mol')
            style.bar_panel(ax, drawn, T_cur, (p_stab, p_hit), sub_labs, yl, (0, 1.15))
            ax.axhline(bs, color='#555555', ls=':', lw=1.1, zorder=2)
            ax.axhline(bh, color='#c0392b', ls='--', lw=1.1, zorder=2)
            handles = [
                Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=sub_labs[0]),
                Patch(fc='#999999', ec='black', label=sub_labs[1]),
                Line2D([0], [0], color='#555555', ls=':', lw=1.1, label=f'base rate ddG > 0 ({bs:.2f})'),
                Line2D([0], [0], color='#c0392b', ls='--', lw=1.1, label=f'base rate ddG > {tau:g} ({bh:.3f})'),
            ]
            ax.legend(handles=handles, ncol=2, loc='upper right', fontsize=8.1, framealpha=0.92)
            finish_bar_fig(
                fig,
                f'{s_title} test, {pool_title}: fraction of suggested variants (predicted > {tau:g} kcal/mol) exceeding threshold ({desc}; B = {B})',
                foot,
            )
            style.save(fig, out / f'suggested_hit_rate{tag}{sfx}')

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width(drawn), 11.2), sharex=True)
        labs_p = ('precision @ ddG > 0 (stabilizing)', f'precision @ ddG > {tau:g} kcal/mol')
        style.bar_panel(ax1, drawn, T_dom[T_dom.subset.isin([p_stab, p_hit])], (p_stab, p_hit), labs_p,
                        'Precision among suggested\n(mean over domains)', (0, 1.15), headers=True)
        ax1.axhline(bs_d, color='#555555', ls=':', lw=1.1, zorder=2)
        ax1.axhline(bh_d, color='#c0392b', ls='--', lw=1.1, zorder=2)
        ax1.legend(handles=[
            Patch(fc='#999999', ec='#999999', alpha=0.55, hatch='////', label=labs_p[0]),
            Patch(fc='#999999', ec='black', label=labs_p[1]),
            Line2D([0], [0], color='#555555', ls=':', lw=1.1, label=f'base rate ddG > 0 ({bs_d:.2f})'),
            Line2D([0], [0], color='#c0392b', ls='--', lw=1.1, label=f'base rate ddG > {tau:g} ({bh_d:.3f})'),
        ], ncol=2, loc='upper right', fontsize=8.1, framealpha=0.92)

        labs_r = (f'recall of true ddG > 0 variants (pred > {tau:g})', f'recall of true ddG > {tau:g} variants (pred > {tau:g})')
        style.bar_panel(ax2, drawn, T_dom[T_dom.subset.isin([r_stab, r_hit])], (r_stab, r_hit), labs_r,
                        f'Recall at pred > {tau:g}\n(mean over domains)', (0, 0.78), headers=False)
        ax2.set_yticks(np.arange(0.0, 0.71, 0.2))

        T3 = T_dom[T_dom.subset == 'n_suggested_per_domain']
        ymax3 = float(T3[T3.column.isin([c['name'] for c in drawn])]['hi'].max()) * 1.25
        oracle_hits = float(th_dom.mean())
        _bars_single(
            ax3, drawn, T3, 'n_suggested_per_domain',
            f'suggested variants per domain (predicted ddG > {tau:g} kcal/mol)',
            'Suggested variants\nper domain (mean)', (0, ymax3),
            ref_lines=[Line2D([0], [0], color='#2c3e50', ls='--', lw=1.2,
                              label=f'true ddG > {tau:g} variants per domain ({oracle_hits:.1f}, total {tot_hits:,})')],
            headers=False, fmt='{:.1f}',
        )
        ax3.axhline(oracle_hits, color='#2c3e50', ls='--', lw=1.2, zorder=2)
        finish_bar_fig(
            fig,
            f'{s_title} test, {pool_title}: precision, recall, and count of variants suggested at predicted ddG > {tau:g} kcal/mol (B = {B})',
            foot,
        )
        style.save(fig, out / f'suggested_hit_rate_with_counts{sfx}')

        # Top-N recovery
        T_frac, T_any1, frac_vals, any1_vals, rand_frac, rand_any1, nth_vals = _compute_topn_recovery(
            fr, cols, pool_mask, libs, counts, top_n=top_n, budgets=budgets,
        )
        b_subs = [f'K = {K}' for K in budgets]
        b_tag = '_'.join(f'k{K}' for K in budgets)
        f_prefix = f'top{top_n}_recovery_{b_tag}{sfx}'
        atomic_to_csv(T_frac, out / f'{f_prefix}.csv', index=False, float_format='%.4f')
        atomic_to_csv(T_any1, out / f'top{top_n}_recovery_atleast1_{b_tag}{sfx}.csv', index=False, float_format='%.4f')
        atomic_to_csv(add_holm_by_group(pd.DataFrame(compare_grouped(fr, frac_vals, b_subs, counts, libs, B_mut=B))),
                      out / f'{f_prefix}_pvalues.csv', index=False, float_format='%.4g')
        atomic_to_csv(add_holm_by_group(pd.DataFrame(compare_grouped(fr, any1_vals, b_subs, counts, libs, B_mut=B))),
                      out / f'top{top_n}_recovery_atleast1_{b_tag}{sfx}_pvalues.csv', index=False, float_format='%.4g')

        sub_labs = [f'budget K = {K}' for K in budgets]
        ref_styles = [('#777777', ':'), ('#2c3e50', '-.'), ('#c0392b', '--')]
        foot_topn = [
            f'Top-{top_n} most stabilizing {pool_title} per domain across {n_libs} test domains '
            f'({top_n}th-best ddG per domain averages {np.mean(nth_vals):+.2f} kcal/mol, min {np.min(nth_vals):+.2f} kcal/mol).'
        ]
        ref_f = [Line2D([0], [0], color=ch, ls=lh, lw=1.1, label=f'random K = {K} ({rand_frac[b_s]:.3f})') for K, b_s, (ch, lh) in zip(budgets, b_subs, ref_styles)]
        ref_a = [Line2D([0], [0], color=ch, ls=lh, lw=1.1, label=f'random K = {K} ({rand_any1[b_s]:.3f})') for K, b_s, (ch, lh) in zip(budgets, b_subs, ref_styles)]

        for T_cur, ref_cur, rdict, yl, ylim, stem_cur in (
            (T_frac, ref_f, rand_frac, f'P(top-{top_n} mutant recovered)\n= fraction of top-{top_n} found per domain', (0, 1.15), f'top{top_n}_recovery_{b_tag}{sfx}'),
            (T_any1, ref_a, rand_any1, f'P(recover ≥ 1 of top-{top_n} per domain)', (0, 1.24), f'top{top_n}_recovery_atleast1_{b_tag}{sfx}'),
        ):
            fig, ax = plt.subplots(figsize=(fig_width(drawn), 5.0))
            _bars_triple(ax, drawn, T_cur, b_subs, sub_labs, yl, ylim, ref_lines=ref_cur, headers=True)
            for b_s, (ch, lh) in zip(b_subs, ref_styles):
                ax.axhline(rdict[b_s], color=ch, ls=lh, lw=1.1, zorder=2)
            finish_bar_fig(fig, f'{s_title} test, {pool_title}: top-{top_n} recovery under budgets K = {", ".join(map(str, budgets))} (B = {B})', foot_topn)
            style.save(fig, out / stem_cur)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width(drawn), 8.6), sharex=True)
        _bars_triple(ax1, drawn, T_frac, b_subs, sub_labs, f'Fraction of top-{top_n} recovered\nper domain', (0, 1.15), ref_lines=ref_f, headers=True)
        for b_s, (ch, lh) in zip(b_subs, ref_styles):
            ax1.axhline(rand_frac[b_s], color=ch, ls=lh, lw=1.1, zorder=2)
        _bars_triple(ax2, drawn, T_any1, b_subs, sub_labs, f'P(recover ≥ 1 of top-{top_n})\nper domain', (0, 1.25), ref_lines=ref_a, headers=False)
        for b_s, (ch, lh) in zip(b_subs, ref_styles):
            ax2.axhline(rand_any1[b_s], color=ch, ls=lh, lw=1.1, zorder=2)
        finish_bar_fig(fig, f'{s_title} test, {pool_title}: top-{top_n} recovery per protein (B = {B})', foot_topn)
        style.save(fig, out / f'top{top_n}_recovery_both_{b_tag}{sfx}')

        # Best ddG vs K (1..96)
        ks_all = list(range(1, max(KS_TICKS) + 1))
        D_all, best_vals = _compute_best_ddg_curve(fr, cols, pool_mask, libs, counts, ks=ks_all)
        D_mark = D_all[D_all.K.isin(KS_TICKS)].copy()
        atomic_to_csv(D_mark, out / f'best_ddg_vs_k{sfx}.csv', index=False, float_format='%.4f')
        atomic_to_csv(D_all, out / f'best_ddg_vs_k_full_1to96{sfx}.csv', index=False, float_format='%.4f')
        atomic_to_csv(add_holm_by_group(pd.DataFrame(compare_grouped(fr, best_vals, [f'K = {K}' for K in KS_TICKS], counts, libs, B_mut=B))),
                      out / f'best_ddg_vs_k{sfx}_pvalues.csv', index=False, float_format='%.4g')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
        retrained = [c for c in drawn if c['cat'] == 'transfer']
        others = [c for c in drawn if c['name'].startswith('ESM-MSR') or c['cat'] in ('zeroshot', 'supervised')]
        _plot_best_ddg_ax(ax1, D_all, retrained, KS_TICKS, title=f'A. Retrained on {s_title}')
        _plot_best_ddg_ax(ax2, D_all, others, KS_TICKS, title='B. ESM-MSR vs Zero-shot & Supervised (not Megascale-trained)')
        ax2.set_ylabel('')
        ax1.legend(loc='lower right', ncol=2, fontsize=7.6, framealpha=0.92)
        ax2.legend(loc='lower right', ncol=2, fontsize=7.6, framealpha=0.92)
        finish_bar_fig(fig, f'{s_title} test, {pool_title}: best measured ddG recovered among top K per protein (K = 1 .. 96; B = {B})', [])
        style.save(fig, out / f'best_ddg_vs_k_panels{sfx}')

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        _plot_best_ddg_ax(ax, D_all, drawn, KS_TICKS)
        ax.legend(loc='lower right', ncol=2, fontsize=7.4, framealpha=0.92)
        finish_bar_fig(fig, f'{s_title} test, {pool_title}: best measured ddG recovered among top K per protein (K = 1 .. 96; B = {B})', [])
        style.save(fig, out / f'best_ddg_vs_k{sfx}')


# ----------------------------------------------------------------------------------------------
# Part 3: 14_property_category_gain (uses style.raincloud with [lo, hi] 95% CI whiskers!)
# ----------------------------------------------------------------------------------------------
def _load_tsu_features(ids: list[str]) -> pd.DataFrame:
    cols = ['code', 'mut_type', 'rel_ASA', 'kdh_wt', 'kdh_mut', 'vol_wt', 'vol_mut', 'conservation', 'n_seqs']
    f = pd.read_csv(TSUBOYAMA_FEATS_CSV, usecols=cols, low_memory=False)
    f['uid'] = (f['code'].astype(str) + '_' + f['mut_type'].astype(str)).str.replace('|', '_', regex=False)
    f = f.drop_duplicates('uid').set_index('uid')
    f['d_hydrophobicity'] = f['kdh_mut'] - f['kdh_wt']
    f['d_volume'] = f['vol_mut'] - f['vol_wt']
    return f.reindex(ids)[['rel_ASA', 'd_hydrophobicity', 'd_volume', 'conservation', 'n_seqs']].reset_index(drop=True)


def _symlog_tr(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= 1.0, y, 1.0 + np.log10(np.maximum(y, 1.0)))


def _symlog_ticks(top: float):
    v = [0, 1] + [10 ** i for i in range(1, 5) if 10 ** i <= top * 1.0001]
    return _symlog_tr(v), [f'{x:g}' for x in v]


def _selection_cat(score: np.ndarray, y: np.ndarray, cat: np.ndarray, ncat: int, doms: list[np.ndarray], k: int, positive_only: bool):
    G = np.zeros((len(doms), ncat))
    N = np.zeros((len(doms), ncat))
    for d, idx in enumerate(doms):
        ii = idx[np.isfinite(score[idx])]
        o = ii[np.argsort(-score[ii], kind='stable')[:k]]
        rect = np.maximum(y[o], 0.0)
        keep = score[o] > 0 if positive_only else np.ones(len(o), bool)
        G[d] = np.bincount(cat[o][keep], weights=rect[keep], minlength=ncat)
        N[d] = np.bincount(cat[o], minlength=ncat)
    return G, N


def _model_cat_stats(fr: Frame, cols: list[dict], y: np.ndarray, cat: np.ndarray, ncat: int, doms: list[np.ndarray], k: int, positive_only: bool):
    g_orc, n_orc = _selection_cat(y, y, cat, ncat, doms, k, positive_only)
    out = {f'Top {k} per Domain': (g_orc[None, :, :], n_orc[None, :, :])}
    for c in cols:
        g = [_selection_cat(p, y, cat, ncat, doms, k, positive_only) for p in fr.preds[c['key']]]
        out[c['name']] = (np.array([a for a, _ in g], dtype=float), np.array([b for _, b in g], dtype=float))
    return out


def _total_cat_stats(y: np.ndarray, cat: np.ndarray, ncat: int, doms: list[np.ndarray]):
    G = np.zeros((len(doms), ncat))
    N = np.zeros((len(doms), ncat))
    for d, idx in enumerate(doms):
        G[d] = np.bincount(cat[idx], weights=np.maximum(y[idx], 0.0), minlength=ncat)
        N[d] = np.bincount(cat[idx], minlength=ncat)
    return G[None, :, :], N[None, :, :]


def _boot_mean_ci(V: np.ndarray, counts: np.ndarray):
    V3 = V[None, :, :] if V.ndim == 2 else V
    n_seeds = V3.shape[0]
    w_dom = counts / counts.sum(axis=1, keepdims=True)
    dom_reps = np.einsum('bd,sdc->bsc', w_dom, V3)
    if n_seeds == 1:
        reps = dom_reps[:, 0, :]
    else:
        seed_w = np.random.default_rng(17).multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=len(counts)).astype(float) / n_seeds
        reps = np.einsum('bs,bsc->bc', seed_w, dom_reps)
    lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
    return V3.mean(axis=(0, 1)), lo, hi


def _with_oracle_col(cols: list[dict], k: int) -> list[dict]:
    return [dict(name=f'Top {k} per Domain', cat='oracle', color='#2c3e50', label=f'Top {k}\nper domain', key='oracle')] + list(cols)


def _distribution_figure(path: Path, dcols: list[dict], stats_k: dict, total: tuple, counts: np.ndarray,
                         cat_labels: list[str], k: int, note: str = '', include_picks: bool = True) -> pd.DataFrame:
    ncat = len(cat_labels)
    mcols = _with_oracle_col(dcols, k)
    cat_gap = 0.28
    xpad_dist = (0.58, 0.58)
    xs, spans = style.layout_grouped_columns(mcols, gap=cat_gap)
    rng = np.random.default_rng(3)
    colors = PALETTE_CAT[:ncat]
    slot = 0.85
    cw = slot / ncat
    x_units = xs[-1] - xs[0] + sum(xpad_dist)
    fig_w = 1.48 * x_units
    row_specs = [('gain', f'Rectified gain @ k={k} (kcal/mol)')]
    if include_picks:
        row_specs.append(('picks', 'Predicted most stabilizing per domain'))
    n_rows = len(row_specs)
    fig_h = 10.2 if include_picks else 6.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    if include_picks:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.86], hspace=0.14, left=0.055, right=0.992, top=0.915, bottom=0.20)
        axs = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
        ax_pref = None
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.095], hspace=0.025, left=0.055, right=0.992, top=0.875, bottom=0.30)
        axs = [fig.add_subplot(gs[0, 0])]
        ax_pref = fig.add_subplot(gs[1, 0])

    cats_hdr = {'oracle': 'Oracle', **style.GROUP_TITLE}
    table = []
    fs_txt = 8.3 if ncat >= 4 else 9.2
    for r, (what, ylab_m) in enumerate(row_specs):
        idx = 0 if what == 'gain' else 1
        omean, _, _ = _boot_mean_ci(stats_k[mcols[0]['name']][idx], counts)
        tmean, tlo, thi = _boot_mean_ci(total[idx], counts)
        top = 0.0
        V_tot = total[idx]
        for c_idx in range(ncat):
            med_tot = float(np.mean(np.median(V_tot[:, :, c_idx], axis=1)))
            table.append(dict(statistic=what, model='Total', category=cat_labels[c_idx], mean=tmean[c_idx], lo=tlo[c_idx], hi=thi[c_idx], median=med_tot, n_domains=V_tot.shape[1]))

        ax_m = axs[r]
        for mc, x_col in zip(mcols, xs):
            m = mc['name']
            V_m = stats_k[m][idx]
            mean, lo, hi = _boot_mean_ci(V_m, counts)
            for c_idx in range(ncat):
                x0 = x_col + (c_idx - (ncat - 1) / 2) * cw
                # BUG FIX: style.raincloud renders BOTH the dark mean bar + [lo, hi] 95% CI whisker AND the darker-hued median bar!
                style.raincloud(
                    ax_m, x0, V_m[:, :, c_idx], colors[c_idx],
                    mean=float(mean[c_idx]), lo=float(lo[c_idx]), hi=float(hi[c_idx]),
                    width=cw * 0.96, rng=rng, transform=_symlog_tr, annotate=True, fs=fs_txt,
                )
                top = max(top, float(_symlog_tr(hi[c_idx])), float(_symlog_tr(V_m[:, :, c_idx]).max()))
                med_m = float(np.mean(np.median(V_m[:, :, c_idx], axis=1)))
                table.append(dict(statistic=what, model=m, category=cat_labels[c_idx], mean=mean[c_idx], lo=lo[c_idx], hi=hi[c_idx], median=med_m, n_domains=V_m.shape[1]))
        _draw_cat_spans(ax_m, xs, spans, headers=(r == 0), cats=cats_hdr, xpad=xpad_dist, gap=cat_gap, header_fs=13.5)

        tk, tl = _symlog_ticks(10 ** (top - 1) if top > 1 else 1)
        ax_m.set_yticks(tk)
        ax_m.set_yticklabels(tl, fontsize=12.5)
        ax_m.set_ylim(-0.18, top + 0.16)
        ax_m.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        y_orc = [float(_symlog_tr(omean[c_idx])) for c_idx in range(ncat)]
        for c_idx in range(ncat):
            near_prev = any(abs(y_orc[c_idx] - y_orc[j]) < 0.025 for j in range(c_idx))
            ls = (4.0, (4.0, 4.0)) if near_prev else (0.0, (4.0, 4.0))
            ax_m.axhline(y_orc[c_idx], color=colors[c_idx], ls=ls, lw=1.15, alpha=0.92, zorder=2)
        ax_m.set_ylabel(ylab_m, fontsize=13.0)

    if include_picks:
        axs[0].set_xticks([])
        bot_ax = axs[n_rows - 1]
    else:
        axs[0].set_xticks([])
        _draw_cat_spans(ax_pref, xs, spans, headers=False, cats=cats_hdr, xpad=xpad_dist, gap=cat_gap)
        ax_pref.set_ylim(0, 1)
        ax_pref.set_yticks([0.5])
        ax_pref.set_yticklabels(['Picks (%)'], fontsize=11.5, fontweight='semibold')
        ax_pref.tick_params(axis='y', length=0, pad=6)
        for sp_name in ('top', 'right', 'left', 'bottom'):
            ax_pref.spines[sp_name].set_visible(False)
        for mc, x_col in zip(mcols, xs):
            m = mc['name']
            p_mean, _, _ = _boot_mean_ci(stats_k[m][1], counts)
            p_sum = float(p_mean.sum())
            pcts = (100.0 * p_mean / p_sum) if p_sum > 0 else np.zeros(ncat)
            x_cur = x_col - slot / 2.0
            for c_idx in range(ncat):
                w_seg = slot * (pcts[c_idx] / 100.0)
                if w_seg <= 0:
                    continue
                ax_pref.add_patch(plt.Rectangle((x_cur, 0.08), w_seg, 0.84, fc=colors[c_idx], alpha=0.55, ec='white', lw=1.0, zorder=3))
                if pcts[c_idx] >= 5.0:
                    ax_pref.text(x_cur + w_seg / 2.0, 0.50, f'{pcts[c_idx]:.0f}', ha='center', va='center',
                                 fontsize=8.6 if pcts[c_idx] >= 12.0 else 7.4, fontweight='bold', color='#1a1a1a', zorder=5)
                x_cur += w_seg
            ax_pref.add_patch(plt.Rectangle((x_col - slot / 2.0, 0.08), slot, 0.84, fc='none', ec='#555555', lw=0.8, zorder=4))
        bot_ax = ax_pref

    bot_ax.set_xticks(xs)
    bot_ax.set_xticklabels([mc['label'] for mc in mcols], fontsize=11.5)
    fig.legend(
        handles=[Patch(fc=colors[c_idx], alpha=0.8, label=cat_labels[c_idx]) for c_idx in range(ncat)],
        loc='lower center', ncol=2 if ncat == 4 else ncat,
        bbox_to_anchor=(0.5, 0.042 if include_picks else 0.058), fontsize=13.0, frameon=False,
    )
    if note:
        fig.text(0.055, 0.008 if include_picks else 0.012, note, fontsize=9.8, va='bottom', ha='left', color='#444444')
    style.save(fig, path)
    return pd.DataFrame(table)


def _stacked_aa_figure(path: Path, dcols: list[dict], G: dict, N: dict, counts: np.ndarray, k: int, note: str = '') -> pd.DataFrame:
    mcols = _with_oracle_col(dcols, k)
    xs, spans = style.layout_grouped_columns(mcols, gap=0.7)
    x = np.array(xs, dtype=float)
    x_units = xs[-1] - xs[0] + 1.45
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.05 * x_units, 6.0))
    cats_hdr = {'oracle': 'Oracle', **style.GROUP_TITLE}
    rows = []
    for ax, D_cur, ylab in ((ax1, G, 'Rectified total gain (kcal/mol)'), (ax2, N, 'Mutation count')):
        tot = np.array([D_cur[mc['name']].mean(axis=0).sum() for mc in mcols])
        ci_bounds = {}
        for mc in mcols:
            m = mc['name']
            dom_tot = D_cur[m].sum(axis=2)
            n_s = dom_tot.shape[0]
            dom_reps = counts @ dom_tot.T
            if n_s == 1:
                reps = dom_reps[:, 0]
            else:
                seed_w = np.random.default_rng(17).multinomial(n_s, np.full(n_s, 1.0 / n_s), size=len(counts)).astype(float) / n_s
                reps = np.sum(dom_reps * seed_w, axis=1)
            ci_bounds[m] = np.quantile(reps, [0.025, 0.975])
        ymax = max(tot.max(), max(hi for _, hi in ci_bounds.values()))
        thr = 0.03 * tot.max()
        bottom = np.zeros(len(mcols))
        for j, aa in enumerate(AA_ORDER):
            v = np.array([D_cur[mc['name']][:, :, j].mean(axis=0).sum() for mc in mcols])
            ax.bar(x, v, bottom=bottom, color=AA_COLOR[aa], edgecolor='white', linewidth=0.5, width=0.68, label=aa, zorder=3)
            for i in np.flatnonzero(v > thr):
                ax.text(x[i], bottom[i] + v[i] / 2, aa, ha='center', va='center', fontsize=7.2, fontweight='bold', zorder=4)
            bottom += v
        for i, mc in enumerate(mcols):
            m = mc['name']
            lo, hi = ci_bounds[m]
            ax.errorbar(x[i], tot[i], yerr=[[max(tot[i] - lo, 0)], [max(hi - tot[i], 0)]],
                        fmt='none', ecolor=style.RING, elinewidth=1.3, capsize=2.8, zorder=6)
            ax.text(x[i], max(hi, tot[i]) + 0.012 * ymax, f'{tot[i]:.0f}', ha='center', va='bottom', fontsize=7.5,
                    zorder=8, bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
            rows.append(dict(
                panel='gain' if D_cur is G else 'picks', model=m, total=tot[i], lo=lo, hi=hi,
                **{aa: D_cur[m][:, :, j].mean(axis=0).sum() for j, aa in enumerate(AA_ORDER)},
            ))
        ax.set_ylabel(ylab)
        ax.set_xticks(xs)
        ax.set_xticklabels([mc['label'] for mc in mcols], fontsize=8.0, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_ylim(0, ymax * 1.12)
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        _draw_cat_spans(ax, xs, spans, headers=True, cats=cats_hdr)
    handles = [Patch(fc=AA_COLOR[aa], label=aa) for aa in AA_ORDER[::-1]]
    ax1.legend(handles=handles, loc='upper right', title='Amino acids', ncol=4, fontsize=7.8, handlelength=1.2, frameon=True, framealpha=0.9)
    if note:
        fig.text(0.012, 0.008, note, fontsize=7.5, va='bottom', ha='left', color='#444444')
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 0.94))
    style.save(fig, path)
    return pd.DataFrame(rows)


def run_14_property_gain(split: str, B: int) -> pd.DataFrame | None:
    if split != 'capped':
        return None
    if not TSUBOYAMA_FEATS_CSV.exists():
        print(f'[p09] Skipping property_gain: {TSUBOYAMA_FEATS_CSV} not found')
        return None

    out = results_dir('property_gain', split)
    fr, _ = load_tsu_test_frame(split)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    y = sp['y'].to_numpy()
    feat = _load_tsu_features(sp['id'].tolist())
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    group = sp['group'].to_numpy()
    best, _ = best_other(fr)
    cols = _single_columns(fr, best)
    s_title = split_title(split)
    counts = make_domain_counts(libs, B=B, seed=5)

    def domains(pool):
        return [np.flatnonzero(pool & (group == lib)) for lib in libs]

    # Stacked AA bars
    aa = sp['mut'].astype(str).str[-1].to_numpy()
    aa_idx = np.array([AA_ORDER.index(c) if c in AA_ORDER else -1 for c in aa])
    dom_aa = domains(single & (aa_idx >= 0))
    for k in (24, 96):
        st = _model_cat_stats(fr, cols, y, np.where(aa_idx >= 0, aa_idx, 0), len(AA_ORDER), dom_aa, k, positive_only=False)
        models_k = list(st)
        t = _stacked_aa_figure(
            out / f'gain_aa_stacked_k{k}', cols,
            {m: st[m][0] for m in models_k}, {m: st[m][1] for m in models_k},
            counts, k,
            note=f'Top {k} of each of the {len(libs)} {s_title} test domains (single mutants); gain = sum of max(ddG, 0) over the picks;\nwhiskers: 95% domain + seed bootstrap interval (B = {B}).',
        )
        atomic_to_csv(t, out / f'gain_aa_stacked_k{k}.csv', index=False, float_format='%.4f')

    # Property-category rainclouds
    asa, dh, dv, cons, nseq = (feat[c].to_numpy() for c in ('rel_ASA', 'd_hydrophobicity', 'd_volume', 'conservation', 'n_seqs'))
    splits_cfg = [
        dict(name='asa_hydrophobicity', ks=(24, 96), pool=single & np.isfinite(asa) & np.isfinite(dh), cat=2 * (dh > 0) + (asa > 0.5),
             labels=['Δ hydrophobicity ≤ 0 & relative ASA ≤ 0.5', 'Δ hydrophobicity ≤ 0 & relative ASA > 0.5',
                     'Δ hydrophobicity > 0 & relative ASA ≤ 0.5', 'Δ hydrophobicity > 0 & relative ASA > 0.5']),
        dict(name='volume', ks=(96,), pool=single & np.isfinite(dv), cat=(dv > -1).astype(int) + (dv > 1),
             labels=['Δ volume ≤ −1', '−1 < Δ volume ≤ 1', 'Δ volume > 1']),
        dict(name='conservation', ks=(96,), pool=single & np.isfinite(cons) & np.isfinite(nseq) & (nseq > 2), cat=(cons > 25).astype(int) + (cons > 75),
             labels=['% conservation ≤ 25', '25 < % conservation ≤ 75', '% conservation > 75']),
    ]
    rules = [
        ('', True, cols, 'the picks predicted stabilizing (> 0)'),
        ('_allpicks', False, cols, 'every pick, whatever its predicted sign'),
    ]
    cmp_rows = []
    asa_k96_allpicks: pd.DataFrame | None = None
    for s in splits_cfg:
        cat = np.nan_to_num(np.asarray(s['cat'], float)).astype(int)
        dom = [d for d in domains(s['pool']) if len(d)]
        cnt = np.random.default_rng(5).multinomial(len(dom), np.full(len(dom), 1.0 / len(dom)), size=B).astype(float)
        tot = _total_cat_stats(y, cat, len(s['labels']), dom)
        for k in s['ks']:
            for suffix, positive_only, dcols, rule in rules:
                st = _model_cat_stats(fr, dcols, y, cat, len(s['labels']), dom, k, positive_only=positive_only)
                note = (
                    f'Top {k} per {s_title} test domain ({len(dom)} domains with the feature, {int(s["pool"].sum()):,} single mutants); '
                    f'gain = sum of max(ddG, 0) over {rule};\n'
                    f'boxes: interquartile range with darker colored median line; dots: domains; dark bars with vertical 95% CI whiskers (B = {B}) & annotations: mean over domains; dashed lines: oracle mean.'
                )
                t = _distribution_figure(out / f'gain_{s["name"]}_k{k}{suffix}', dcols, st, tot, cnt, s['labels'], k, note=note, include_picks=True)
                atomic_to_csv(t, out / f'gain_{s["name"]}_k{k}{suffix}.csv', index=False, float_format='%.4f')
                _distribution_figure(out / f'gain_{s["name"]}_k{k}{suffix}_gainonly', dcols, st, tot, cnt, s['labels'], k, note=note, include_picks=False)
                if s['name'] == 'asa_hydrophobicity' and k == 96 and suffix == '_allpicks':
                    asa_k96_allpicks = t
                if s['name'] == 'asa_hydrophobicity':
                    for m, (G_m, N_m) in st.items():
                        mean, lo, hi = _boot_mean_ci(G_m.sum(axis=2, keepdims=True), cnt)
                        cmp_rows.append(dict(k=k, rule='notebook' if positive_only else 'all picks', model=m,
                                             mean_gain=float(mean[0]), lo=float(lo[0]), hi=float(hi[0]), picks_counted=float(N_m.sum(axis=2).mean())))

    # Rule comparison figure
    cmp_df = pd.DataFrame(cmp_rows)
    atomic_to_csv(cmp_df, out / 'gain_rule_comparison.csv', index=False, float_format='%.4f')
    ks_cmp = sorted(cmp_df['k'].unique())
    mcols_ref = _with_oracle_col(cols, ks_cmp[0])
    xs_r, _ = style.layout_grouped_columns(mcols_ref, gap=0.7)
    x_units = xs_r[-1] - xs_r[0] + 1.45
    fig, axs = plt.subplots(1, len(ks_cmp), figsize=(1.02 * x_units * len(ks_cmp), 5.0), sharey=False)
    cats_hdr = {'oracle': 'Oracle', **style.GROUP_TITLE}
    for ax, k in zip(np.atleast_1d(axs), ks_cmp):
        mcols = _with_oracle_col(cols, k)
        xs_k, spans_k = style.layout_grouped_columns(mcols, gap=0.7)
        x = np.array(xs_k, dtype=float)
        models_k = [mc['name'] for mc in mcols]
        sub = cmp_df[cmp_df['k'] == k]
        for off, rule, col, mk in ((-0.14, 'notebook', '#555555', 'o'), (0.14, 'all picks', '#c0392b', 'D')):
            r = sub[sub['rule'] == rule].set_index('model').reindex(models_k)
            ax.errorbar(
                x + off, r['mean_gain'],
                yerr=[r['mean_gain'] - r['lo'], r['hi'] - r['mean_gain']],
                fmt=mk, ms=5, color=col, capsize=2.2, lw=1.2, zorder=4,
                label={'notebook': 'picks predicted stabilizing (notebook rule)', 'all picks': 'every pick'}[rule],
            )
        ax.set_xticks(xs_k)
        ax.set_xticklabels([mc['label'] for mc in mcols], fontsize=8.0, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_ylabel(f'Rectified gain @ k={k}, per domain (kcal/mol)')
        ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_ylim(0, float(sub['hi'].max()) * 1.15)
        _draw_cat_spans(ax, xs_k, spans_k, headers=True, cats=cats_hdr)
    np.atleast_1d(axs)[0].legend(loc='upper right', fontsize=8.2, framealpha=0.92)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save(fig, out / 'gain_rule_comparison')
    return asa_k96_allpicks


# ----------------------------------------------------------------------------------------------
# Part 4: pg_activity (ProteinGym Activity wild-type drop / hurt-rate screening)
# ----------------------------------------------------------------------------------------------
def _cutoff_on_scale(cut: float, score: np.ndarray) -> float:
    return -cut if (cut < score.min() or cut > score.max()) and score.min() <= -cut <= score.max() else cut


def _plateau(score: np.ndarray, cut: float) -> tuple[float, float]:
    frac = (score > cut).mean()
    if frac < 0.05 or frac > 0.95:
        cut = float(np.median(score))
    up = score[score > cut]
    xs = np.linspace(up.min(), up.max(), 400)
    w = float(xs[np.argmax(gaussian_kde(up)(xs))])
    return w, float(1.4826 * np.median(np.abs(up - w)))


def _short_names(ids: list[str]) -> dict[str, str]:
    base = {d: f"{d.split('_')[0]}_{d.split('_')[1][:6]}" for d in ids}
    dup = pd.Series(base).value_counts()
    return {d: (b if dup[b] == 1 else f"{b.split('_')[0]} {d.split('_', 2)[2]}") for d, b in base.items()}


def _noise(t, plateau_sd: float) -> tuple[float, str]:
    if np.isfinite(t.manual_noise):
        return float(t.manual_noise), 'paper'
    if t.n_syn >= 20 and t.syn_distinct >= 0.5 * t.n_syn and t.syn_sd > 0:
        return float(t.syn_sd), 'synonymous rows'
    return plateau_sd, 'plateau'


def load_pg_activity_assays() -> tuple[Frame, pd.DataFrame]:
    pg_ref_dir = REFERENCE / 'pg_activity'
    pg_dir = ROOT / 'final_pgym_wt' / 'FINAL_p0' / 's1'
    esm3_s0 = ROOT / 'msr_predictions' / 'released_capped_w2a4_m16a16' / 'sigma0' / 'proteingym_wt_only'
    esm3_s0_fb = ROOT / 'pgym_results' / 'unmasked-esm-msr-small' / 'sigma0.0'
    rosetta_pg = PUBLIC / 'rosetta' / 'raw_narval' / 'proteingym_cartesian_preds_09_11.csv'

    ref = pd.read_csv(PGYM_REF_CSV)
    row_of = {d: i for i, d in enumerate(ref.DMS_id)}
    ref = ref[ref.coarse_selection_type == 'Activity'].set_index('DMS_id')
    ref = ref[~ref.index.isin(excluded('proteingym'))]

    ros = pd.read_csv(rosetta_pg).rename(columns={'Unnamed: 0': 'key', 'cartesian_ddg_dir': 'cart'})
    ros['i'] = ros.key.str.extract(r'^P(\d+)_')[0].astype(int)
    ros['pm'] = ros.key.str.split('_', n=1).str[1]
    ros_map = {i: g.drop_duplicates('pm').set_index('pm')['cart'] for i, g in ros.groupby('i')}

    wtref = pd.read_csv(pg_ref_dir / 'wt_reference.csv').set_index('DMS_id')
    aicda = pd.read_csv(pg_ref_dir / 'aicda_wt_by_position.csv').set_index('pos')
    seeds = (1, 2, 3)
    srcs = {
        'msrp0:wt': (lambda k: ROOT / 'final_pgym_wt' / 'FINAL_p0' / f's{k}', 'wt_lora_pred'),
        'me1': (lambda k: RETRAINED / 'mutate_everything' / 'predictions' / 'stage1' / f'seed{k}' / 'proteingym', 'pred_combined'),
        'tmpnn': (lambda k: RETRAINED / 'thermompnn' / 'predictions' / 'single_additive' / f'seed{k}' / 'proteingym', 'pred_single_additive'),
        'spurs': (lambda k: RETRAINED / 'spurs' / 'predictions' / f'seed{k}' / 'proteingym', 'SPURS_single_additive'),
        'pmpnn': (lambda k: PUBLIC / 'proteinmpnn' / 'proteingym' / {1: 'fast', 2: 'fast_seed1', 3: 'fast_seed2'}[k], 'mpnn_score'),
    }
    pub_cols = {'prosst': 'ProSST-2048', 'pmpnn_pub': 'ProteinMPNN', 'esm3_pub': 'ESM3'}
    esm3dg_pg_dir = PUBLIC / 'esm3dg' / 'proteingym'
    rows, preds, info = [], {**{k: {sd: [] for sd in seeds} for k in srcs}, **{k: [] for k in list(pub_cols) + ['rosetta', 'msrrel@0:wt', 'esm3dg']}}, []

    for dms, r in ref.iterrows():
        base_full = pd.read_csv(pg_dir / f'{dms}.csv', usecols=['mutant', 'DMS_score'])
        base_sing = base_full[~base_full['mutant'].str.contains(':')]
        sing_row_ids = base_sing.index
        base = base_sing.reset_index(drop=True)
        x = base['DMS_score'].to_numpy()
        cut = _cutoff_on_scale(r.DMS_binarization_cutoff, x)
        w_pl, sd_pl = _plateau(x, cut)
        t = wtref.loc[dms]
        use = np.ones(len(x), bool)
        if t.tier == 'C-position':
            pos = base['mutant'].str.extract(r'(\d+)')[0].astype(int)
            wt = pos.map(aicda['wt_score']).to_numpy(float)
            use = (pos.map(aicda['wt_count_selected']) >= 100).to_numpy()
            kind, wt_scalar = 'paper (per position)', np.nan
        elif t.tier == 'C-open':
            wt, kind, wt_scalar = np.full(len(x), w_pl), 'plateau proxy', w_pl
        else:
            wt = np.full(len(x), t.wt_value)
            kind, wt_scalar = {'A': 'explicit WT row', 'B': 'synonymous rows', 'C-paper': 'paper'}[t.tier], float(t.wt_value)
        sd, sd_src = _noise(t, sd_pl)
        var_all = float(x[use].var())
        sd_v5, sd_v10 = np.sqrt(0.05 * var_all), np.sqrt(0.10 * var_all)
        if t.n_null >= 10:
            fl, fl_src, n_fl, fl_sd = float(t.null_stop), 'stop codons', int(t.n_null), float(t.null_sd)
        elif np.isfinite(t.manual_null_value):
            fl, fl_src, n_fl, fl_sd = float(t.manual_null_value), 'paper', 0, np.nan
        else:
            fl, fl_src, n_fl, fl_sd = np.nan, '', 0, np.nan
        span = wt - fl
        floor_ok = bool(np.isfinite(fl) and (span[use] > 0).all() and kind != 'plateau proxy')
        r_floor = (wt - x) / np.where(floor_ok, span, np.nan)
        span_med = float(np.median(span[use])) if np.isfinite(fl) else np.nan
        window = span_med / np.hypot(sd, 0 if np.isnan(fl_sd) else fl_sd) if np.isfinite(fl) else np.nan
        z, z_v5, z_v10, z_pl, z_old = (x - wt) / sd, (x - wt) / sd_v5, (x - wt) / sd_v10, (x - wt) / sd_pl, (x - w_pl) / sd_pl
        z_sd = (x - wt) / np.sqrt(var_all)
        cls = 'enzyme' if dms in PG_ENZYME else ('fluorescence' if dms in PG_FLUOR else 'other')
        frac_fn = lambda m: float(m[use].mean()) if m is not None else np.nan  # noqa: E731
        info.append(dict(
            DMS_id=dms, cls=cls, n_singles=len(x), n_used=int(use.sum()), tier=t.tier, wt_kind=kind, wt=wt_scalar, sd=sd, sd_source=sd_src,
            var_all=var_all, assay_sd=np.sqrt(var_all), sd_var5=sd_v5, sd_var10=sd_v10, sd_share_of_variance=sd ** 2 / var_all,
            floor=fl, floor_source=fl_src, n_floor=n_fl, floor_sd=fl_sd, floor_ok=floor_ok, span=span_med, window_over_noise=window,
            floor_resolved=bool(floor_ok and window >= 3.0), frac_within_sd_of_min=float((x < x.min() + sd).mean()),
            loss5_over_sd=0.05 * span_med / sd if np.isfinite(span_med) else np.nan,
            plateau_w=w_pl, plateau_sd=sd_pl, plateau_minus_wt_in_sd=(w_pl - wt_scalar) / sd if (np.isfinite(wt_scalar) and kind != 'plateau proxy') else np.nan,
            pg_cutoff=cut, pg_cutoff_method=r.DMS_binarization_method,
            hurt_floor5=frac_fn(r_floor > 0.05) if floor_ok else np.nan, hurt_floor10=frac_fn(r_floor > 0.10) if floor_ok else np.nan,
            hurt_sd=frac_fn(z_sd < -0.5), hurt_var5=frac_fn(z_v5 < -2.0), hurt_var10=frac_fn(z_v10 < -2.0), hurt_ladder=frac_fn(z < -2.0),
            hurt_plateau_sd=frac_fn(z_pl < -2.0), hurt_old_proxy=float((z_old < -2.0).mean()),
        ))
        for k, (dirs, col) in srcs.items():
            for sd_i in seeds:
                s = pd.read_csv(dirs(sd_i) / f'{dms}.csv', usecols=['mutant', col]).drop_duplicates('mutant').set_index('mutant')[col]
                preds[k][sd_i].append(s.reindex(base['mutant']).to_numpy(dtype=float))
        preds['rosetta'].append(ros_map.get(row_of[dms], pd.Series(dtype=float)).reindex(base['mutant'].str[1:]).to_numpy(dtype=float))
        f0 = esm3_s0 / f'{dms}.csv' if (esm3_s0 / f'{dms}.csv').exists() else esm3_s0_fb / f'{dms}.csv'
        s0 = pd.read_csv(f0, usecols=['mutant', 'wt_lora_pred']).drop_duplicates('mutant').set_index('mutant')['wt_lora_pred']
        preds['msrrel@0:wt'].append(s0.reindex(base['mutant']).to_numpy(dtype=float))
        pub_df = pd.read_csv(PUBLIC / 'proteingym_published' / f'{dms}.csv', usecols=['mutant'] + list(pub_cols.values())).drop_duplicates('mutant').set_index('mutant')
        for k, col in pub_cols.items():
            preds[k].append(pub_df[col].reindex(base['mutant']).to_numpy(dtype=float))
        e3dg_p = esm3dg_pg_dir / f'{dms}.csv'
        if e3dg_p.exists():
            e3dg_s = pd.read_csv(e3dg_p, usecols=['row_id', 'pred']).drop_duplicates('row_id').set_index('row_id')['pred']
            preds['esm3dg'].append(e3dg_s.reindex(sing_row_ids).to_numpy(dtype=float))
        else:
            preds['esm3dg'].append(np.full(len(base), np.nan, dtype=float))
        rows.append(pd.DataFrame({
            'id': dms + '_' + base['mutant'], 'y': z, 'score': x, 'wt': wt, 'z': z, 'z_sd': z_sd, 'z_var5': z_v5, 'z_var10': z_v10,
            'z_plateau_sd': z_pl, 'z_old_proxy': z_old, 'r_floor': r_floor, 'use_wt': use, 'use_floor': use & floor_ok,
            'group': dms, 'mut': base['mutant'], 'n_mut': 1, 'use': True,
        }))

    spine = pd.concat(rows, ignore_index=True)
    info_df = pd.DataFrame(info).set_index('DMS_id')
    arr = {k: (np.stack([np.concatenate(v[sd_i]) for sd_i in seeds]) if k in srcs else np.concatenate(v)[None, :]) for k, v in preds.items()}
    return Frame('pg_activity', spine, arr), info_df


def _label_frame(fr: Frame, groups: list[str], label: str, sign: int, use_col: str = 'use_wt') -> Frame:
    sp = fr.spine
    sel = sp['group'].isin(groups).to_numpy()
    s = sp[sel].reset_index(drop=True)
    s = s.assign(y=sign * s[label], use=s[use_col])
    return Frame('pg_activity', s, {k: v[:, sel] for k, v in fr.preds.items()})


def run_pg_activity(specs: dict, B: int) -> dict[str, pd.DataFrame]:
    out = results_dir('proteingym_screening')
    fr, info = load_pg_activity_assays()
    atomic_to_csv(info, out / 'assay_definitions.csv', float_format='%.4f')

    pg_fam = specs['fam_transfer']
    pg_fam_all = [f for f in specs['fam_all'] if f != 'geoddg']
    n_enz = int(sum(d in PG_ENZYME for d in info.index))
    n_all = len(info)

    # Diagnostic figures: wt_vs_plateau & floor_audit
    d_wp = info[np.isfinite(info['plateau_minus_wt_in_sd'])].copy()
    d_wp = d_wp.sort_values(['cls', 'plateau_minus_wt_in_sd'], key=lambda c: c.map({'enzyme': 0, 'other': 1, 'fluorescence': 2}) if c.name == 'cls' else c)
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * len(d_wp) + 1.4))
    cls_cols = {'enzyme': '#1f4e79', 'other': '#000000', 'fluorescence': '#7f7f7f'}
    yv = np.arange(len(d_wp))[::-1]
    ax.barh(yv, d_wp['plateau_minus_wt_in_sd'], color=[cls_cols[c] for c in d_wp['cls']], height=0.7)
    ax.axvline(0, color='#333333', lw=0.8)
    for vv in (-1, 1):
        ax.axvline(vv, color='#999999', lw=0.6, ls=':')
    names = _short_names(list(info.index))
    ax.set_yticks(yv)
    ax.set_yticklabels([f"{names[i]}  (WT tier {t})" for i, t in zip(d_wp.index, d_wp['tier'])], fontsize=6.8)
    for tl, c in zip(ax.get_yticklabels(), d_wp['cls']):
        tl.set_color(cls_cols[c])
    ax.set_xlabel('(plateau estimate − measured WT) / sd')
    ax.set_title('How far the old WT estimate (plateau peak) was from the measured wild type', loc='left', fontsize=9.5)
    fig.tight_layout()
    style.save(fig, out / 'wt_vs_plateau')

    # Curves for enzymes (dynamic count: 16 held-out assays!) and all (34 held-out assays)
    act_tables: dict[str, pd.DataFrame] = {}
    sets = (
        ('enzymes', PG_ENZYME, f'ProteinGym Activity: enzymes ({n_enz} held-out assays), single mutants'),
        ('all', set(info.index), f'ProteinGym Activity: all {n_all} held-out assays, single mutants'),
    )
    math_note = (
        "per assay (single mutants; x = score, higher = more active; WT = measured wild-type score on the assay's own scale):\n"
        f"  hurts: x < WT - 0.5 x SD, where SD is the standard deviation of all single-mutant scores of the assay\n"
        f"curve: mean over assays of (variants that hurt among the K best-scored) / K; dashed: random picks; band: 95% bootstrap (B = {B}), ESM-MSR only; lower is better"
    )
    for name, gs, set_title in sets:
        g_list = [d for d in info.index if d in gs]
        lf = _label_frame(fr, g_list, 'z_sd', -1, 'use_wt')
        for fams, tag, markers in ((pg_fam, '', True), (pg_fam_all, '_all_models', False)):
            d = per_protein_curves('pg', KS_CURVE, 0.5, B, specs, mode='singles', fams=fams, fr=lf)
            d = d.assign(definition='sd', n_assays=len(g_list))
            atomic_to_csv(d, out / f'screening_{name}{tag}.csv', index=False, float_format='%.4f')
            act_tables[f'screening_{name}{tag}'] = d

            fig, ax = plt.subplots(figsize=(9.8, 5.7))
            for f in fams:
                x = d[(d.family == f) & (d.metric == 'hit_rate')].sort_values('K')
                if x.empty:
                    continue
                if markers:
                    k_ = specs['fam_style'][f]
                    sp_s = style.style_of(k_)
                    ax.plot(x.K, x.value, color=sp_s.color, lw=2.2 if f == 'msr' else 1.3, marker=sp_s.marker, ms=4.5,
                            label=specs['fam_label_singles'][f], zorder=4 if f == 'msr' else 3)
                    if f == 'msr':
                        ax.fill_between(x.K, x.lo, x.hi, color=sp_s.color, alpha=0.18, lw=0, zorder=1)
                else:
                    col, ls, lw, zo = specs['fam_color_all'][f]
                    ax.plot(x.K, x.value, color=col, ls=ls, lw=lw, label=specs['fam_label_singles'][f], zorder=zo)
                    if f == 'msr':
                        ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=0.15, lw=0, zorder=1)
            rr = d[(d.family == 'msr') & (d.metric == 'rand_hit')].sort_values('K')
            ax.plot(rr.K, rr.value, color='#c0392b', ls=(0, (3, 2)), lw=1.4, zorder=3)
            ax.set_xscale('log')
            ax.set_xlabel('variants screened per assay (K)')
            ax.set_ylabel('fraction of the K picks that hurt (lower is better)')
            ax.set_xticks(KS_CURVE)
            ax.set_xticklabels(KS_CURVE)
            ax.set_ylim(0, None)
            ax.set_title(f'hurts if x < WT − 0.5 assay SD\n({len(g_list)} held-out assays)', loc='left', fontsize=9.2)
            handles = []
            for f in fams:
                if f in d.family.values:
                    if markers:
                        sp_s = style.style_of(specs['fam_style'][f])
                        handles.append(Line2D([0], [0], color=sp_s.color, marker=sp_s.marker, lw=2.0, ms=5, label=specs['fam_label_singles'][f]))
                    else:
                        col, ls, lw, _ = specs['fam_color_all'][f]
                        handles.append(Line2D([0], [0], color=col, ls=ls, lw=lw + 0.4, label=specs['fam_label_singles'][f]))
            handles.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='random picks'))
            fig.suptitle(set_title, x=0.008, ha='left', fontsize=11, fontweight='bold', y=0.995)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.962), ncol=min(len(handles), 5),
                       fontsize=8.4, handlelength=3.2, frameon=True, facecolor='white', edgecolor='#cccccc')
            fig.text(0.008, 0.01, math_note, fontsize=7.0, family='monospace', color='#333333', va='bottom', ha='left')
            fig.tight_layout(rect=(0, 0.09, 1, 0.90))
            style.save(fig, out / f'screening_{name}{tag}')

    # Paired difference vs ESM-MSR across definitions
    defs_all = {
        'sd': ('z_sd', -1, 0.5, 'use_wt', lambda i, d: True),
        'var5': ('z_var5', -1, 2.0, 'use_wt', lambda i, d: True),
        'var10': ('z_var10', -1, 2.0, 'use_wt', lambda i, d: True),
        'floor': ('r_floor', +1, 0.05, 'use_floor', lambda i, d: bool(i.loc[d, 'floor_ok'])),
        'ladder': ('z', -1, 2.0, 'use_wt', lambda i, d: True),
    }
    p_rows = []
    for setname, gs in (('enzymes', PG_ENZYME), ('all', set(info.index))):
        for dkey, (lab, sign, tau_d, use_col, ok_fn) in defs_all.items():
            g_list = [d for d in info.index if d in gs and ok_fn(info, d)]
            lf = _label_frame(fr, g_list, lab, sign, use_col)
            sp = lf.spine
            for K in (10, 100):
                tabs = {}
                for fam in pg_fam_all:
                    k = fam_key(fam, lf, specs, mode='singles')
                    if k is None:
                        continue
                    m = sp['use'].to_numpy() & np.isfinite(lf.preds[k]).all(axis=0)
                    t = pd.concat([scr_core.per_protein_table(sp['group'].to_numpy()[m], pr[m], sp['y'].to_numpy()[m], K, tau_d) for pr in lf.preds[k]]).groupby('group').mean(numeric_only=True)
                    tabs[fam] = t['hit']
                ref_s = tabs['msr']
                for fam, t in tabs.items():
                    if fam == 'msr':
                        continue
                    dv = (t - ref_s).dropna().to_numpy()
                    boot = dv[np.random.default_rng(0).integers(0, len(dv), (max(B, 1000), len(dv)))].mean(axis=1)
                    p_rows.append(dict(set=setname, definition=dkey, K=K, family=fam, n_assays=len(dv),
                                       diff=float(dv.mean()), lo=float(np.quantile(boot, 0.025)), hi=float(np.quantile(boot, 0.975))))
    atomic_to_csv(pd.DataFrame(p_rows), out / 'paired_vs_esm_msr.csv', index=False, float_format='%.4f')
    return act_tables


# ----------------------------------------------------------------------------------------------
# Part 5: 16_screening_overview (2x3 overview + rectified gain per variant)
# ----------------------------------------------------------------------------------------------
def _draw_curve_panel(ax: plt.Axes, d: pd.DataFrame, fams: list[str], flab: dict[str, str], specs: dict,
                      markers: bool, metric: str, ref: str, orc: str | None, ylabel: str) -> None:
    for f in fams:
        x = d[(d.family == f) & (d.metric == metric)].sort_values('K')
        if x.empty:
            continue
        if markers:
            sp_s = style.style_of(specs['fam_style'][f])
            ax.plot(x.K, x.value, color=sp_s.color, lw=2.2 if f == 'msr' else 1.3, marker=sp_s.marker, ms=4.5,
                    label=flab[f], zorder=4 if f == 'msr' else 3)
            if f == 'msr':
                ax.fill_between(x.K, x.lo, x.hi, color=sp_s.color, alpha=0.18, lw=0, zorder=1)
        else:
            col, ls, lw, zo = specs['fam_color_all'][f]
            ax.plot(x.K, x.value, color=col, ls=ls, lw=lw, label=flab[f], zorder=zo)
            if f == 'msr':
                ax.fill_between(x.K, x.lo, x.hi, color=col, alpha=0.15, lw=0, zorder=1)
    rr = d[(d.family == 'msr') & (d.metric == ref)].sort_values('K')
    ax.plot(rr.K, rr.value, color='#c0392b', ls=(0, (3, 2)), lw=1.4, zorder=3)
    if orc:
        oo = d[(d.family == 'msr') & (d.metric == orc)].sort_values('K')
        ax.plot(oo.K, oo.value, color='#2c3e50', ls=':', lw=1.7, zorder=7)
    ax.set_xscale('log')
    ks = sorted(d.K.unique())
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)


def _overview_legend(fig, d: pd.DataFrame, fams: list[str], flab: dict[str, str], specs: dict, markers: bool, oracle: bool = True):
    handles = []
    for f in fams:
        if f in d.family.values:
            if markers:
                sp_s = style.style_of(specs['fam_style'][f])
                handles.append(Line2D([0], [0], color=sp_s.color, marker=sp_s.marker, lw=2.0, ms=5, label=flab[f]))
            else:
                col, ls, lw, _ = specs['fam_color_all'][f]
                handles.append(Line2D([0], [0], color=col, ls=ls, lw=lw + 0.4, label=flab[f]))
    if oracle:
        handles.append(Line2D([0], [0], color='#2c3e50', ls=':', lw=2.0, label='oracle'))
    handles.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='random picks'))
    return fig.legend(
        handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995),
        ncol=6 if len(handles) > 8 else 5, fontsize=8.8, handlelength=3.4,
        frameon=True, facecolor='white', edgecolor='#cccccc',
    )


def _finish_overview(fig, ax_titles: list[tuple[plt.Axes, str]], note: str, path: Path) -> None:
    r = fig.canvas.get_renderer()
    kw = dict(fontsize=7.2, family='monospace', color='#333333')
    t = fig.text(0.012, 0.01, style.wrap_to_px(fig, note, 0.955 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
    foot = t.get_window_extent(r).height / fig.bbox.height
    top = 1 - fig.legends[0].get_window_extent(r).height / fig.bbox.height - 0.02
    for _ in range(3):
        fig.tight_layout(rect=(0.0, foot + 0.02, 1.0, top))
        for ax, raw in ax_titles:
            ax.set_title(style.wrap_to_px(fig, raw, 0.97 * ax.get_window_extent(r).width, fontsize=9.6, fontweight='bold'), loc='left', fontsize=9.6)
    style.save(fig, path)


def run_16_screening_overview(tables_08: dict[str, pd.DataFrame], act_tables: dict[str, pd.DataFrame], specs: dict, B: int) -> None:
    out = results_dir('screening_overview')
    act_note = (
        f'ProteinGym Activity column: single mutants only, scored by the single-mutant models; a variant hurts when its activity is '
        f'more than half an assay standard deviation below the measured wild type; lower is better.  Shaded band: 95% bootstrap interval (B = {B}) of ESM-MSR only.'
    )

    def title_of(ds: str, version: str, n: int) -> str:
        if ds == 'domainome':
            t = 'Domainome (held-out), single mutants only'
        elif version == 'single':
            t = 'Homology-Restricted MegaScale test set, single mutants only [excl. Cys]'
        else:
            t = 'Homology-Restricted MegaScale test set, singles + doubles [excl. Cys]: singles scored by the single-mutant model, doubles by the multi-mutant model'
        return f'{t} ({n} proteins)'

    for version, mtag in (('single', '_singles'), ('multi', '_composite')):
        flab = specs['fam_label_singles'] if version == 'single' else specs['fam_label_composite']
        for vtag, markers in (('', True), ('_all_models', False)):
            fams = specs['fam_transfer'] if markers else list(specs['fam_all'])
            D = tables_08[f'screening_per_protein{mtag}{vtag}']
            A_all = act_tables[f'screening_all{vtag}']
            A_enz = act_tables[f'screening_enzymes{vtag}']

            # 2x3 overview
            fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
            ax_titles = []
            for c, ds in enumerate(('tsuboyama_capped_test', 'domainome')):
                d = D[D.dataset == ds]
                _draw_curve_panel(axes[0, c], d, fams, flab, specs, markers, 'hit_rate', 'rand_hit', 'oracle_hit',
                                  'hit rate (abundance > wild type)' if ds == 'domainome' else 'hit rate (> 0.5 kcal/mol)')
                _draw_curve_panel(axes[1, c], d, fams, flab, specs, markers, 'p_recover', 'rand_rec', 'oracle_rec',
                                  'P(at least one hit in top K)')
                ax_titles.append((axes[0, c], title_of(ds, version, int(d.n_proteins.iloc[0]))))
                axes[1, c].set_xlabel('variants screened per protein (K)')
            for r_i, (A_cur, nm) in enumerate(((A_all, 'all held-out assays'), (A_enz, 'held-out enzymes'))):
                _draw_curve_panel(axes[r_i, 2], A_cur, fams, flab, specs, markers, 'hit_rate', 'rand_hit', None,
                                  'fraction of the K picks\nthat hurt (lower is better)')
                ax_titles.append((axes[r_i, 2], f'ProteinGym Activity, {nm}, single mutants ({int(A_cur.n_assays.iloc[0])} assays)'))
                axes[r_i, 2].set_ylim(0, None)
            axes[1, 2].set_xlabel('variants screened per assay (K)')
            _overview_legend(fig, D, fams, flab, specs, markers)
            note = act_note + ('' if version == 'single' else '  The legend names the models of the Tsuboyama panels (single | multi).')
            _finish_overview(fig, ax_titles, note, out / f'screening_overview_{version}{vtag}')

            # Gain per variant
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))
            ax_titles = []
            for ax, ds in zip(axes, ('tsuboyama_capped_test', 'domainome')):
                d = D[D.dataset == ds]
                g = d[d.metric.isin(['gain', 'rand_gain', 'oracle_gain'])].copy()
                for col_c in ('value', 'lo', 'hi'):
                    g[col_c] = g[col_c] / g['K']
                _draw_curve_panel(ax, g, fams, flab, specs, markers, 'gain', 'rand_gain',
                                  None if ds == 'domainome' else 'oracle_gain',
                                  'rectified gain per variant (abundance)' if ds == 'domainome' else 'rectified gain per variant (kcal/mol)')
                ax.set_ylim(0, None)
                ax.set_xlabel('variants screened per protein (K)')
                ax_titles.append((ax, title_of(ds, version, int(d.n_proteins.iloc[0]))))
            _overview_legend(fig, D, fams, flab, specs, markers)
            _finish_overview(
                fig, ax_titles,
                f'Rectified gain per variant = sum of max(y, 0) over the K best-scored variants of a protein divided by K, averaged over proteins; '
                f'shaded band: 95% bootstrap interval (B = {B}) of ESM-MSR only.',
                out / f'screening_gain_per_variant_{version}{vtag}',
            )


# ----------------------------------------------------------------------------------------------
# Part 6: 17_gain_per_variant_fit (joint [protein x seed] bootstrap + fixed multi footnote!)
# ----------------------------------------------------------------------------------------------
def _fit_line(lk: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(lk, g, 1)
    return float(intercept), float(-slope)


def _boot_line_joint(G: np.ndarray, lk: np.ndarray, B: int, seed: int = 0) -> np.ndarray:
    """Joint [protein × seed] bootstrap of ``g(K) = a - b ln K``.

    ``G`` has shape ``(n_seeds, n_proteins, len(KS))`` (fixing the legacy bug where seeds were averaged
    before bootstrapping).
    """
    G3 = G[None, :, :] if G.ndim == 2 else G
    n_seeds, n_prots, n_k = G3.shape
    rng_p = np.random.default_rng(seed)
    rng_s = np.random.default_rng(seed + 17)
    ab = np.empty((B, 2), dtype=float)
    fin = np.isfinite(G3)
    G_zero = np.where(fin, G3, 0.0)

    for b_idx in range(B):
        wp = rng_p.multinomial(n_prots, np.full(n_prots, 1.0 / n_prots)).astype(float)
        ws = (rng_s.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds)).astype(float) / n_seeds) if n_seeds > 1 else np.ones(1, dtype=float)
        # Weighted protein mean per seed, then seed-weighted mean -> (n_k,)
        num = np.einsum('s,p,spk->k', ws, wp, G_zero)
        den = np.einsum('s,p,spk->k', ws, wp, fin.astype(float))
        g_rep = num / np.maximum(den, 1e-12)
        ab[b_idx] = _fit_line(lk, g_rep)
    return ab


def _fit_shapes(ks: list[int], g: np.ndarray) -> dict:
    out = {}
    k_arr = np.asarray(ks, dtype=float)
    for name, (fn, p0) in SHAPES.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p, _ = curve_fit(fn, k_arr, g, p0=p0, maxfev=20000)
            r = g - fn(k_arr, *p)
            out[name] = (float(np.sqrt((r ** 2).mean())), float(np.abs(r).max()), p)
        except Exception:
            out[name] = (np.nan, np.nan, None)
    return out


def run_17_gain_fit(split: str, specs: dict, B: int) -> tuple[pd.DataFrame, dict]:
    out = results_dir('gain_fit', split)
    ds = tsu_datasets(split)[1]
    raw = load_frame(ds)
    fr = Frame(raw.name, raw.spine, dict(raw.preds), dict(raw.notes))
    sp = fr.spine
    to_cys = np.array([any(part.endswith('C') for part in str(m).split(':')) for m in sp['mut']], dtype=bool)
    lk = np.log(KS_CURVE)

    extra = [f for f in specs['fam_all'] if f in ('prosst', 'pmpnn', 'esm3', 'rosetta', 'spursu', 'tmpub')]
    fams_fit = list(specs['fam_transfer']) + [f for f in extra if f not in specs['fam_transfer']]
    main_fams = list(specs['fam_transfer'])
    s_title = split_title(split)

    def seed_tensor(fam: str, mode: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
        k = fam_key(fam, fr, specs, mode=mode)
        if k is None:
            return None, None
        keep = ~to_cys & ((sp['n_mut'].to_numpy() == 1) if mode == 'singles' else True)
        ok = sp['use'].to_numpy() & keep & np.isfinite(fr.preds[k]).all(axis=0)
        if not ok.any():
            return None, None
        y, g = sp['y'].to_numpy()[ok], sp['group'].to_numpy()[ok]
        n_seeds = fr.preds[k].shape[0]
        # Per-seed tables at each K
        per_seed_tabs = {
            K: [scr_core.per_protein_table(g, pr[ok], y, K, 0.5) for pr in fr.preds[k]]
            for K in KS_CURVE
        }
        prots = sorted(set().union(*[set(t['group']) for K in KS_CURVE for t in per_seed_tabs[K]]))
        G = np.full((n_seeds, len(prots), len(KS_CURVE)), np.nan)
        for j, K in enumerate(KS_CURVE):
            for s_i, tab in enumerate(per_seed_tabs[K]):
                G[s_i, :, j] = (tab.set_index('group')['gain'] / K).reindex(prots).to_numpy()
        return G, per_seed_tabs

    rows, srows, curves = [], [], {}
    for ver, mode in (('single', 'singles'), ('multi', 'composite')):
        cur: dict = {}
        nmed_v = np.nan
        for fam in fams_fit:
            G, per_seed_tabs = seed_tensor(fam, mode)
            if G is None:
                continue
            G_mean_s = np.nanmean(G, axis=0)  # (n_prots, n_k)
            fin = np.isfinite(G_mean_s)
            cur[fam] = np.array([G_mean_s[fin[:, j], j].mean() for j in range(len(KS_CURVE))])
            cur[fam + '_G'] = G
            if 'oracle' not in cur:
                prots = sorted(set().union(*[set(t['group']) for K in KS_CURVE for t in per_seed_tabs[K]]))
                for nm, col in (('oracle', 'oracle_gain'), ('random', 'rand_gain')):
                    Go = np.full((1, len(prots), len(KS_CURVE)), np.nan)
                    for j, K in enumerate(KS_CURVE):
                        Go[0, :, j] = (per_seed_tabs[K][0].set_index('group')[col] / K).reindex(prots).to_numpy()
                    fino = np.isfinite(Go[0])
                    cur[nm] = np.array([Go[0, fino[:, j], j].mean() for j in range(len(KS_CURVE))])
                    cur[nm + '_G'] = Go
                nmed_v = float(np.median(per_seed_tabs[1][0]['n']))

        for name, g_vec in [(n, v) for n, v in cur.items() if not n.endswith('_G')]:
            G_cur = cur[name + '_G']
            if name == 'random':
                rows.append(dict(
                    version=ver, model=name, a=float(g_vec.mean()), a_lo=np.nan, a_hi=np.nan,
                    b=0.0, b_lo=np.nan, b_hi=np.nan, rmse=float(g_vec.std()),
                    max_err=float(np.abs(g_vec - g_vec.mean()).max()), r2=np.nan,
                    n_proteins=int(G_cur.shape[1]), k0=np.nan, n_variants_median=nmed_v,
                ))
                continue
            ab = _boot_line_joint(G_cur, lk, B=B, seed=0)
            a, b = _fit_line(lk, g_vec)
            pred = a - b * lk
            ss_tot = float(((g_vec - g_vec.mean()) ** 2).sum())
            r2_val = float(1.0 - ((g_vec - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
            rows.append(dict(
                version=ver, model=name, a=a, a_lo=float(np.quantile(ab[:, 0], 0.025)), a_hi=float(np.quantile(ab[:, 0], 0.975)),
                b=b, b_lo=float(np.quantile(ab[:, 1], 0.025)), b_hi=float(np.quantile(ab[:, 1], 0.975)),
                rmse=float(np.sqrt(((g_vec - pred) ** 2).mean())), max_err=float(np.abs(g_vec - pred).max()),
                r2=r2_val, n_proteins=int(G_cur.shape[1]),
                k0=float(np.exp((a - float(cur['random'].mean())) / b)) if b > 0 else np.nan,
                n_variants_median=nmed_v,
            ))
            for sh, (rm, mx, p) in _fit_shapes(KS_CURVE, g_vec).items():
                srows.append(dict(version=ver, model=name, shape=sh, rmse=rm, max_err=mx,
                                  params=' '.join(f'{v:.4g}' for v in p) if p is not None else ''))
        curves[ver] = cur

    P = pd.DataFrame(rows)
    S = pd.DataFrame(srows)
    atomic_to_csv(P, out / 'gain_fit_params.csv', index=False, float_format='%.4f')
    atomic_to_csv(S, out / 'gain_fit_shapes.csv', index=False, float_format='%.4f')

    lbl_map = {**{f: n for f, n in specs['fam_name'].items()}, 'oracle': 'oracle (K largest labels)', 'random': 'random picks'}
    col_map = {**{f: M.color(n) if n in M.BY_NAME else specs['fam_color_all'][f][0] for f, n in specs['fam_name'].items()},
               'oracle': '#2c3e50', 'random': '#c0392b'}

    for ver in ('single', 'multi'):
        P_v = P[P.version == ver].set_index('model')
        cur = curves[ver]
        fig, (ax, rx) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
        xx = np.geomspace(1, 100, 100)
        for name in [f for f in main_fams if f in P_v.index] + ['oracle']:
            r = P_v.loc[name]
            col = col_map[name]
            ax.plot(KS_CURVE, cur[name], 'o', color=col, ms=5.5, zorder=4, mec='white', mew=0.8)
            ax.plot(xx, r.a - r.b * np.log(xx), color=col, lw=1.8 if name != 'oracle' else 1.4,
                    ls='-' if name != 'oracle' else ':', zorder=3,
                    label=f'{lbl_map[name]}:  {r.a:.2f} − {r.b:.3f} ln K   (rmse {r.rmse:.3f})')
            rx.plot(KS_CURVE, cur[name] - (r.a - r.b * np.log(KS_CURVE)), 'o-', color=col, ms=4.5, lw=1.1, label=lbl_map[name])
        ax.plot(KS_CURVE, cur['random'], 's', color=col_map['random'], ms=4)
        ax.axhline(P_v.loc['random', 'a'], color=col_map['random'], ls=(0, (3, 2)), lw=1.4,
                   label=f'random picks:  {P_v.loc["random", "a"]:.3f}')
        for ax_i in (ax, rx):
            ax_i.set_xscale('log')
            ax_i.set_xticks(KS_CURVE)
            ax_i.set_xticklabels(KS_CURVE)
            ax_i.set_xlabel('variants screened per protein (K)')
            ax_i.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax.set_ylabel('rectified gain per variant (kcal/mol)')
        ax.set_ylim(0, None)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.95)
        rx.axhline(0, color='#888888', lw=0.8)
        rx.set_ylabel('data − fit (kcal/mol)')
        rx.set_ylim(-0.06, 0.06)
        rx.legend(fontsize=8, loc='upper right', framealpha=0.95)
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))

        # BUG FIX: accurate pool description for 'multi' ('singles + doubles, Cys excluded' instead of 'single mutants, Cys excluded')!
        pool_desc = (
            'singles + doubles, Cys excluded; singles scored by the single-mutant model, doubles by the multi-mutant model'
            if ver == 'multi'
            else 'single mutants, Cys excluded; single-mutant models'
        )
        foot_txt = (
            f'Points: rectified gain per variant of the {s_title} test set ({pool_desc}), '
            f'the mean over proteins of (sum of max(ddG, 0) over the K best-scored variants) / K.  '
            f'Lines: least-squares line a − b ln K through the seven K (95% joint protein × seed bootstrap, B = {B}).  '
            f'ESM-MSR reaches the random level at K ≈ {P_v.loc["msr", "k0"]:.0f} (median library size {P_v.loc["msr", "n_variants_median"]:.0f}).'
        )
        kw = dict(fontsize=8.0, color='#333333')
        fig.text(0.012, 0.012, style.wrap_to_px(fig, foot_txt, 0.97 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)
        fig.suptitle('Rectified gain per variant falls linearly in ln K: g(K) = a − b ln K', x=0.012, ha='left', fontsize=12, fontweight='bold')
        style.save(fig, out / f'gain_fit_{ver}')

    return P, curves


# ----------------------------------------------------------------------------------------------
# Part 7: New High-Impact Composite Figure #3 — screening_gain_law_composite.{png,pdf}
# ----------------------------------------------------------------------------------------------
def _panel_label(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.09) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')


def build_screening_gain_law_composite(
    tables_08: dict[str, pd.DataFrame],
    act_tables: dict[str, pd.DataFrame],
    fit_params: pd.DataFrame,
    fit_curves: dict,
    asa_k96_df: pd.DataFrame | None,
    specs: dict,
    B: int,
) -> None:
    out = results_dir('supplementary_figs')
    fams = specs['fam_transfer']
    flab_m = specs['fam_label_composite']
    flab_s = specs['fam_label_singles']
    D_comp = tables_08['screening_per_protein_composite']
    A_enz = act_tables['screening_enzymes']
    n_enz = int(A_enz.n_assays.iloc[0])

    fig = plt.figure(figsize=(15.2, 12.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.12], hspace=0.40, wspace=0.46)

    # Row 1 (panels a, b, c): Screening hit rate & Activity preservation
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])

    d_tsu = D_comp[D_comp.dataset == 'tsuboyama_capped_test']
    d_dom = D_comp[D_comp.dataset == 'domainome']
    _draw_curve_panel(ax_a, d_tsu, fams, flab_m, specs, True, 'hit_rate', 'rand_hit', 'oracle_hit', 'Hit rate (ΔΔG > 0.5 kcal/mol)')
    ax_a.set_xlabel('Variants screened per protein (K)')
    ax_a.set_title(f'MegaScale test (singles+doubles, {int(d_tsu.n_proteins.iloc[0])} domains)', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_a, 'a')

    _draw_curve_panel(ax_b, d_dom, fams, flab_s, specs, True, 'hit_rate', 'rand_hit', 'oracle_hit', 'Hit rate (abundance > WT)')
    ax_b.set_xlabel('Variants screened per domain (K)')
    ax_b.set_title(f'Domainome held-out ({int(d_dom.n_proteins.iloc[0])} human domains)', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_b, 'b')

    _draw_curve_panel(ax_c, A_enz, fams, flab_s, specs, True, 'hit_rate', 'rand_hit', None, 'Deleterious pick rate (↓ better)')
    ax_c.set_ylim(0, None)
    ax_c.set_xlabel('Variants screened per enzyme (K)')
    ax_c.set_title(f'ProteinGym Activity ({n_enz} held-out enzymes)', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_c, 'c')

    # Shared top legend
    leg_handles = []
    for f in fams:
        sp_s = style.style_of(specs['fam_style'][f])
        leg_handles.append(Line2D([0], [0], color=sp_s.color, marker=sp_s.marker, lw=2.0, ms=5, label=specs['fam_name'][f]))
    leg_handles.append(Line2D([0], [0], color='#2c3e50', ls=':', lw=2.0, label='Oracle'))
    leg_handles.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='Random picks'))
    fig.legend(handles=leg_handles, loc='upper center', bbox_to_anchor=(0.5, 0.992), ncol=6, fontsize=8.8, frameon=True, facecolor='white', edgecolor='#cccccc')

    # Row 2 (panels d, e): Logarithmic diminishing-returns law g(K) = a - b ln K + residuals (multi)
    ax_d = fig.add_subplot(gs[1, 0:4])
    ax_e = fig.add_subplot(gs[1, 4:6])
    P_m = fit_params[fit_params.version == 'multi'].set_index('model')
    cur_m = fit_curves['multi']
    xx = np.geomspace(1, 100, 100)
    lbl_map = {**{f: n for f, n in specs['fam_name'].items()}, 'oracle': 'Oracle', 'random': 'Random picks'}
    col_map = {**{f: M.color(n) for f, n in specs['fam_name'].items() if n in M.BY_NAME}, 'oracle': '#2c3e50', 'random': '#c0392b'}
    for name in [f for f in fams if f in P_m.index] + ['oracle']:
        r = P_m.loc[name]
        col = col_map[name]
        ax_d.plot(KS_CURVE, cur_m[name], 'o', color=col, ms=5.5, zorder=4, mec='white', mew=0.8)
        ax_d.plot(xx, r.a - r.b * np.log(xx), color=col, lw=1.9 if name != 'oracle' else 1.4,
                  ls='-' if name != 'oracle' else ':', zorder=3,
                  label=f'{lbl_map[name]}: {r.a:.2f} − {r.b:.3f} ln K  (R²={r.r2:.3f})')
        ax_e.plot(KS_CURVE, cur_m[name] - (r.a - r.b * np.log(KS_CURVE)), 'o-', color=col, ms=4.5, lw=1.1, label=lbl_map[name])
    ax_d.plot(KS_CURVE, cur_m['random'], 's', color=col_map['random'], ms=4)
    ax_d.axhline(P_m.loc['random', 'a'], color=col_map['random'], ls=(0, (3, 2)), lw=1.4, label=f'Random: {P_m.loc["random", "a"]:.3f}')
    for ax_i in (ax_d, ax_e):
        ax_i.set_xscale('log')
        ax_i.set_xticks(KS_CURVE)
        ax_i.set_xticklabels(KS_CURVE)
        ax_i.set_xlabel('Variants screened per protein (K)')
        ax_i.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    ax_d.set_ylabel('Rectified gain per variant (kcal/mol)')
    ax_d.set_ylim(0, None)
    ax_d.legend(fontsize=7.9, loc='upper right', ncol=2, framealpha=0.94)
    ax_d.set_title('Diminishing-returns screening law: g(K) = a − b ln K (singles + doubles, Cys excl.)', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_d, 'd', x=-0.048)

    ax_e.axhline(0, color='#888888', lw=0.8)
    ax_e.set_ylabel('Residual: data − fit (kcal/mol)')
    ax_e.set_ylim(-0.06, 0.06)
    ax_e.set_title('Log-linear fit residuals across K', loc='left', fontsize=9.4, fontweight='bold')
    _panel_label(ax_e, 'e')

    # Row 3 (panel f): Biophysical origin of top-96 gain by burial × hydrophobicity
    ax_f = fig.add_subplot(gs[2, :])
    if asa_k96_df is not None and not asa_k96_df.empty:
        g_df = asa_k96_df[(asa_k96_df.statistic == 'gain') & (asa_k96_df.model != 'Total')].copy()
        cat_order = [
            'Δ hydrophobicity ≤ 0 & relative ASA ≤ 0.5',
            'Δ hydrophobicity ≤ 0 & relative ASA > 0.5',
            'Δ hydrophobicity > 0 & relative ASA ≤ 0.5',
            'Δ hydrophobicity > 0 & relative ASA > 0.5',
        ]
        short_cats = [
            'Polar/less apolar, buried (ASA ≤ 0.5)',
            'Polar/less apolar, exposed (ASA > 0.5)',
            'More hydrophobic, buried (ASA ≤ 0.5)',
            'More hydrophobic, exposed (ASA > 0.5)',
        ]
        models_f = [m for m in g_df.model.drop_duplicates().tolist()]
        x_m = np.arange(len(models_f), dtype=float)
        w = 0.19
        for c_i, (cat_full, cat_short) in enumerate(zip(cat_order, short_cats)):
            sub = g_df[g_df.category == cat_full].set_index('model').reindex(models_f)
            offs = x_m + (c_i - 1.5) * w
            vals = sub['mean'].to_numpy(float)
            los = sub['lo'].to_numpy(float)
            his = sub['hi'].to_numpy(float)
            ax_f.bar(offs, vals, w, color=PALETTE_CAT[c_i], edgecolor='black', lw=0.6, alpha=0.90, label=cat_short, zorder=3)
            ax_f.errorbar(offs, vals, yerr=[np.maximum(vals - los, 0), np.maximum(his - vals, 0)],
                          fmt='none', ecolor='#222222', elinewidth=1.0, capsize=2.0, zorder=5)
        ax_f.set_xticks(x_m)
        ax_f.set_xticklabels([m.replace('Top 96 per Domain', 'Oracle\n(Top 96)') for m in models_f], fontsize=8.4, rotation=18, ha='right')
        ax_f.set_ylabel('Mean rectified gain @ K=96\nper domain (kcal/mol)')
        ax_f.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax_f.legend(loc='upper right', ncol=2, fontsize=8.1, framealpha=0.94)
        ax_f.set_title('Biophysical decomposition of top-96 single-mutant gain by solvent accessibility × hydrophobicity change (mean ± 95% CI)', loc='left', fontsize=9.4, fontweight='bold')
    else:
        ax_f.text(0.5, 0.5, 'Biophysical feature CSV unavailable', ha='center', va='center', transform=ax_f.transAxes)
    _panel_label(ax_f, 'f', x=-0.031)

    fig.text(
        0.01, 0.006,
        f'Shaded bands & whiskers denote 95% joint domain × seed bootstrap intervals (B = {B}).  '
        f'ESM-MSR achieves higher stabilizing hit rates across MegaScale and held-out Domainome while selecting fewer activity-disrupting mutations on {n_enz} held-out enzymes.',
        fontsize=8.0, color='#444444', ha='left', va='bottom',
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.935, bottom=0.075)
    style.save(fig, out / 'screening_gain_law_composite')


# ----------------------------------------------------------------------------------------------
# Part 8: Main-text Figure 4 — Per-property gain (a) stacked above 6-panel screening overview (b, c)
# ----------------------------------------------------------------------------------------------
ASA_HYDRO_LABELS = [
    'Δ hydrophobicity ≤ 0 & relative ASA ≤ 0.5',
    'Δ hydrophobicity ≤ 0 & relative ASA > 0.5',
    'Δ hydrophobicity > 0 & relative ASA ≤ 0.5',
    'Δ hydrophobicity > 0 & relative ASA > 0.5',
]


def _compute_asa_k96_allpicks_bundle(split: str, B: int) -> dict | None:
    if split != 'capped' or not TSUBOYAMA_FEATS_CSV.exists():
        return None
    fr, _ = load_tsu_test_frame(split)
    sp = fr.spine
    single = (sp['use'] & (sp['n_mut'] == 1)).to_numpy()
    y = sp['y'].to_numpy()
    feat = _load_tsu_features(sp['id'].tolist())
    libs = sorted(np.unique(sp['group'].to_numpy()[single]))
    group = sp['group'].to_numpy()
    best, _ = best_other(fr)
    cols = _single_columns(fr, best)
    asa = feat['rel_ASA'].to_numpy()
    dh = feat['d_hydrophobicity'].to_numpy()
    pool = single & np.isfinite(asa) & np.isfinite(dh)
    cat = np.nan_to_num(np.asarray(2 * (dh > 0) + (asa > 0.5), float)).astype(int)
    dom = [d for d in (np.flatnonzero(pool & (group == lib)) for lib in libs) if len(d)]
    cnt = np.random.default_rng(5).multinomial(len(dom), np.full(len(dom), 1.0 / len(dom)), size=B).astype(float)
    st = _model_cat_stats(fr, cols, y, cat, len(ASA_HYDRO_LABELS), dom, 96, positive_only=False)
    return dict(
        dcols=cols,
        stats_k=st,
        counts=cnt,
        cat_labels=list(ASA_HYDRO_LABELS),
        k=96,
        n_domains=len(dom),
        n_muts=int(pool.sum()),
        s_title=split_title(split),
    )


def build_figure4(
    tables_08: dict[str, pd.DataFrame],
    act_tables: dict[str, pd.DataFrame],
    specs: dict,
    B: int,
    prop_bundle: dict | None = None,
) -> None:
    """Build main-text Figure 4 (00_main_text_figs/figure4.{png,pdf}).

    Panel a: Per-property rectified gain @ k=96 (all picks, ASA x hydrophobicity) box plots + Picks (%) strip.
    Panel b: Homology-Restricted MegaScale test set [excl. Cys] single-mutant all-models screening curves (hit rate & P(at least one hit)).
    Panel c: Domainome (held-out) single-mutant all-models screening curves (hit rate & P(at least one hit)).
    Panel d: ProteinGym Activity single-mutant all-models screening curves (all held-out assays & held-out enzymes).
    """
    if prop_bundle is None:
        prop_bundle = _compute_asa_k96_allpicks_bundle(split='capped', B=B)
    if prop_bundle is None:
        print('[p09] Skipping figure4: Tsuboyama feature CSV not available for panel a')
        return

    out_main = results_dir('main_text_figs')
    dcols = prop_bundle['dcols']
    stats_k = prop_bundle['stats_k']
    counts = prop_bundle['counts']
    cat_labels = prop_bundle['cat_labels']
    k = prop_bundle['k']
    ncat = len(cat_labels)

    mcols = _with_oracle_col(dcols, k)
    cat_gap = 0.28
    xpad_dist = (0.58, 0.58)
    xs, spans = style.layout_grouped_columns(mcols, gap=cat_gap)
    rng = np.random.default_rng(3)
    colors = PALETTE_CAT[:ncat]
    slot = 0.85
    cw = slot / ncat
    cats_hdr = {'oracle': 'Oracle', **style.GROUP_TITLE}

    fig = plt.figure(figsize=(16.0, 14.8))

    # --- Top section: Panel a (per-property gain box plots + Picks % strip) ---
    gs_a = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.105], hspace=0.025,
        left=0.055, right=0.988, top=0.945, bottom=0.675,
    )
    ax_gain = fig.add_subplot(gs_a[0, 0])
    ax_pref = fig.add_subplot(gs_a[1, 0])

    omean, _, _ = _boot_mean_ci(stats_k[mcols[0]['name']][0], counts)
    top_val = 0.0
    fs_txt = 7.4
    for mc, x_col in zip(mcols, xs):
        m = mc['name']
        V_m = stats_k[m][0]
        mean, lo, hi = _boot_mean_ci(V_m, counts)
        for c_idx in range(ncat):
            x0 = x_col + (c_idx - (ncat - 1) / 2) * cw
            style.raincloud(
                ax_gain, x0, V_m[:, :, c_idx], colors[c_idx],
                mean=float(mean[c_idx]), lo=float(lo[c_idx]), hi=float(hi[c_idx]),
                width=cw * 0.96, rng=rng, transform=_symlog_tr, annotate=True, fs=fs_txt,
            )
            top_val = max(top_val, float(_symlog_tr(hi[c_idx])), float(_symlog_tr(V_m[:, :, c_idx]).max()))
    _draw_cat_spans(ax_gain, xs, spans, headers=True, cats=cats_hdr, xpad=xpad_dist, gap=cat_gap, header_fs=11.8)

    tk, tl = _symlog_ticks(10 ** (top_val - 1) if top_val > 1 else 1)
    ax_gain.set_yticks(tk)
    ax_gain.set_yticklabels(tl, fontsize=10.8)
    ax_gain.set_ylim(-0.18, top_val + 0.16)
    ax_gain.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
    y_orc = [float(_symlog_tr(omean[c_idx])) for c_idx in range(ncat)]
    for c_idx in range(ncat):
        near_prev = any(abs(y_orc[c_idx] - y_orc[j]) < 0.025 for j in range(c_idx))
        ls = (4.0, (4.0, 4.0)) if near_prev else (0.0, (4.0, 4.0))
        ax_gain.axhline(y_orc[c_idx], color=colors[c_idx], ls=ls, lw=1.15, alpha=0.92, zorder=2)
    ax_gain.set_ylabel(f'Rectified gain @ k={k} (kcal/mol)', fontsize=11.2)
    ax_gain.set_xticks([])

    _draw_cat_spans(ax_pref, xs, spans, headers=False, cats=cats_hdr, xpad=xpad_dist, gap=cat_gap)
    ax_pref.set_ylim(0, 1)
    ax_pref.set_yticks([0.5])
    ax_pref.set_yticklabels(['Picks (%)'], fontsize=10.2, fontweight='semibold')
    ax_pref.tick_params(axis='y', length=0, pad=6)
    for sp_name in ('top', 'right', 'left', 'bottom'):
        ax_pref.spines[sp_name].set_visible(False)
    for mc, x_col in zip(mcols, xs):
        m = mc['name']
        p_mean, _, _ = _boot_mean_ci(stats_k[m][1], counts)
        p_sum = float(p_mean.sum())
        pcts = (100.0 * p_mean / p_sum) if p_sum > 0 else np.zeros(ncat)
        x_cur = x_col - slot / 2.0
        for c_idx in range(ncat):
            w_seg = slot * (pcts[c_idx] / 100.0)
            if w_seg <= 0:
                continue
            ax_pref.add_patch(plt.Rectangle((x_cur, 0.08), w_seg, 0.84, fc=colors[c_idx], alpha=0.55, ec='white', lw=1.0, zorder=3))
            if pcts[c_idx] >= 5.0:
                ax_pref.text(
                    x_cur + w_seg / 2.0, 0.50, f'{pcts[c_idx]:.0f}', ha='center', va='center',
                    fontsize=8.2 if pcts[c_idx] >= 12.0 else 7.2, fontweight='bold', color='#1a1a1a', zorder=5,
                )
            x_cur += w_seg
        ax_pref.add_patch(plt.Rectangle((x_col - slot / 2.0, 0.08), slot, 0.84, fc='none', ec='#555555', lw=0.8, zorder=4))
    ax_pref.set_xticks(xs)
    ax_pref.set_xticklabels([mc['label'] for mc in mcols], fontsize=10.2)

    # Panel a category legend (centered directly beneath panel a x-tick labels)
    leg_a = fig.legend(
        handles=[Patch(fc=colors[c_idx], alpha=0.8, label=cat_labels[c_idx]) for c_idx in range(ncat)],
        loc='upper center', ncol=2, bbox_to_anchor=(0.52, 0.632),
        fontsize=10.4, frameon=False, columnspacing=2.2,
    )
    fig.add_artist(leg_a)

    # --- Bottom section: Panels b & c (6-panel single-mutant all-models screening overview) ---
    fams = list(specs['fam_all'])
    flab = specs['fam_label_singles']
    D = tables_08['screening_per_protein_singles_all_models']
    A_all = act_tables['screening_all_all_models']
    A_enz = act_tables['screening_enzymes_all_models']

    # Shared model legend for panels b & c
    handles_bc = []
    for f in fams:
        if f in D.family.values:
            col, ls, lw, _ = specs['fam_color_all'][f]
            handles_bc.append(Line2D([0], [0], color=col, ls=ls, lw=lw + 0.4, label=flab[f]))
    handles_bc.append(Line2D([0], [0], color='#2c3e50', ls=':', lw=2.0, label='oracle'))
    handles_bc.append(Line2D([0], [0], color='#c0392b', ls=(0, (3, 2)), lw=1.8, label='random picks'))
    fig.legend(
        handles=handles_bc, loc='upper center', bbox_to_anchor=(0.52, 0.568),
        ncol=6, fontsize=8.8, handlelength=3.2,
        frameon=True, facecolor='white', edgecolor='#cccccc',
    )

    gs_bc = fig.add_gridspec(
        2, 3, left=0.055, right=0.988, top=0.486, bottom=0.082,
        hspace=0.34, wspace=0.20,
    )
    axes_bc = np.array([[fig.add_subplot(gs_bc[r, c]) for c in range(3)] for r in range(2)])
    ax_titles: list[tuple[plt.Axes, str]] = []

    for c, ds in enumerate(('tsuboyama_capped_test', 'domainome')):
        d = D[D.dataset == ds]
        _draw_curve_panel(
            axes_bc[0, c], d, fams, flab, specs, False, 'hit_rate', 'rand_hit', 'oracle_hit',
            'hit rate (abundance > wild type)' if ds == 'domainome' else 'hit rate (> 0.5 kcal/mol)',
        )
        _draw_curve_panel(
            axes_bc[1, c], d, fams, flab, specs, False, 'p_recover', 'rand_rec', 'oracle_rec',
            'P(at least one hit in top K)',
        )
        n_prot = int(d.n_proteins.iloc[0])
        if ds == 'domainome':
            ttl = f'Domainome (held-out), single mutants only ({n_prot} proteins)'
        else:
            ttl = f'Homology-Restricted MegaScale test set, single mutants only [excl. Cys] ({n_prot} proteins)'
        ax_titles.append((axes_bc[0, c], ttl))
        axes_bc[1, c].set_xlabel('variants screened per protein (K)')

    for r_i, (A_cur, nm) in enumerate(((A_all, 'all held-out assays'), (A_enz, 'held-out enzymes'))):
        _draw_curve_panel(
            axes_bc[r_i, 2], A_cur, fams, flab, specs, False, 'hit_rate', 'rand_hit', None,
            'fraction of the K picks\nthat hurt (lower is better)',
        )
        ax_titles.append((axes_bc[r_i, 2], f'ProteinGym Activity, {nm}, single mutants ({int(A_cur.n_assays.iloc[0])} assays)'))
        axes_bc[r_i, 2].set_ylim(0, None)
    axes_bc[1, 2].set_xlabel('variants screened per assay (K)')

    # Wrap panel titles to column width
    fig.canvas.draw()
    r_rend = fig.canvas.get_renderer()
    for ax_t, raw_t in ax_titles:
        ax_t.set_title(
            style.wrap_to_px(fig, raw_t, 0.97 * ax_t.get_window_extent(r_rend).width, fontsize=9.8, fontweight='bold'),
            loc='left', fontsize=9.8, pad=6,
        )

    # Nature-style sub-figure annotations: lowercase bold sans-serif 'a' (top) and 'b', 'c', 'd' (per column)
    fig.canvas.draw()
    y_bcd = axes_bc[0, 0].get_position().y1 + 0.036
    fig.text(0.008, 0.992, 'a', ha='left', va='top', fontsize=15, fontweight='bold')
    for col_i, letter in enumerate(('b', 'c', 'd')):
        pos_c = axes_bc[0, col_i].get_position()
        x_lbl = 0.008 if col_i == 0 else pos_c.x0 - 0.047
        fig.text(x_lbl, y_bcd, letter, ha='left', va='top', fontsize=15, fontweight='bold')

    # Combined footnote
    note4 = (
        f'a. Top {k} per {prop_bundle["s_title"]} test domain ({prop_bundle["n_domains"]} domains with the feature, {prop_bundle["n_muts"]:,} single mutants); '
        f'gain = sum of max(ddG, 0) over every pick, whatever its predicted sign; boxes: interquartile range with darker colored median line; '
        f'dots: domains; dark bars with vertical 95% CI whiskers (B = {B}) & annotations: mean over domains; dashed lines: oracle mean.\n'
        f'b–d. Single mutants only, scored by the single-mutant models; MegaScale excludes mutations to Cys. '
        f'ProteinGym Activity column: a variant hurts when its activity is more than half an assay standard deviation below the measured wild type; lower is better. '
        f'Shaded band: 95% bootstrap interval (B = {B}) of ESM-MSR only. ESM3ΔG is shown on Domainome and ProteinGym Activity (omitted on MegaScale due to 41% strong-homolog overlap with its MGnify training set).'
    )
    kw_n = dict(fontsize=7.6, color='#444444')
    fig.text(0.012, 0.006, style.wrap_to_px(fig, note4, 0.975 * fig.bbox.width, **kw_n), ha='left', va='bottom', **kw_n)

    style.save(fig, out_main / 'figure4')


def build_supp_figure7(
    D_all: pd.DataFrame | None = None,
    specs: dict | None = None,
    B: int = 1000,
) -> Path:
    """Build Supplementary Figure 7 (`00_supplementary_figs/supp_figure7.{png,pdf}`).

    Rectified gain per variant g(K) = a - b ln K across ALL models on Homology-Restricted
    MegaScale test single mutants (Cys excluded, 33 proteins):
      a. Empirical rectified gain per variant vs K (log scale) with least-squares fits g(K) = a - b ln K
         and each model's fitted equation in the legend.
      b. Fit residuals (data - fit) across K for all models and the oracle.
    """
    out_supp = results_dir('supplementary_figs')
    if specs is None:
        specs = build_family_specs('capped')
    if D_all is None:
        mega_csv = results_dir('megascale_screening', 'capped', create=False) / 'screening_per_protein_singles_all_models.csv'
        D_all = pd.read_csv(mega_csv)
    d = D_all[D_all['dataset'] == 'tsuboyama_capped_test'].copy()

    fams = [f for f in specs['fam_all'] if f in d['family'].values]
    flab = specs['fam_label_singles']
    ks = np.array(sorted(d['K'].unique()), dtype=float)
    lk = np.log(ks)
    n_prots = int(d['n_proteins'].iloc[0])

    # Extract curves & fit g(K) = a - b ln K for every model, oracle, and random
    curves: dict[str, np.ndarray] = {}
    fits: dict[str, tuple[float, float, float, float]] = {}  # (a, b, rmse, r2)
    msr_ci = None

    for f in fams:
        sub = d[(d['family'] == f) & (d['metric'] == 'gain')].sort_values('K')
        if len(sub) != len(ks):
            continue
        gv = (sub['value'].to_numpy(float)) / ks
        curves[f] = gv
        a_v, b_v = _fit_line(lk, gv)
        pred = a_v - b_v * lk
        rmse_v = float(np.sqrt(np.mean((gv - pred) ** 2)))
        ss_tot = float(np.sum((gv - gv.mean()) ** 2))
        r2_v = float(1.0 - np.sum((gv - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan
        fits[f] = (a_v, b_v, rmse_v, r2_v)
        if f == 'msr':
            msr_ci = (sub['lo'].to_numpy(float) / ks, sub['hi'].to_numpy(float) / ks)

    sub_orc = d[(d['family'] == 'msr') & (d['metric'] == 'oracle_gain')].sort_values('K')
    g_orc = sub_orc['value'].to_numpy(float) / ks
    a_o, b_o = _fit_line(lk, g_orc)
    pred_o = a_o - b_o * lk
    rmse_o = float(np.sqrt(np.mean((g_orc - pred_o) ** 2)))
    r2_o = float(1.0 - np.sum((g_orc - pred_o) ** 2) / np.sum((g_orc - g_orc.mean()) ** 2))

    sub_rnd = d[(d['family'] == 'msr') & (d['metric'] == 'rand_gain')].sort_values('K')
    g_rnd = sub_rnd['value'].to_numpy(float) / ks
    a_r = float(g_rnd.mean())

    # Order models by fitted intercept a (highest first) so legend reads cleanly top-to-bottom
    fams_ordered = sorted(curves.keys(), key=lambda f: -fits[f][0])

    fig, (ax, rx) = plt.subplots(1, 2, figsize=(14.6, 6.2), gridspec_kw=dict(width_ratios=[1.38, 1.0]), facecolor='white')
    xx = np.geomspace(1, 100, 120)

    # Oracle
    ax.plot(ks, g_orc, 'o', color='#2c3e50', ms=4.8, mec='white', mew=0.6, zorder=4)
    ax.plot(xx, a_o - b_o * np.log(xx), color='#2c3e50', ls=':', lw=1.6, zorder=3,
            label=f'Oracle (top K labels):  {a_o:.2f} − {b_o:.3f} ln K  (R²={r2_o:.3f})')
    rx.plot(ks, g_orc - (a_o - b_o * lk), marker='o', color='#2c3e50', ls=':', ms=4.0, lw=1.1, label='Oracle')

    # ESM-MSR 95% CI band
    if msr_ci is not None:
        ax.fill_between(ks, msr_ci[0], msr_ci[1], color='#000000', alpha=0.11, lw=0, zorder=2)

    # All models
    for f in fams_ordered:
        col, ls, lw, _ = specs['fam_color_all'][f]
        a_v, b_v, rmse_v, r2_v = fits[f]
        gv = curves[f]
        is_msr = (f == 'msr')
        ax.plot(ks, gv, 'o', color=col, ms=5.4 if is_msr else 4.2, mec='white', mew=0.6, zorder=6 if is_msr else 4)
        ax.plot(
            xx, a_v - b_v * np.log(xx), color=col, ls=ls, lw=2.3 if is_msr else max(lw, 1.3),
            zorder=5 if is_msr else 3,
            label=f'{flab[f]}:  {a_v:.2f} − {b_v:.3f} ln K  (R²={r2_v:.3f})',
        )
        rx.plot(
            ks, gv - (a_v - b_v * lk), marker='o', color=col, ls=ls,
            ms=4.4 if is_msr else 3.5, lw=1.8 if is_msr else 1.1,
            zorder=5 if is_msr else 3, label=flab[f],
        )

    # Random picks
    ax.plot(ks, g_rnd, 's', color='#c0392b', ms=3.8, zorder=4)
    ax.axhline(a_r, color='#c0392b', ls=(0, (3, 2)), lw=1.4, zorder=3, label=f'Random picks:  {a_r:.3f}')

    for ax_i in (ax, rx):
        ax_i.set_xscale('log')
        ax_i.set_xticks(ks)
        ax_i.set_xticklabels([str(int(k)) for k in ks])
        ax_i.set_xlabel('variants screened per protein (K)', fontsize=10.2)
        ax_i.grid(axis='y', color=style.GRID, lw=0.6, zorder=0)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)

    ax.set_ylabel('rectified gain per variant (kcal/mol)', fontsize=10.2)
    ax.set_ylim(0.0, 1.62)
    ax.set_title('Rectified gain per variant: g(K) = a − b ln K (all models)', fontsize=10.4, fontweight='bold', loc='left', pad=6)
    ax.legend(fontsize=7.3, loc='upper right', framealpha=0.94, edgecolor='#cccccc', handlelength=2.4, labelspacing=0.28)

    rx.axhline(0.0, color='#888888', lw=0.9)
    rx.set_ylabel('data − fit (kcal/mol)', fontsize=10.2)
    rx.set_ylim(-0.08, 0.08)
    rx.set_title('Log-linear fit residuals across K', fontsize=10.4, fontweight='bold', loc='left', pad=6)
    rx.legend(fontsize=7.3, loc='upper right', ncol=2, framealpha=0.94, edgecolor='#cccccc', handlelength=2.0, columnspacing=0.9, labelspacing=0.25)

    fig.subplots_adjust(left=0.058, right=0.988, top=0.900, bottom=0.215, wspace=0.18)
    fig.canvas.draw()
    pr = rx.get_position()
    fig.text(0.010, 0.975, 'a', ha='left', va='top', fontsize=14, fontweight='bold')
    fig.text(pr.x0 - 0.052, 0.975, 'b', ha='left', va='top', fontsize=14, fontweight='bold')

    a_m, b_m, _, _ = fits['msr']
    k0_msr = float(np.exp((a_m - a_r) / b_m)) if b_m > 0 else np.nan
    note = (
        f'Homology-Restricted MegaScale test set ({n_prots} proteins; single mutants, Cys excluded; single-mutant models; 3 seeds averaged for retrained models).  '
        f'a. Points show mean rectified gain per variant = (1/K) Σ max(ΔΔG, 0) over the K highest-scored variants per protein; lines show least-squares fits '
        f'g(K) = a − b ln K across K ∈ {{1, 3, 5, 10, 20, 50, 100}} (shaded band: 95% joint domain × seed bootstrap CI for ESM-MSR, B = {B:,}; '
        f'ESM-MSR reaches the random baseline {a_r:.3f} kcal/mol at K ≈ {k0_msr:.0f}).  '
        f'b. Residuals (empirical − linear fit) across K for all models and the oracle.'
    )
    kw = dict(fontsize=7.8, color='#333333', linespacing=1.25)
    fig.text(0.014, 0.012, style.wrap_to_px(fig, note, 0.972 * fig.bbox.width, **kw), ha='left', va='bottom', **kw)

    out_path = out_supp / 'supp_figure7'
    style.save(fig, out_path)
    return out_path


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------
def add_args(ap) -> None:
    ap.add_argument('--tau', type=float, default=0.5, help='Stabilizing ddG threshold in kcal/mol (default: 0.5).')
    ap.add_argument('--frac', type=float, default=0.10, help='Top fraction for pooled external screens (default: 0.10).')
    ap.add_argument('--top-n', type=int, default=6, help='Top-N most stabilizing single mutants per domain (default: 6).')
    ap.add_argument('--budgets', nargs='+', type=int, default=[6, 24, 96], help='Fixed screening budgets K (default: 6 24 96).')
    ap.add_argument(
        '--only',
        choices=('all', 'domainome_and_overview', 'figure4', 'supp_figure7'),
        default='all',
        help='Optional substage selector (domainome_and_overview, figure4, or supp_figure7).',
    )


def run(args=None) -> int:
    if args is None:
        ap = base_parser(STAGE, DESCRIPTION, with_split=True)
        add_args(ap)
        args = ap.parse_args()
    B = resolve_b(args)
    split = getattr(args, 'split', 'capped')
    tau = getattr(args, 'tau', 0.5)
    frac = getattr(args, 'frac', 0.10)
    top_n = getattr(args, 'top_n', 6)
    budgets = tuple(getattr(args, 'budgets', [6, 24, 96]))
    only = getattr(args, 'only', 'all')

    register_ext_styles()
    print(f'[p09:{STAGE}] Running split={split}, B={B}, only={only} ...')

    if only == 'supp_figure7':
        out = build_supp_figure7(B=B)
        print(f'[p09:screening] Saved {out}.{{png,pdf}}')
        return 0

    if only == 'figure4':
        specs = build_family_specs('capped')
        out_mega = results_dir('megascale_screening', 'capped', create=False)
        out_dom = results_dir('domainome_screening', create=False)
        pg_scr = results_dir('proteingym_screening', create=False)
        tables_08 = {
            'screening_per_protein_singles_all_models': pd.concat([
                pd.read_csv(out_mega / 'screening_per_protein_singles_all_models.csv'),
                pd.read_csv(out_dom / 'screening_per_protein_all_models.csv'),
            ], ignore_index=True),
        }
        act_tables = {
            nm: pd.read_csv(pg_scr / f'{nm}.csv')
            for nm in ('screening_all_all_models', 'screening_enzymes_all_models')
        }
        build_figure4(tables_08=tables_08, act_tables=act_tables, specs=specs, B=B)
        build_supp_figure7(D_all=tables_08['screening_per_protein_singles_all_models'], specs=specs, B=B)
        print('[p09:screening] Completed figure4.')
        return 0

    if only == 'domainome_and_overview':
        tables_08, specs = run_08_screening(split='capped', tau=tau, frac=frac, B=B, only_domainome=True)
        pg_scr = results_dir('proteingym_screening', create=False)
        act_tables = {
            nm: pd.read_csv(pg_scr / f'{nm}.csv')
            for nm in ('screening_all', 'screening_all_all_models', 'screening_enzymes', 'screening_enzymes_all_models')
        }
        run_16_screening_overview(tables_08=tables_08, act_tables=act_tables, specs=specs, B=B)
        prop_csv = results_dir('property_gain', 'capped', create=False) / 'gain_asa_hydrophobicity_k96_allpicks.csv'
        asa_k96_df = pd.read_csv(prop_csv) if prop_csv.exists() else None
        fit_params, fit_curves = run_17_gain_fit(split='capped', specs=specs, B=B)
        build_screening_gain_law_composite(
            tables_08=tables_08, act_tables=act_tables,
            fit_params=fit_params, fit_curves=fit_curves,
            asa_k96_df=asa_k96_df, specs=specs, B=B,
        )
        build_figure4(tables_08=tables_08, act_tables=act_tables, specs=specs, B=B)
        build_supp_figure7(D_all=tables_08['screening_per_protein_singles_all_models'], specs=specs, B=B)
        print('[p09:screening] Completed domainome_and_overview.')
        return 0

    tables_08, specs = run_08_screening(split=split, tau=tau, frac=frac, B=B)
    run_15_single_screening(split=split, tau=tau, B=B, top_n=top_n, budgets=budgets)
    fit_params, fit_curves = run_17_gain_fit(split=split, specs=specs, B=B)

    if split == 'capped':
        asa_k96_df = run_14_property_gain(split=split, B=B)
        act_tables = run_pg_activity(specs=specs, B=B)
        run_16_screening_overview(tables_08=tables_08, act_tables=act_tables, specs=specs, B=B)
        build_screening_gain_law_composite(
            tables_08=tables_08, act_tables=act_tables,
            fit_params=fit_params, fit_curves=fit_curves,
            asa_k96_df=asa_k96_df, specs=specs, B=B,
        )
        build_figure4(tables_08=tables_08, act_tables=act_tables, specs=specs, B=B)
        build_supp_figure7(D_all=tables_08['screening_per_protein_singles_all_models'], specs=specs, B=B)
    print(f'[p09:{STAGE}] Completed split={split}.')
    return 0


def main(argv=None) -> int:
    ap = base_parser(STAGE, DESCRIPTION, with_split=True)
    add_args(ap)
    return run(ap.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
