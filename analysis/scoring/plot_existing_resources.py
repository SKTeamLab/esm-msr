"""Visualize the fixed-slot RTX A4500 (20 GB) inference resource benchmarks across all manuscript datasets.

Benchmark groups (matching manuscript conventions; DMS excluded):
  1. Megascale Test  (tsuboyama_test / oct06_capped_allbl_test: 37 domains, ~52k variants)
  2. External        (12 external thermostability sets)
  3. Domainome       (522 domains, ~572k variants)
  4. ProteinGym      (217 DMS assays, ~2.47M variants)
  5. Rocklin         (rocklin_denovo: 1,000 de novo scaffolds, 209,818 variants)
  6. MGnify          (mgnify: 180,367 scaffolds, 766,790 variants)

Produces two publication-style figures (PNG + PDF) under
analysis/results/10_inference_resources/:
  1. 01_existing_benchmark_resources_a4500.{png,pdf}
     Uses MutateEverything St2 (St1 omitted), includes ESM-MSR masking variants and ProSST,
     and covers all 6 manuscript groups (marking queued/pending cells clearly).
  2. 02_existing_proteingym_scaling_a4500.{png,pdf}
     Per-protein length scaling on the 217 ProteinGym assays (MutateEverything St2 omitted;
     includes ESM-MSR WT-view, Dual-view, Independent masking, plus Masked marginal & ProSST once tracked).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'analysis'))

from msr_analysis.paths import INFERENCE_RESOURCES_DIR, MSR_ROOT  # noqa: E402
from msr_analysis import style  # noqa: E402
from msr_analysis.models import BY_NAME  # noqa: E402

style.setup()

OUT_DIR = INFERENCE_RESOURCES_DIR
TRACKED_ROOT = MSR_ROOT / 'resources_tracked'

GROUPS = ['Megascale Test', 'External', 'Domainome', 'ProteinGym', 'Rocklin', 'MGnify']
CORE_GROUPS = ['Megascale Test', 'External', 'Domainome', 'ProteinGym']
EXT_SETS = {'s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld'}

ROCKLIN_ROWS = 209818
MGNIFY_ROWS = 766790

# Exact gpu4 (RTX A4500 20 GB) Slurm chunk-log timings for MGnify/Rocklin runs that did not emit resources_scopes.csv
# (verified directly from /home/sreeves/esm-msr-devel/outputs/tracked/mgnify/*/slurm-bench_inf-*.out on gpu4):
GPU4_MGNIFY_SLURM_FALLBACK = {
    'spurs_retrain_s1': {
        'Rocklin': dict(wall_s=2.9 * 60.0, n_rows=ROCKLIN_ROWS, job='389871'),
        'MGnify': dict(wall_s=309.6 * 60.0, n_rows=MGNIFY_ROWS, job='389871'),
    },
    'proteinmpnn': {
        'Rocklin': dict(wall_s=9.9 * 60.0, n_rows=ROCKLIN_ROWS, job='389885'),
        'MGnify': dict(wall_s=65.9 * 60.0, n_rows=MGNIFY_ROWS, job='389885'),
    },
    'thermompnn_s1': {
        'Rocklin': dict(wall_s=10.7 * 60.0, n_rows=ROCKLIN_ROWS, job='389874'),
        'MGnify': dict(wall_s=72.3 * 60.0, n_rows=MGNIFY_ROWS, job='389874'),
    },
    'prosst_a4500_full': {
        'Rocklin': dict(wall_s=2.3 * 60.0, n_rows=ROCKLIN_ROWS, job='389890'),
        'MGnify': dict(wall_s=231.9 * 60.0, n_rows=MGNIFY_ROWS, job='389890'),
    },
}

# Models shown in Figure 1 (ME St2 only; includes ProSST and ESM-MSR masking variants)
FIG1_MODELS = [
    dict(
        key='esm_msr_final_p0_wt_s1',
        mgnify_dir='esm_msr_final_p0_wt_s1_mgnify',
        line='ESM-MSR (WT-view)',
        color=BY_NAME['ESM-MSR'].color, marker='D', hollow=True, ls='--', lw=2.2, batched=False,
        pending_note='queued',
    ),
    dict(
        key='spurs_retrain_s1',
        mgnify_dir=None,
        line='SPURS (+multi)',
        color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0, batched=False,
    ),
    dict(
        key='prosst_a4500_full',
        mgnify_dir=None,
        line='ProSST-2048',
        color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0, batched=False,
        pending_note='queued',
    ),
    dict(
        key='proteinmpnn',
        mgnify_dir=None,
        line='ProteinMPNN',
        color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0, batched=True,
    ),
    dict(
        key='thermompnn_s1',
        mgnify_dir=None,
        line='ThermoMPNN (+D)',
        color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=False, ls='-', lw=2.0, batched=True,
    ),
    dict(
        key='me_st2_s1',
        mgnify_dir='me_st2_s1_rocklin',
        line='MutateEverything (St2)',
        color=BY_NAME['MutateEverything'].color, marker='h', hollow=False, ls='-', lw=2.0, batched=False,
        mgnify_pending='MSAs',
    ),
    dict(
        key='esm_msr_final_p1_marg_s1_full',
        mgnify_dir='esm_msr_final_p1_marg_s1_mgnify',
        line='ESM-MSR (Masked marginal)',
        color='#8c8c8c', marker='^', hollow=False, ls='-', lw=2.0, batched=True,
        pending_note='queued',
    ),
    dict(
        key='esm_msr_final_p1_s1_indep',
        mgnify_dir='esm_msr_final_p1_s1_indep_mgnify',
        line='ESM-MSR (Indep. masking)',
        color='#4d4d4d', marker='s', hollow=False, ls='-', lw=2.1, batched=True,
        pending_note='queued',
    ),
    dict(
        key='esm_msr_final_p0_s1_full',
        fallback_key='esm_msr_devel_s1',
        mgnify_dir='esm_msr_final_p0_mgnify',
        line='ESM-MSR (Dual-view)',
        color=BY_NAME['ESM-MSR'].color, marker='D', hollow=False, ls='-', lw=2.5, batched=True,
    ),
]

# Models shown in Figure 2 (ProteinGym scaling: ME St1 only, St2 removed; includes ESM-MSR masking variants & ProSST)
FIG2_MODELS = [
    dict(key='esm_msr_final_p0_wt_s1', line='ESM-MSR (WT-view)', color=BY_NAME['ESM-MSR'].color, marker='D', hollow=True, ls='--', lw=2.2, batched=False, est_mode='oneshot'),
    dict(key='spurs_retrain_s1', line='SPURS (+multi)', color=BY_NAME['SPURS'].color, marker='s', hollow=False, ls='-', lw=2.0, batched=False, est_mode='spurs_single'),
    dict(key='prosst_a4500_full', line='ProSST-2048', color=BY_NAME['ProSST'].color, marker='X', hollow=False, ls=':', lw=2.0, batched=False, est_mode='oneshot'),
    dict(key='proteinmpnn', line='ProteinMPNN', color=BY_NAME['ProteinMPNN'].color, marker='d', hollow=False, ls='-.', lw=2.0, batched=True, est_mode=None),
    dict(key='thermompnn_s1', line='ThermoMPNN (+D)', color=BY_NAME['ThermoMPNN'].color, marker='p', hollow=False, ls='-', lw=2.0, batched=True, est_mode='linear_single'),
    dict(key='me_st1_s1_a4500', line='MutateEverything (St1)', color=BY_NAME['MutateEverything'].color, marker='h', hollow=True, ls='--', lw=2.0, batched=False, est_mode='oneshot'),
    dict(key='esm_msr_final_p1_marg_s1_full', line='ESM-MSR (Masked marginal)', color='#8c8c8c', marker='^', hollow=False, ls='-', lw=2.0, batched=True, est_mode=None),
    dict(key='esm_msr_final_p1_s1_indep', line='ESM-MSR (Indep. masking)', color='#4d4d4d', marker='s', hollow=False, ls='-', lw=2.1, batched=True, est_mode=None),
    dict(key='esm_msr_final_p0_s1_full', fallback_key='esm_msr_devel_s1', line='ESM-MSR (Dual-view)', color=BY_NAME['ESM-MSR'].color, marker='D', hollow=False, ls='-', lw=2.5, batched=True, est_mode='linear'),
]


def fmt_time_s(sec: float) -> str:
    if not np.isfinite(sec):
        return '—'
    if sec < 60:
        return f'{sec:.1f} s'
    if sec < 3600:
        return f'{sec / 60:.1f} m'
    return f'{sec / 3600:.2f} h'


def _load_tracked_dir(dir_name: str | None) -> tuple[pd.DataFrame, list[dict]]:
    if not dir_name:
        return pd.DataFrame(), []
    d = TRACKED_ROOT / dir_name
    if not d.is_dir():
        return pd.DataFrame(), []
    runs = sorted(p.parent for p in d.glob('*/resources_run.json'))
    dfs, metas = [], []
    for r in runs:
        csv_p = r / 'resources_scopes.csv'
        jl_p = r / 'resources_scopes.jsonl'
        if csv_p.exists():
            dfs.append(pd.read_csv(csv_p))
        elif jl_p.exists() and jl_p.stat().st_size > 0:
            dfs.append(pd.read_json(jl_p, lines=True))
        metas.append(json.loads((r / 'resources_run.json').read_text()))
    return (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()), metas


def _group_of_dataset(ds: str, pgym_names: set[str]) -> str | None:
    d = str(ds)
    for pre in ('external_', 'dms_'):
        if d.startswith(pre):
            d = d[len(pre):]
    if d in ('tsuboyama_test', 'oct06_capped_allbl_test'):
        return 'Megascale Test'
    if d in ('tsuboyama_val', 'oct06_capped_allbl_val', 'oct06_capped_allbl_testcomp'):
        return None  # excluded from Megascale Test
    if d in EXT_SETS:
        return 'External'
    if d.endswith(('_Faure_2021_abundance_domain', '_Faure_2021_binding_domain')) or d in (
        'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain',
    ):
        return None  # DMS excluded
    if d == 'domainome':
        return 'Domainome'
    if d == 'rocklin_denovo':
        return 'Rocklin'
    if d == 'proteingym' or d in pgym_names:
        return 'ProteinGym'
    return None


def collect_fig1_table(pgym_names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-(model, group) and per-model summary tables across all 6 manuscript groups."""
    group_rows = []
    model_rows = []
    for spec in FIG1_MODELS:
        key = spec['key']
        sc, metas = _load_tracked_dir(key)
        if sc.empty and spec.get('fallback_key'):
            sc, metas = _load_tracked_dir(spec['fallback_key'])

        per_g: dict[str, dict] = {}
        n_failed_core = 0
        ram_peak = np.nan
        vram_dev_peak = np.nan
        load_s = np.nan

        if not sc.empty:
            prot = sc[sc['level'] == 'protein'].copy()
            prot['group'] = prot['dataset'].map(lambda x: _group_of_dataset(x, pgym_names))
            prot = prot[prot['group'].notna()]
            ok = prot[prot['status'] == 'ok']
            n_failed_core = int(prot['status'].astype(str).str.contains('OutOfMemory|oom|RuntimeError', case=False, na=False).sum())
            if len(ok):
                ram_peak = float(ok['ram_rss_peak_gb'].max())
                vram_dev_peak = float(ok['vram_device_peak_gb'].max())
            st = sc[sc['level'] == 'stage']
            if len(st):
                load_s = float(st['wall_s'].sum())
            for g, sub in ok.groupby('group'):
                w = float(sub['wall_s'].sum())
                nr = float(sub['n_rows'].sum()) if 'n_rows' in sub.columns else np.nan
                per_g[g] = dict(wall_s=w, n_rows=nr, rows_per_s=nr / w if (w > 0 and np.isfinite(nr)) else np.nan, source='tracked')

        # Load dedicated MGnify/Rocklin tracked run if present
        mg_sc, _ = _load_tracked_dir(spec.get('mgnify_dir'))
        if not mg_sc.empty:
            mp = mg_sc[(mg_sc['level'] == 'protein') & (mg_sc['status'] == 'ok')].copy()
            if len(mp):
                # esm_msr_testing logs dataset='mgnify' in both rocklin (1000 scaffolds) and mgnify (180k scaffolds) runs;
                # distinguish by run_id scope count (~1000 = Rocklin, >10000 = MGnify) or explicit dataset='rocklin_denovo'
                for rid, gdf in mp.groupby('run_id'):
                    ds_vals = set(gdf['dataset'].astype(str))
                    if 'rocklin_denovo' in ds_vals or len(gdf) <= 2000:
                        grp = 'Rocklin'
                    else:
                        grp = 'MGnify'
                    w = float(gdf['wall_s'].sum())
                    nr = float(gdf['n_rows'].sum()) if 'n_rows' in gdf.columns else (ROCKLIN_ROWS if grp == 'Rocklin' else MGNIFY_ROWS)
                    per_g[grp] = dict(wall_s=w, n_rows=nr, rows_per_s=nr / w if w > 0 else np.nan, source='tracked')
                if not np.isfinite(ram_peak):
                    ram_peak = float(mp['ram_rss_peak_gb'].max())
                if not np.isfinite(vram_dev_peak):
                    vram_dev_peak = float(mp['vram_device_peak_gb'].max())

        # Apply verified gpu4 Slurm chunk-log timings for MGnify/Rocklin where no resources_scopes.csv was written
        for grp, fb in GPU4_MGNIFY_SLURM_FALLBACK.get(key, {}).items():
            if grp not in per_g:
                w = fb['wall_s']
                nr = fb['n_rows']
                per_g[grp] = dict(wall_s=w, n_rows=nr, rows_per_s=nr / w, source=f'gpu4_log:{fb["job"]}')

        for grp in GROUPS:
            info = per_g.get(grp)
            status = 'measured' if info else (spec.get('mgnify_pending', 'queued') if grp == 'MGnify' and spec.get('mgnify_pending') else spec.get('pending_note', 'pending'))
            group_rows.append(dict(
                key=key, line=spec['line'], group=grp, status=status,
                wall_s=info['wall_s'] if info else np.nan,
                n_rows=info['n_rows'] if info else np.nan,
                rows_per_s=info['rows_per_s'] if info else np.nan,
                source=info['source'] if info else status,
            ))

        core_ok = all(g in per_g for g in CORE_GROUPS)
        all6_ok = all(g in per_g for g in GROUPS)
        core_h = sum(per_g[g]['wall_s'] for g in CORE_GROUPS) / 3600.0 if core_ok else np.nan
        mg_rock_h = sum(per_g[g]['wall_s'] for g in ('Rocklin', 'MGnify') if g in per_g) / 3600.0
        total6_h = sum(per_g[g]['wall_s'] for g in GROUPS) / 3600.0 if all6_ok else np.nan
        model_rows.append(dict(
            key=key, line=spec['line'], core_ok=core_ok, all6_ok=all6_ok,
            core_h=core_h, mg_rock_h=mg_rock_h, total6_h=total6_h,
            n_failed=n_failed_core, ram_peak_gb=ram_peak, vram_device_peak_gb=vram_dev_peak, model_load_s=load_s,
        ))
    return pd.DataFrame(group_rows), pd.DataFrame(model_rows)


def plot_fig1_overview(g_df: pd.DataFrame, m_df: pd.DataFrame) -> Path:
    """2x2 overview across all 6 manuscript benchmark groups (Megascale Test, External, Domainome, ProteinGym, Rocklin, MGnify)."""
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.2))
    ax_tot, ax_hm, ax_rps, ax_mem = axes.ravel()

    m_idx = m_df.set_index('key')
    # Order measured models by core_h ascending, with queued-only models placed in logical position
    def _sort_key(spec):
        r = m_idx.loc[spec['key']]
        if np.isfinite(r['core_h']):
            return (0, float(r['core_h']))
        return (1, FIG1_MODELS.index(spec))

    ordered_specs = sorted(FIG1_MODELS, key=_sort_key)
    keys_ord = [s['key'] for s in ordered_specs]
    y_pos = np.arange(len(ordered_specs))

    # ---- Panel A: Core 4-benchmark wall time (+ Rocklin/MGnify extension where measured) ----
    dual_core_h = float(m_idx.loc['esm_msr_final_p0_s1_full', 'core_h'])
    for i, s in enumerate(ordered_specs):
        r = m_idx.loc[s['key']]
        core_h = float(r['core_h'])
        all6_h = float(r['total6_h'])
        if not np.isfinite(core_h):
            mg_h = float(r['mg_rock_h'])
            msg = f'core 4 queued on gpu4 (Rocklin+MGnify: {mg_h:.2f} h)' if mg_h > 0 else 'queued on gpu4 (RTX A4500)'
            ax_tot.text(0.06, i, msg, va='center', ha='left',
                        fontsize=8.0, style='italic', color='#666666')
            continue
        hatched = s['hollow']
        ax_tot.barh(
            i, core_h, height=0.58,
            color=s['color'], alpha=0.45 if hatched else 0.90,
            edgecolor=s['color'] if hatched else 'black',
            hatch='////' if hatched else None, lw=1.0, zorder=3,
        )
        # If Rocklin + MGnify are also measured, show total-6 marker
        if np.isfinite(all6_h) and all6_h > core_h * 1.02:
            ax_tot.plot([core_h, all6_h], [i, i], color=s['color'], lw=2.2, ls=':', zorder=4)
            ax_tot.scatter([all6_h], [i], marker='|', s=95, color=s['color'], linewidths=2.0, zorder=5)
        ratio = dual_core_h / core_h
        tag = f'{core_h:.2f} h'
        if np.isfinite(all6_h):
            tag += f'  (all 6: {all6_h:.2f} h)'
        if s['key'] == 'esm_msr_final_p0_wt_s1':
            tag += f'  [{ratio:.0f}× faster]'
        elif s['key'] != 'esm_msr_final_p0_s1_full':
            tag += f'  [{ratio:.1f}×]'
        if int(r['n_failed']) > 0:
            tag += f'  ({int(r["n_failed"])} OOM)'
        anchor_x = max(core_h, all6_h) if np.isfinite(all6_h) else core_h
        ax_tot.text(anchor_x * 1.16, i, tag, va='center', ha='left', fontsize=7.8,
                    fontweight='bold' if 'esm_msr' in s['key'] else 'normal', color='#222222')

    ax_tot.set_xscale('log')
    ax_tot.set_xlim(0.04, 1500)
    ax_tot.set_ylim(len(ordered_specs) - 0.35, -0.60)
    ax_tot.set_yticks(y_pos)
    ax_tot.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.5)
    ax_tot.set_xlabel('Inference wall time: bar = 4 core benchmarks; dotted = + Rocklin & MGnify (h, log)')
    ax_tot.set_title('A. Benchmark inference time (RTX A4500 20 GB)', loc='left')
    ax_tot.grid(axis='x', color=style.GRID, lw=0.6, which='major', zorder=0)

    # ---- Panel B: Per-group wall time heatmap across all 6 manuscript groups ----
    piv_s = g_df.pivot(index='key', columns='group', values='wall_s').loc[keys_ord, GROUPS]
    piv_st = g_df.pivot(index='key', columns='group', values='status').loc[keys_ord, GROUPS]
    raw = piv_s.to_numpy(dtype=float)
    mat_log = np.where(np.isfinite(raw), np.log10(np.maximum(raw, 0.5)), np.nan)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad(color='#f0f0f0')
    im = ax_hm.imshow(mat_log, aspect='auto', cmap=cmap, vmin=0.3, vmax=5.2)
    for i in range(len(keys_ord)):
        for j in range(len(GROUPS)):
            val = raw[i, j]
            if np.isfinite(val):
                txt = fmt_time_s(val)
                dark = mat_log[i, j] > 3.2
                ax_hm.text(j, i, txt, ha='center', va='center', fontsize=7.6,
                           color='white' if dark else '#111111',
                           fontweight='bold' if val > 3600 else 'normal')
            else:
                st_lbl = str(piv_st.iloc[i, j])
                ax_hm.text(j, i, st_lbl, ha='center', va='center', fontsize=7.0,
                           style='italic', color='#777777')
    ax_hm.set_xticks(range(len(GROUPS)))
    ax_hm.set_xticklabels(GROUPS, fontsize=8.3, rotation=24, ha='right')
    ax_hm.set_yticks(range(len(keys_ord)))
    ax_hm.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.2)
    ax_hm.set_title('B. Wall time across all 6 manuscript benchmark groups', loc='left')
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.040, pad=0.02)
    cb.set_label('log₁₀(wall time [s])', fontsize=8.4)
    cb.ax.tick_params(labelsize=8)

    # ---- Panel C: Throughput (variants / s) across all 6 groups ----
    piv_rps = g_df.pivot(index='key', columns='group', values='rows_per_s').loc[keys_ord, GROUPS]
    x_g = np.arange(len(GROUPS))
    style.band_axis(ax_rps, len(GROUPS))
    offs = np.linspace(-0.32, 0.32, len(ordered_specs))
    for k, s in enumerate(ordered_specs):
        ys = piv_rps.loc[s['key'], GROUPS].to_numpy(dtype=float)
        if not np.isfinite(ys).any():
            continue
        ok_m = np.isfinite(ys)
        mk = dict(
            marker=s['marker'], ms=6.6, color=s['color'], mec=s['color'],
            mfc='white' if s['hollow'] else s['color'], mew=1.7 if s['hollow'] else 0.9,
            ls=s['ls'], lw=1.2, alpha=0.92,
        )
        ax_rps.plot(x_g[ok_m] + offs[k], ys[ok_m], label=s['line'], zorder=5 if 'esm_msr' in s['key'] else 3, **mk)
    ax_rps.set_yscale('log')
    ax_rps.set_ylim(0.10, 1.5e5)
    ax_rps.set_xticks(x_g)
    ax_rps.set_xticklabels(GROUPS, fontsize=8.3, rotation=20, ha='right')
    ax_rps.set_ylabel('Throughput (variants / second, log scale)')
    ax_rps.set_title('C. Scoring throughput across all 6 benchmark groups', loc='left')
    ax_rps.grid(axis='y', color=style.GRID, lw=0.6, which='major', zorder=0)
    ax_rps.legend(loc='lower left', ncol=2, fontsize=7.0, framealpha=0.94, handlelength=1.8, columnspacing=0.9)

    # ---- Panel D: Peak Device VRAM & Host RAM ----
    bar_h = 0.36
    for i, s in enumerate(ordered_specs):
        r = m_idx.loc[s['key']]
        vd = float(r['vram_device_peak_gb'])
        rp = float(r['ram_peak_gb'])
        ls = float(r['model_load_s'])
        if not np.isfinite(vd):
            ax_mem.text(0.4, i, 'queued on gpu4 (RTX A4500)', va='center', fontsize=7.8, style='italic', color='#777777')
            continue
        ax_mem.barh(
            i - bar_h / 2, vd, height=bar_h,
            color=s['color'], alpha=0.88 if not s['batched'] else 0.40,
            edgecolor=s['color'] if s['batched'] else 'black',
            hatch='///' if s['batched'] else None, lw=0.9, zorder=3,
        )
        ax_mem.barh(
            i + bar_h / 2, rp, height=bar_h,
            color='#7f8c8d', alpha=0.65, edgecolor='#333333', lw=0.7, zorder=3,
        )
        v_note = f'{vd:.1f} GB' + (' (batch-filled)' if s['batched'] else ' (B=1 / native)')
        ax_mem.text(vd + 0.35, i - bar_h / 2, v_note, va='center', fontsize=7.2, color='#222222')
        load_txt = f' (load {ls:.0f} s)' if np.isfinite(ls) else ''
        ax_mem.text(rp + 0.35, i + bar_h / 2, f'{rp:.1f} GB{load_txt}', va='center', fontsize=7.0, color='#555555')

    ax_mem.axvline(20.0, color=style.RING, ls='--', lw=1.1, zorder=4)
    ax_mem.text(20.4, 1.0, '20 GB A4500\ncap', ha='left', va='center', fontsize=7.4, color=style.RING, fontweight='bold')
    ax_mem.set_xlim(0, 32.5)
    ax_mem.set_ylim(len(ordered_specs) + 0.95, -0.55)
    ax_mem.set_yticks(y_pos)
    ax_mem.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.2)
    ax_mem.set_xlabel('Peak memory across tracked runs (GB)')
    ax_mem.set_title('D. Peak GPU VRAM & Host RAM (hatched = 90% batch-filled)', loc='left')
    ax_mem.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    mem_handles = [
        Patch(fc='#555555', ec='black', alpha=0.88, label='Peak VRAM (native / B=1)'),
        Patch(fc='#555555', ec='#555555', alpha=0.40, hatch='///', label='Peak VRAM (90% batch-filled)'),
        Patch(fc='#7f8c8d', ec='#333333', alpha=0.65, label='Peak host RAM (RSS)'),
    ]
    ax_mem.legend(handles=mem_handles, loc='lower center', ncol=2, fontsize=7.0, framealpha=0.95, columnspacing=1.0)

    stem = OUT_DIR / '01_existing_benchmark_resources_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.93,
        title='Fixed-Slot Inference Resource Benchmark Across Manuscript Datasets (1× RTX A4500 20 GB, 6 CPUs, 60 GB RAM)',
        note=(
            'Measured on gpu4 (RTX A4500 20 GB fixed slot). Groups: Megascale Test (37 domains, 52k rows), 12 External sets, '
            'Domainome (522 domains), ProteinGym (217 assays), Rocklin (1,000 de novo designs, 210k rows), and MGnify (180,367 scaffolds, 767k rows); DMS excluded. '
            'ESM-MSR Indep. masking (FINAL_p1) achieves an 8× speedup over unmasked Dual-view on Megascale Test and Domainome via single-site state deduplication. '
            'Grey cells indicate runs currently queued on gpu4 (or pending ColabFold MSAs for MutateEverything on MGnify).'
        ),
    )
    return stem


def plot_fig2_proteingym_scaling(pgym_seqlen: dict[str, float]) -> Path:
    """Per-protein length scaling on the 217 ProteinGym assays (ME St1 only; St2 omitted)."""
    pg_data = {}
    present_specs = []
    for s in FIG2_MODELS:
        sc, _ = _load_tracked_dir(s['key'])
        if sc.empty and s.get('fallback_key'):
            sc, _ = _load_tracked_dir(s['fallback_key'])
        if sc.empty:
            continue
        prot = sc[sc['level'] == 'protein'].copy()
        if s['key'].startswith('me_'):
            pg = prot[prot['dataset'].isin(pgym_seqlen.keys())].copy()
            pg['assay'] = pg['dataset']
        else:
            pg = prot[prot['dataset'] == 'proteingym'].copy()
            pg['assay'] = pg['name']
        if pg.empty:
            continue
        if s['key'] == 'thermompnn_s1':
            pg_single = pg[pg['model'] == 'single'].copy()
            pg_all = pg.groupby('assay', as_index=False).agg(
                wall_s=('wall_s', 'sum'),
                n_rows=('n_rows', 'sum'),
                vram_alloc_peak_gb=('vram_alloc_peak_gb', 'max'),
                vram_device_peak_gb=('vram_device_peak_gb', 'max'),
                seq_len=('seq_len', 'first'),
                status=('status', 'first'),
            )
            pg_all['wall_s_single'] = pg_all['assay'].map(pg_single.groupby('assay')['wall_s'].sum())
            pg_all['n_rows_single'] = pg_all['assay'].map(pg_single.groupby('assay')['n_rows'].sum())
            pg = pg_all
        if 'seq_len' not in pg.columns or pg['seq_len'].isna().any():
            pg['seq_len'] = pg['assay'].map(pgym_seqlen)
        pg_data[s['key']] = pg
        present_specs.append(s)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2))
    ax_time, ax_est, ax_vram = axes

    # ---- Panel A: Raw per-assay inference time vs L ----
    for s in present_specs:
        pg = pg_data[s['key']]
        ok = pg[(pg['status'] == 'ok') & pg['seq_len'].notna() & (pg['wall_s'] > 0)].sort_values('seq_len')
        ax_time.scatter(
            ok['seq_len'], ok['wall_s'],
            s=18 if 'esm_msr' in s['key'] else 13,
            marker=s['marker'],
            facecolors='none' if s['hollow'] else s['color'],
            edgecolors=s['color'],
            linewidths=1.1 if s['hollow'] else 0.4,
            alpha=0.70, label=s['line'],
            zorder=5 if 'esm_msr' in s['key'] else 3,
        )
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_time.set_ylabel('Observed ProteinGym assay wall time (s)')
    ax_time.set_title('A. Per-assay inference time (217 ProteinGym assays)', loc='left')
    ax_time.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)

    # ---- Panel B: Estimated full 19L single-mutant screen time ----
    for s in present_specs:
        mode = s.get('est_mode')
        if not mode:
            continue
        pg = pg_data[s['key']]
        ok = pg[(pg['status'] == 'ok') & pg['seq_len'].notna() & (pg['wall_s'] > 0)].copy()
        if mode == 'oneshot':
            ok['est_19L_s'] = ok['wall_s']
            leg = f"{s['line']} (1 pass)"
        elif mode == 'spurs_single':
            ok = ok[ok['t_single_forward_s'].notna() & (ok['t_single_forward_s'] > 0)]
            ok['est_19L_s'] = ok['t_single_forward_s']
            leg = 'SPURS single (1 pass)'
        elif mode == 'linear_single':
            ok = ok[ok['wall_s_single'].notna() & (ok['n_rows_single'] >= 50)]
            ok['est_19L_s'] = ok['wall_s_single'] * (19.0 * ok['seq_len'] / ok['n_rows_single'])
            leg = 'ThermoMPNN single (scaled 19L)'
        elif mode == 'linear':
            ok = ok[ok['n_rows'] >= 50]
            ok['est_19L_s'] = ok['wall_s'] * (19.0 * ok['seq_len'] / ok['n_rows'])
            leg = f"{s['line']} (scaled 19L)"
        ok = ok.sort_values('seq_len')
        ax_est.scatter(
            ok['seq_len'], ok['est_19L_s'], s=13, marker=s['marker'],
            facecolors='none' if s['hollow'] else s['color'], edgecolors=s['color'],
            linewidths=0.9 if s['hollow'] else 0.3, alpha=0.35, zorder=3,
        )
        bins = np.geomspace(32, 3000, 14)
        ok['bin'] = pd.cut(ok['seq_len'], bins)
        med = ok.groupby('bin', observed=True).agg(L=('seq_len', 'median'), t=('est_19L_s', 'median')).dropna()
        ax_est.plot(
            med['L'], med['t'], color=s['color'], ls=s['ls'], lw=s['lw'],
            marker=s['marker'], ms=5.5,
            mfc='white' if s['hollow'] else s['color'], mec=s['color'], mew=1.4 if s['hollow'] else 0.8,
            label=leg, zorder=5 if 'esm_msr' in s['key'] else 4,
        )
    # Also add ESM-MSR Indep. masking single-mutant-only assays (n_rows <= 19*L) scaled to L states if present
    if 'esm_msr_final_p1_s1_indep' in pg_data:
        s_ind = next(x for x in present_specs if x['key'] == 'esm_msr_final_p1_s1_indep')
        pg_i = pg_data['esm_msr_final_p1_s1_indep']
        ok_i = pg_i[(pg_i['status'] == 'ok') & pg_i['seq_len'].notna() & (pg_i['wall_s'] > 0) & (pg_i['n_rows'] <= 19 * pg_i['seq_len'] * 1.05)].copy()
        if len(ok_i):
            bins = np.geomspace(32, 3000, 14)
            ok_i['bin'] = pd.cut(ok_i['seq_len'], bins)
            med_i = ok_i.groupby('bin', observed=True).agg(L=('seq_len', 'median'), t=('wall_s', 'median')).dropna()
            ax_est.plot(
                med_i['L'], med_i['t'], color=s_ind['color'], ls=s_ind['ls'], lw=s_ind['lw'],
                marker=s_ind['marker'], ms=5.5, mfc=s_ind['color'], mec=s_ind['color'],
                label='ESM-MSR Indep. masking (1-mut PG)', zorder=5,
            )

    ax_est.set_xscale('log')
    ax_est.set_yscale('log')
    ax_est.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_est.set_ylabel('Estimated full 1-mut screen (19$L$ variants) time (s)')
    ax_est.set_title('B. Estimated full single-mutant screen time (from PG)', loc='left')
    ax_est.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)
    for yref, txt in [(1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h')]:
        ax_est.axhline(yref, color='#888888', ls=':', lw=0.8, zorder=1)
        ax_est.text(2850, yref * 1.15, txt, fontsize=7.4, color='#666666', ha='right')
    ax_est.legend(loc='upper left', fontsize=6.9, framealpha=0.94)

    # ---- Panel C: Observed peak allocated GPU VRAM vs L ----
    for s in present_specs:
        pg = pg_data[s['key']]
        ok = pg[(pg['status'] == 'ok') & pg['seq_len'].notna() & pg['vram_alloc_peak_gb'].notna()].sort_values('seq_len')
        fail = pg[(pg['status'] != 'ok') & pg['seq_len'].notna()]
        ax_vram.scatter(
            ok['seq_len'], ok['vram_alloc_peak_gb'],
            s=18 if not s['batched'] else 11,
            marker=s['marker'],
            facecolors='none' if s['hollow'] else s['color'],
            edgecolors=s['color'],
            linewidths=1.1 if s['hollow'] else 0.4,
            alpha=0.78 if not s['batched'] else 0.30,
            label=s['line'] + (' [batch-filled]' if s['batched'] else ' [B=1]'),
            zorder=5 if not s['batched'] else 2,
        )
        if len(fail):
            ax_vram.scatter(
                fail['seq_len'], np.full(len(fail), 19.6),
                s=55, marker='x', color=s['color'], linewidths=1.8, zorder=7,
            )
    ax_vram.axhline(20.0, color=style.RING, ls='--', lw=1.2, zorder=6)
    ax_vram.text(34, 20.35, '20 GB RTX A4500 limit (× = OOM failure)', fontsize=7.5,
                 color=style.RING, fontweight='bold', va='bottom')
    ax_vram.set_xscale('log')
    ax_vram.set_ylim(0, 22.2)
    ax_vram.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_vram.set_ylabel('Peak PyTorch allocated VRAM (GB)')
    ax_vram.set_title('C. Observed VRAM vs $L$ (batched models fill card)', loc='left')
    ax_vram.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)
    ax_vram.legend(loc='center left', bbox_to_anchor=(0.01, 0.49), fontsize=6.6, framealpha=0.94, markerscale=1.1)

    for ax in (ax_time, ax_est, ax_vram):
        ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    stem = OUT_DIR / '02_existing_proteingym_scaling_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.88,
        title='Length Scaling Extracted From Existing ProteinGym Fixed-Slot Runs (RTX A4500 20 GB)',
        note=(
            'Per-protein scopes across the 217 ProteinGym assays (L = 37–2,832) on gpu4 (RTX A4500 20 GB). '
            'MutateEverything St2 omitted (St1 shown); ProSST-2048, ESM-MSR Independent masking (FINAL_p1), and ESM-MSR Masked marginal (FINAL_p1_marg) included.'
        ),
    )
    return stem


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp, _ = _load_tracked_dir('spurs_retrain_s1')
    sp_pg = sp[(sp['level'] == 'protein') & (sp['dataset'] == 'proteingym')]
    pgym_seqlen = sp_pg.set_index('name')['seq_len'].to_dict()

    g_df, m_df = collect_fig1_table(set(pgym_seqlen.keys()))
    g_df.to_csv(OUT_DIR / 'manuscript_resources_per_group_a4500.csv', index=False)
    m_df.to_csv(OUT_DIR / 'manuscript_resources_per_model_a4500.csv', index=False)

    s1 = plot_fig1_overview(g_df, m_df)
    s2 = plot_fig2_proteingym_scaling(pgym_seqlen)
    print('Wrote:', s1.with_suffix('.png'), s2.with_suffix('.png'))


if __name__ == '__main__':
    main()
