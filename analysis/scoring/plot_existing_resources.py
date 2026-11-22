"""Visualize the existing fixed-slot RTX A4500 (20 GB) inference resource benchmark data.

Reads:
  - analysis/results/10_inference_resources/{per_run.csv, per_group.csv}
  - docs/benchmarks/inference_resources_a4500.csv
  - <MSR_ROOT>/resources_tracked/<label>/*/resources_scopes.csv

Produces two publication-style figures (PNG + PDF) under
analysis/results/10_inference_resources/:
  1. 01_existing_benchmark_resources_a4500.{png,pdf}
     Overview of full-benchmark wall time, per-group breakdown, throughput (rows/s),
     and memory / model-load footprint across all 7 fixed-slot A4500 runs.
  2. 02_existing_proteingym_scaling_a4500.{png,pdf}
     Per-protein scaling with sequence length L across the 217 ProteinGym assays
     (raw per-protein wall time, estimated full single-mutant 19L screen time,
     and peak allocated VRAM highlighting B=1 vs batch-filled models and OOM limits).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Canonical model ordering & styling aligned with msr_analysis.models.ROSTER
MODEL_SPECS = [
    dict(
        label='esm_msr_final_p0_wt_s1',
        short='ESM-MSR\n(WT-view)',
        line='ESM-MSR (WT-view)',
        color=BY_NAME['ESM-MSR'].color,
        marker='D',
        hollow=True,
        ls='--',
        lw=2.2,
        batched=False,
        doubles_capable=False,
    ),
    dict(
        label='esm_msr_devel_s1',
        short='ESM-MSR\n(Dual-view)',
        line='ESM-MSR (Dual-view)',
        color=BY_NAME['ESM-MSR'].color,
        marker='D',
        hollow=False,
        ls='-',
        lw=2.5,
        batched=True,
        doubles_capable=True,
    ),
    dict(
        label='spurs_retrain_s1',
        short='SPURS\n(+multi)',
        line='SPURS (+multi)',
        color=BY_NAME['SPURS'].color,
        marker='s',
        hollow=False,
        ls='-',
        lw=2.0,
        batched=False,
        doubles_capable=True,
    ),
    dict(
        label='proteinmpnn',
        short='ProteinMPNN',
        line='ProteinMPNN',
        color=BY_NAME['ProteinMPNN'].color,
        marker='d',
        hollow=False,
        ls='-.',
        lw=2.0,
        batched=True,
        doubles_capable=True,
    ),
    dict(
        label='thermompnn_s1',
        short='ThermoMPNN\n(+D)',
        line='ThermoMPNN (+D)',
        color=BY_NAME['ThermoMPNN'].color,
        marker='p',
        hollow=False,
        ls='-',
        lw=2.0,
        batched=True,
        doubles_capable=True,
    ),
    dict(
        label='me_st1_s1_a4500',
        short='MutateEverything\n(Stage 1)',
        line='MutateEverything (St1)',
        color=BY_NAME['MutateEverything'].color,
        marker='h',
        hollow=True,
        ls='--',
        lw=1.9,
        batched=False,
        doubles_capable=False,
    ),
    dict(
        label='me_st2_s1',
        short='MutateEverything\n(Stage 2)',
        line='MutateEverything (St2)',
        color=BY_NAME['MutateEverything'].color,
        marker='h',
        hollow=False,
        ls='-',
        lw=2.0,
        batched=False,
        doubles_capable=True,
    ),
]
SPEC_BY_LABEL = {s['label']: s for s in MODEL_SPECS}
GROUPS = ['Tsuboyama', 'External', 'DMS', 'Domainome', 'ProteinGym']


def fmt_time_s(sec: float) -> str:
    if not np.isfinite(sec):
        return '—'
    if sec < 60:
        return f'{sec:.1f} s'
    if sec < 3600:
        return f'{sec / 60:.1f} m'
    return f'{sec / 3600:.2f} h'


def load_scopes_for_label(label: str) -> pd.DataFrame:
    d = TRACKED_ROOT / label
    csvs = sorted(d.glob('*/resources_scopes.csv'))
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def plot_fig1_overview(doc_df: pd.DataFrame, pg_df: pd.DataFrame) -> Path:
    """2x2 overview of the fixed-slot RTX A4500 (20 GB) benchmark resource runs."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    ax_tot, ax_hm, ax_rps, ax_mem = axes.ravel()

    # Order models by total inference_h (fastest -> slowest) for Panel A, keep consistent order across panels
    doc_idx = doc_df.set_index('label')
    ordered_specs = sorted(MODEL_SPECS, key=lambda s: float(doc_idx.loc[s['label'], 'inference_h']))
    labels_ord = [s['label'] for s in ordered_specs]
    y_pos = np.arange(len(ordered_specs))

    # ---- Panel A: Total full-benchmark wall time (hours, log10 scale) ----
    dual_h = float(doc_idx.loc['esm_msr_devel_s1', 'inference_h'])
    for i, s in enumerate(ordered_specs):
        h = float(doc_idx.loc[s['label'], 'inference_h'])
        n_fail = int(doc_idx.loc[s['label'], 'n_failed'])
        hatched = s['hollow']
        ax_tot.barh(
            i, h, height=0.62,
            color=s['color'],
            alpha=0.45 if hatched else 0.92,
            edgecolor=s['color'] if hatched else 'black',
            hatch='////' if hatched else None,
            lw=1.0, zorder=3,
        )
        ratio = dual_h / h
        tag = f'{h:.2f} h'
        if s['label'] == 'esm_msr_final_p0_wt_s1':
            tag += f'  ({ratio:.0f}× faster than dual-view)'
        elif s['label'] != 'esm_msr_devel_s1':
            tag += f'  ({ratio:.1f}×)'
        else:
            tag += '  (1.0× ref)'
        if n_fail > 0:
            tag += f'  [{n_fail} OOM]'
        ax_tot.text(h * 1.15, i, tag, va='center', ha='left', fontsize=8.2,
                    fontweight='bold' if 'esm_msr' in s['label'] else 'normal', color='#222222')

    ax_tot.set_xscale('log')
    ax_tot.set_xlim(0.05, 500)
    ax_tot.set_yticks(y_pos)
    ax_tot.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.8)
    ax_tot.invert_yaxis()
    ax_tot.set_xlabel('Total inference wall time across all 5 benchmark groups (hours, log scale)')
    ax_tot.set_title('A. Total benchmark inference time (RTX A4500 20 GB)', loc='left')
    ax_tot.grid(axis='x', color=style.GRID, lw=0.6, which='major', zorder=0)

    # ---- Panel B: Per-group wall time heatmap (log10 seconds, annotated with exact durations) ----
    sub_pg = pg_df[pg_df['label'].isin(labels_ord)]
    piv_s = sub_pg.pivot(index='label', columns='group', values='wall_h').loc[labels_ord, GROUPS] * 3600.0
    mat_log = np.log10(np.maximum(piv_s.to_numpy(dtype=float), 0.5))
    im = ax_hm.imshow(mat_log, aspect='auto', cmap='YlOrRd', vmin=0.3, vmax=5.2)
    for i in range(len(labels_ord)):
        for j in range(len(GROUPS)):
            val = piv_s.iloc[i, j]
            txt = fmt_time_s(val)
            dark = mat_log[i, j] > 3.2
            ax_hm.text(j, i, txt, ha='center', va='center', fontsize=8.0,
                       color='white' if dark else '#111111',
                       fontweight='bold' if val > 3600 else 'normal')
    ax_hm.set_xticks(range(len(GROUPS)))
    ax_hm.set_xticklabels(GROUPS, fontsize=8.8, rotation=20, ha='right')
    ax_hm.set_yticks(range(len(labels_ord)))
    ax_hm.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.4)
    ax_hm.set_title('B. Inference time by benchmark group (seconds / min / h)', loc='left')
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.042, pad=0.02)
    cb.set_label('log₁₀(wall time [s])', fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # ---- Panel C: Throughput (rows/s) by benchmark group ----
    piv_rps = sub_pg.pivot(index='label', columns='group', values='rows_per_s').loc[labels_ord, GROUPS]
    x_g = np.arange(len(GROUPS))
    style.band_axis(ax_rps, len(GROUPS))
    offs = np.linspace(-0.30, 0.30, len(ordered_specs))
    for k, s in enumerate(ordered_specs):
        ys = piv_rps.loc[s['label'], GROUPS].to_numpy(dtype=float)
        mk = dict(
            marker=s['marker'], ms=7.2, color=s['color'], mec=s['color'],
            mfc='white' if s['hollow'] else s['color'], mew=1.8 if s['hollow'] else 1.0,
            ls='-', lw=1.1, alpha=0.92,
        )
        ax_rps.plot(x_g + offs[k], ys, label=s['line'], zorder=4 if 'esm_msr' in s['label'] else 3, **mk)
    ax_rps.set_yscale('log')
    ax_rps.set_ylim(0.12, 2.0e5)
    ax_rps.set_xticks(x_g)
    ax_rps.set_xticklabels(GROUPS, fontsize=8.8, rotation=15, ha='right')
    ax_rps.set_ylabel('Throughput (variants / second, log scale)')
    ax_rps.set_title('C. Scoring throughput across benchmark groups', loc='left')
    ax_rps.grid(axis='y', color=style.GRID, lw=0.6, which='major', zorder=0)
    ax_rps.legend(loc='lower left', ncol=2, fontsize=7.4, framealpha=0.94, handlelength=1.8, columnspacing=1.0)

    # ---- Panel D: Peak Device VRAM vs Peak Host RAM (with batch-policy caveat) ----
    vram_dev = [float(doc_idx.loc[s['label'], 'vram_device_peak_gb']) for s in ordered_specs]
    ram_pk = [float(doc_idx.loc[s['label'], 'ram_peak_gb']) for s in ordered_specs]
    load_s = [float(doc_idx.loc[s['label'], 'model_load_s']) for s in ordered_specs]

    bar_h = 0.36
    for i, s in enumerate(ordered_specs):
        ax_mem.barh(
            i - bar_h / 2, vram_dev[i], height=bar_h,
            color=s['color'], alpha=0.88 if not s['batched'] else 0.40,
            edgecolor=s['color'] if s['batched'] else 'black',
            hatch='///' if s['batched'] else None, lw=0.9, zorder=3,
        )
        ax_mem.barh(
            i + bar_h / 2, ram_pk[i], height=bar_h,
            color='#7f8c8d', alpha=0.65, edgecolor='#333333', lw=0.7, zorder=3,
        )
        v_note = f'{vram_dev[i]:.1f} GB' + (' (batch-filled)' if s['batched'] else ' (B=1 / native)')
        ax_mem.text(vram_dev[i] + 0.35, i - bar_h / 2, v_note, va='center', fontsize=7.4, color='#222222')
        ax_mem.text(ram_pk[i] + 0.35, i + bar_h / 2, f'{ram_pk[i]:.1f} GB  (load {load_s[i]:.0f} s)',
                    va='center', fontsize=7.2, color='#555555')

    ax_mem.axvline(20.0, color=style.RING, ls='--', lw=1.1, zorder=4)
    ax_mem.text(20.4, 1.0, '20 GB A4500\ncap', ha='left', va='center',
                fontsize=7.5, color=style.RING, fontweight='bold')
    ax_mem.set_xlim(0, 32.0)
    ax_mem.set_ylim(len(ordered_specs) + 0.95, -0.55)
    ax_mem.set_yticks(y_pos)
    ax_mem.set_yticklabels([s['line'] for s in ordered_specs], fontsize=8.4)
    ax_mem.set_xlabel('Peak memory across full benchmark run (GB)')
    ax_mem.set_title('D. Peak GPU VRAM & Host RAM (hatched = 90% batch-filled)', loc='left')
    ax_mem.grid(axis='x', color=style.GRID, lw=0.6, zorder=0)
    from matplotlib.patches import Patch
    mem_handles = [
        Patch(fc='#555555', ec='black', alpha=0.88, label='Peak VRAM (native / B=1)'),
        Patch(fc='#555555', ec='#555555', alpha=0.40, hatch='///', label='Peak VRAM (90% batch-filled)'),
        Patch(fc='#7f8c8d', ec='#333333', alpha=0.65, label='Peak host RAM (RSS)'),
    ]
    ax_mem.legend(handles=mem_handles, loc='lower center', ncol=2, fontsize=7.2, framealpha=0.95, columnspacing=1.0)

    stem = OUT_DIR / '01_existing_benchmark_resources_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.93,
        title='Fixed-Slot Inference Resource Benchmark Across All Datasets (1× RTX A4500 20 GB, 6 CPUs, 60 GB RAM)',
        note=(
            'Source: docs/benchmarks/inference_resources_a4500.csv and analysis/results/10_inference_resources/per_group.csv '
            '(gpu4 fixed-slot runs, slot_ok=True). ESM-MSR WT-view (--skip_reverse) scores the full benchmark in 0.10 h '
            '(348× faster than ESM-MSR Dual-view at 34.82 h; 1,233× faster on ProteinGym). Peak VRAM of batched models '
            '(ESM-MSR Dual-view, ThermoMPNN, ProteinMPNN) reflects --batch_mode table filling ~90% of VRAM, not minimum memory.'
        ),
    )
    return stem


def plot_fig2_proteingym_scaling() -> Path:
    """Per-protein scaling with sequence length L from existing 217 ProteinGym A4500 runs."""
    # Reference ProteinGym lengths & row counts from SPURS (all 217 succeeded)
    sp = load_scopes_for_label('spurs_retrain_s1')
    sp_pg = sp[(sp['level'] == 'protein') & (sp['dataset'] == 'proteingym')].copy()
    seqlen_map = sp_pg.set_index('name')['seq_len'].to_dict()
    nrows_map = sp_pg.set_index('name')['n_rows'].to_dict()

    # Load ProteinGym protein scopes for all 7 models
    pg_data = {}
    for s in MODEL_SPECS:
        lab = s['label']
        df = load_scopes_for_label(lab)
        prot = df[df['level'] == 'protein'].copy()
        if lab.startswith('me_'):
            # ME stores assay name in `dataset`, protein id in `name`
            pg = prot[prot['dataset'].isin(seqlen_map.keys())].copy()
            pg['assay'] = pg['dataset']
        else:
            pg = prot[prot['dataset'] == 'proteingym'].copy()
            pg['assay'] = pg['name']
        if lab == 'thermompnn_s1':
            # Aggregate ThermoMPNN single + epistatic per assay (and keep single separately for 1-mut estimate)
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
            pg['seq_len'] = pg['assay'].map(seqlen_map)
        pg_data[lab] = pg

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    ax_time, ax_est, ax_vram = axes

    # ---- Panel A: Raw per-protein inference wall time vs L (ProteinGym) ----
    for s in MODEL_SPECS:
        lab = s['label']
        pg = pg_data[lab]
        ok = pg[(pg['status'] == 'ok') & pg['seq_len'].notna() & (pg['wall_s'] > 0)].sort_values('seq_len')
        ax_time.scatter(
            ok['seq_len'], ok['wall_s'],
            s=18 if 'esm_msr' in lab else 13,
            marker=s['marker'],
            facecolors='none' if s['hollow'] else s['color'],
            edgecolors=s['color'],
            linewidths=1.1 if s['hollow'] else 0.4,
            alpha=0.72,
            label=s['line'],
            zorder=5 if 'esm_msr' in lab else 3,
        )
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_time.set_ylabel('Observed ProteinGym assay wall time (s)')
    ax_time.set_title('A. Per-assay inference time (217 ProteinGym assays)', loc='left')
    ax_time.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)

    # ---- Panel B: Estimated full single-mutant (19L) screen time from existing A4500 data ----
    # Note: One-shot models (ESM-MSR WT, ME St1/St2, SPURS single phase) score all 19L in 1 pass;
    # per-variant models (ESM-MSR Dual, ThermoMPNN single) scale ~linearly with n_rows -> wall_s * (19L / n_rows).
    # For ProteinMPNN, singles take L unique decoder rows, whereas PG assays have mixed singles/multis;
    # we only plot models whose 19L screen time is cleanly estimable from single-pass or linear per-variant PG runs.
    est_specs = [
        ('esm_msr_final_p0_wt_s1', 'ESM-MSR WT-view (1 pass)', 'oneshot'),
        ('spurs_retrain_s1', 'SPURS single (1 pass, t_single)', 'spurs_single'),
        ('me_st1_s1_a4500', 'MutateEverything St1 (1 pass)', 'oneshot'),
        ('me_st2_s1', 'MutateEverything St2 (1 pass)', 'oneshot'),
        ('thermompnn_s1', 'ThermoMPNN single (scaled to 19L)', 'linear_single'),
        ('esm_msr_devel_s1', 'ESM-MSR Dual-view (scaled to 19L)', 'linear'),
    ]
    for lab, leg_lbl, mode in est_specs:
        s = SPEC_BY_LABEL[lab]
        pg = pg_data[lab]
        ok = pg[(pg['status'] == 'ok') & pg['seq_len'].notna() & (pg['wall_s'] > 0)].copy()
        if mode == 'oneshot':
            ok['est_19L_s'] = ok['wall_s']
        elif mode == 'spurs_single':
            ok = ok[ok['t_single_forward_s'].notna() & (ok['t_single_forward_s'] > 0)]
            ok['est_19L_s'] = ok['t_single_forward_s']
        elif mode == 'linear_single':
            ok = ok[ok['wall_s_single'].notna() & (ok['n_rows_single'] >= 50)]
            ok['est_19L_s'] = ok['wall_s_single'] * (19.0 * ok['seq_len'] / ok['n_rows_single'])
        elif mode == 'linear':
            ok = ok[ok['n_rows'] >= 50]
            ok['est_19L_s'] = ok['wall_s'] * (19.0 * ok['seq_len'] / ok['n_rows'])
        ok = ok.sort_values('seq_len')
        ax_est.scatter(
            ok['seq_len'], ok['est_19L_s'],
            s=14, marker=s['marker'],
            facecolors='none' if s['hollow'] else s['color'],
            edgecolors=s['color'],
            linewidths=0.9 if s['hollow'] else 0.3,
            alpha=0.40, zorder=3,
        )
        # Binned median trend line (log-spaced length bins)
        bins = np.geomspace(35, 3000, 14)
        ok['bin'] = pd.cut(ok['seq_len'], bins)
        med = ok.groupby('bin', observed=True).agg(L=('seq_len', 'median'), t=('est_19L_s', 'median')).dropna()
        ax_est.plot(
            med['L'], med['t'],
            color=s['color'], ls=s['ls'], lw=s['lw'],
            marker=s['marker'], ms=5.5,
            mfc='white' if s['hollow'] else s['color'], mec=s['color'], mew=1.4 if s['hollow'] else 0.8,
            label=leg_lbl, zorder=5 if 'esm_msr' in lab else 4,
        )
    ax_est.set_xscale('log')
    ax_est.set_yscale('log')
    ax_est.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_est.set_ylabel('Estimated full 1-mut screen (19$L$ variants) time (s)')
    ax_est.set_title('B. Estimated full single-mutant screen time (from PG data)', loc='left')
    ax_est.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)
    # Reference time lines (1 s, 1 min, 1 h)
    for yref, txt in [(1.0, '1 s'), (60.0, '1 min'), (3600.0, '1 h')]:
        ax_est.axhline(yref, color='#888888', ls=':', lw=0.8, zorder=1)
        ax_est.text(2850, yref * 1.15, txt, fontsize=7.5, color='#666666', ha='right')
    ax_est.legend(loc='upper left', fontsize=7.1, framealpha=0.94)

    # ---- Panel C: Peak allocated GPU VRAM vs L (ProteinGym) ----
    for s in MODEL_SPECS:
        lab = s['label']
        pg = pg_data[lab]
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
    ax_vram.text(40, 20.35, '20 GB RTX A4500 limit (× = OOM failure)', fontsize=7.6,
                 color=style.RING, fontweight='bold', va='bottom')
    ax_vram.set_xscale('log')
    ax_vram.set_ylim(0, 22.2)
    ax_vram.set_xlabel('Protein length $L$ (residues, log scale)')
    ax_vram.set_ylabel('Peak PyTorch allocated VRAM (GB)')
    ax_vram.set_title('C. Observed VRAM vs $L$ (batched models fill card)', loc='left')
    ax_vram.grid(True, color=style.GRID, lw=0.6, which='major', zorder=0)
    ax_vram.legend(loc='center left', bbox_to_anchor=(0.01, 0.49), fontsize=6.7, framealpha=0.94, markerscale=1.1)

    for ax in (ax_time, ax_est, ax_vram):
        ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    stem = OUT_DIR / '02_existing_proteingym_scaling_a4500'
    style.finish_figure(
        fig, stem=stem, top=0.91,
        title='Length Scaling Extracted From Existing ProteinGym Fixed-Slot Runs (RTX A4500 20 GB)',
        note=(
            'Per-protein scopes from ../resources_tracked/ (217 ProteinGym assays, L = 37–2,832). '
            'Panel B estimates full 19L single-mutant screening time directly from one-shot wall times (WT-view, SPURS, ME) '
            'or linear scaling of per-variant assays (ESM-MSR Dual-view, ThermoMPNN single); ProteinMPNN and double-mutant '
            'epistatic costs require dedicated controlled tests. Panel C shows true B=1 VRAM for ESM-MSR WT-view, ME, and SPURS, '
            'whereas batched models (faded) fill the 20 GB card—motivating controlled B=1 scaling benchmarks.'
        ),
    )
    return stem


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc_df = pd.read_csv(_REPO_ROOT / 'docs' / 'benchmarks' / 'inference_resources_a4500.csv')
    pg_df = pd.read_csv(OUT_DIR / 'per_group.csv')
    s1 = plot_fig1_overview(doc_df, pg_df)
    s2 = plot_fig2_proteingym_scaling()
    print('Wrote:', s1.with_suffix('.png'), s2.with_suffix('.png'))


if __name__ == '__main__':
    main()
