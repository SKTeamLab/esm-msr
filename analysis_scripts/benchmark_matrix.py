"""
The all-by-all benchmark matrix behind docs/BENCHMARKS_OCT2026.md: every model scored on every dataset, gathered
from wherever each run left its results on the subhak-gpu cluster, in one long table (model, dataset, rho, sd,
n_seeds, source) and rendered as the doc's markdown tables.

Run on the cluster (it reads the result trees in place); pandas only, CPU, about a minute.

  python analysis_scripts/benchmark_matrix.py collect --out matrix_long.csv     # gather + score
  python analysis_scripts/benchmark_matrix.py tables matrix_long.csv           # markdown for the doc
  sacct -u sreeves -S 2026-10-01 -X -P -o JobID,JobName%40,NodeList,Start,End,Elapsed,State > sacct.txt
  python analysis_scripts/benchmark_matrix.py ledger sacct.txt --out slurm_jobs.csv                # gpu4 ledger

Metric conventions, the same as esm_msr_testing.update_stats and the stability-baselines harness: external sets and
DMS are one pooled Spearman per set; Tsuboyama and Domainome are the mean over proteins of the per-protein Spearman,
Tsuboyama on MEASURED rows only (cens == 0). ProteinGym: per-DMS Spearman, on the 213 DMS every model scored, as the
mean of the five category means (PG avg), the Stability category, and the plain mean (DMS mean). Higher = more
stable / fitter for every model.

Sources:
  ESM-MSR (capped, dc8, ε=0)      esm-msr-devel/analysis_notebooks/stats/<set>/<ckpt dir>/<ckpt>_epsilon<ε>_unmasked_<head>[_avg].csv
  MutateEverything / ThermoMPNN   esm-msr-devel/outputs/baselines_capped/{combined_*,final_*}_table.csv (3-seed means)
  third-party (old runs)          stability-baselines/results/FINAL_TABLE.csv (Tsuboyama on hyperopt_splits)
  third-party (capped, ProteinGym), SPURS / ProteinMPNN gamut
                                  stability-baselines/results/<model>/predictions[_tag]/<dataset>.csv, scored here
  ESM3 Forge zero-shot            esm-msr-devel/analysis_notebooks/stats/esm3_forge/zero_shot/ (pooled sets only)
  ProteinGym                      esm-msr-devel/outputs/pgym/
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

DEV = '/home/sreeves/esm-msr-devel'
SB = '/home/sreeves/stability-baselines'
EXT = ['s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld']
DMS = ['DLG4_HUMAN_Faure_2021_abundance_domain', 'DLG4_HUMAN_Faure_2021_binding_domain',
       'GRB2_HUMAN_Faure_2021_abundance_domain', 'GRB2_HUMAN_Faure_2021_binding_domain',
       'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain']
# >1,800 residues: out of memory for at least one model, so left out of every ProteinGym comparison
PG_DROP = {'BRCA1_HUMAN_Findlay_2018', 'SCN5A_HUMAN_Glazer_2019', 'POLG_CXB3N_Mattenberger_2021',
           'BRCA2_HUMAN_Erwood_2022_HEK293T'}
CAPPED = {'splits_oct06_capped-val': 'tsu_capped_val', 'splits_oct06_capped-test': 'tsu_capped_test'}


class Rows(list):
    def add(self, model, ds, v, src, n_seeds=1, sd=np.nan):
        if v is None or pd.isna(v):
            return
        self.append(dict(model=model, dataset=ds, rho=float(v), sd=sd, n_seeds=n_seeds, source=src))


def spearman(a, b):
    v = pd.DataFrame({'a': a, 'b': b}).dropna()
    return v.corr('spearman').iloc[0, 1] if len(v) > 2 else np.nan


def pg_categories():
    pg = pd.read_csv(f'{DEV}/outputs/pgym/pgym_all_models_per_dms.csv')
    return pg.set_index('DMS_id')['category']


def pg_summaries(rows, model, per_dms, src, n_seeds=1):
    """per_dms: Series DMS_id -> rho. Adds PG avg / PG Stability / PG DMS mean when all 213 common DMS are present."""
    cat = pg_categories()
    keep = cat.index[~cat.index.isin(PG_DROP)]
    s = per_dms.reindex(keep)
    if s.notna().sum() < len(keep):
        print(f'[pg] {model}: {s.notna().sum()}/{len(keep)} DMS, not summarised')
        return
    g = s.groupby(cat.reindex(keep)).mean()
    rows.add(model, 'PG avg', g.mean(), src, n_seeds)
    rows.add(model, 'PG Stability', g['Stability'], src, n_seeds)
    rows.add(model, 'PG DMS mean', s.mean(), src, n_seeds)


def collect_msr(rows, ckpt_dirs, model, eps='1'):
    stats = f'{DEV}/analysis_notebooks/stats'
    for head, label in [('Combined', ''), ('WT_LoRA', ' WT'), ('MT_LoRA', ' MT')]:
        acc = {}
        for ck in ckpt_dirs:
            for sub, names in [('external', EXT), ('DMS', DMS)]:
                f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}.csv')
                if f:
                    d = pd.read_csv(f[0], index_col=0)
                    for ds in names:
                        if ds in d.index:
                            acc.setdefault(ds, []).append(d.loc[ds, 'spearman_all'])
            for sub, ds in [('domainome', 'domainome'), ('splits_oct06_capped-val', 'tsu_capped_val'),
                            ('splits_oct06_capped-test', 'tsu_capped_test')]:
                f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}_avg.csv')
                if f:
                    acc.setdefault(ds, []).append(pd.read_csv(f[0], index_col=0).loc['spearman_all'].iloc[0])
        for ds, v in acc.items():
            rows.add(model + label, ds, np.mean(v), 'esm_msr_testing stats', len(v), np.std(v) if len(v) > 1 else np.nan)


def collect_retrained(rows):
    o6 = f'{DEV}/outputs/baselines_capped'
    ext = pd.read_csv(f'{o6}/combined_external_table.csv', index_col=0)
    for c, m in [('ME stage1', 'ME stage1'), ('ME stage2', 'ME stage2'), ('TMPNN add', 'TMPNN'), ('TMPNN comp', 'TMPNN+D')]:
        for ds in ext.index:
            rows.add(m, ds, ext.loc[ds, c], 'combined_external_table', 3, ext.loc[ds, c + ' sd'])
    dms = pd.read_csv(f'{o6}/final_dms_table.csv', header=[0, 1], index_col=0).loc['spearman_all']
    for (ds, c), v in dms.items():
        if not c.endswith(' sd') and not c.startswith('MSR'):
            rows.add(c, ds, float(v), 'final_dms_table', 3, float(dms.get((ds, c + ' sd'), np.nan)))
    dom = pd.read_csv(f'{o6}/final_domainome_table.csv', index_col=0).loc['spearman_all']
    for c in ['TMPNN', 'ME stage1', 'ME stage2']:
        rows.add(c, 'domainome', dom[c], 'final_domainome_table', 3, dom[c + ' sd'])
    for split, f in [('tsu_capped_val', 'combined_val_table.csv'), ('tsu_capped_test', 'combined_test_table.csv')]:
        d = pd.read_csv(f'{o6}/{f}', index_col=0).loc['spearman_all']
        for c in ['ME stage1', 'ME stage2', 'TMPNN', 'TMPNN+D']:
            rows.add(c, split, d[c], f, 3, d[c + ' sd'])


THIRD = {'ESMtherm': 'ESMtherm', 'DDGemb (server)': 'DDGemb', 'DDMut (server)': 'DDMut',
         'IFUM (WT backbone)': 'IFUM (WT backbone)', 'IFUM (native, ESMFold/variant)': 'IFUM (native, ESMFold)',
         'GeoDDG-3D (FASPR)': 'GeoDDG-3D', 'ProSST (zero-shot)': 'ProSST',
         'ESM3dG (ens.; trained on Megascale)': 'ESM3dG', 'ref: esm-msr-small, crystal': 'ESM-MSR-small (March, crystal)'}


def collect_final_table(rows):
    ft = pd.read_csv(f'{SB}/results/FINAL_TABLE.csv').set_index('dataset')
    ft = ft.rename(index=lambda s: 'tsu_hyperopt_val' if s.startswith('Tsuboyama val') else
                   ('tsu_hyperopt_test' if s.startswith('Tsuboyama test') else s))
    for c, m in THIRD.items():
        for ds in ft.index:
            if ds.startswith('MEAN'):
                continue
            v, cov = ft.loc[ds, c], ft.loc[ds, c + ' cov']
            if pd.notna(v):
                full = pd.isna(cov) or cov >= 0.995
                rows.add(m if full else m + ' [partial]', ds, v, 'FINAL_TABLE' + ('' if full else f' cov={cov:.2f}'))


def collect_forge(rows):
    for size in ['small-2024-08', 'medium-2024-08', 'large-2024-03']:
        for f in ['external', 'dms']:
            p = f'{DEV}/analysis_notebooks/stats/esm3_forge/zero_shot/esm3-{size}/{f}_global_stats.csv'
            if os.path.exists(p):
                d = pd.read_csv(p, index_col=0)
                for ds in d.index:
                    rows.add(f'ESM3-{size.split("-")[0]} zero-shot', {'PTMUL': 'ptmul'}.get(ds, ds),
                             d.loc[ds, 'spearman_all'], 'esm3_forge stats')


def collect_pgym(rows):
    pg = pd.read_csv(f'{DEV}/outputs/pgym/pgym_all_models_per_dms.csv').set_index('DMS_id')
    cols = {'ESM-MSR full': ('ESM-MSR capped', 1), 'ESM-MSR WT eps=1.0': ('ESM-MSR capped WT', 1),
            'ESM-MSR WT eps=0': ('ESM3-small-open (ε=0, MSR readout)', 1),
            'SPURS single-additive': ('SPURS (single, additive)', 1), 'SPURS multi': ('SPURS (multi)', 1),
            'MutateEverything single': ('ME stage1', 3), 'MutateEverything double': ('ME stage2', 3),
            'ThermoMPNN': ('TMPNN', 3), 'ProteinMPNN': ('ProteinMPNN', 3)}
    for c, (m, n) in cols.items():
        pg_summaries(rows, m, pg[c], 'pgym_all_models_per_dms', n)
    per = []
    for s in (1, 2, 3):
        f = f'{DEV}/outputs/pgym/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{s}/sigma1.0/summary.csv'
        d = pd.read_csv(f).drop_duplicates('DMS_id', keep='last')
        per.append(d[d.status == 'done'].set_index('DMS_id').spearman_combined)
    pg_summaries(rows, 'ESM-MSR dc8', pd.concat(per, axis=1).mean(axis=1), 'pgym summary.csv', 3)


# harness runs added for the all-by-all (2026-10-21): model dir, prediction tag, {readout column: model label}
HARNESS = [('esmtherm', '', {'pred': 'ESMtherm'}), ('ifum', '', {'pred': 'IFUM (WT backbone)'}),
           ('geoddg', '_faspr', {'pred': 'GeoDDG-3D'}), ('esm3dg', '', {'pred': 'ESM3dG'}), ('prosst', '', {'pred': 'ProSST'}),
           ('spurs', '', {'pred': 'SPURS (multi)', 'pred_additive': 'SPURS (single, additive)'}),
           ('proteinmpnn', '', {'pred': 'ProteinMPNN'}), ('ddgemb', '', {'pred': 'DDGemb'})]


def harness_frame(model, tag, name):
    p = f'{SB}/results/{model}/predictions{tag}/{name}.csv'
    t = f'{SB}/tasks/{name}.csv.gz'
    if not (os.path.exists(p) and os.path.exists(t)):
        return None
    task = pd.read_csv(t, keep_default_na=False, na_values=[''], low_memory=False)
    pred = pd.read_csv(p)
    return task.merge(pred[['row_id', 'pred', 'pred_additive']], on='row_id', how='left')


def collect_harness(rows, done_by_old_runs):
    for model, tag, readouts in HARNESS:
        for col, label in readouts.items():
            for name in EXT + DMS + ['domainome']:
                if (label, name) in done_by_old_runs:
                    continue
                d = harness_frame(model, tag, name)
                if d is None or d[col].isna().all():
                    continue
                if name == 'domainome':
                    v = np.nanmean([spearman(g[col], g['truth']) for _, g in d.groupby('group')])
                else:
                    v = spearman(d[col], d['truth'])
                rows.add(label, name, v, f'harness {model}{tag}')
            for name, ds in CAPPED.items():
                d = harness_frame(model, tag, name)
                if d is None:
                    continue
                d = d[d['cens'] == 0]   # measured rows only, as reported for ESM-MSR
                per = [spearman(g[col], g['truth']) for _, g in d.groupby('code_wt')]
                per = [v for v in per if pd.notna(v)]
                cov = d[col].notna().mean()
                # a model that cannot represent mutant backgrounds (SPURS, ProteinMPNN) is scored on the rest
                rows.add(label, ds, np.mean(per) if per else np.nan, f'harness {model}{tag} cov={cov:.2f}')
            pg = {}
            for f in glob.glob(f'{SB}/results/{model}/predictions{tag}/pg_*.csv'):
                name = os.path.basename(f)[:-4]
                d = harness_frame(model, tag, name)
                if d is not None:
                    pg[name[3:]] = spearman(d[col], d['truth'])
            if pg and label not in {'SPURS (multi)', 'SPURS (single, additive)', 'ProteinMPNN'}:
                pg_summaries(rows, label, pd.Series(pg), f'harness {model}{tag}')


def cmd_collect(a):
    rows = Rows()
    collect_msr(rows, ['esm-msr-baseline-w2a4-m16a16-capped'], 'ESM-MSR capped')
    collect_msr(rows, [f'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{i}' for i in (1, 2, 3)], 'ESM-MSR dc8')
    collect_msr(rows, ['esm-msr-baseline-w2a4-m16a16-capped'], 'ESM3-small-open (ε=0, MSR readout)', eps='0.0')
    collect_retrained(rows)
    collect_final_table(rows)
    collect_forge(rows)
    collect_pgym(rows)
    collect_harness(rows, {(r['model'], r['dataset']) for r in rows})
    L = pd.DataFrame(rows)
    L.to_csv(a.out, index=False)
    print(L.groupby('model').dataset.count().sort_values().to_string())


# ---------------------------------------------------------------------------------------------- tables
ROWS = [('ESM-MSR capped', 'ESM-MSR capped (combined)'), ('ESM-MSR capped WT', 'ESM-MSR capped, WT head'),
        ('ESM-MSR capped MT', 'ESM-MSR capped, MT head'), ('ESM-MSR dc8', 'ESM-MSR dc8 (combined, 3 seeds)'),
        ('ESM-MSR dc8 WT', 'ESM-MSR dc8, WT head'), ('ESM-MSR dc8 MT', 'ESM-MSR dc8, MT head'),
        ('ME stage1', 'MutateEverything st1 (3 seeds)'), ('ME stage1+', 'MutateEverything st1, additive'),
        ('ME stage2', 'MutateEverything st2 (3 seeds)'), ('ME stage2+', 'MutateEverything st2, additive'),
        ('TMPNN', 'ThermoMPNN (additive, 3 seeds)'), ('TMPNN+D', 'ThermoMPNN + ThermoMPNN-D (3 seeds)'),
        ('SPURS (single, additive)', 'SPURS single-additive ‡'), ('SPURS (multi)', 'SPURS multi ‡'),
        ('ESMtherm', 'ESMtherm ‡'), ('DDGemb', 'DDGemb (server)'), ('DDMut', 'DDMut (server)'),
        ('IFUM (WT backbone)', 'IFUM, WT backbone ‡'), ('IFUM (native, ESMFold)', 'IFUM, ESMFold per variant ‡'),
        ('GeoDDG-3D', 'GeoDDG-3D (FASPR)'), ('ESM3dG', 'ESM3ΔG ‡'), ('ProSST', 'ProSST (zero-shot)'),
        ('ProteinMPNN', 'ProteinMPNN (zero-shot)'), ('ESM3-small-open (ε=0, MSR readout)', 'ESM3-small-open, ε = 0 (zero-shot)'),
        ('ESM3-large zero-shot', 'ESM3-large (Forge, zero-shot)'), ('ESM3-medium zero-shot', 'ESM3-medium (Forge, zero-shot)'),
        ('ESM-MSR-small (March, crystal)', 'ESM-MSR-small (March reference)')]
TSU_ALL = ['tsu_capped_val', 'tsu_capped_test', 'tsu_hyperopt_val', 'tsu_hyperopt_test']
PG = ['PG avg', 'PG Stability', 'PG DMS mean']
_FULL = EXT + DMS + ['domainome', 'tsu_capped_val', 'tsu_capped_test']
# cells a run queued on 2026-10-21 will fill (gpu4 jobs 386274-386282; DDGemb server stream ddgemb_state_capped)
QUEUED = {**{m: _FULL for m in ['SPURS (single, additive)', 'SPURS (multi)', 'ProteinMPNN', 'ESM3-small-open (ε=0, MSR readout)']},
          **{m: ['tsu_capped_val', 'tsu_capped_test'] + PG for m in ['ESMtherm', 'GeoDDG-3D', 'ESM3dG', 'ProSST']},
          # IFUM's ProteinGym run was cancelled 2026-10-22 (not worth ~10 days; see GPU4_INFERENCE_LEDGER.md 3.2),
          # so its PG cells read as not run rather than queued.
          'IFUM (WT backbone)': ['tsu_capped_val', 'tsu_capped_test']}
# documented training-set overlap with a test set (authors' training data; not exhaustive)
LEAK = {'DDGemb': {'s2648', 's8754'}, 'DDMut': {'s2648', 's8754', 'q3421', 'k3822', 'k2369'}, 'GeoDDG-3D': {'s2648', 's8754', 's783'},
        **{m: set(TSU_ALL) for m in ['ESMtherm', 'IFUM (WT backbone)', 'IFUM (native, ESMFold)', 'ESM3dG',
                                     'SPURS (single, additive)', 'SPURS (multi)']}}


def cmd_tables(a):
    L = pd.read_csv(a.long)
    part = L[L.model.str.endswith(' [partial]')].copy()
    part['model'] = part.model.str.replace(' [partial]', '', regex=False)
    W = L[~L.model.str.endswith(' [partial]')].pivot_table(index='model', columns='dataset', values='rho', aggfunc='first')
    P = part.pivot_table(index='model', columns='dataset', values='rho', aggfunc='first')

    def get(T, m, d):
        return T.loc[m, d] if (m in T.index and d in T.columns) else np.nan

    def cell(m, d):
        v, star = get(W, m, d), ''
        if pd.isna(v) and pd.notna(get(P, m, d)):
            v, star = get(P, m, d), '*'
        if pd.isna(v):
            return 'q' if d in QUEUED.get(m, []) else '–'
        return f'{v:.3f}' + ('†' if d in LEAK.get(m, set()) else '') + star

    def table(cols, names, mean_label=None):
        out = ['| model | ' + ' | '.join(names) + (f' | {mean_label}' if mean_label else '') + ' |',
               '|---|' + '---|' * (len(names) + bool(mean_label))]
        for m, lab in ROWS:
            cs = [cell(m, d) for d in cols]
            if all(c == '–' for c in cs):
                continue
            tail = ''
            if mean_label:
                vals = [get(W, m, d) for d in cols]
                tail = ' | ' + (f'{np.mean(vals):.3f}' if not any(pd.isna(vals)) else ('q' if 'q' in cs else '–'))
            out.append(f'| {lab} | ' + ' | '.join(cs) + tail + ' |')
        return '\n'.join(out)

    short = ['DLG4 abund.', 'DLG4 bind.', 'GRB2 abund.', 'GRB2 bind.', 'MYO', 'ESTA dTm', 'GB1']
    print('#### Thermostability: 12 external sets\n\n' + table(EXT, EXT, 'mean of 12'))
    print('\n#### Tsuboyama\n\n' + table(TSU_ALL, ['capped val', 'capped test', 'hyperopt val', 'hyperopt test']))
    print('\n#### DMS and Domainome\n\n' + table(DMS + ['domainome'], short + ['Domainome']))
    print('\n#### ProteinGym (213 DMS)\n\n' + table(PG, ['PG avg', 'Stability', 'DMS mean']))


# ---------------------------------------------------------------------------------------------- ledger
GPU = {'gpu1': 'RTX 6000', 'gpu2': 'RTX 6000', 'gpu3': 'RTX 6000', 'gpu4': 'RTX A4500'}
JOBS = [  # job-name regex -> model, benchmark, kind
    (r'^esmtherm_(full|s669)$', 'ESMtherm', 'gamut (22 sets, hyperopt Tsuboyama)', 'inference'),
    (r'^ifum_wt_full$', 'IFUM (WT backbone)', 'gamut (22 sets)', 'inference'),
    (r'^(esmfold_s|fold_w|foldS_w|ifum_foldbench)', 'IFUM (ESMFold per variant)', 'ESMFold folding of variant structures', 'structure prep'),
    (r'^(ifum_esmfold_ext|ifum_multi_|ifum_sing_)', 'IFUM (ESMFold per variant)', 'gamut scoring on folded structures', 'inference'),
    (r'^(geoddg_full|geoddg_retry|geoddg_s669)$', 'GeoDDG-3D (FASPR)', 'gamut (22 sets)', 'inference'),
    (r'^(foldx_full|foldx_fix|fx_test)$', 'GeoDDG-3D (FoldX)', 'FoldX structure building (CPU)', 'structure prep'),
    (r'^geoddg_fx_(full|rescore)$', 'GeoDDG-3D (FoldX)', '12 external sets', 'inference'),
    (r'^prosst_all$', 'ProSST', 'gamut (22 sets)', 'inference'),
    (r'^esm3dg_', 'ESM3ΔG', 'gamut (22 sets, sharded)', 'inference'),
    (r'^af2ext_\d', 'ColabFold AF2', 'WT structures for external sets', 'structure prep'),
    (r'^msr_small_af2$', 'ESM-MSR-small (March)', '12 external sets, AF2 structures', 'inference'),
    (r'^(me_oct06|me_capped|me_capped_s\d|me_nobb_s\d)$', 'MutateEverything', 'training (st1+st2)', 'training'),
    (r'^(tm_sin_c|tm_epi_c|tmpnn_(epi|single)_s|tm_nobb_single_s)', 'ThermoMPNN / -D', 'training', 'training'),
    (r'^(me_ext|me_ext2|me_eval|me_vf|me_nobb_eval)', 'MutateEverything', 'Tsuboyama val/test + externals', 'inference'),
    (r'^(me_dms|me_gb1|me_nobb_dms|me_nobb_fix)', 'MutateEverything', 'DMS', 'inference'),
    (r'^(me_dom|me_nobb_dom)', 'MutateEverything', 'Domainome', 'inference'),
    (r'^(tm_eval_c|tm_val_c|tm_vf_s|tm_nobb_eval|tmpnn_eval|tmpnn_ext)', 'ThermoMPNN / -D', 'Tsuboyama val/test + externals', 'inference'),
    (r'^(tm_dms_c|tm_gb1|tm_nobb_dms)', 'ThermoMPNN / -D', 'DMS', 'inference'),
    (r'^(tm_dom_c|tm_nobb_domainome)', 'ThermoMPNN / -D', 'Domainome', 'inference'),
    (r'^msr_(ext|tsu|dms|dom)$', 'ESM-MSR capped', 'gamut (one job per block)', 'inference'),
    (r'^msr_gb1$', 'ESM-MSR capped', 'GB1 rerun', 'inference'),
    (r'^msr_dc8_s\d$', 'ESM-MSR dc8', 'gamut', 'inference'),
    (r'^pgym_msr_(a4500|fp32)$', 'ESM-MSR capped', 'ProteinGym full', 'inference'),
    (r'^pgym_msr_(wtonly|wtsweep)$', 'ESM-MSR capped', 'ProteinGym WT-only ε sweep', 'inference'),
    (r'^pgym_dc8_s\d_e[\d.]+(_rest)?$', 'ESM-MSR dc8', 'ProteinGym (ε sweep)', 'inference'),
    (r'^pgym_bigalloc$', 'ESM-MSR dc8', 'ProteinGym BRCA1/SCN5A retry', 'inference'),
    (r'^pgym_dc8_s1_e1.0_mtpass', 'ESM-MSR dc8', 'ProteinGym MT-pass ablation', 'inference'),
    (r'^pgym_spurs$', 'SPURS', 'ProteinGym', 'inference'),
    (r'^pgym_mpnn(_fast|_s\d)?$', 'ProteinMPNN', 'ProteinGym', 'inference'),
    (r'^pgym_mpnn_tta', 'ProteinMPNN', 'ProteinGym TTA pilot (73 DMS)', 'inference'),
    (r'^pgym_me_(single|double)_s\d|^pgym_me_retry', 'MutateEverything', 'ProteinGym', 'inference'),
    (r'^pgym_tmpnn(_s\d)?$', 'ThermoMPNN / -D', 'ProteinGym', 'inference'),
    (r'^gpu4_esm3dg', 'ESM3ΔG', 'capped Tsuboyama + ProteinGym', 'inference'),
    (r'^gpu4_ifum', 'IFUM (WT backbone)', 'capped Tsuboyama + ProteinGym', 'inference'),
    (r'^gpu4_esmtherm', 'ESMtherm', 'capped Tsuboyama + ProteinGym', 'inference'),
    (r'^gpu4_geoddg', 'GeoDDG-3D (FASPR)', 'capped Tsuboyama + ProteinGym', 'inference'),
    (r'^gpu4_prosst', 'ProSST', 'capped Tsuboyama + ProteinGym', 'inference'),
    (r'^gpu4_msr_eps0', 'ESM3-small-open (ε=0)', 'gamut', 'inference'),
    (r'^gpu4_spurs', 'SPURS', 'gamut (incl. capped Tsuboyama)', 'inference'),
    (r'^gpu4_mpnn', 'ProteinMPNN', 'gamut (incl. capped Tsuboyama)', 'inference'),
]


def cmd_ledger(a):
    d = pd.read_csv(a.sacct, sep='|')
    rows = []
    for _, r in d[d.JobName.notna()].iterrows():
        for pat, m, b, k in JOBS:
            if re.search(pat, r.JobName):
                rows.append(dict(job=r.JobID, name=r.JobName, model=m, benchmark=b, kind=k, node=r.NodeList,
                                 gpu=GPU.get(r.NodeList, r.NodeList), start=str(r.Start)[:16], elapsed=r.Elapsed,
                                 state=r.State.split()[0]))
                break
    J = pd.DataFrame(rows)
    J.to_csv(a.out, index=False)
    print(f'{len(J)} jobs -> {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect'); c.add_argument('--out', default='matrix_long.csv'); c.set_defaults(fn=cmd_collect)
    t = sub.add_parser('tables'); t.add_argument('long'); t.set_defaults(fn=cmd_tables)
    l = sub.add_parser('ledger'); l.add_argument('sacct'); l.add_argument('--out', default='slurm_jobs.csv'); l.set_defaults(fn=cmd_ledger)
    a = ap.parse_args()
    a.fn(a)
