"""
The all-by-all benchmark matrix behind docs/BENCHMARKS_OCT2026.md: every model scored on every dataset, gathered
from wherever each run left its results on the subhak-gpu cluster, in one long table (model, dataset, rho, sd,
n_seeds, source) and rendered as the doc's markdown tables.

Run on the cluster (it reads the result trees in place); pandas only, CPU, about a minute.

  python analysis_scripts/benchmark_matrix.py collect --out matrix_long.csv     # gather + score
  python analysis_scripts/benchmark_matrix.py tables matrix_long.csv           # markdown for the doc
  python analysis_scripts/benchmark_matrix.py epi                              # the epistasis tables
  sacct -u sreeves -S 2026-10-01 -X -P -o JobID,JobName%40,NodeList,Start,End,Elapsed,State > sacct.txt
  python analysis_scripts/benchmark_matrix.py ledger sacct.txt --out slurm_jobs.csv                # gpu4 ledger

Metric conventions, the same as esm_msr_testing.update_stats and the stability-baselines harness: external sets and
DMS are one pooled Spearman per set; Tsuboyama and Domainome are the mean over proteins of the per-protein Spearman,
Tsuboyama on MEASURED rows only (cens == 0). ProteinGym: per-DMS Spearman, on the 213 DMS every model scored, as the
mean of the five category means (PG avg), the Stability category, and the plain mean (DMS mean). Higher = more
stable / fitter for every model.

Sources:
  ESM-MSR (released, devel, ε sweep, ε=0)
                                  esm-msr-devel/analysis_notebooks/stats/<set>/<ckpt dir>/<ckpt>_epsilon<ε>_unmasked_<head>[_avg].csv
  MutateEverything / ThermoMPNN   esm-msr-devel/outputs/baselines_capped/{combined_*,final_*}_table.csv (3-seed means)
  third-party (old runs)          stability-baselines/results/FINAL_TABLE.csv (Tsuboyama on hyperopt_splits)
  third-party (capped, ProteinGym), SPURS / ProteinMPNN gamut
                                  stability-baselines/results/<model>/predictions[_tag]/<dataset>.csv, scored here
  ESM3 Forge zero-shot            esm-msr-devel/analysis_notebooks/predictions/<set>/esm3_forge/zero_shot/ (per variant;
                                  S461, K2369 and PTMUL-D are scored from their parent sets' predictions)
  ProteinGym                      esm-msr-devel/outputs/pgym/; ProSST from the authors' published scores
                                  (analysis_scripts/pgym_public_scores.py)
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

# The two ESM-MSR checkpoints compared throughout. "released": the released code retrained on the capped split
# (LoRA_models/esm-msr-baseline-w2a4-m16a16-capped). "devel": the three-seed dc8-floorlink series from devel-revision.
REL, DEVEL = 'ESM-MSR (released)', 'ESM-MSR (devel)'
REL_CK = ['esm-msr-baseline-w2a4-m16a16-capped']
DEVEL_CK = [f'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{i}' for i in (1, 2, 3)]
EPS_SWEEP = ['0.75', '0.5', '0.25']          # LoRA scales run on the DMS and Domainome blocks, besides 1 and 0


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


def pg_summaries(rows, model, per_dms, src, n_seeds=1, allow_partial=False):
    """per_dms: Series DMS_id -> rho. Adds PG avg / PG Stability / PG DMS mean when all 213 common DMS are present.
    allow_partial summarises over the DMS a model does cover (the count goes in `src`) -- used only for a model that
    cannot score some DMS at all, never for one that is simply unfinished."""
    cat = pg_categories()
    keep = cat.index[~cat.index.isin(PG_DROP)]
    s = per_dms.reindex(keep)
    if s.notna().sum() < len(keep):
        if not allow_partial:
            print(f'[pg] {model}: {s.notna().sum()}/{len(keep)} DMS, not summarised')
            return
        src = f'{src}; {int(s.notna().sum())} of {len(keep)} DMS'
        keep = s.dropna().index
        s = s.reindex(keep)
    g = s.groupby(cat.reindex(keep)).mean()
    rows.add(model, 'PG avg', g.mean(), src, n_seeds)
    rows.add(model, 'PG Stability', g['Stability'], src, n_seeds)
    rows.add(model, 'PG DMS mean', s.mean(), src, n_seeds)


def collect_msr(rows, ckpt_dirs, model, eps='1', heads=None, blocks=('external', 'DMS', 'domainome', 'tsu')):
    """ESM-MSR statistics written by esm_msr_testing.py for one LoRA scale (the ε is in the file name).
    A model is the mean over the checkpoints given (seeds); heads default to combined / WT / MT."""
    stats = f'{DEV}/analysis_notebooks/stats'
    heads = heads or [('Combined', ''), ('WT_LoRA', ' WT'), ('MT_LoRA', ' MT')]
    for head, label in heads:
        acc = {}
        for ck in ckpt_dirs:
            for sub, names in [('external', EXT), ('DMS', DMS)]:
                if sub not in blocks:
                    continue
                f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}.csv')
                if f:
                    d = pd.read_csv(f[0], index_col=0)
                    for ds in names:
                        if ds in d.index:
                            acc.setdefault(ds, []).append(d.loc[ds, 'spearman_all'])
            for sub, ds, blk in [('domainome', 'domainome', 'domainome'), ('splits_oct06_capped-val', 'tsu_capped_val', 'tsu'),
                                 ('splits_oct06_capped-test', 'tsu_capped_test', 'tsu')]:
                if blk not in blocks:
                    continue
                f = glob.glob(f'{stats}/{sub}/{ck}/*_epsilon{eps}_unmasked_{head}_avg.csv')
                if f:
                    acc.setdefault(ds, []).append(pd.read_csv(f[0], index_col=0).loc['spearman_all'].iloc[0])
        for ds, v in acc.items():
            # a DMS/Domainome sweep point counts only once every seed has it, so a half-finished sweep never
            # shows up as a seed mean over fewer seeds than its neighbours
            if len(v) < len(ckpt_dirs):
                continue
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


# ESM3 zero-shot through the Forge API (small / medium / large). Predictions exist per variant for the parent sets
# only. S461, K2369 and PTMUL-D are subsets of S669, K3822 and PTMUL: every child row is found in its parent by
# (code, mut_type) with the same reference sequence and the same PDB file (checked: 461/461, 2369/2369, 536/536), so
# a zero-shot score -- a function of structure and mutation alone -- is the parent's score, and only the LABELS are
# the child's own. (They are not all identical: S461 revised 40 of its 461 labels relative to S669, 12 of them sign
# flips, and PTMUL-D differs from PTMUL on 41 of 536.)
FORGE_DIR = {'s571': 's571', 's783': 's783', 's2648': 's2648', 's8754': 's8754', 's669': 's669', 'ssym': 'ssym',
             'q3421': 'q3421', 'k3822': 'k3822', 'ptmul': 'PTMUL',
             's461': 's669', 'k2369': 'k3822', 'ptmuld': 'PTMUL'}
FORGE_SUBSET = {'s461': 's669', 'k2369': 'k3822', 'ptmuld': 'ptmul'}


def collect_forge(rows):
    P = f'{DEV}/analysis_notebooks/predictions'
    for size in ('small-2024-08', 'medium-2024-08', 'large-2024-03'):
        label = f'ESM3-{size.split("-")[0]} zero-shot'
        zs = f'esm3_forge/zero_shot/esm3-{size}/predictions.csv'
        for ds in EXT:
            f = f'{P}/{FORGE_DIR[ds]}/{zs}'
            if not os.path.exists(f):
                continue
            pred = pd.read_csv(f, usecols=['id', 'code', 'mut_type', 'esm3_score'], low_memory=False)
            # PTMUL leaves `code` blank for its 3+ mutants (343 of 914 rows) but the id is intact: '1A5EA_W15D:L37S:L121R'
            # is PDB code + chain + '_' + mutations, so the code is the first four characters of the id
            pred['code'] = pred['code'].fillna(pred['id'].astype(str).str[:4])
            pred = pred.drop_duplicates(['code', 'mut_type'])
            task = pd.read_csv(f'{SB}/tasks/{ds}.csv.gz', usecols=['code', 'mut_type', 'truth'], keep_default_na=False,
                               na_values=[''], low_memory=False)
            j = task.merge(pred, on=['code', 'mut_type'], how='left')
            cov = j.esm3_score.notna().mean()
            if cov < 0.99:
                print(f'[forge] {label} {ds}: only {cov:.2f} of rows matched, skipped')
                continue
            via = f' via parent {FORGE_SUBSET[ds]}' if ds in FORGE_SUBSET else ''
            rows.add(label, ds, spearman(j.esm3_score, j.truth), f'esm3 forge zero-shot{via}')
        for ds in DMS:
            f = f'{P}/{ds}/{zs}'
            if os.path.exists(f):
                d = pd.read_csv(f, usecols=['ddG_ML', 'esm3_score'], low_memory=False)
                rows.add(label, ds, spearman(d.esm3_score, d.ddG_ML), 'esm3 forge zero-shot')
        f = f'{P}/domainome/{zs}'
        if os.path.exists(f):                       # per-domain mean, as every other Domainome number here
            d = pd.read_csv(f, usecols=['code', 'ddG_ML', 'esm3_score'], low_memory=False)
            per = [spearman(g.esm3_score, g.ddG_ML) for _, g in d.groupby('code')]
            rows.add(label, 'domainome', np.nanmean(per), 'esm3 forge zero-shot, per-domain mean')


def collect_pgym(rows):
    pg = pd.read_csv(f'{DEV}/outputs/pgym/pgym_all_models_per_dms.csv').set_index('DMS_id')
    cols = {'ESM-MSR full': (REL, 1), 'ESM-MSR WT eps=1.0': (REL + ' WT', 1),
            'ESM-MSR WT eps=0': ('ESM3-small-open (ε=0, MSR readout)', 1),
            'SPURS single-additive': ('SPURS', 1), 'SPURS multi': ('SPURS multi', 1),
            'MutateEverything single': ('ME stage1', 3), 'MutateEverything double': ('ME stage2', 3),
            'ThermoMPNN': ('TMPNN', 3), 'ProteinMPNN': ('ProteinMPNN', 3)}
    for c, (m, n) in cols.items():
        pg_summaries(rows, m, pg[c], 'pgym_all_models_per_dms', n)
    per = []
    for s in (1, 2, 3):
        f = f'{DEV}/outputs/pgym/esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{s}/sigma1.0/summary.csv'
        d = pd.read_csv(f).drop_duplicates('DMS_id', keep='last')
        per.append(d[d.status == 'done'].set_index('DMS_id').spearman_combined)
    pg_summaries(rows, DEVEL, pd.concat(per, axis=1).mean(axis=1), 'pgym summary.csv', 3)


def collect_pgym_public(rows):
    """ProSST on ProteinGym is not rerun: the authors' own ProSST-2048 scores ship with ProteinGym
    (analysis_scripts/pgym_public_scores.py), scored here on exactly the variants every other model sees."""
    f = f'{DEV}/outputs/pgym/pgym_public_per_dms.csv'
    if os.path.exists(f):
        d = pd.read_csv(f).set_index('DMS_id')
        pg_summaries(rows, 'ProSST', d['ProSST-2048 rho_mapped'], 'ProteinGym published scores (ProSST-2048)')
        # the same two models read out by their authors' ProteinGym recipe, beside ours (our ProteinMPNN is the unmasked
        # conditional score, our ESM3-small-open is the ESM-MSR readout with the LoRA off): the difference is inference
        for col, name, what in [('ProteinMPNN', 'ProteinMPNN (published)', 'whole-sequence likelihood'),
                                ('ESM3', 'ESM3-open (published)', 'the authors\' own ESM3 readout, an additive wild-type-marginal sum')]:
            if f'{col} rho_mapped' in d.columns:
                pg_summaries(rows, name, d[f'{col} rho_mapped'], f'ProteinGym published scores ({col}, {what})')


def collect_pgym_rosetta(rows):
    """Rosetta Cartesian ddG covers ProteinGym's SINGLE mutants only (analysis_scripts/pgym_rosetta_singles.py), so its
    numbers are on a different variant set from every other model's and are shown in italics, not compared."""
    f = f'{DEV}/outputs/pgym/pgym_rosetta_singles_per_dms.csv'
    if os.path.exists(f):
        d = pd.read_csv(f).set_index('DMS_id')
        d = d[d.n_scored > 0]                      # POLG_HCVJF has no scored single; BRCA2 is excluded anyway
        pg_summaries(rows, 'Rosetta', d['rho'], 'ProteinGym singles only (Rosetta Cartesian ddG)', allow_partial=True)


# harness runs: (model dir, prediction tag, {readout column: model label}, scope). `scope` 'capped' restricts a run
# to the capped Tsuboyama split, for runs whose other datasets live under a different (older, partial) tag.
#   SPURS has two models. tag '' is the first run, which used the single model for singles and the multi model for
#   multi-mutants; its pred_additive is the single model summed over substitutions, i.e. SPURS proper. Its `pred` is a
#   hybrid and is NOT read. SPURS multi is the multi-mutant model applied to every variant, singles included, and
#   comes from the `--readout multi` rerun (tag _multi).
HARNESS = [('esmtherm', '', {'pred': 'ESMtherm'}, 'all'), ('ifum', '', {'pred': 'IFUM (WT backbone)'}, 'all'),
           ('ifum', '_esmfold', {'pred': 'IFUM (native, ESMFold)'}, 'capped'),
           ('geoddg', '_faspr', {'pred': 'GeoDDG-3D'}, 'all'), ('esm3dg', '', {'pred': 'ESM3dG'}, 'all'),
           ('prosst', '', {'pred': 'ProSST'}, 'all'),
           ('spurs', '', {'pred_additive': 'SPURS'}, 'all'), ('spurs', '_multi', {'pred': 'SPURS multi'}, 'all'),
           ('proteinmpnn', '', {'pred': 'ProteinMPNN'}, 'all'), ('ddgemb', '', {'pred': 'DDGemb'}, 'all'),
           ('rosetta', '', {'pred': 'Rosetta'}, 'capped')]    # cartesian_ddg on Narval; doubles scored directly
NO_HARNESS_PG = {'SPURS', 'SPURS multi', 'ProteinMPNN', 'ProSST'}   # their ProteinGym numbers come from elsewhere


def harness_frame(model, tag, name):
    p = f'{SB}/results/{model}/predictions{tag}/{name}.csv'
    t = f'{SB}/tasks/{name}.csv.gz'
    if not (os.path.exists(p) and os.path.exists(t)):
        return None
    task = pd.read_csv(t, keep_default_na=False, na_values=[''], low_memory=False)
    pred = pd.read_csv(p)
    return task.merge(pred[['row_id', 'pred', 'pred_additive']], on='row_id', how='left')


def collect_harness(rows, done_by_old_runs):
    for model, tag, readouts, scope in HARNESS:
        for col, label in readouts.items():
            if scope == 'all':
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
            if scope != 'all' or label in NO_HARNESS_PG:
                continue
            pg = {}
            for f in glob.glob(f'{SB}/results/{model}/predictions{tag}/pg_*.csv'):
                name = os.path.basename(f)[:-4]
                d = harness_frame(model, tag, name)
                if d is not None:
                    pg[name[3:]] = spearman(d[col], d['truth'])
            if pg:
                pg_summaries(rows, label, pd.Series(pg), f'harness {model}{tag}')


def cmd_collect(a):
    rows = Rows()
    collect_msr(rows, REL_CK, REL)
    collect_msr(rows, DEVEL_CK, DEVEL)
    collect_msr(rows, REL_CK, 'ESM3-small-open (ε=0, MSR readout)', eps='0.0')
    for e in EPS_SWEEP:                    # DMS + Domainome only; the combined head
        for ck, name in [(REL_CK, REL), (DEVEL_CK, DEVEL)]:
            collect_msr(rows, ck, f'{name} ε={e}', eps=e, heads=[('Combined', '')], blocks=('DMS', 'domainome'))
    collect_retrained(rows)
    collect_final_table(rows)
    collect_forge(rows)
    collect_pgym(rows)
    collect_pgym_public(rows)
    collect_pgym_rosetta(rows)
    collect_harness(rows, {(r['model'], r['dataset']) for r in rows})
    L = pd.DataFrame(rows)
    L.to_csv(a.out, index=False)
    print(L.groupby('model').dataset.count().sort_values().to_string())


# ---------------------------------------------------------------------------------------------- tables
ROWS = [(REL, 'ESM-MSR (released), combined'), (REL + ' WT', 'ESM-MSR (released), WT head'),
        (DEVEL, 'ESM-MSR (devel), combined (3 seeds)'), (DEVEL + ' WT', 'ESM-MSR (devel), WT head'),
        ('ME stage1', 'MutateEverything st1 (3 seeds)'), ('ME stage1+', 'MutateEverything st1, additive'),
        ('ME stage2', 'MutateEverything st2 (3 seeds)'), ('ME stage2+', 'MutateEverything st2, additive'),
        ('TMPNN', 'ThermoMPNN (additive, 3 seeds)'), ('TMPNN+D', 'ThermoMPNN + ThermoMPNN-D (3 seeds)'),
        ('SPURS', 'SPURS (single model) ‡'), ('SPURS multi', 'SPURS multi (multi-mutant model) ‡'),
        ('ESMtherm', 'ESMtherm ‡'), ('DDGemb', 'DDGemb (server)'), ('DDMut', 'DDMut (server)'),
        ('IFUM (WT backbone)', 'IFUM, WT backbone ‡'), ('IFUM (native, ESMFold)', 'IFUM, ESMFold per variant ‡'),
        ('GeoDDG-3D', 'GeoDDG-3D (FASPR)'), ('ESM3dG', 'ESM3ΔG ‡'), ('ProSST', 'ProSST (zero-shot)'),
        ('Rosetta', 'Rosetta Cartesian ΔΔG'),
        ('ProteinMPNN', 'ProteinMPNN (zero-shot)'), ('ProteinMPNN (published)', 'ProteinMPNN, ProteinGym published recipe'),
        ('ESM3-small-open (ε=0, MSR readout)', 'ESM3-small-open, ε = 0 (zero-shot)'),
        ('ESM3-open (published)', 'ESM3-open, ProteinGym published recipe'),
        ('ESM3-large zero-shot', 'ESM3-large (Forge, zero-shot)'), ('ESM3-medium zero-shot', 'ESM3-medium (Forge, zero-shot)'),
        ('ESM3-small zero-shot', 'ESM3-small (Forge, zero-shot)'),
        ('ESM-MSR-small (March, crystal)', 'ESM-MSR-small (March reference)')]
TSU_ALL = ['tsu_capped_val', 'tsu_capped_test', 'tsu_hyperopt_val', 'tsu_hyperopt_test']
PG = ['PG avg', 'PG Stability', 'PG DMS mean']
_FULL = EXT + DMS + ['domainome', 'tsu_capped_val', 'tsu_capped_test']
_CAP = ['tsu_capped_val', 'tsu_capped_test']
# cells a run still in the queue will fill (a filled cell always overrides the marker)
QUEUED = {'SPURS multi': _FULL,                                                   # --readout multi rerun
          'IFUM (native, ESMFold)': _CAP,                                         # fold 64,008 sequences, then score
          'GeoDDG-3D': _CAP + PG, 'ESM3dG': PG, 'ESMtherm': PG,                   # runs in progress / one DMS retried
          'ProSST': _CAP,                                                         # ProteinGym comes from published scores
          'IFUM (WT backbone)': _CAP,
          # the ε sweep on DMS + Domainome, one lane per checkpoint
          **{f'{n} ε={e}': DMS + ['domainome'] for n in (REL, DEVEL) for e in EPS_SWEEP}}
# cells that are on a different variant set from the rest of their column and are shown in italics, never bolded:
# Rosetta's ProteinGym numbers cover single mutants only
ITALIC = {('Rosetta', d) for d in PG}
# documented training-set overlap with a test set (authors' training data; not exhaustive)
LEAK = {'DDGemb': {'s2648', 's8754'}, 'DDMut': {'s2648', 's8754', 'q3421', 'k3822', 'k2369'}, 'GeoDDG-3D': {'s2648', 's8754', 's783'},
        **{m: set(TSU_ALL) for m in ['ESMtherm', 'IFUM (WT backbone)', 'IFUM (native, ESMFold)', 'ESM3dG', 'SPURS', 'SPURS multi']}}


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
        if (m, d) in ITALIC:
            return f'_{v:.3f}_'
        return f'{v:.3f}' + ('†' if d in LEAK.get(m, set()) else '') + star

    def plain(c):
        """the number in a cell, or None when it is absent, queued, flagged as overlapping, partial, or italic (not comparable)"""
        if any(f in c for f in ('†', '‡', '*', '_')) or c in ('–', 'q'):
            return None
        try:
            return float(c)
        except ValueError:
            return None

    def table(cols, names, mean_label=None, rows_=None):
        body = []
        for m, lab in (rows_ or ROWS):
            cs = [cell(m, d) for d in cols]
            if all(c == '–' for c in cs):
                continue
            tail = None
            if mean_label:
                vals = [get(W, m, d) for d in cols]
                tail = f'{np.mean(vals):.3f}' if not any(pd.isna(vals)) else ('q' if 'q' in cs else '–')
            body.append([lab, cs, tail])
        # bold the best value in each dataset column, among the cells that are a like-for-like comparison.
        # The aggregate column is left unbolded: it averages over sets that are flagged for some models and not
        # others, so its maximum is not a comparison (the overlap-free ranking is in the doc's bullets instead).
        for j in range(len(cols)):
            vals = [(plain(r[1][j]), i) for i, r in enumerate(body)]
            vals = [(v, i) for v, i in vals if v is not None]
            if vals:
                best = max(v for v, _ in vals)
                for v, i in vals:
                    if v == best:
                        body[i][1][j] = f'**{body[i][1][j]}**'
        out = ['| model | ' + ' | '.join(names) + (f' | {mean_label}' if mean_label else '') + ' |',
               '|---|' + '---|' * (len(names) + bool(mean_label))]
        for lab, cs, tail in body:
            out.append(f'| {lab} | ' + ' | '.join(cs) + (f' | {tail}' if tail is not None else '') + ' |')
        return '\n'.join(out)

    short = ['DLG4 abund.', 'DLG4 bind.', 'GRB2 abund.', 'GRB2 bind.', 'MYO', 'ESTA dTm', 'GB1']
    # the ε sweep: combined head, ε = 1 (the main tables), the three sweep points, and the ε = 0 control
    eps_rows = []
    for name in (REL, DEVEL):
        tag = name.split('(')[1].rstrip(')')
        eps_rows += [(name, f'{tag}, ε = 1')] + [(f'{name} ε={e}', f'{tag}, ε = {e}') for e in EPS_SWEEP]
    eps_rows += [('ESM3-small-open (ε=0, MSR readout)', 'ε = 0 (zero-shot control)')]
    print('#### Thermostability: 12 external sets\n\n' + table(EXT, EXT, 'mean of 12'))
    print('\n#### Tsuboyama\n\n' + table(TSU_ALL, ['capped val', 'capped test', 'hyperopt val', 'hyperopt test']))
    print('\n#### DMS and Domainome\n\n' + table(DMS + ['domainome'], short + ['Domainome']))
    print('\n#### ESM-MSR LoRA scale\n\n' + table(DMS + ['domainome'], short + ['Domainome'], rows_=eps_rows))
    print('\n#### ProteinGym (213 DMS)\n\n' + table(PG, ['PG avg', 'Stability', 'DMS mean']))


# ---------------------------------------------------------------------------------------------- epistasis tables
EPI_COLS = [('epi_naive_rho', 'naive ρ', 1), ('epi_beyond_rho', 'beyond ρ', 1), ('epi_pair_rho', 'pair ρ', 1),
            ('epi_line_rho', 'line ρ', 1), ('epi_cell_mag', 'cell_mag', 1), ('epi_cell_rank', 'cell_rank', 1),
            ('epi_cell_flipacc', 'flipacc', 1), ('epi_global_err', 'global_err ↓', -1)]


def cmd_epi(a):
    """The epistasis hierarchy, one row per MODEL. ESM-MSR is shown by its combined head only (what is deployed); a
    baseline by its direct double head, which is its only prediction of an interaction (its additive head is by
    construction 0 on every residual and rank metric, the suite's own validity check). Seed means; the sd is printed
    so the text can say how far to trust a difference."""
    B = f'{DEV}/outputs/baselines_capped'
    pieces = {  # split -> (baselines + released, devel seeds, Rosetta, harness models)
        'valfull (34,957 rows)': ('epi_test_valfull_allbb.csv', 'epi_test_valfull_allbb_msr_dc8_s1-dc8_s2-dc8_s3.csv',
                                  'epi_rosetta_valfull_allbb.csv', 'epi_harness_valfull_allbb.csv'),
        'test (40,866 rows, destabilized backbones dropped)': ('epi_test.csv', 'epi_test_allbb_msr_dc8_s1-dc8_s2-dc8_s3.csv',
                                                               'epi_rosetta_test.csv', 'epi_harness_test.csv')}
    # harness models with a real double prediction (analysis_scripts/epi_harness.py), labelled as in the other tables;
    # ProSST is absent because its double is the sum of its singles, so it has no interaction to score
    harness = [('ESMtherm', 'ESMtherm ‡'), ('IFUM (WT backbone)', 'IFUM, WT backbone ‡'),
               ('IFUM (native, ESMFold)', 'IFUM, ESMFold per variant ‡'), ('GeoDDG-3D (FASPR)', 'GeoDDG-3D (FASPR)'),
               ('ESM3dG', 'ESM3ΔG ‡'), ('SPURS multi', 'SPURS multi (multi-mutant model) ‡'),
               ('ProteinMPNN', 'ProteinMPNN (zero-shot)'), ('DDGemb', 'DDGemb (server)')]
    for split, (base_f, dev_f, ros_f, har_f) in pieces.items():
        base = pd.read_csv(f'{B}/{base_f}').set_index('run/head')
        dev = pd.read_csv(f'{B}/{dev_f}').set_index('run/head')
        ros = pd.read_csv(f'{B}/{ros_f}').set_index('run/head')
        har = pd.read_csv(f'{B}/{har_f}').set_index('run/head')
        seeds = lambda df, stem: df[[i.startswith(stem) for i in df.index]]
        spec = [('MutateEverything stage 1', seeds(base, 'ME_stage1_s')[lambda x: x.index.str.endswith('/mt')]),
                ('MutateEverything stage 2', seeds(base, 'ME_stage2_s')[lambda x: x.index.str.endswith('/mt')]),
                ('ThermoMPNN-D', seeds(base, 'TMPNN_s')[lambda x: x.index.str.endswith('/mt')]),
                ('Rosetta Cartesian ΔΔG', ros[ros.index == 'Rosetta/mt']),
                *[(label, har[har.index == f'{m}/mt']) for m, label in harness],
                (REL, base[base.index == 'MSR/comb']),
                (DEVEL + ' (3 seeds)', seeds(dev, 'MSR_dc8_s')[lambda x: x.index.str.endswith('/comb')])]
        body = []
        for name, df in spec:
            body.append((name, [df[c].mean() for c, _, _ in EPI_COLS], [df[c].std() if len(df) > 1 else np.nan for c, _, _ in EPI_COLS]))
        # best value per column among models that did not train on Tsuboyama (‡), as in the other tables
        fair = [r for r in body if '‡' not in r[0]]
        best = [(max if sign > 0 else min)(r[1][j] for r in fair) for j, (_, _, sign) in enumerate(EPI_COLS)]
        print(f'\n##### {split}\n')
        print('| model | ' + ' | '.join(n for _, n, _ in EPI_COLS) + ' |')
        print('|---|' + '---|' * len(EPI_COLS))
        for name, m, sd in body:
            cells = [f'**{v:.3f}**' if v == best[j] and '‡' not in name else f'{v:.3f}' for j, v in enumerate(m)]
            print(f'| {name} | ' + ' | '.join(cells) + ' |')
        print('\nseed sd (largest per model): ' + '; '.join(f'{n} {np.nanmax(sd):.3f}' for n, _, sd in body if not np.isnan(sd).all()))


DDG_ROWS = [('spearman_all', 'all variants'), ('spearman_doubles', 'doubles'), ('spearman_doubles_epi', 'dddG of doubles')]


def cmd_ddg(a):
    """ddG ranking (mean per-protein Spearman) on capped val and test: the retrained baselines against both ESM-MSR
    checkpoints, each shown by its WT (additive) head and by its combined prediction. Baselines are 3-seed means from
    combined_{val,test}_table.csv (TMPNN, the singles-only ThermoMPNN, has no dddG, the one structural gap); ESM-MSR
    comes from esm_msr_testing.py's *_avg.csv, the devel columns being seed means ± sd over the 3 seeds."""
    o6 = f'{DEV}/outputs/baselines_capped'
    stats = f'{DEV}/analysis_notebooks/stats'
    seeds = [f'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{i}' for i in (1, 2, 3)]

    def msr(split, cks, head):
        out = {m: [] for m, _ in DDG_ROWS}
        for ck in cks:
            f = glob.glob(f'{stats}/splits_oct06_capped-{split}/{ck}/*_epsilon1_unmasked_{head}_avg.csv')
            assert len(f) == 1, (split, ck, head, f)
            d = pd.read_csv(f[0], index_col=0).iloc[:, 0]
            for m, _ in DDG_ROWS:
                out[m].append(d[m])
        return out

    cols = ['ME st1', 'ME st2', 'TMPNN', 'TMPNN+D', f'{REL} WT', f'{REL} combined', f'{DEVEL} WT', f'{DEVEL} combined']
    print('| | ' + ' | '.join(cols) + ' |')
    print('|---|' + '---|' * len(cols))
    for split in ('val', 'test'):
        base = pd.read_csv(f'{o6}/combined_{split}_table.csv', index_col=0)
        pieces = {'ME st1': 'ME stage1', 'ME st2': 'ME stage2', 'TMPNN': 'TMPNN', 'TMPNN+D': 'TMPNN+D'}
        m_rel = {h: msr(split, REL_CK, h) for h in ('WT_LoRA', 'Combined')}
        m_dev = {h: msr(split, seeds, h) for h in ('WT_LoRA', 'Combined')}
        for metric, name in DDG_ROWS:
            vals = [base.loc[metric, pieces[c]] if c in pieces else None for c in cols]
            vals[4], vals[5] = m_rel['WT_LoRA'][metric][0], m_rel['Combined'][metric][0]
            vals[6], vals[7] = np.mean(m_dev['WT_LoRA'][metric]), np.mean(m_dev['Combined'][metric])
            sd = [np.nan] * 6 + [np.std(m_dev['WT_LoRA'][metric], ddof=1), np.std(m_dev['Combined'][metric], ddof=1)]
            ok = [v for v in vals if v is not None and np.isfinite(v)]
            cells = []
            for v, e in zip(vals, sd):
                if v is None or not np.isfinite(v):
                    cells.append('–')
                    continue
                t = f'{v:.3f}' + (f' ± {e:.3f}' if np.isfinite(e) else '')
                cells.append(f'**{t}**' if v == max(ok) else t)
            print(f'| {split} {name} (`{metric}`) | ' + ' | '.join(cells) + ' |')


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
    (r'^gpu4_ifumfold', 'IFUM (ESMFold per variant)', 'ESMFold folding, capped Tsuboyama (64,008 sequences)', 'structure prep'),
    (r'^gpu4_ifumnat', 'IFUM (ESMFold per variant)', 'capped Tsuboyama scoring', 'inference'),
    (r'^gpu4_spurs_multi', 'SPURS multi', 'gamut, multi model on every variant', 'inference'),
    (r'^gpu4_eps', 'ESM-MSR', 'LoRA scale sweep, DMS + Domainome', 'inference'),
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
    e = sub.add_parser('epi'); e.set_defaults(fn=cmd_epi)
    g = sub.add_parser('ddg'); g.set_defaults(fn=cmd_ddg)
    l = sub.add_parser('ledger'); l.add_argument('sacct'); l.add_argument('--out', default='slurm_jobs.csv'); l.set_defaults(fn=cmd_ledger)
    a = ap.parse_args()
    a.fn(a)
