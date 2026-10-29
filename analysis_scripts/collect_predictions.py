"""
Collect the predictions behind docs/BENCHMARKS_OCT2026.md into one local layout, from the subhak-gpu cluster and
from earlier local pulls.

  <root>/msr_predictions/<spec>[/seed<k>]/sigma<s>/<dataset>.csv         ESM-MSR (also .../proteingym[_wt_only]/<DMS>.csv, stats/)
  <root>/baselines/retrained/<model>/predictions/<variant>/seed<k>/...    models retrained on the capped split
  <root>/baselines/public/<model>[/<variant>]/<dataset>.csv               every other model in the tables

Files are copied byte for byte; only their path changes. Dataset names are the ones benchmark_matrix.py uses (s571 ...
k2369, ptmul, ptmuld, the seven DMS ids, domainome, tsuboyama_capped_{val,test}). Each model directory gets a
MANIFEST.tsv (path, bytes, md5, origin on the cluster, origin mtime), so every rename can be undone.

  python analysis_scripts/collect_predictions.py plan                 # list what would move or be fetched, nothing is touched
  python analysis_scripts/collect_predictions.py run                  # do it (rsync from the cluster, mv from earlier pulls)
  python analysis_scripts/collect_predictions.py manifest             # (re)write the MANIFEST.tsv files
  python analysis_scripts/collect_predictions.py pgpublished          # ProteinGym's published ProSST / ProteinMPNN / ESM3 scores, per DMS

Selection rules (what counts as "behind the tables") are in the SPECS / RETRAINED / PUBLIC tables below. Left out on
purpose: files a later fix superseded (`_pre_pmb/`, `.pre_gb1_fix`, `.old`, `.partial`, `.pre_bbpatch`), the withdrawn
item-filtered ME stage 2 / ThermoMPNN-D runs, the published SPURS checkpoint, smoke tests, and arms the tables do not cite.
Re-running is safe: a destination that already exists with the same size is skipped.
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = os.environ.get('MSR_CLUSTER', 'subhak-gpu.uwaterloo.ca')
RHOME = os.environ.get('MSR_CLUSTER_HOME', '/home/sreeves')
ROOT = Path(os.environ.get('MSR_ROOT', '~/playground/esm-msr-devel')).expanduser()
DEV, SB = f'{RHOME}/esm-msr-devel', f'{RHOME}/stability-baselines'
SNAP = ROOT / 'results_cluster'                       # the 2026-10-27 pull of ESM-MSR predictions
STAGE = ROOT / '.collect_staging'

TSU = {'splits_oct06_capped-val': 'tsuboyama_capped_val', 'splits_oct06_capped-test': 'tsuboyama_capped_test',
       'hyperopt_splits-val': 'tsuboyama_hyperopt_val', 'hyperopt_splits-validation': 'tsuboyama_hyperopt_val',
       'hyperopt_splits-test': 'tsuboyama_hyperopt_test'}


def ds_name(raw):
    raw = TSU.get(raw, raw)
    return 'ptmul' if raw == 'PTMUL' else raw


def sigma_dir(tok):
    return f'sigma{float(tok):g}'                       # 1.0 / 1 -> sigma1, 0.0 -> sigma0, 0.25 -> sigma0.25


# ------------------------------------------------------------------------------------------------ cluster access
_find_cache = {}


def remote_find(root):
    """[(size, mtime, path)] for every regular file under a cluster directory (one ssh, read only)."""
    if root not in _find_cache:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, f"find {root} -type f -printf '%s\\t%T@\\t%p\\n'"],
                             capture_output=True, text=True, check=True).stdout
        rows = []
        for ln in out.splitlines():
            s, m, p = ln.split('\t', 2)
            rows.append((int(s), float(m), p))
        _find_cache[root] = rows
    return _find_cache[root]


class Plan(list):
    def add(self, dst, src, size, mtime, local=None, group=''):
        """dst relative to ROOT; src the cluster path the file came from; local an earlier pull that already holds it."""
        self.append(dict(dst=str(dst), src=src, size=size, mtime=mtime, local=local, group=group))


def same_file(path, size, mtime):
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size == size and abs(st.st_mtime - mtime) < 2


def local_twin(rel, size, mtime):
    p = SNAP / rel
    return str(p) if same_file(p, size, mtime) else None


# ------------------------------------------------------------------------------------------------ ESM-MSR
# (spec directory, cluster checkpoint dir template, seeds); ProteinGym dirs sit under outputs/pgym/<ckpt>/<sigma dir>.
SPECS = [
    ('baseline_w2a4_m16a16', 'release-baseline/s{k}', (1, 2, 3), {'sigma1.0': 'sigma1/proteingym'}),
    ('devel_dc8_floorlink_w2a4_m16a16', 'esm-msr-dc8-floorlink-canon-w2a4-m16a16-s{k}', (1, 2, 3), {'sigma1.0': 'sigma1/proteingym'}),
    ('released_capped_w2a4_m16a16', 'esm-msr-baseline-w2a4-m16a16-capped', (None,),
     {'sigma1.0': 'sigma1/proteingym', 'sigma1.0_wt_only': 'sigma1/proteingym_wt_only', 'sigma0_wt_only': 'sigma0/proteingym_wt_only'}),
]
GAMUT_RX = re.compile(r'^.+\.ckpt_sigma(?P<sig>[0-9.]+)_unmasked_predictions\.csv$')


def plan_msr(plan):
    P = f'{DEV}/analysis_notebooks/predictions'
    pred = remote_find(P)
    pg_all = remote_find(f'{DEV}/outputs/pgym')
    stats = remote_find(f'{DEV}/analysis_notebooks/stats')
    for spec, tmpl, seeds, pgdirs in SPECS:
        for k in seeds:
            ck = tmpl.format(k=k)
            base = f'{spec}/seed{k}' if k else spec
            # gamut predictions: the files directly in <block>/<ckpt>/ (the _pre_pmb/ subdirectory holds the superseded convention)
            for size, mt, path in pred:
                rel = path[len(P) + 1:]
                parts = rel.split('/')
                block, rest = parts[0], '/'.join(parts[1:])
                if not rest.startswith(ck + '/') or '/' in rest[len(ck) + 1:]:
                    continue
                m = GAMUT_RX.match(parts[-1])
                if not m or block.startswith(('hyperopt_splits', 'esm3_forge')) or '_valprobe' in block:
                    continue
                plan.add(f'msr_predictions/{base}/{sigma_dir(m["sig"])}/{ds_name(block)}.csv', path, size, mt,
                         local_twin(f'predictions/{rel}', size, mt), f'msr:{spec}')
            # ProteinGym runs
            for pgdir, sub in pgdirs.items():
                pre = f'{DEV}/outputs/pgym/{ck}/{pgdir}/'
                for size, mt, path in pg_all:
                    if path.startswith(pre) and '/' not in path[len(pre):]:
                        rel = f'{ck}/{pgdir}/{path[len(pre):]}'
                        plan.add(f'msr_predictions/{base}/{sub}/{path[len(pre):]}', path, size, mt,
                                 local_twin(f'pgym/{rel}', size, mt), f'msr:{spec}')
            # the per-dataset summaries esm_msr_testing.py wrote next to the predictions
            sp = f'{DEV}/analysis_notebooks/stats/'
            for size, mt, path in stats:
                rel = path[len(sp):]
                top, rest = rel.split('/', 1)
                if rest.startswith(ck + '/') and '/' not in rest[len(ck) + 1:] and not top.startswith(('hyperopt', 'esm3_forge')) and '_valprobe' not in top:
                    plan.add(f'msr_predictions/{base}/stats/{ds_name(top)}/{rest[len(ck) + 1:]}', path, size, mt, None, f'msr:{spec}')
    # the March ESM-MSR-small reference run in the third-party harness (crystal structures)
    R = f'{SB}/results/esm-msr-small_crystal'
    for size, mt, path in remote_find(R):
        rel = path[len(R) + 1:]
        if rel.startswith('predictions/') and rel.endswith('.csv'):
            plan.add(f'msr_predictions/esm-msr-small_march_crystal/{ds_name(Path(rel).stem)}.csv', path, size, mt, None, 'msr:small')
        elif rel.startswith('stats/') or rel == 'timing.csv':
            plan.add(f'msr_predictions/esm-msr-small_march_crystal/{rel}', path, size, mt, None, 'msr:small')


# ------------------------------------------------------------------------------------------------ retrained baselines
def plan_retrained(plan):
    RT = 'baselines/retrained'
    # SPURS (single + multi in one file per dataset): the tracked fixed-slot runs the BENCHMARKS rows are scored from
    for k in (1, 2, 3):
        run = f'{DEV}/outputs/tracked/spurs_retrain_s{k}'
        for size, mt, path in remote_find(run):
            rel = path[len(run) + 1:]
            if rel.startswith('proteingym/'):
                dst = f'{RT}/spurs/predictions/seed{k}/{rel}'
            elif rel.startswith('resources/'):
                dst = f'{RT}/spurs/resources/seed{k}/{rel[len("resources/"):]}'
            elif rel.startswith('slurm-') and rel.endswith('.out'):
                dst = f'{RT}/spurs/logs/seed{k}/{rel}'
            elif rel.endswith('.csv') and '/' not in rel:
                stem = rel[:-4]
                stem = {'tsuboyama_val': 'tsuboyama_capped_val', 'tsuboyama_test': 'tsuboyama_capped_test'}.get(stem, stem)
                stem = re.sub(r'^(dms|external)_', '', stem)
                dst = f'{RT}/spurs/predictions/seed{k}/{stem}.csv'
            else:
                continue
            plan.add(dst, path, size, mt, None, 'retrained:spurs')
    plan.add(f'{RT}/spurs/docs/HANDOFF_SPURS_BENCHMARK.md', f'{RHOME}/spurs_retrain/HANDOFF_SPURS_BENCHMARK.md', *stat_remote(f'{RHOME}/spurs_retrain/HANDOFF_SPURS_BENCHMARK.md'), None, 'retrained:spurs')

    # ThermoMPNN: the single (additive) rows come from the capped-split evaluation, ThermoMPNN-D from the corrected one
    def tm(src_dir, variant, names):
        for k in (1, 2, 3):
            d = src_dir.format(k=k)
            for size, mt, path in remote_find(d):
                rel = path[len(d) + 1:]
                if '/' in rel:
                    continue
                if rel.endswith('.md'):
                    dst = f'{RT}/thermompnn/predictions/{variant}/seed{k}/{rel}'
                elif rel.endswith('.csv'):
                    stem = names.get(rel[:-4], re.sub(r'^(dms|external)_', '', rel[:-4]))
                    dst = f'{RT}/thermompnn/predictions/{variant}/seed{k}/{stem}.csv'
                else:
                    continue
                plan.add(dst, path, size, mt, None, 'retrained:thermompnn')
    tm(f'{DEV}/outputs/baselines_capped/thermompnn/eval_seed{{k}}', 'single_additive',
       {'predictions_test': 'tsuboyama_capped_test', 'predictions_val': 'tsuboyama_capped_val_itemfiltered', 'predictions_valfull': 'tsuboyama_capped_val'})
    tm(f'{DEV}/outputs/baselines_corrected/thermompnn/eval_seed{{k}}', 'epistatic_corrected',
       {'predictions_test': 'tsuboyama_capped_test', 'predictions_val': 'tsuboyama_capped_val'})
    for k in (1, 2, 3):
        d = f'{DEV}/outputs/pgym/thermompnn_capped_seed{k}'
        for size, mt, path in remote_find(d):
            plan.add(f'{RT}/thermompnn/predictions/single_additive/seed{k}/proteingym/{path[len(d) + 1:]}', path, size, mt, None, 'retrained:thermompnn')

    # MutateEverything: stage 1 (singles) and the corrected stage 2 (all measured doubles); raw upstream result files
    me_tsu = {'stage1': {'oct06_capped_val': 'tsuboyama_capped_val_itemfiltered', 'oct06_capped_valfull': 'tsuboyama_capped_val',
                         'oct06_capped_test': 'tsuboyama_capped_test', 'oct06_capped_testcomp': 'tsuboyama_capped_testcomp'},
              'stage2': {'oct06_capped_allbl_val': 'tsuboyama_capped_val', 'oct06_capped_allbl_test': 'tsuboyama_capped_test',
                         'oct06_capped_allbl_testcomp': 'tsuboyama_capped_testcomp'}}
    MEB = f'{RHOME}/software/MutateEverything-oct06'
    # stage 1's DMS and Domainome evaluations ran in their own folders (the stage-2 ones are in the capped_allbl run)
    runs = {'stage1': ['logs_eval/af_single_oct06_capped_seed{k}', 'logs_eval_dms/af_single_seed{k}', 'logs_eval_dom/af_single_seed{k}'],
            'stage2': ['logs_eval/af_double_oct06_capped_allbl_seed{k}']}
    for stage, run_dirs in runs.items():
        for k in (1, 2, 3):
            for run in run_dirs:
                d = f'{MEB}/{run.format(k=k)}'
                for size, mt, path in remote_find(d):
                    rel = path[len(d) + 1:]
                    m = re.match(r'^(results|metrics)_(.+)\.csv$', rel)
                    if m:
                        name = me_tsu[stage].get(m.group(2), m.group(2))
                        sub = 'predictions' if m.group(1) == 'results' else 'metrics'
                        if name == 's461':      # ME's run on the S461 table ME ships (another construct and MSA per protein than its S669
                                                # inputs); the S461 column scores the s669 predictions, so no table cell reads this file
                            plan.add(f'msr_predictions/_not_in_benchmarks/me_upstream_s461/{sub}/{stage}/seed{k}.csv', path, size, mt, None, 'retrained:me')
                        else:
                            plan.add(f'{RT}/mutate_everything/{sub}/{stage}/seed{k}/{name}.csv', path, size, mt, None, 'retrained:me')
                    elif rel == 'timing.csv':
                        plan.add(f'{RT}/mutate_everything/resources/{stage}/seed{k}/timing.csv', path, size, mt, None, 'retrained:me')
    for k in (1, 2, 3):                        # stage 1 ProteinGym (stage 2's is withdrawn / not in the tables)
        d = f'{DEV}/outputs/pgym/mutateeverything_single_seed{k}'
        for size, mt, path in remote_find(d):
            plan.add(f'{RT}/mutate_everything/predictions/stage1/seed{k}/proteingym/{path[len(d) + 1:]}', path, size, mt, None, 'retrained:me')
    # scored per-variant tables (labels attached) behind the corrected rows
    sc = f'{RHOME}/me_audit/out/scores'
    for size, mt, path in remote_find(sc):
        top, _, name = path[len(sc) + 1:].partition('/')
        if re.match(r'^me_st2_s\d$', top):
            plan.add(f'{RT}/mutate_everything/scored/stage2/seed{top[-1]}/{name}', path, size, mt, None, 'retrained:me')
        elif re.match(r'^tmpnn_s\d$', top):
            plan.add(f'{RT}/thermompnn/scored/epistatic_corrected/seed{top[-1]}/{name}', path, size, mt, None, 'retrained:thermompnn')
        elif re.match(r'^C_single_s\d$', top):
            plan.add(f'{RT}/mutate_everything/scored/stage1_valfull_rescore/seed{top[-1]}/{name}', path, size, mt, None, 'retrained:me')
    for k in (1, 2, 3):
        for split in ('val', 'test'):
            d = f'{DEV}/outputs/baselines_capped/score_me_single_{split}_s{k}'
            for size, mt, path in remote_find(d):
                plan.add(f'{RT}/mutate_everything/scored/stage1/seed{k}/{split}/{path[len(d) + 1:]}', path, size, mt, None, 'retrained:me')
    for rel, dst in (('project_oct06_baselines/README.md', 'docs/project_oct06_baselines_README.md'),
                     ('project_oct06_baselines/RUNS.md', 'docs/project_oct06_baselines_RUNS.md')):
        plan.add(f'{RT}/{dst}', f'{RHOME}/{rel}', *stat_remote(f'{RHOME}/{rel}'), None, 'retrained:docs')


def stat_remote(path):
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', HOST, f"stat -c '%s %Y' {path}"], capture_output=True, text=True, check=True).stdout.split()
    return int(out[0]), float(out[1])


# ------------------------------------------------------------------------------------------------ public baselines
# (destination under baselines/public, cluster directory under stability-baselines/results)
PUBLIC = [('esmtherm', 'esmtherm/predictions'), ('esm3dg', 'esm3dg/predictions'), ('ddgemb', 'ddgemb/predictions'),
          ('ddmut', 'ddmut/predictions'), ('prosst', 'prosst/predictions'), ('proteinmpnn', 'proteinmpnn/predictions'),
          ('rosetta', 'rosetta/predictions'), ('geoddg/faspr', 'geoddg/predictions_faspr'),
          ('ifum/wt_backbone', 'ifum/predictions'), ('ifum/native_esmfold', 'ifum/predictions_esmfold'),
          ('ifum/native_esmfold_full', 'ifum/predictions_esmfold_full')]
PUBLIC_STATS = [('esmtherm', 'esmtherm/stats'), ('esm3dg', 'esm3dg/stats'), ('prosst', 'prosst/stats'), ('geoddg/faspr', 'geoddg/stats_faspr'),
                ('ifum/wt_backbone', 'ifum/stats'), ('ifum/native_esmfold', 'ifum/stats_esmfold'), ('ifum/native_esmfold_full', 'ifum/stats_esmfold_full')]


def plan_public(plan):
    PB = 'baselines/public'
    R = f'{SB}/results'
    for model, rdir in PUBLIC:
        d = f'{R}/{rdir}'
        for size, mt, path in remote_find(d):
            rel = path[len(d) + 1:]
            if '/' in rel or not rel.endswith('.csv') or rel.count('.') > 1:
                continue                                     # .old / .partial / .pre_bbpatch side files stay behind
            stem = rel[:-4]
            if stem == 'timing':
                dst = f'{PB}/{model}/timing.csv'
            elif stem.startswith('pg_'):
                dst = f'{PB}/{model}/proteingym/{stem[3:]}.csv'
            else:
                dst = f'{PB}/{model}/{ds_name(stem)}.csv'
            plan.add(dst, path, size, mt, None, f'public:{model}')
    for model, rdir in PUBLIC_STATS:
        d = f'{R}/{rdir}'
        for size, mt, path in remote_find(d):
            plan.add(f'{PB}/{model}/stats/{path[len(d) + 1:]}', path, size, mt, None, f'public:{model}')
    # ProteinMPNN's ProteinGym runs (three, averaged in the tables)
    for tag, run in (('fast', 'proteinmpnn_020_unmasked_fast'), ('fast_seed1', 'proteinmpnn_020_unmasked_fast_seed1'), ('fast_seed2', 'proteinmpnn_020_unmasked_fast_seed2')):
        d = f'{DEV}/outputs/pgym/{run}'
        for size, mt, path in remote_find(d):
            plan.add(f'{PB}/proteinmpnn/proteingym/{tag}/{path[len(d) + 1:]}', path, size, mt, None, 'public:proteinmpnn')
    # ESM3 zero-shot through the Forge API
    P = f'{DEV}/analysis_notebooks/predictions'
    for size, mt, path in remote_find(P):
        m = re.match(rf'^{re.escape(P)}/([^/]+)/esm3_forge/zero_shot/(esm3-[a-z]+-\d{{4}}-\d\d)/predictions\.csv$', path)
        if m:
            plan.add(f'{PB}/esm3_forge_zero_shot/{m.group(2)}/{ds_name(m.group(1))}.csv', path, size, mt, None, 'public:esm3')
    # the per-DMS ProteinGym tables every model's PG rows are summarised from (derived results, a few hundred KB)
    for name in ('pgym_all_models_per_dms.csv', 'pgym_dc8_per_dms.csv', 'pgym_public_per_dms.csv', 'pgym_rosetta_singles_per_dms.csv',
                 'proteinmpnn_020_unmasked_fast_ensemble3_per_dms.csv'):
        plan.add(f'baselines/pgym_per_dms/{name}', f'{DEV}/outputs/pgym/{name}', *stat_remote(f'{DEV}/outputs/pgym/{name}'), None, 'public:pgym_tables')
    # label tables the harness predictions are keyed to (row_id), and the harness's own documentation and timings
    for size, mt, path in remote_find(f'{SB}/tasks'):
        plan.add(f'{PB}/_tasks/{ds_name(Path(path).name[:-7])}.csv.gz' if path.endswith('.csv.gz') else f'{PB}/_tasks/{Path(path).name}', path, size, mt, None, 'public:docs')
    for rel in ('AUDIT.md', 'PROGRESS.md', 'DDGemb.md', 'DDMut.md', 'ESM3dG.md', 'ESMtherm.md', 'GeoDDG.md', 'IFUM.md', 'ProSST.md', 'UniMutStab.md',
                'results/FINAL_TABLE.csv', 'results/FINAL_TABLE.md', 'results/SUMMARY.csv', 'results/SUMMARY.md', 'results/TIMING.csv', 'results/TIMING.md',
                'results/ifum_esmfold_cost_projection.csv', 'results/ifum_fold_cost_benchmark.csv', 'results/ifum_unfolded_seq_stats.csv',
                'results/rosetta/README.txt', 'closed_servers/PLAN.md'):
        plan.add(f'{PB}/docs/{rel.replace("results/", "").replace("/", "_")}', f'{SB}/{rel}', *stat_remote(f'{SB}/{rel}'), None, 'public:docs')


# ProteinGym ships every benchmarked model's published predictions (one CSV per DMS, ~99 model columns, 1.9 GB zipped).
# The tables use three of them (ProSST-2048, ProteinMPNN, ESM3); this keeps those columns and the labels, per DMS.
PG_PUBLISHED_COLS = ['mutant', 'DMS_score', 'ProSST-2048', 'ProteinMPNN', 'ESM3']
_PG_REMOTE = r'''
import csv, io, sys, tarfile, time, zipfile
cols = %r
z = zipfile.ZipFile(%r)
tar = tarfile.open(fileobj=sys.stdout.buffer, mode='w|')
for info in z.infolist():
    if info.is_dir() or not info.filename.endswith('.csv'):
        continue
    with z.open(info) as f:
        rd = csv.reader(io.TextIOWrapper(f, newline=''))
        head = next(rd)
        idx = [head.index(c) for c in cols]
        out = io.StringIO(newline='')
        w = csv.writer(out, lineterminator='\n')
        w.writerow(cols)
        for row in rd:
            w.writerow([row[i] for i in idx])
    data = out.getvalue().encode()
    ti = tarfile.TarInfo(info.filename.rsplit('/', 1)[-1])
    ti.size, ti.mtime = len(data), time.mktime(info.date_time + (0, 0, -1))
    tar.addfile(ti, io.BytesIO(data))
tar.close()
'''


def pg_published():
    """Extract the three published-score columns per DMS into baselines/public/proteingym_published/ (streams a tar over ssh)."""
    dest = ROOT / 'baselines/public/proteingym_published'
    dest.mkdir(parents=True, exist_ok=True)
    code = _PG_REMOTE % (PG_PUBLISHED_COLS, f'{SB}/pgym_public/zero_shot_substitutions_scores.zip')
    ssh = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', HOST, 'python3 -'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ssh.stdin.write(code.encode())
    ssh.stdin.close()
    tar = subprocess.run(['tar', '-x', '-C', str(dest)], stdin=ssh.stdout)
    if ssh.wait() or tar.returncode:
        sys.exit('ProteinGym published-score extraction failed')
    print(f'{len(list(dest.glob("*.csv")))} DMS files in {dest}')


# ------------------------------------------------------------------------------------------------ execution
def build_plan(groups):
    plan = Plan()
    for g in groups:
        {'msr': plan_msr, 'retrained': plan_retrained, 'public': plan_public}[g](plan)
    dsts = [p['dst'] for p in plan]
    dup = {d for d in dsts if dsts.count(d) > 1}
    if dup:
        sys.exit(f'destination collisions: {sorted(dup)[:5]}')
    return plan


def summarise(plan):
    import collections
    agg = collections.defaultdict(lambda: [0] * 6)
    for p in plan:
        a = agg[p['group']]
        have = (ROOT / p['dst']).exists() and (ROOT / p['dst']).stat().st_size == p['size']
        k = 4 if have else (2 if p['local'] else 0)
        a[k] += 1
        a[k + 1] += p['size']
    print(f'{"group":28s} {"fetch":>7s} {"GB":>7s} {"move":>7s} {"GB":>7s} {"done":>7s} {"GB":>7s}')
    for g, a in sorted(agg.items()):
        print(f'{g:28s} {a[0]:7d} {a[1] / 1e9:7.2f} {a[2]:7d} {a[3] / 1e9:7.2f} {a[4]:7d} {a[5] / 1e9:7.2f}')


def run(plan):
    fetch = []
    for p in plan:
        dst = ROOT / p['dst']
        if dst.exists() and dst.stat().st_size == p['size']:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p['local']:
            os.replace(p['local'], dst)                          # same filesystem: a rename, mtime kept
        else:
            fetch.append(p)
    print(f'moved local files; {len(fetch)} to fetch ({sum(p["size"] for p in fetch) / 1e9:.2f} GB)', flush=True)
    if fetch:
        STAGE.mkdir(exist_ok=True)
        lst = STAGE / 'files.txt'
        lst.write_text('\n'.join(p['src'][len(RHOME) + 1:] for p in fetch) + '\n')
        subprocess.run(['rsync', '-a', '--partial', '-e', 'ssh -o BatchMode=yes', f'--files-from={lst}', f'{HOST}:{RHOME}/', str(STAGE) + '/'], check=True)
        grew = []
        for p in fetch:
            staged = STAGE / p['src'][len(RHOME) + 1:]
            dst = ROOT / p['dst']
            if not staged.exists():
                sys.exit(f'fetch incomplete: {p["src"]}')
            st = staged.stat()
            if st.st_size != p['size']:
                # written to while we copied (a run still in flight on the cluster, usually a run.log); rsync's copy is a
                # consistent snapshot, so keep it, record what we really hold, and say so
                grew.append(f'{p["dst"]}: {p["size"]} -> {st.st_size} bytes')
                p['size'], p['mtime'] = st.st_size, st.st_mtime
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, dst)
        shutil.rmtree(STAGE, ignore_errors=True)
        if grew:
            print(f'{len(grew)} files changed on the cluster while they were copied (runs still in flight); the snapshot was kept:')
            print('\n'.join('  ' + g for g in grew[:30]))
    return plan


def write_manifests(plan):
    by_dir = {}
    for p in plan:
        parts = Path(p['dst']).parts
        # one manifest per model folder: baselines/<retrained|public>/<model>, msr_predictions/<spec>, baselines/<table folder>
        n = 3 if parts[0] == 'baselines' else 2
        top = parts[:n] if len(parts) > n else parts[:-1]
        by_dir.setdefault(Path(*top), []).append(p)
    for d, items in by_dir.items():
        def md5(p):
            h = hashlib.md5()
            with open(ROOT / p['dst'], 'rb') as f:
                for b in iter(lambda: f.read(1 << 22), b''):
                    h.update(b)
            return h.hexdigest()
        with ThreadPoolExecutor(4) as ex:
            sums = list(ex.map(md5, items))
        with open(ROOT / d / 'MANIFEST.tsv', 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(['path', 'bytes', 'md5', 'origin_on_cluster', 'origin_mtime'])
            for p, s in sorted(zip(items, sums), key=lambda x: x[0]['dst']):
                w.writerow([str(Path(p['dst']).relative_to(d)), (ROOT / p['dst']).stat().st_size, s, p['src'].replace(RHOME, '~'),
                            time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(p['mtime']))])
        print(f'manifest {d}: {len(items)} files')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('cmd', choices=['plan', 'run', 'manifest', 'pgpublished'])
    ap.add_argument('--groups', default='msr,retrained,public')
    a = ap.parse_args()
    if a.cmd == 'pgpublished':
        pg_published()
        sys.exit()
    plan = build_plan(a.groups.split(','))
    if a.cmd == 'plan':
        summarise(plan)
    elif a.cmd == 'run':
        run(plan)
        write_manifests(plan)
    else:
        write_manifests(plan)
