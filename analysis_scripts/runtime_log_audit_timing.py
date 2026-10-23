import glob, os, io, subprocess
import pandas as pd
SB = '/home/sreeves/stability-baselines/results'
fs = sorted(glob.glob(f'{SB}/*/predictions*/timing.csv'))
print(f'{len(fs)} timing.csv files')
allrows = []
for f in fs:
    m, tag = f.split('/results/')[1].split('/')[:2]
    try: d = pd.read_csv(f)
    except Exception as e: print('UNREADABLE', f, e); continue
    d['model'], d['tag'] = m, tag; allrows.append(d)
D = pd.concat(allrows, ignore_index=True)
print(f'{len(D)} rows; columns: {list(D.columns)[:11]}')
print('\nrows per model/tag, hosts, gpus:')
g = D.groupby(['model', 'tag']).agg(rows=('dataset', 'count'), uniq=('dataset', 'nunique'), hosts=('host', lambda s: ','.join(sorted(set(map(str, s))))),
                                    gpus=('gpu', lambda s: ','.join(sorted(set(map(str, s))))[:36]), wall_h=('wall_s', lambda s: round(s.sum() / 3600, 2)), nan_total=('n_nan', 'sum')).reset_index()
print(g.to_string(index=False))
# problems
print('\n--- datasets run more than once in the same predictions dir (reruns / retries):')
dup = D[D.duplicated(['model', 'tag', 'dataset'], keep=False)].sort_values(['model', 'tag', 'dataset', 'unix_end'])
print(dup[['model', 'tag', 'dataset', 'n_rows', 'wall_s', 'n_nan', 'host', 'slurm_job']].to_string(index=False) if len(dup) else '(none)')
print('\n--- suspicious rows: wall_s<=0, ms_per_row inconsistent with wall/rows, missing host/gpu/slurm_job')
D['ms_chk'] = 1000 * D.wall_s / D.n_rows.clip(lower=1)
bad = D[(D.wall_s <= 0) | ((D.ms_chk - D.ms_per_row).abs() > 0.05) | D.host.isna() | D.gpu.isna() | D.slurm_job.isna()]
print(bad[['model', 'tag', 'dataset', 'n_rows', 'wall_s', 'ms_per_row', 'ms_chk', 'host', 'gpu', 'slurm_job']].to_string(index=False) if len(bad) else '(none)')
# job-level cross-check: the sum of dataset wall times of one job must not exceed that job's elapsed
ids = sorted({int(x) for x in D.slurm_job.dropna().astype(float).astype(int)})
out = subprocess.run(['sacct', '-j', ','.join(map(str, ids)), '-X', '-P', '-n', '--format=JobID,Elapsed,State,NodeList'], capture_output=True, text=True).stdout
S = pd.read_csv(io.StringIO(out), sep='|', header=None, names=['JobID', 'Elapsed', 'State', 'Node'])
def secs(x):
    p = x.split('-'); d0 = int(p[0]) if len(p) == 2 else 0; t = p[-1].split(':'); return d0 * 86400 + int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
S['el'] = S.Elapsed.map(secs); S['JobID'] = S.JobID.astype(str)
j = D.assign(JobID=D.slurm_job.dropna().astype(float).astype(int).astype(str)).groupby('JobID').agg(loop_s=('wall_s', 'sum'), n=('dataset', 'count')).reset_index().merge(S, on='JobID', how='left')
print(f'\n--- loop time vs job elapsed ({len(j)} jobs): loop time must be <= elapsed')
over = j[j.loop_s > j.el + 5]
print('jobs where summed dataset time EXCEEDS job elapsed:', len(over)); 
if len(over): print(over.to_string(index=False))
print('node mismatch (timing host vs sacct node):', int(((D.merge(S, left_on=D.slurm_job.astype(float).astype(int).astype(str), right_on='JobID', how='left').pipe(lambda x: x.host != x.Node)) & D.slurm_job.notna()).sum()))
