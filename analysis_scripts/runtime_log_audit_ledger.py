import re, subprocess, io, sys, importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location('bm', '/home/sreeves/consolidation/benchmark_matrix.py'); bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
out = subprocess.run(['sacct', '-u', 'sreeves', '-S', '2026-10-01', '-X', '-P', '--format=JobID,JobName,Start,End,Elapsed,State,NodeList,ReqMem,AllocTRES'],
                     capture_output=True, text=True).stdout
d = pd.read_csv(io.StringIO(out), sep='|')
d = d[d.JobName.notna() & ~d.JobName.isin(['bash', 'interactive'])]
print(f'{len(d)} jobs since 2026-10-01; states: {d.State.str.split().str[0].value_counts().to_dict()}')
cls, un = [], []
for _, r in d.iterrows():
    hit = next((i for i, (p, *_) in enumerate(bm.JOBS) if re.search(p, r.JobName)), None)
    (cls if hit is not None else un).append(r)
U = pd.DataFrame(un)
print(f'matched by a ledger pattern: {len(cls)}; NOT matched: {len(U)}')
if len(U):
    g = U.groupby('JobName').agg(n=('JobID', 'count'), node=('NodeList', lambda s: ','.join(sorted(set(s.astype(str))))), state=('State', lambda s: ','.join(sorted(set(x.split()[0] for x in s)))),
                                 first=('Start', 'min')).reset_index()
    print(g.sort_values('first').to_string(index=False, max_colwidth=34))
# jobs the ledger would count that did not finish: still running, cancelled, failed
cl = pd.DataFrame(cls)
print('\nmatched jobs by state:', cl.State.str.split().str[0].value_counts().to_dict())
# elapsed sanity: sacct Elapsed vs End-Start
def secs(x):
    h = x.split('-'); d0 = int(h[0]) if len(h) == 2 else 0; t = h[-1].split(':'); return d0 * 86400 + int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
fin = cl[cl.End.notna() & (cl.End != 'Unknown')]
dt = (pd.to_datetime(fin.End) - pd.to_datetime(fin.Start)).dt.total_seconds()
bad = fin[(dt - fin.Elapsed.map(secs)).abs() > 120]
print(f'finished jobs whose Elapsed disagrees with End-Start by >2 min: {len(bad)}')
# the committed CSV vs now
old = pd.read_csv('/home/sreeves/esm-msr-devel/docs/benchmarks/slurm_jobs_oct2026.csv') if False else None
