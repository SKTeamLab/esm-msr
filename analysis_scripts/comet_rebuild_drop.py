"""
Remove metrics from FINISHED Comet experiments by rebuilding them without those series.

Comet has no delete-metric call (APIExperiment deletes parameters, tags and assets, never a metric), so the only way to take a series off an
existing run is to copy the run into a new experiment WITHOUT the series and delete the old one. This does that, one run at a time and in an order
that cannot lose a run it could not verify:

  1. read the old experiment through the API (metrics, parameters, tags, assets, git metadata);
  2. create the new experiment in the same project with the same name and tags (+ a 'rebuilt' tag), re-log every kept metric at its original
     step and epoch, every parameter, the code / conda assets, and record the old key, git parent / branch and start / end time as 'orig_*' others;
  3. read the NEW experiment back and compare every kept (metric, step) value with the old one;
  4. only if all of them match (and the parameters and tag sets match): delete the old experiment and append old key -> new key to the ledger.

Lost by construction: the dropped series, the sys.* / 'process duration' system series (they are not metrics Comet lets a script write), the
original timestamps and run duration, and the experiment key / URL (the ledger maps old to new). Never run it on a running experiment: it
refuses any whose metadata says running or whose last metric is newer than --min-idle-min minutes, and the selection is explicit (--names-file).

  # dry run (the default): reads only, prints what would be dropped and what would be lost
  python analysis_scripts/comet_rebuild_drop.py --project esm-msr-oct24-final --match '^(blk|ts_)'
  # real run, only the experiments named in a file (one experiment NAME per line; every non-archived experiment of that name is rebuilt)
  python analysis_scripts/comet_rebuild_drop.py --project esm-msr-oct24-final --names-file done.txt --execute

The Comet key is read by comet_ml from COMET_API_KEY or ~/.comet.config; it is never printed.
"""
import argparse
import os
import re
import sys
import time

import comet_ml

WORKSPACE = 'sareeves96'
# val_[unmasked_][diag_]epi_<metric>_epi: the MT head at twice the reported scale; the _comb twins (and the MT legacy cell rank) are kept.
DROP = r'^val_(unmasked_)?(diag_)?epi_.+_epi$'
SYSTEM = ('sys.', 'process duration')
AUTO_PARAMS = ('curr_step', 'curr_epoch')                       # the SDK maintains these itself from the last logged step / epoch
LEDGER = '/home/sareeves/playground/esm-msr-devel/dc_runs/comet_rebuild_ledger.txt'
TAG = 'rebuilt_2026-11-03_no_epi_head'


def listing(api, project, match):
    pat = re.compile(match, re.I)
    out = []
    r = api._client.get_from_endpoint('experiments', {'workspaceName': WORKSPACE, 'projectName': project, 'archived': False})
    for e in r['experiments']:
        name = e.get('experimentName') or ''
        if pat.search(name):
            out.append((name, e['experimentKey']))
    return sorted(out)


def read(api, key):
    e = api.get_experiment_by_key(key)
    md = e.get_metadata()
    pts = e.get_metrics()
    keep, dropped, system = [], [], []
    drop = re.compile(DROP)
    for p in pts:
        n = p['metricName']
        (system if n.startswith(SYSTEM) else dropped if drop.match(n) else keep).append(p)
    return dict(exp=e, meta=md, keep=keep, dropped=dropped, system=system, params={p['name']: p['valueCurrent'] for p in e.get_parameters_summary() if p['name'] not in AUTO_PARAMS},
                tags=sorted(e.get_tags() or []), assets=e.get_asset_list(), git=e.get_git_metadata() or {},
                last=max([float(p['timestamp']) for p in pts if p.get('timestamp')] or [0.0]))


def report(name, key, d, idle_min):
    why = []
    if d['meta'].get('running'):
        why.append('RUNNING')
    if idle_min is not None and (time.time() * 1000 - d['last']) < idle_min * 60_000:
        why.append(f'metric < {idle_min} min old')
    print(f"{name:34s} {key[:8]}  points {len(d['keep']) + len(d['dropped']) + len(d['system']):6d}: keep {len(d['keep']):6d}  drop {len(d['dropped']):5d}  "
          f"system(lost) {len(d['system']):5d}  assets {len(d['assets'])}  tags {d['tags']}  {'SKIP: ' + ', '.join(why) if why else ''}")
    return not why


def group(points):
    g = {}
    for p in points:
        step = None if p['step'] is None else int(p['step'])
        epoch = None if p['epoch'] is None else int(p['epoch'])
        g.setdefault((step, epoch), {})[p['metricName']] = float(p['metricValue'])
    return sorted(g.items(), key=lambda kv: (-1 if kv[0][0] is None else kv[0][0], kv[0][1] if kv[0][1] is not None else -1))


def rebuild(api, project, name, key, d):
    off = dict(auto_param_logging=False, auto_metric_logging=False, auto_output_logging=False, log_code=False, log_graph=False, log_env_details=False,
               log_git_metadata=False, log_git_patch=False, log_env_gpu=False, log_env_cpu=False, log_env_host=False, log_env_network=False,
               log_env_disk=False, auto_log_co2=False, display_summary_level=0, parse_args=False)
    new = comet_ml.Experiment(workspace=WORKSPACE, project_name=project, **off)
    new.set_name(name)
    new.add_tags(d['tags'] + [TAG])
    new.log_parameters(d['params'])
    orig = {'orig_experiment_key': key, 'orig_start_time_ms': d['meta'].get('startTimeMillis'), 'orig_end_time_ms': d['meta'].get('endTimeMillis')}
    for k in ('parent', 'branch', 'origin'):
        if d['git'].get(k):
            orig[f'orig_git_{k}'] = d['git'][k]
    for k, v in orig.items():
        new.log_other(k, v)
    for a in d['assets']:                                       # the code snapshot is the run's provenance: copy every asset
        data = d['exp'].get_asset(a['assetId'], return_type='binary')
        if a.get('type') == 'source_code':
            new.log_code(code=data.decode('utf-8', 'replace'), code_name=a['fileName'])
        else:
            new.log_asset_data(data, name=a['fileName'])
    for (step, epoch), vals in group(d['keep']):
        new.log_metrics(vals, step=step, epoch=epoch)
    new_key = new.get_key()
    new.end()
    return new_key


def verify(api, new_key, d, timeout_s=900):
    """Poll until the new experiment holds every kept point, every parameter and the tags (the server ingests asynchronously); returns (ok, message)."""
    want = {(p['metricName'], None if p['step'] is None else int(p['step'])): float(p['metricValue']) for p in d['keep']}
    t0, why = time.time(), ''
    while True:
        e = api.get_experiment_by_key(new_key)
        got = {}
        for p in e.get_metrics():
            if not p['metricName'].startswith(SYSTEM):
                got[(p['metricName'], None if p['step'] is None else int(p['step']))] = float(p['metricValue'])
        params = {p['name']: p['valueCurrent'] for p in e.get_parameters_summary() if p['name'] not in AUTO_PARAMS}
        missing = [k for k in want if k not in got]
        if missing:
            why = f'{len(missing)} of {len(want)} kept points not there yet'
        elif params != d['params']:
            why = f'parameters differ ({len(params)} of {len(d["params"])} there)'
        elif sorted(e.get_tags() or []) != sorted(d['tags'] + [TAG]):
            why = 'tags differ'
        else:
            bad = [k for k, v in want.items() if abs(got[k] - v) > 1e-9 * max(1.0, abs(v))]
            extra = [k for k in got if k not in want]
            return (False, f'{len(bad)} values differ, {len(extra)} unexpected points') if (bad or extra) else (True, f'{len(want)} points, {len(params)} parameters identical')
        if time.time() - t0 > timeout_s:
            return False, why
        time.sleep(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--match', default='^(blk|ts_)', help='experiment-name regex (dry run listing)')
    ap.add_argument('--names-file', help='with --execute: only experiments with these NAMES (one per line) are rebuilt')
    ap.add_argument('--min-idle-min', type=float, default=90.0, help='skip an experiment whose newest metric is younger than this')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--target-project', help='create the rebuilt copies in this project instead (a dry rehearsal: use with --keep-old)')
    ap.add_argument('--keep-old', action='store_true', help='verify the copy but do NOT delete the original')
    ap.add_argument('--stop-file', help='checked before every rebuild: if this file exists the run stops cleanly')
    a = ap.parse_args()
    api = comet_ml.API()
    names = None
    if a.execute:
        if not a.names_file:
            sys.exit('--execute needs --names-file: the selection is explicit')
        names = {l.strip() for l in open(a.names_file) if l.strip() and not l.startswith('#')}
    todo = [(n, k) for n, k in listing(api, a.project, a.match) if names is None or n in names]
    print(f"{len(todo)} experiments selected; dropping /{DROP}/; {'EXECUTING' if a.execute else 'dry run'}")
    done = 0
    for name, key in todo:
        if a.stop_file and os.path.exists(a.stop_file):
            print(f'stop file {a.stop_file} exists: stopping before {name}')
            break
        d = read(api, key)
        ok = report(name, key, d, a.min_idle_min)
        if not a.execute or not ok or not d['dropped']:
            continue
        new_key = rebuild(api, a.target_project or a.project, name, key, d)
        good, msg = verify(api, new_key, d)
        if not good:
            sys.exit(f'STOP: {name} {key[:8]} -> {new_key[:8]} did not verify ({msg}); the old experiment is untouched, the new one is left for inspection')
        if not a.keep_old:
            api.delete_experiment(key)
        with open(LEDGER, 'a') as f:
            f.write(f"{time.strftime('%F %T')} {name} {key} -> {new_key}  ({msg}; dropped {len(d['dropped'])} points{'; OLD KEPT' if a.keep_old else ''})\n")
        print(f"   rebuilt {name}: {key[:8]} -> {new_key[:8]}  {msg}{'  (old kept)' if a.keep_old else ''}")
        done += 1
    print(f'done: {done} rebuilt')


if __name__ == '__main__':
    main()
