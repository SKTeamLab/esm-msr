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

Copies are made back to back and verified in the background of the next one (the server can take many minutes to show a copy's parameters);
an original is deleted only after its OWN copy verifies, a copy that arrived WRONG stops the run with every original untouched, and a copy left
by an interrupted run is found again (same name, rebuild tag, 'orig_experiment_key') instead of being made twice.

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


# A _comb twin that was BACKFILLED onto an older run was recomputed from the validation dump, so it equals the logged _epi value only to the dump's precision
# (about 1e-4); a twin that differs by more than this is a different quantity and the _epi series stays.
EXACT_TOL = 2e-3
WORST = [0.0]                                     # the largest _epi-vs-twin difference accepted so far (reported per run)
EXACT = ('beyond_rho', 'beyond_lib', 'beyond_rho_pairavg', 'pair_rho', 'line_rho')   # bit-identical to their _comb twin (the diag clean err / bias too)
TWIN = re.compile(r'^(val_(?:unmasked_)?)(diag_)?epi_(.+)_epi$')


def twin_of(name):
    """(the metric that replaces a dropped ``_epi`` series, whether the two must be numerically identical)."""
    pre, diag, metric = TWIN.match(name).groups()
    if diag:
        return f'{pre}diag_epi_{metric}_comb', True
    if metric == 'cell_rank':
        return f'{pre}diag_epi_cell_rank_mt_legacy', True        # the MT head alone, which cell_rank_epi equals exactly
    return f'{pre}epi_{metric}_comb', metric in EXACT


def droppable(series):
    """The ``_epi`` series that may go: only those whose replacement is in the same experiment at every step (and equal to it where they must be).
    Older runs have no _comb or MT-legacy twin, and for them the _epi series is the only copy, so it stays."""
    out = set()
    for name in series:
        if not re.match(DROP, name):
            continue
        twin, exact = twin_of(name)
        if twin not in series or not set(series[name]) <= set(series[twin]):
            continue
        if exact:
            worst = max(abs(v - series[twin][st]) for st, v in series[name].items())
            WORST[0] = max(WORST[0], worst)
            if worst > EXACT_TOL:
                continue
        out.add(name)
    return out


def read(api, key):
    e = api.get_experiment_by_key(key)
    md = e.get_metadata()
    pts = e.get_metrics()
    series = {}
    for p in pts:
        if not p['metricName'].startswith(SYSTEM) and re.match(r'^val_', p['metricName']):
            series.setdefault(p['metricName'], {})[None if p['step'] is None else int(p['step'])] = float(p['metricValue'])
    WORST[0] = 0.0
    gone = droppable(series)
    worst = WORST[0]
    keep, dropped, system, no_twin = [], [], [], []
    for p in pts:
        n = p['metricName']
        if n.startswith(SYSTEM):
            system.append(p)
        elif n in gone:
            dropped.append(p)
        else:
            keep.append(p)
            if re.match(DROP, n):
                no_twin.append(n)
    return dict(worst=worst, exp=e, meta=md, keep=keep, dropped=dropped, system=system, no_twin=sorted(set(no_twin)),
                params={p['name']: p['valueCurrent'] for p in e.get_parameters_summary() if p['name'] not in AUTO_PARAMS},
                tags=sorted(e.get_tags() or []), assets=e.get_asset_list(), git=e.get_git_metadata() or {},
                last=max([float(p['timestamp']) for p in pts if p.get('timestamp')] or [0.0]))


def report(name, key, d, idle_min):
    why = []
    if d['meta'].get('running'):
        why.append('RUNNING')
    if idle_min is not None and (time.time() * 1000 - d['last']) < idle_min * 60_000:
        why.append(f'metric < {idle_min} min old')
    print(f"{name:34s} {key[:8]}  points {len(d['keep']) + len(d['dropped']) + len(d['system']):6d}: keep {len(d['keep']):6d}  drop {len(d['dropped']):5d}  "
          f"system(lost) {len(d['system']):5d}  assets {len(d['assets'])}  tags {d['tags']}  max|twin diff| {d['worst']:.1e}  {'KEPT (no twin): ' + str(len(d['no_twin'])) + ' _epi series  ' if d['no_twin'] else ''}{'SKIP: ' + ', '.join(why) if why else ''}")
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


def verify(api, new_key, d):
    """One look at the new experiment (the server ingests asynchronously, and parameters can lag the metrics by many minutes).
    Returns (True, msg) when every kept point, every parameter and the tags are identical, (None, why) while something has not arrived yet, and
    (False, why) when what has arrived is WRONG, which no waiting fixes."""
    want = {(p['metricName'], None if p['step'] is None else int(p['step'])): float(p['metricValue']) for p in d['keep']}
    e = api.get_experiment_by_key(new_key)
    got = {}
    for p in e.get_metrics():
        if not p['metricName'].startswith(SYSTEM):
            got[(p['metricName'], None if p['step'] is None else int(p['step']))] = float(p['metricValue'])
    params = {p['name']: p['valueCurrent'] for p in e.get_parameters_summary() if p['name'] not in AUTO_PARAMS}
    missing = [k for k in want if k not in got]
    if missing:
        return None, f'{len(missing)} of {len(want)} kept points not there yet'
    if len(params) < len(d['params']):
        return None, f'parameters not all there yet ({len(params)} of {len(d["params"])})'
    if params != d['params']:
        return False, 'parameters differ'
    if sorted(e.get_tags() or []) != sorted(d['tags'] + [TAG]):
        return None, 'tags not there yet' if TAG not in (e.get_tags() or []) else 'tags differ'
    bad = [k for k, v in want.items() if abs(got[k] - v) > 1e-9 * max(1.0, abs(v))]
    extra = [k for k in got if k not in want]
    if bad or extra:
        return False, f'{len(bad)} values differ, {len(extra)} unexpected points'
    return True, f'{len(want)} points, {len(params)} parameters identical'


def find_copy(api, project, name, key):
    """A copy of ``key`` made by an earlier (interrupted) run: same name, the rebuild tag, and its 'orig_experiment_key' other."""
    r = api._client.get_from_endpoint('experiments', {'workspaceName': WORKSPACE, 'projectName': project, 'archived': False})
    for e in r['experiments']:
        if e.get('experimentName') == name and e['experimentKey'] != key:
            x = api.get_experiment_by_key(e['experimentKey'])
            if TAG in (x.get_tags() or []) and any(o.get('name') == 'orig_experiment_key' and o.get('valueCurrent') == key for o in x.get_others_summary()):
                return e['experimentKey']
    return None


def finalize(api, pending, keep_old):
    """Delete the original of every pending (name, key, new_key, d) whose copy now verifies; returns the ones still waiting. A copy that is WRONG stops the run."""
    left = []
    for name, key, new_key, d in pending:
        ok, msg = verify(api, new_key, d)
        if ok is None:
            left.append((name, key, new_key, d))
            continue
        if not ok:
            sys.exit(f'STOP: {name} {key[:8]} -> {new_key[:8]} is wrong ({msg}); every original is untouched, the copy is left for inspection')
        if not keep_old:
            api.delete_experiment(key)
        with open(LEDGER, 'a') as f:
            f.write(f"{time.strftime('%F %T')} {name} {key} -> {new_key}  ({msg}; dropped {len(d['dropped'])} points{'; OLD KEPT' if keep_old else ''})\n")
        print(f"   rebuilt {name}: {key[:8]} -> {new_key[:8]}  {msg}{'  (old kept)' if keep_old else ''}", flush=True)
    return left


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
    ap.add_argument('--finalize-wait-min', type=float, default=120.0, help='after the last copy is made, keep checking this long for slow ones to arrive')
    a = ap.parse_args()
    api = comet_ml.API()
    names = None
    if a.execute:
        if not a.names_file:
            sys.exit('--execute needs --names-file: the selection is explicit')
        names = {l.strip() for l in open(a.names_file) if l.strip() and not l.startswith('#')}
    todo = [(n, k) for n, k in listing(api, a.project, a.match) if names is None or n in names]
    print(f"{len(todo)} experiments selected; dropping /{DROP}/; {'EXECUTING' if a.execute else 'dry run'}", flush=True)
    target = a.target_project or a.project
    pending = []
    for name, key in todo:
        if a.stop_file and os.path.exists(a.stop_file):
            print(f'stop file {a.stop_file} exists: no more copies', flush=True)
            break
        d = read(api, key)
        ok = report(name, key, d, a.min_idle_min)
        if not a.execute or not ok or not d['dropped']:
            continue
        new_key = find_copy(api, target, name, key) or rebuild(api, target, name, key, d)
        pending.append((name, key, new_key, d))
        pending = finalize(api, pending, a.keep_old)      # copies are made back to back; the originals go as soon as their copy has arrived
    t0 = time.time()
    while pending and time.time() - t0 < a.finalize_wait_min * 60 and not (a.stop_file and os.path.exists(a.stop_file)):
        time.sleep(60)
        pending = finalize(api, pending, a.keep_old)
    for name, key, new_key, d in pending:
        print(f'UNVERIFIED {name} {key[:8]} -> {new_key[:8]}: original kept, copy left in place', flush=True)
    print(f'done: {len(pending)} still waiting', flush=True)


if __name__ == '__main__':
    main()
