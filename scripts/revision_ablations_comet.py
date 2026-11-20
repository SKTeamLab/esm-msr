#!/usr/bin/env python
"""Apply scripts/revision_ablations_rescore.py's output to the Comet project revision-ablations. Dry run by default; nothing is ever deleted.

  list     read-only inventory of the project: experiment, key, running / idle, which have the pooled series, which still carry a misapplied link.
  append   add the NEW series (val_[unmasked_]rho_combined_pooled / rho_singles_wt_pooled) to existing experiments. Adding a metric name an experiment does not have yet is
           non-destructive, so this is done in place (comet_ml.ExistingExperiment), once per (metric, step); what is already there is skipped (idempotent).
  correct  experiments whose logged RMSE / observed-scale epistasis went through a link they should not have had (the --reg_nll none runs): Comet has no per-metric overwrite, so
           the experiment is REBUILT: a copy with the same name, parameters, tags (+ link_corrected_2026-11-20) and code assets, every point at its original step and epoch except
           the corrected ones (and with the new pooled series), verified value by value against the original; only then is the original RETIRED (renamed SUPERSEDED_<name>, tags
           superseded + link_misapplied), never deleted (the user keeps superseded runs as references). --delete-original exists and is not the default.

The Comet key is read by comet_ml from COMET_API_KEY or ~/.comet.config and never printed. The uncached API client is used throughout: the cached one replays stale reads (one
experiment once looked 571 points short for 80 minutes). Comet shows sent points minutes late: a run's points are sent ONCE, then polled with back-off, and only what is still
absent is re-sent (re-sending on a timer once left an experiment with up to 38 copies of a point, which Comet cannot remove).

  PYTHONPATH=src python scripts/revision_ablations_comet.py list
  PYTHONPATH=src python scripts/revision_ablations_comet.py append  --points DIR [--names A B ...] [--execute]
  PYTHONPATH=src python scripts/revision_ablations_comet.py correct --points DIR [--names A B ...] [--execute] [--target-project SCRATCH] [--keep-original-name]

--target-project builds the corrected copy in another project and leaves the original untouched (a rehearsal on one run first, as the earlier rebuilds did).
An experiment is refused while it is running or its newest metric is younger than --min-idle-min (a run whose flag still says running long after its last point needs
--ignore-running-flag, after checking that its job is really over).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import comet_ml

WORKSPACE = 'sareeves96'
PROJECT = 'revision-ablations'
SYSTEM = ('sys.', 'process duration')
AUTO_PARAMS = ('curr_step', 'curr_epoch')                      # the SDK maintains these itself from the last logged step / epoch
TAG = 'link_corrected_2026-11-20'
RETIRE_TAGS = ['superseded', 'link_misapplied']
POOLED = ('rho_combined_pooled', 'rho_singles_wt_pooled')
OFF = dict(auto_param_logging=False, auto_metric_logging=False, auto_output_logging=False, log_code=False, log_graph=False, log_env_details=False, log_git_metadata=False,
           log_git_patch=False, log_env_gpu=False, log_env_cpu=False, log_env_host=False, log_env_network=False, log_env_disk=False, auto_log_co2=False,
           display_summary_level=0, parse_args=False)


def new_api():
    return comet_ml.API(cache=False)


def listing(api, project=PROJECT):
    r = api._client.get_from_endpoint('experiments', {'workspaceName': WORKSPACE, 'projectName': project, 'archived': False})
    return sorted(((e.get('experimentName') or '', e['experimentKey'], bool(e.get('running'))) for e in r['experiments']))


def find(api, name, project=PROJECT):
    """The experiment of a points file: its name as given, or with the '-0' version suffix the training runs carry."""
    hits = [x for x in listing(api, project) if x[0] in (name, f'{name}-0')]
    if len(hits) != 1:
        sys.exit(f'{name}: {len(hits)} experiments match in {project} ({[h[0] for h in hits]})')
    return hits[0]


def metrics_of(e):
    pts = [p for p in e.get_metrics() if not p['metricName'].startswith(SYSTEM)]
    return pts


def idle_minutes(pts):
    ts = [float(p['timestamp']) for p in pts if p.get('timestamp')]
    return (time.time() * 1000 - max(ts)) / 60_000 if ts else float('inf')


def refusal(e, meta_running, pts, a):
    why = []
    idle = idle_minutes(pts)
    if meta_running and not a.ignore_running_flag:
        why.append('RUNNING (flag)')
    if idle < a.min_idle_min:
        why.append(f'newest metric {idle:.0f} min old')
    return why


def group(points):
    g = {}
    for p in points:
        g.setdefault((None if p['step'] is None else int(p['step']), None if p['epoch'] is None else int(p['epoch'])), {})[p['metricName']] = float(p['metricValue'])
    return sorted(g.items(), key=lambda kv: (-1 if kv[0][0] is None else kv[0][0], -1 if kv[0][1] is None else kv[0][1]))


def have(pts):
    return {(p['metricName'], None if p['step'] is None else int(p['step'])): float(p['metricValue']) for p in pts}


def load_points(d, names):
    out = []
    for f in sorted(Path(d).glob('*.json')):
        r = json.loads(f.read_text())
        if names and r['name'] not in names:
            continue
        out.append(r)
    return out


def aligned(r, pts):
    """The pooled points sit at the steps of the validations Comet already holds (val_rho_combined_avg): refuse a points file that does not line up."""
    steps = {s for (m, s) in have(pts) if m == 'val_rho_combined_avg'}
    want = {x['step'] for x in r['new']} | {x['step'] for x in r['corrected']}
    return want <= steps, sorted(want - steps)


def settle_check(api, key, want, wait_min, interval=60):
    """Poll an experiment until every wanted (metric, step) is there with its value; returns (n_missing, n_wrong, n_duplicate_names). Waits up to wait_min minutes."""
    t0, backoff = time.time(), interval
    while True:
        e = api.get_experiment_by_key(key)
        got, dup = {}, 0
        for p in metrics_of(e):
            k = (p['metricName'], None if p['step'] is None else int(p['step']))
            dup += k in got
            got[k] = float(p['metricValue'])
        missing = [k for k in want if k not in got]
        wrong = [k for k, v in want.items() if k in got and abs(got[k] - v) > 1e-9 * max(1.0, abs(v))]
        if not missing or time.time() - t0 > wait_min * 60:
            return len(missing), len(wrong), dup
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 300)


def cmd_list(a):
    api = new_api()
    print(f'{"experiment":42s} {"key":8s} run   idle(min) points  val series  pooled  link-sensitive')
    for name, key, running in listing(api, a.project):
        e = api.get_experiment_by_key(key)
        pts = metrics_of(e)
        names = {p['metricName'] for p in pts}
        print(f'{name:42s} {key[:8]} {"RUN " if running else "    "}  {idle_minutes(pts):9.0f} {len(pts):6d} {sum(n.startswith("val_") for n in names):10d} '
              f'{"yes" if any(n.endswith(POOLED) for n in names) else "no":>6s}  {sum("rmse" in n or n.endswith("epi_naive_rho_comb") for n in names)}', flush=True)


def cmd_append(a):
    api = new_api()
    for r in load_points(a.points, a.names):
        name, key, running = find(api, r['name'], a.project)
        e = api.get_experiment_by_key(key)
        pts = metrics_of(e)
        present = have(pts)
        todo = [x for x in r['new'] if (x['metric'], x['step']) not in present]
        ok, off = aligned(r, pts)
        why = refusal(e, running, pts, a)
        print(f"{name:34s} {key[:8]} new pooled points {len(r['new'])}, to send {len(todo)}, already there {len(r['new']) - len(todo)}; steps aligned: {ok}{'' if ok else ' (missing ' + str(off) + ')'}"
              f"{'  REFUSED: ' + ', '.join(why) if why else ''}")
        if not a.execute or not todo or why or not ok:
            continue
        exp = comet_ml.ExistingExperiment(previous_experiment=key, **OFF)
        for (step, epoch), vals in group([{'metricName': x['metric'], 'step': x['step'], 'epoch': x['epoch'], 'metricValue': x['value']} for x in todo]):
            exp.log_metrics(vals, step=step, epoch=epoch)
        exp.end()
        want = {(x['metric'], x['step']): x['value'] for x in r['new']}
        miss, wrong, dup = settle_check(api, key, want, a.settle_min)
        print(f'   sent {len(todo)} points once; read back: missing {miss}, wrong {wrong}, duplicate points {dup}', flush=True)
        if miss or wrong or dup:
            print('   NOT SETTLED: do not re-send by hand; run append again later (it only sends what is still absent)')


def read_old(api, key):
    e = api.get_experiment_by_key(key)
    return dict(exp=e, meta=e.get_metadata(), pts=metrics_of(e), params={p['name']: p['valueCurrent'] for p in e.get_parameters_summary() if p['name'] not in AUTO_PARAMS},
                tags=sorted(e.get_tags() or []), assets=e.get_asset_list(), git=e.get_git_metadata() or {})


def desired_points(old_pts, r):
    """The corrected experiment's points: every old point except the corrected ones, which take their new value, plus the new pooled series."""
    new_val = {(x['metric'], x['step']): x['new'] for x in r['corrected']}
    out, seen = [], set()
    for p in old_pts:
        k = (p['metricName'], None if p['step'] is None else int(p['step']))
        if k in seen:
            continue                                            # a duplicate point of the original (Comet keeps both): carried once
        seen.add(k)
        out.append({**p, 'metricValue': new_val.get(k, float(p['metricValue']))})
    out += [{'metricName': x['metric'], 'step': x['step'], 'epoch': x['epoch'], 'metricValue': x['value']} for x in r['new'] if (x['metric'], x['step']) not in seen]
    return out


def find_copy(api, project, name, key):
    for n, k, _ in listing(api, project):
        if n == name and k != key:
            x = api.get_experiment_by_key(k)
            if TAG in (x.get_tags() or []) and any(o.get('name') == 'orig_experiment_key' and o.get('valueCurrent') == key for o in x.get_others_summary()):
                return k
    return None


def rebuild(api, project, name, key, d, pts):
    new = comet_ml.Experiment(workspace=WORKSPACE, project_name=project, **OFF)
    new.set_name(name)
    new.add_tags(d['tags'] + [TAG])
    new.log_parameters(d['params'])
    for k, v in {'orig_experiment_key': key, 'orig_start_time_ms': d['meta'].get('startTimeMillis'), 'orig_end_time_ms': d['meta'].get('endTimeMillis'),
                 'link_correction': 'RMSE / observed-scale epistasis recomputed without the Tobit link (run trained with --reg_nll none); pooled rho series added'}.items():
        new.log_other(k, v)
    for gk in ('parent', 'branch', 'origin'):
        if d['git'].get(gk):
            new.log_other(f'orig_git_{gk}', d['git'][gk])
    for asset in d['assets']:
        data = d['exp'].get_asset(asset['assetId'], return_type='binary')
        if asset.get('type') == 'source_code':
            new.log_code(code=data.decode('utf-8', 'replace'), code_name=asset['fileName'])
        else:
            new.log_asset_data(data, name=asset['fileName'])
    for (step, epoch), vals in group(pts):
        new.log_metrics(vals, step=step, epoch=epoch)
    new_key = new.get_key()
    new.end()
    return new_key


def retire(api, key, name):
    e = api.get_experiment_by_key(key)
    if not name.startswith('SUPERSEDED_'):
        e.set_name('SUPERSEDED_' + name)
    e.add_tags(RETIRE_TAGS)


def cmd_correct(a):
    api = new_api()
    target = a.target_project or a.project
    for r in load_points(a.points, a.names):
        if not r['corrected']:
            print(f"{r['name']}: nothing to correct (the link was right for this run); use `append` for the pooled series")
            continue
        name, key, running = find(api, r['name'], a.project)
        d = read_old(api, key)
        why = refusal(d['exp'], running, d['pts'], a)
        ok, off = aligned(r, d['pts'])
        pts = desired_points(d['pts'], r)
        changed = len(r['corrected'])
        print(f"{name:34s} {key[:8]} points {len(d['pts'])} -> {len(pts)} ({changed} corrected values, {len(r['new'])} pooled), assets {len(d['assets'])}, tags {d['tags']}; "
              f"steps aligned: {ok}{'' if ok else ' (missing ' + str(off) + ')'}{'  REFUSED: ' + ', '.join(why) if why else ''}")
        if not a.execute or why or not ok:
            continue
        prior = find_copy(api, target, name, key)
        new_key = prior or rebuild(api, target, name, key, d, pts)
        want = have(pts)
        miss, wrong, dup = settle_check(api, new_key, want, a.settle_min)
        print(f'   copy {new_key[:8]}{" (found from an earlier run)" if prior else ""}: missing {miss}, wrong {wrong}, duplicate points {dup}', flush=True)
        if wrong or dup:
            sys.exit(f'STOP: the copy of {name} is WRONG; the original is untouched, the copy is left for inspection')
        if miss:
            print('   copy not complete yet: run again later (the same copy is found and checked; nothing was changed on the original)')
            continue
        if not a.target_project:
            retire(api, key, name)
            if a.delete_original:
                api.delete_experiment(key)
        Path(a.ledger).parent.mkdir(parents=True, exist_ok=True)
        with open(a.ledger, 'a') as f:
            f.write(f"{time.strftime('%F %T')} {name} {key} -> {new_key} ({len(pts)} points; {changed} corrected; {'copy only' if a.target_project else ('original DELETED' if a.delete_original else 'original retired')})\n")
        print(f"   corrected {name}: {key[:8]} -> {new_key[:8]}{'' if a.target_project else '; original retired as SUPERSEDED_' + name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['list', 'append', 'correct'])
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--points', help='directory of the rescoring script output (one <experiment name>.json each)')
    ap.add_argument('--names', nargs='*', help='only these points files / experiments')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--min-idle-min', type=float, default=90.0)
    ap.add_argument('--ignore-running-flag', action='store_true')
    ap.add_argument('--settle-min', type=float, default=20.0, help='how long to poll a sent experiment before reporting it as not settled')
    ap.add_argument('--target-project', help='correct: build the copy here and leave the original alone (rehearsal)')
    ap.add_argument('--delete-original', action='store_true', help='correct: delete instead of retiring the original (not the default; needs the user to have said so)')
    ap.add_argument('--ledger', default=str(Path.home() / 'playground' / 'esm-msr-devel' / 'final_rescore_nov20' / 'comet_correction_ledger.txt'))
    a = ap.parse_args()
    if a.cmd != 'list' and not a.points:
        sys.exit('--points DIR is required')
    {'list': cmd_list, 'append': cmd_append, 'correct': cmd_correct}[a.cmd](a)


if __name__ == '__main__':
    main()
