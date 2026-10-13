"""
Adds rescored validation metrics (the CSV of baseline/epi_from_dump.py) to an existing Comet experiment, at the step and epoch of the
validation each dump came from (read from the run's metrics.csv), under the names training logs today (val_epi_<metric>_<head>,
val_rho_wt_valid_avg, ...). Names the experiment already has are skipped, so nothing logged by the run itself is overwritten. The API key
is read from ~/.comet.config.

  PYTHONPATH=src python baseline/comet_upload_rescored.py RESCORED_CSV RUN_METRICS_CSV EXPERIMENT_KEY [--dry_run]
"""
import argparse
import configparser
import os
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rescored_csv')
    ap.add_argument('run_metrics_csv')
    ap.add_argument('experiment_key')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--provenance', default='')
    a = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser('~/.comet.config'))
    key = cfg['comet']['api_key'].strip()

    import comet_ml
    api = comet_ml.API(api_key=key)
    existing = {m['name'] for m in api.get_experiment_by_key(a.experiment_key).get_metrics_summary()}

    res = pd.read_csv(a.rescored_csv, index_col=0)
    run = pd.read_csv(a.run_metrics_csv)
    val = run[run['val_rho_combined_avg'].notna()][['epoch', 'step']].reset_index(drop=True)
    when = {'zs': (0, 0)}                                                     # the validation before training
    for i, r in val.iloc[1:].iterrows():
        when[f'e{int(r.epoch)}'] = (int(r.epoch), int(r.step))

    plan, skipped = [], set()
    for dump, row in res.iterrows():
        tag = re.search(r'val_dump_(\w+)\.npz', dump).group(1)
        epoch, step = when[tag]
        for col, v in row.items():
            if col.startswith('link_') or not pd.notna(v):
                continue
            name = f'val_{col}'
            if name in existing:
                skipped.add(name)
                continue
            plan.append((name, float(v), step, epoch))
    names = sorted({p[0] for p in plan})
    print(f'{len(plan)} values, {len(names)} new names, at {sorted(when.items(), key=lambda kv: kv[1][1])}')
    print('skipped (already on the experiment):', sorted(skipped))
    if a.dry_run:
        print('new names:', names)
        return
    exp = comet_ml.ExistingExperiment(api_key=key, experiment_key=a.experiment_key, log_env_details=False, log_env_gpu=False,
                                      log_env_cpu=False, log_env_host=False, log_git_metadata=False, log_git_patch=False,
                                      auto_output_logging=False, auto_metric_logging=False, auto_param_logging=False, parse_args=False,
                                      display_summary_level=0)
    for name, v, step, epoch in plan:
        exp.log_metric(name, v, step=step, epoch=epoch)
    if a.provenance:
        exp.log_other('rescored_metrics', a.provenance)
    exp.end()
    print('uploaded')


if __name__ == '__main__':
    main()
