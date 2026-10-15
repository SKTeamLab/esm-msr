"""Rebuild a run's Comet experiment with validation metrics scored on the FULL population.

Comet has no per-metric overwrite - APIExperiment exposes delete_asset/delete_parameter/delete_tags but no
delete_metric - so an in-place correction is impossible. The equivalent is to delete the experiment and re-upload
it complete, which is lossless here: metrics.csv holds every metric Comet had except `process duration` and the
sys.* system series, hparams.yaml holds the config, and the val dumps regenerate every censoring-affected metric.

Non-val metrics (train/*, mem/*, lr, link/*) are copied verbatim at their original steps. Every val_* metric is
recomputed from that epoch's dump with cens forced to 0.

Usage (from the repo root, with PYTHONPATH=src and COMET_API_KEY set). Dry-run first; it prints the row counts and
the before/after headline, and refuses to run if the validation-row count and the dump count disagree:

    python analysis_scripts/comet_reupload.py <log_dir>/0 --name "<comet experiment name>" \
        --delete-key <key of the experiment being replaced> [--execute]

Omit --delete-key to upload alongside an existing experiment. Never pass the key of a RUNNING experiment: the live
job is still writing to it, and deleting it server-side breaks its logger.
"""
import argparse, glob, os, sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comet_rescore import load_dump, recompute

PROJECT = 'esm-msr-agent-oct06-capped'


def plan(run_dir):
    csv = pd.read_csv(os.path.join(run_dir, 'metrics.csv'), low_memory=False)
    val_cols = [c for c in csv.columns if c.startswith('val_')]
    other_cols = [c for c in csv.columns if c not in val_cols and c not in ('epoch', 'step')]
    is_val = csv[val_cols].notna().any(axis=1)

    dumps = [os.path.join(run_dir, 'val_dump_zs.npz')] + sorted(
        glob.glob(os.path.join(run_dir, 'val_dump_e*.npz')),
        key=lambda p: int(p.rsplit('_e', 1)[1].split('.')[0]))
    dumps = [p for p in dumps if os.path.exists(p)]
    val_idx = list(csv.index[is_val])
    if len(val_idx) != len(dumps):
        raise SystemExit(f"{run_dir}: {len(val_idx)} validation rows but {len(dumps)} dumps - refusing to guess")

    records, dump_for = [], dict(zip(val_idx, dumps))
    for i, row in csv.iterrows():
        payload = {c: float(row[c]) for c in other_cols if pd.notna(row[c])}
        if i in dump_for:
            payload.update(recompute(load_dump(dump_for[i]), use_cens=False))
        if payload:
            records.append((int(row['step']), int(row['epoch']), payload, dump_for.get(i)))
    return csv, records, dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--name', required=True, help='Comet experiment name to create')
    ap.add_argument('--delete-key', default=None, help='experiment key to delete first (the one being replaced)')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()

    csv, records, dumps = plan(a.run_dir)
    nval = sum(1 for r in records if r[3])
    print(f"{a.run_dir}\n  rows to log: {len(records)}   validation rows rescored: {nval}")
    print(f"  dumps: {[os.path.basename(d) for d in dumps]}")
    ex = next(r for r in records if r[3])
    print(f"  example rescored row: step={ex[0]} epoch={ex[1]} metrics={len(ex[2])}  "
          f"val_rho_combined_avg={ex[2].get('val_rho_combined_avg'):.4f}")
    last = [r for r in records if r[3]][-1]
    print(f"  final rescored row:   step={last[0]} epoch={last[1]}  "
          f"val_rho_combined_avg={last[2].get('val_rho_combined_avg'):.4f}  "
          f"(was {csv['val_rho_combined_avg'].dropna().iloc[-1]:.4f})")
    print(f"  will delete key: {a.delete_key or '(none)'}\n  will create name: {a.name}")
    if not a.execute:
        print("\n  DRY RUN - pass --execute to apply")
        return

    import comet_ml
    api = comet_ml.API()
    if a.delete_key:
        print(f"  deleting {a.delete_key} ...", api.delete_experiment(a.delete_key))
    exp = comet_ml.Experiment(project_name=PROJECT, log_code=False, log_graph=False,
                              auto_metric_logging=False, auto_param_logging=False,
                              auto_output_logging='simple', log_env_details=False)
    exp.set_name(a.name)
    exp.add_tags(['rescored_full_population'])
    exp.log_other('rescored_note', 'val_* metrics recomputed over all items (censoring applies to the loss only)')
    with open(os.path.join(a.run_dir, 'hparams.yaml')) as f:
        exp.log_parameters(yaml.safe_load(f))
    tr = os.path.join('src', 'esm_msr', 'training.py')
    if os.path.exists(tr):
        exp.log_code(file_name=tr)
    for step, epoch, payload, _ in records:
        exp.log_metrics(payload, step=step, epoch=epoch)
    exp.end()
    print(f"  uploaded: {exp.url}")


if __name__ == '__main__':
    main()
