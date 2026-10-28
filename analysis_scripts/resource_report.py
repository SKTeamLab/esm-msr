"""Summarise resource-tracker runs (inference_scripts/resource_tracker.py) into one table per model and dataset.

    python analysis_scripts/resource_report.py RUN_DIR [RUN_DIR ...] [--out summary.csv] [--proteins proteins.csv]

A RUN_DIR holds resources_run.json and resources_scopes.jsonl (give a parent directory and every run below it is read).
Inference cost of a dataset = its `protein` rows only (the inference calls; data handling and model loading are not in
them). Per (model, dataset): proteins, rows, inference wall seconds, rows per second, CPU-core-seconds and mean cores busy,
peak host RAM and peak GPU memory over its proteins, plus the cost outside inference that the run recorded (`load` rows,
the ME featurization; `stage` rows, model loading). Each row also says which run it came from, the GPU, and whether the
run was in the fixed slot and alone on the node, so a contaminated measurement is visible instead of averaged in.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'inference_scripts'))
from resource_tracker import load_scopes  # noqa: E402


def find_runs(paths):
    runs = []
    for p in map(Path, paths):
        runs += [p] if (p / 'resources_run.json').exists() else sorted(q.parent for q in p.rglob('resources_run.json'))
    return runs


def run_flags(meta):
    chk = (meta.get('limits') or {}).get('check')
    start, end = meta.get('neighbours_at_start'), meta.get('neighbours_at_end')
    return {
        'slot_checked': chk is not None,
        'slot_ok': None if chk is None else bool(chk['ok']) and 'ram_limit_violation' not in meta,
        'neighbours_start': None if start is None else start.get('other_jobs'),
        'neighbours_end': None if end is None else end.get('other_jobs'),
        'gpu': (meta.get('gpu') or {}).get('name'),
        'job': (meta.get('slurm') or {}).get('SLURM_JOB_ID'),
    }


def summarise(run_dir):
    meta = json.load(open(Path(run_dir) / 'resources_run.json'))
    sc = load_scopes(run_dir)
    if sc.empty:
        return pd.DataFrame(), pd.DataFrame()
    flags = run_flags(meta)
    prot = sc[sc.level == 'protein'].copy()
    rows = []
    for ds, g in prot.groupby('dataset', sort=False):
        other = sc[(sc.dataset == ds) & (sc.level != 'protein')]
        wall = g.wall_s.sum()
        rows.append({
            'model': meta['label'], 'run_id': meta['run_id'], 'dataset': ds, 'n_proteins': len(g),
            'n_rows': g.get('n_rows', pd.Series(dtype=float)).sum(), 'inference_wall_s': wall,
            'rows_per_s': g.get('n_rows', pd.Series(dtype=float)).sum() / wall if wall else None,
            'cpu_core_s': g.cpu_s.sum(), 'cpu_cores_mean': g.cpu_s.sum() / wall if wall else None,
            'ram_peak_gb': g.ram_rss_peak_gb.max(), 'vram_alloc_peak_gb': g.vram_alloc_peak_gb.max(),
            'vram_device_peak_gb': g.vram_device_peak_gb.max(),
            'load_wall_s': sc[(sc.level == 'load') & (sc.dataset == ds)].wall_s.sum() if (sc.level == 'load').any() else 0.0,
            'dataset_wall_s': other[other.level == 'dataset'].wall_s.sum(), **flags})
    stages = sc[sc.level == 'stage']
    for _, r in stages.iterrows():
        rows.append({'model': meta['label'], 'run_id': meta['run_id'], 'dataset': f'(stage) {r["name"]}', 'n_proteins': 0,
                     'inference_wall_s': 0.0, 'dataset_wall_s': r.wall_s, 'cpu_core_s': r.cpu_s, 'ram_peak_gb': r.ram_rss_peak_gb,
                     'vram_alloc_peak_gb': r.vram_alloc_peak_gb, 'vram_device_peak_gb': r.vram_device_peak_gb, **flags})
    prot.insert(0, 'model', meta['label'])
    for k, v in flags.items():
        prot[k] = v
    return pd.DataFrame(rows), prot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--proteins', default=None, help='also write every protein row of every run to this CSV')
    a = ap.parse_args()
    runs = find_runs(a.paths)
    if not runs:
        sys.exit('no resources_run.json under the given paths')
    parts = [summarise(r) for r in runs]
    summary = pd.concat([s for s, _ in parts if len(s)], ignore_index=True)
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(summary.round(3).to_string(index=False))
    if a.out:
        summary.to_csv(a.out, index=False)
    if a.proteins:
        pd.concat([p for _, p in parts if len(p)], ignore_index=True).to_csv(a.proteins, index=False)


if __name__ == '__main__':
    main()
