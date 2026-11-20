#!/usr/bin/env python
"""Inventory of the baseline-augmentation runs ``bl_w2a4_m16a16_capped_*``: main-baseline code plus one change at a time
towards the ``FINAL_p0`` recipe (Tobit regression, no rank loss, censored ListMLE, links).

Reads ``<TRAINING_LOGS>/<run>/0/{hparams.yaml,metrics.csv}`` (default ``msr_analysis.paths.TRAINING_LOGS``; override via
``MSR_TRAINING_LOGS`` or ``--logs``). Only whitelisted hyperparameter fields are read (never secrets/API keys).

Writes ``inventory.csv`` and ``README.md`` in ``analysis/audits/baseline_bridge/``.
Usage: ``python -m audits.baseline_bridge.inventory [--logs DIR]``
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from msr_analysis.paths import TRAINING_LOGS, atomic_to_csv, atomic_write_text

HERE = Path(__file__).resolve().parent
LOGS = TRAINING_LOGS
CONTROL = 'bl_w2a4_m16a16_capped_s1'
SKIP = {
    'experiment_name', 'checkpoint_filename', 'log_dir', 'version', 'offline', 'checkpoint_path', 'cache_path',
    'raw_data_file', 'benchmark_data_path', 'mut_structures_root', 'af_model_folder', 'load_lora_checkpoint',
    'model_device', 'gpus', 'num_workers', 'log_every_n_steps', 'regenerate_cache', 'offline_model', 'seed',
    'num_sanity_val_steps',
}
SECRET = re.compile(r'comet|key|token|secret', re.I)
METRICS = ['val_rho_combined_avg', 'val_rho_wt_avg', 'val_rho_combined_external', 'val_rmse_combined_external', 'val_epi_composite']

GROUPS = [
    ('0 control', lambda n: re.fullmatch(r'bl_w2a4_m16a16_capped_s[123]', n),
     'main-baseline recipe (MSE + hinge regression, ListMLE rank terms): the released "ESM-MSR baseline" triplicate'),
    ('1 link', lambda n: '_link' in n,
     'learned observed-scale links (old code only: tag archive/link-code-2026-11-05)'),
    ('2 tobit', lambda n: re.fullmatch(r'.*_tobit_s[123]', n),
     'regression = noisy-floor Tobit NLL (--reg_nll tobit), everything else the baseline'),
    ('3 tobit, no rank loss', lambda n: '_tobit_norank_' in n,
     'Tobit + both ListMLE rank terms off'),
    ('4 tobit + censored ListMLE', lambda n: '_tobit_rank05_reg15_cens' in n,
     'Tobit + rank 0.5 / reg 15 + censored ListMLE at --censor_threshold'),
]


def group_of(name: str) -> str:
    for g, f, _ in GROUPS:
        if f(name):
            return g
    return '9 other'


def read_run(d: Path) -> dict:
    hp_path = d / '0' / 'hparams.yaml'
    h = yaml.safe_load(hp_path.read_text(encoding='utf-8')) if hp_path.exists() else {}
    row = {'run': d.name, 'group': group_of(d.name), 'seed': h.get('seed')}
    mpath = d / '0' / 'metrics.csv'
    if mpath.exists():
        m = pd.read_csv(mpath, low_memory=False)
        ep = m.dropna(subset=['val_rho_combined_avg']) if 'val_rho_combined_avg' in m else m.iloc[0:0]
        row['epochs_validated'] = int(ep['epoch'].nunique()) if len(ep) else 0
        row['num_epochs_planned'] = h.get('num_epochs')
        if len(ep):
            best = ep.loc[ep['val_rho_combined_avg'].idxmax()]
            row['best_epoch'] = int(best['epoch'])
            for c in METRICS:
                if c in ep:
                    row[f'best_{c}'] = round(float(best[c]), 4) if pd.notna(best[c]) else None
                    row[f'last_{c}'] = round(float(ep.iloc[-1][c]), 4) if pd.notna(ep.iloc[-1][c]) else None
    sub_dir = d / '0'
    files = sorted(p.name for p in sub_dir.glob('*') if p.is_file()) if sub_dir.is_dir() else []
    row['checkpoints'] = sum(f.endswith('.ckpt') for f in files)
    row['dumps'] = sum(f.startswith('val_dump') for f in files)
    row['has_last_ckpt'] = 'last.ckpt' in files
    done, plan = row.get('epochs_validated', 0), row.get('num_epochs_planned')
    row['status'] = (
        'aborted before the first epoch (hparams only)' if not row.get('epochs_validated') else
        'complete' if plan and done >= plan else f'incomplete ({done}/{plan} epochs)'
    )
    if d.name.startswith('_aborted'):
        row['status'] = f'aborted ({done} epochs, power outage)'
    row['size_gb'] = round(sum(p.stat().st_size for p in sub_dir.glob('*') if p.is_file()) / 1e9, 2) if sub_dir.is_dir() else 0.0
    row['_hparams'] = {k: v for k, v in h.items() if not SECRET.search(k)}
    return row


def differs(h: dict, h0: dict) -> dict:
    return {
        k: (h.get(k), h0.get(k))
        for k in sorted(set(h) | set(h0))
        if k not in SKIP and not SECRET.search(k) and h.get(k) != h0.get(k)
    }


def build_inventory(logs: str | Path = LOGS) -> pd.DataFrame:
    logs_path = Path(logs).expanduser()
    runs = [read_run(d) for d in sorted(logs_path.iterdir()) if d.is_dir()]
    h0 = next((r['_hparams'] for r in runs if r['run'] == CONTROL), {})
    for r in runs:
        r['changes_vs_control_s1'] = '; '.join(
            f'{k}: {a} (control {b})' for k, (a, b) in differs(r['_hparams'], h0).items()
        )
    df = pd.DataFrame([{k: v for k, v in r.items() if k != '_hparams'} for r in runs]).sort_values(['group', 'run'])
    atomic_to_csv(df, HERE / 'inventory.csv', index=False)

    lines = [
        '# Baseline augmentation runs `bl_w2a4_m16a16_capped_*`',
        '',
        f'{len(df)} run directories in `{logs_path}` (configured via `MSR_TRAINING_LOGS` / `--logs`). '
        'Generated by `audits/baseline_bridge/inventory.py`; `inventory.csv` holds every column. '
        'Metrics are validation `val_rho_combined_avg` (best epoch / last epoch) from `metrics.csv`; '
        '"changes" list hyperparameters differing from control seed 1.',
        '',
    ]
    for g, _, desc in GROUPS + [('9 other', None, 'not matched by a group')]:
        sub = df[df['group'] == g]
        if sub.empty:
            continue
        lines += [f'## {g}', '', desc, '', '| run | seed | status | best (epoch) | last | ckpts | dumps | GB |', '|---|---|---|---|---|---|---|---|']
        for _, r in sub.iterrows():
            ep_str = int(r.best_epoch) if pd.notna(r.get('best_epoch')) else '-'
            lines.append(
                f"| `{r.run}` | {r.seed} | {r.status} | {r.get('best_val_rho_combined_avg')} ({ep_str}) | "
                f"{r.get('last_val_rho_combined_avg')} | {r.checkpoints} | {r.dumps} | {r.size_gb} |"
            )
        lines.append('')
        for _, r in sub.iterrows():
            if r.changes_vs_control_s1:
                lines.append(f'- `{r.run}`: {r.changes_vs_control_s1}')
        lines.append('')

    cl = HERE / 'inventory_cluster.csv'
    if cl.exists():
        c = pd.read_csv(cl)
        lines += [
            '## Cluster only (not copied to WSL)',
            '',
            'Found on `subhak-gpu` in `~/esm-msr-devel/sweep_baseline_nov07/runs/` (Nov 7 main-baseline sweep, code `main-baseline @ 8ddc2c2`). '
            'Checkpoints and val dumps remain on the cluster; `inventory_cluster.csv` stores the metadata.',
            '',
            '| run | seed | status | best (epoch) | last | ckpts | dumps | GB |',
            '|---|---|---|---|---|---|---|---|',
        ]
        lines += [
            f"| `{x.run}` | {x.seed} | {x.status} | {x.get('best_val_rho_combined_avg')} ({int(x.best_epoch)}) | "
            f"{x.get('last_val_rho_combined_avg')} | {x.checkpoints} | {x.dumps} | {x.size_gb} |"
            for _, x in c.iterrows()
        ]
        lines.append('')

    rem = HERE / 'removed_from_wsl.csv'
    if rem.exists():
        r = pd.read_csv(rem)
        lines += [
            '## Removed from WSL (Nov 19)',
            '',
            'Copies deleted from `training_logs/` because they do not match the finalized ablations; archived identically in `/mnt/g/msr_revisions_extras/archive/training_logs/`.',
            '',
            '| run | status then | best val_rho_combined_avg | reason |',
            '|---|---|---|---|',
        ]
        lines += [f"| `{x.run}` | {x.status} | {x.get('best_val_rho_combined_avg')} | {x.reason} |" for _, x in r.iterrows()]
        lines.append('')

    atomic_write_text(HERE / 'README.md', '\n'.join(lines))
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--logs', default=str(LOGS), help='Path to training_logs directory (default: MSR_TRAINING_LOGS)')
    args = ap.parse_args(argv)
    df = build_inventory(args.logs)
    print(df[['run', 'group', 'status', 'best_epoch', 'best_val_rho_combined_avg', 'checkpoints', 'size_gb']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
