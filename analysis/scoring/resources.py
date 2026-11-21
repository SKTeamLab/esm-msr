"""Unified resource-tracker run reporting and cross-model slot summary.

Subcommands:
  python analysis/scoring/resources.py report RUN_DIR [RUN_DIR ...] [--out summary.csv] [--proteins proteins.csv]
  python analysis/scoring/resources.py summary [RUNS_ROOT] [--out DIR] [--docs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT / 'analysis'), str(_REPO_ROOT / 'inference_scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from msr_analysis.paths import (  # noqa: E402
    INFERENCE_RESOURCES_DIR,
    MSR_ROOT,
    REPO_ROOT,
    RESULTS_DIR,
    atomic_to_csv,
    atomic_write_text,
)
from resource_tracker import load_scopes  # noqa: E402

ROOT = MSR_ROOT
OUT = INFERENCE_RESOURCES_DIR
DOCS = REPO_ROOT / 'docs'

# label -> (model, variant, seed, transfer-learned on Homology-Restricted MegaScale)
LABELS = {
    'esm_msr_baseline_s1': ('ESM-MSR', 'baseline recipe', 1, True),
    'esm_msr_devel_s1': ('ESM-MSR', 'dc8 (devel)', 1, True),
    'esm_msr_released': ('ESM-MSR', 'released checkpoint', None, True),
    'esm_msr_fl0_s1': ('ESM-MSR', 'fl0 (PG WT-only)', 1, True),
    'esm_msr_fl0_s2': ('ESM-MSR', 'fl0 (PG WT-only)', 2, True),
    'esm_msr_fl0_s3': ('ESM-MSR', 'fl0 (PG WT-only)', 3, True),
    'esm_msr_final_p0_wt_s1': ('ESM-MSR', 'FINAL_p0, WT head only (all sets)', 1, True),
    'me_st1_s1': ('MutateEverything', 'stage 1', 1, True),
    'me_st1_s1_a4500': ('MutateEverything', 'stage 1', 1, True),
    'me_st2_s1': ('MutateEverything', 'stage 2', 1, True),
    'me_st2_s2': ('MutateEverything', 'stage 2', 2, True),
    'me_st2_s3': ('MutateEverything', 'stage 2', 3, True),
    'thermompnn_s1': ('ThermoMPNN', 'single + ThermoMPNN-D', 1, True),
    'spurs_retrain_s1': ('SPURS', 'retrained (single + multi)', 1, True),
    'spurs_retrain_s2': ('SPURS', 'retrained (single + multi)', 2, True),
    'spurs_retrain_s3': ('SPURS', 'retrained (single + multi)', 3, True),
    'spurs_published': ('SPURS', 'published checkpoint', None, False),
    'proteinmpnn': ('ProteinMPNN', 'zero-shot', None, False),
}
GROUPS = ['Tsuboyama', 'External', 'DMS', 'Domainome', 'ProteinGym']
DOC_ROWS = [
    ('ESM-MSR, WT + MT passes', ['esm_msr_devel_s1'], 'dc8 (devel) seed 1'),
    ('ESM-MSR, WT head only', ['esm_msr_final_p0_wt_s1', 'esm_msr_fl0_s1'], None),
    ('MutateEverything stage 2', ['me_st2_s1'], 'seed 1'),
    ('MutateEverything stage 1', ['me_st1_s1_a4500'], 'seed 1, singles'),
    ('ThermoMPNN + ThermoMPNN-D', ['thermompnn_s1'], 'seed 1'),
    ('SPURS + SPURS-multi', ['spurs_retrain_s1'], 'retrained, seed 1'),
    ('ProteinMPNN', ['proteinmpnn'], 'zero-shot'),
]
PENDING = {'me_st1_s1_a4500': 'gpu4 job 388811', 'esm_msr_final_p0_wt_s1': 'gpu4 job 388812'}
BEGIN, END = '<!-- BEGIN resource_summary.py --docs -->', '<!-- END resource_summary.py --docs -->'
EXT = {'s571', 's783', 's2648', 's8754', 's669', 's461', 'ssym', 'q3421', 'k3822', 'k2369', 'ptmul', 'ptmuld'}


def find_runs(paths) -> list[Path]:
    runs: list[Path] = []
    for p in map(Path, paths):
        runs += [p] if (p / 'resources_run.json').exists() else sorted(q.parent for q in p.rglob('resources_run.json'))
    return runs


def run_flags(meta: dict) -> dict:
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


def summarise_run(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise one resource-tracker run directory into (per_dataset_df, per_protein_df)."""
    run_path = Path(run_dir)
    meta = json.loads((run_path / 'resources_run.json').read_text())
    sc = load_scopes(run_path)
    if sc.empty:
        return pd.DataFrame(), pd.DataFrame()
    flags = run_flags(meta)
    prot = sc[sc.level == 'protein'].copy().rename(columns={'model': 'network'})
    rows = []
    for ds, g in prot.groupby('dataset', sort=False):
        other = sc[(sc.dataset == ds) & (sc.level != 'protein')]
        wall = g.wall_s.sum()
        rows.append({
            'model': meta['label'],
            'run_id': meta['run_id'],
            'dataset': ds,
            'n_proteins': len(g),
            'n_rows': g.get('n_rows', pd.Series(dtype=float)).sum(),
            'inference_wall_s': wall,
            'rows_per_s': g.get('n_rows', pd.Series(dtype=float)).sum() / wall if wall else None,
            'cpu_core_s': g.cpu_s.sum(),
            'cpu_cores_mean': g.cpu_s.sum() / wall if wall else None,
            'ram_peak_gb': g.ram_rss_peak_gb.max(),
            'vram_alloc_peak_gb': g.vram_alloc_peak_gb.max(),
            'vram_device_peak_gb': g.vram_device_peak_gb.max(),
            'load_wall_s': sc[(sc.level == 'load') & (sc.dataset == ds)].wall_s.sum() if (sc.level == 'load').any() else 0.0,
            'dataset_wall_s': other[other.level == 'dataset'].wall_s.sum(),
            **flags,
        })
    stages = sc[sc.level == 'stage']
    for _, r in stages.iterrows():
        rows.append({
            'model': meta['label'],
            'run_id': meta['run_id'],
            'dataset': f'(stage) {r["name"]}',
            'n_proteins': 0,
            'inference_wall_s': 0.0,
            'dataset_wall_s': r.wall_s,
            'cpu_core_s': r.cpu_s,
            'ram_peak_gb': r.ram_rss_peak_gb,
            'vram_alloc_peak_gb': r.vram_alloc_peak_gb,
            'vram_device_peak_gb': r.vram_device_peak_gb,
            **flags,
        })
    prot.insert(0, 'model', meta['label'])
    for k, v in flags.items():
        prot[k] = v
    return pd.DataFrame(rows), prot


def group_of(ds) -> str:
    d = str(ds)
    for pre in ('external_', 'dms_'):
        if d.startswith(pre):
            d = d[len(pre):]
    if d.startswith(('tsuboyama', 'oct06_capped_allbl')):
        return 'Tsuboyama'
    if d in EXT:
        return 'External'
    if d.endswith(('_Faure_2021_abundance_domain', '_Faure_2021_binding_domain')) or d in (
        'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain',
    ):
        return 'DMS'
    if d == 'domainome':
        return 'Domainome'
    return 'ProteinGym'


def load_label(label_dir: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    label_path = Path(label_dir)
    runs = sorted(p.parent for p in label_path.glob('*/resources_run.json'))
    scopes, metas = [], []
    for r in runs:
        s = load_scopes(r)
        if len(s):
            scopes.append(s)
        metas.append(json.loads((r / 'resources_run.json').read_text()))
    return (pd.concat(scopes, ignore_index=True) if scopes else pd.DataFrame()), metas


def summarise_tracked(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise fixed-slot benchmark resource runs across ``LABELS`` into (per_group_df, per_run_df)."""
    root_path = Path(root)
    per_group, per_run = [], []
    for label, (model, variant, seed, tl) in LABELS.items():
        d = root_path / label
        if not d.is_dir():
            continue
        s, metas = load_label(d)
        if s.empty:
            continue
        flags = [run_flags(m) for m in metas]
        prot = s[s['level'] == 'protein'].copy()
        prot['group'] = prot['dataset'].map(group_of)
        ok = prot['status'] == 'ok'
        load = s[s['level'] == 'stage']['wall_s'].sum()
        for g in GROUPS:
            p = prot[prot['group'] == g]
            if p.empty:
                continue
            po = p[p['status'] == 'ok']
            wall = po['wall_s'].sum()
            per_group.append(dict(
                label=label, model=model, variant=variant, seed=seed, transfer_learned=tl, group=g,
                n_proteins=len(po), n_failed=int((p['status'] != 'ok').sum()), n_rows=po['n_rows'].sum(),
                wall_h=wall / 3600, rows_per_s=po['n_rows'].sum() / wall if wall else np.nan,
                cpu_cores_mean=po['cpu_s'].sum() / wall if wall else np.nan,
                ram_peak_gb=po['ram_rss_peak_gb'].max(), vram_device_peak_gb=po['vram_device_peak_gb'].max(),
                vram_alloc_peak_gb=po['vram_alloc_peak_gb'].max(),
            ))
        po = prot[ok]
        per_run.append(dict(
            label=label, model=model, variant=variant, seed=seed, transfer_learned=tl,
            gpu='; '.join(sorted({f['gpu'] for f in flags if f['gpu']})),
            jobs=','.join(sorted({str(f['job']) for f in flags if f['job']})),
            slot_ok=all(f['slot_ok'] for f in flags if f['slot_checked']) if any(f['slot_checked'] for f in flags) else None,
            n_proteins=len(po), n_failed=int((~ok).sum()),
            failed=', '.join(sorted(set(prot.loc[~ok, 'name'].astype(str))))[:300],
            n_rows=po['n_rows'].sum(), inference_h=po['wall_s'].sum() / 3600, model_load_s=load,
            ram_peak_gb=po['ram_rss_peak_gb'].max(), vram_device_peak_gb=po['vram_device_peak_gb'].max(),
        ))
    return pd.DataFrame(per_group), pd.DataFrame(per_run)


def markdown(pg: pd.DataFrame, pr: pd.DataFrame) -> str:
    lines = [
        '| Model | Variant | Seed | GPU | Inference h | Proteins (failed) | Peak RAM GB | Peak GPU GB | '
        + ' | '.join(f'{g} h' for g in GROUPS) + ' |',
        '|' + '---|' * (8 + len(GROUPS)),
    ]
    for r in pr.itertuples():
        hours = pg[pg['label'] == r.label].set_index('group')['wall_h']
        lines.append(
            f'| {r.model}{"" if r.transfer_learned else " †"} | {r.variant} | {int(r.seed) if pd.notna(r.seed) else "—"} | '
            f'{r.gpu.replace("NVIDIA ", "")} | {r.inference_h:.2f} | {r.n_proteins} ({r.n_failed}) | {r.ram_peak_gb:.1f} | '
            f'{r.vram_device_peak_gb:.1f} | ' + ' | '.join(f'{hours[g]:.2f}' if g in hours else '—' for g in GROUPS) + ' |'
        )
    return '\n'.join(lines) + '\n\n† not transfer-learned by us (published checkpoint / zero-shot).\n'


def docs_table(pg: pd.DataFrame, pr: pd.DataFrame, docs_dir: str | Path | None = None) -> str:
    """Write ``docs/benchmarks/inference_resources_a4500.csv`` and update ``docs/INFERENCE_RESOURCES.md``."""
    d_dir = Path(docs_dir) if docs_dir is not None else DOCS
    rows = []
    md = [
        '| Model | Run | Job | Inference h | Tsuboyama h | External h | DMS h | Domainome h | ProteinGym h | '
        'Proteins (failed) | Peak RAM GB | Peak GPU GB |',
        '|' + '---|' * 12,
    ]
    present = set(pr['label']) if not pr.empty and 'label' in pr.columns else set()
    for name, cands, note in DOC_ROWS:
        lab = next((c for c in cands if c in present), None)
        r = pr[pr['label'] == lab].iloc[0] if lab else None
        if r is None or 'A4500' not in str(r['gpu']):
            md.append(f'| {name} | pending | {PENDING.get(cands[0], "—")} |' + ' — |' * 9)
            rows.append(dict(model=name, label=cands[0], status='pending'))
            continue
        hours = pg[pg['label'] == lab].set_index('group')['wall_h']
        run_note = note or r['variant']
        if lab != cands[0]:
            run_note += f' (until {cands[0]} lands)'
        md.append(
            f"| {name} | {run_note} | {r['jobs']} | {r['inference_h']:.2f} | "
            + ' | '.join(f'{hours[g]:.2f}' if g in hours else '—' for g in GROUPS)
            + f" | {r['n_proteins']} ({r['n_failed']}) | {r['ram_peak_gb']:.1f} | {r['vram_device_peak_gb']:.1f} |"
        )
        rows.append(dict(
            model=name, label=lab, status='measured', run=run_note, job=r['jobs'], gpu=r['gpu'],
            inference_h=r['inference_h'], **{f'{g.lower()}_h': hours.get(g, np.nan) for g in GROUPS},
            n_proteins=r['n_proteins'], n_failed=r['n_failed'], failed=r['failed'],
            ram_peak_gb=r['ram_peak_gb'], vram_device_peak_gb=r['vram_device_peak_gb'],
            model_load_s=r['model_load_s'],
        ))
    atomic_to_csv(pd.DataFrame(rows), d_dir / 'benchmarks' / 'inference_resources_a4500.csv', index=False, float_format='%.4f')
    block = '\n'.join(md)
    doc = d_dir / 'INFERENCE_RESOURCES.md'
    if doc.exists():
        t = doc.read_text()
        if BEGIN in t and END in t:
            atomic_write_text(doc, t[:t.index(BEGIN) + len(BEGIN)] + '\n' + block + '\n' + t[t.index(END):])
    return block


def cli_report(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Summarise resource-tracker runs into per-(model, dataset) tables.')
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--proteins', default=None, help='also write every protein row of every run to this CSV')
    a = ap.parse_args(argv)
    runs = find_runs(a.paths)
    if not runs:
        sys.exit('no resources_run.json under the given paths')
    parts = [summarise_run(r) for r in runs]
    summary = pd.concat([s for s, _ in parts if len(s)], ignore_index=True)
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(summary.round(3).to_string(index=False))
    if a.out:
        atomic_to_csv(summary, a.out, index=False)
    if a.proteins:
        prot_df = pd.concat([p for _, p in parts if len(p)], ignore_index=True)
        atomic_to_csv(prot_df, a.proteins, index=False)


def cli_summary(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='Summarise fixed-slot resource-tracker runs across all benchmark models.')
    ap.add_argument('root', nargs='?', default=str(ROOT / 'resources_tracked'))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--docs', action='store_true', help='also refresh docs/INFERENCE_RESOURCES.md and docs/benchmarks/inference_resources_a4500.csv')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pg, pr = summarise_tracked(Path(a.root))
    atomic_to_csv(pg, out / 'per_group.csv', index=False, float_format='%.4f')
    atomic_to_csv(pr, out / 'per_run.csv', index=False, float_format='%.4f')
    md = markdown(pg, pr)
    atomic_write_text(out / 'summary.md', md)
    print(md)
    if a.docs:
        print(docs_table(pg, pr))


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in ('-h', '--help'):
        ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
        ap.add_argument('cmd', choices=['report', 'summary'])
        ap.parse_args(args_list)
        return
    cmd, rest = args_list[0], args_list[1:]
    if cmd == 'report':
        cli_report(rest)
    elif cmd == 'summary':
        cli_summary(rest)
    else:
        raise SystemExit(f'unknown subcommand: {cmd}')


if __name__ == '__main__':
    main()
