"""Inference cost of every model in the benchmark, from the fixed-slot resource-tracker runs.

    python analysis/scoring/resource_summary.py [RUNS_ROOT] [--out DIR]

RUNS_ROOT (default <MSR_ROOT>/resources_tracked, filled by ``collect_predictions.py run --groups resources``) holds
<label>/<run>/resources_{run.json,scopes.jsonl} for each tracked job (outputs/tracked/<label> on gpu4). Attempts whose label
ends in _failed_<job> or _cancelled_<job> are skipped; an ESM-MSR label has two runs (the gamut, then ProteinGym), both read.

Inference cost = the tracker's `protein` scopes only (one inference call per protein; data handling and model loading are
excluded, as in resource_report.py). Datasets are mapped onto five groups whatever each model calls them (s669 /
external_s669, DLG4_... / dms_DLG4_..., tsuboyama_test / oct06_capped_allbl_test, ProteinGym as one scope or per assay).
Rows are what each model's own benchmark scored, so totals compare work done, and rows/s compares speed.

Writes to analysis/results/inference_resources/: per_group.csv (label x group), per_run.csv (one row per label: totals,
GPU, slot check), summary.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'inference_scripts'))
sys.path.insert(0, str(HERE))
from resource_report import run_flags  # noqa: E402
from resource_tracker import load_scopes  # noqa: E402

ROOT = Path(os.environ.get('MSR_ROOT', '~/playground/esm-msr-devel')).expanduser()
OUT = HERE.parent / 'results' / 'inference_resources'

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
DOCS = HERE.parents[1] / 'docs'
# The official table (docs/INFERENCE_RESOURCES.md): one run per model type, all in the gpu4 fixed slot (RTX A4500).
# Each row lists candidate labels in order of preference; the first that exists is used, else the row says `pending`.
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


def group_of(ds):
    d = str(ds)
    for pre in ('external_', 'dms_'):
        if d.startswith(pre):
            d = d[len(pre):]
    if d.startswith(('tsuboyama', 'oct06_capped_allbl')):
        return 'Tsuboyama'
    if d in EXT:
        return 'External'
    if d.endswith(('_Faure_2021_abundance_domain', '_Faure_2021_binding_domain')) or d in (
            'MYO_HUMAN_Kung_2025_display', 'ESTA_BACSU_Nutschel_2020_dTm', 'GB1_Wu_2016_binding_domain'):
        return 'DMS'
    if d == 'domainome':
        return 'Domainome'
    return 'ProteinGym'          # 'proteingym' as one scope, or one scope per assay (ME)


def load_label(label_dir):
    runs = sorted(p.parent for p in label_dir.glob('*/resources_run.json'))
    scopes, metas = [], []
    for r in runs:
        s = load_scopes(r)
        if len(s):
            scopes.append(s)
        metas.append(json.loads((r / 'resources_run.json').read_text()))
    return (pd.concat(scopes, ignore_index=True) if scopes else pd.DataFrame()), metas


def summarise(root):
    per_group, per_run = [], []
    for label, (model, variant, seed, tl) in LABELS.items():
        d = root / label
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
            per_group.append(dict(label=label, model=model, variant=variant, seed=seed, transfer_learned=tl, group=g,
                                  n_proteins=len(po), n_failed=int((p['status'] != 'ok').sum()), n_rows=po['n_rows'].sum(),
                                  wall_h=wall / 3600, rows_per_s=po['n_rows'].sum() / wall if wall else np.nan,
                                  cpu_cores_mean=po['cpu_s'].sum() / wall if wall else np.nan,
                                  ram_peak_gb=po['ram_rss_peak_gb'].max(), vram_device_peak_gb=po['vram_device_peak_gb'].max(),
                                  vram_alloc_peak_gb=po['vram_alloc_peak_gb'].max()))
        po = prot[ok]
        per_run.append(dict(label=label, model=model, variant=variant, seed=seed, transfer_learned=tl,
                            gpu='; '.join(sorted({f['gpu'] for f in flags if f['gpu']})),
                            jobs=','.join(sorted({str(f['job']) for f in flags if f['job']})),
                            slot_ok=all(f['slot_ok'] for f in flags if f['slot_checked']) if any(f['slot_checked'] for f in flags) else None,
                            n_proteins=len(po), n_failed=int((~ok).sum()),
                            failed=', '.join(sorted(set(prot.loc[~ok, 'name'].astype(str))))[:300],
                            n_rows=po['n_rows'].sum(), inference_h=po['wall_s'].sum() / 3600, model_load_s=load,
                            ram_peak_gb=po['ram_rss_peak_gb'].max(), vram_device_peak_gb=po['vram_device_peak_gb'].max()))
    return pd.DataFrame(per_group), pd.DataFrame(per_run)


def markdown(pg, pr):
    lines = ['| Model | Variant | Seed | GPU | Inference h | Proteins (failed) | Peak RAM GB | Peak GPU GB | '
             + ' | '.join(f'{g} h' for g in GROUPS) + ' |', '|' + '---|' * (8 + len(GROUPS))]
    for r in pr.itertuples():
        hours = pg[pg['label'] == r.label].set_index('group')['wall_h']
        lines.append(f'| {r.model}{"" if r.transfer_learned else " †"} | {r.variant} | {int(r.seed) if pd.notna(r.seed) else "—"} | '
                     f'{r.gpu.replace("NVIDIA ", "")} | {r.inference_h:.2f} | {r.n_proteins} ({r.n_failed}) | {r.ram_peak_gb:.1f} | '
                     f'{r.vram_device_peak_gb:.1f} | ' + ' | '.join(f'{hours[g]:.2f}' if g in hours else '—' for g in GROUPS) + ' |')
    return '\n'.join(lines) + '\n\n† not transfer-learned by us (published checkpoint / zero-shot).\n'


def docs_table(pg, pr):
    """Rows of the official A4500 table; writes docs/benchmarks/inference_resources_a4500.csv and fills the marked block of
    docs/INFERENCE_RESOURCES.md."""
    rows, md = [], ['| Model | Run | Job | Inference h | Tsuboyama h | External h | DMS h | Domainome h | ProteinGym h | '
                    'Proteins (failed) | Peak RAM GB | Peak GPU GB |', '|' + '---|' * 12]
    for name, cands, note in DOC_ROWS:
        lab = next((c for c in cands if c in set(pr['label'])), None)
        r = pr[pr['label'] == lab].iloc[0] if lab else None
        if r is None or 'A4500' not in str(r['gpu']):
            md.append(f'| {name} | pending | {PENDING.get(cands[0], "—")} |' + ' — |' * 9)
            rows.append(dict(model=name, label=cands[0], status='pending'))
            continue
        hours = pg[pg['label'] == lab].set_index('group')['wall_h']
        run = note or r['variant']
        if lab != cands[0]:
            run += f' (until {cands[0]} lands)'
        md.append(f"| {name} | {run} | {r['jobs']} | {r['inference_h']:.2f} | "
                  + ' | '.join(f'{hours[g]:.2f}' if g in hours else '—' for g in GROUPS)
                  + f" | {r['n_proteins']} ({r['n_failed']}) | {r['ram_peak_gb']:.1f} | {r['vram_device_peak_gb']:.1f} |")
        rows.append(dict(model=name, label=lab, status='measured', run=run, job=r['jobs'], gpu=r['gpu'],
                         inference_h=r['inference_h'], **{f'{g.lower()}_h': hours.get(g, np.nan) for g in GROUPS},
                         n_proteins=r['n_proteins'], n_failed=r['n_failed'], failed=r['failed'],
                         ram_peak_gb=r['ram_peak_gb'], vram_device_peak_gb=r['vram_device_peak_gb'],
                         model_load_s=r['model_load_s']))
    pd.DataFrame(rows).to_csv(DOCS / 'benchmarks' / 'inference_resources_a4500.csv', index=False, float_format='%.4f')
    block = '\n'.join(md)
    doc = DOCS / 'INFERENCE_RESOURCES.md'
    if doc.exists():
        t = doc.read_text()
        if BEGIN in t and END in t:
            doc.write_text(t[:t.index(BEGIN) + len(BEGIN)] + '\n' + block + '\n' + t[t.index(END):])
    return block


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('root', nargs='?', default=str(ROOT / 'resources_tracked'))
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--docs', action='store_true', help='also refresh docs/INFERENCE_RESOURCES.md and docs/benchmarks/inference_resources_a4500.csv')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pg, pr = summarise(Path(a.root))
    pg.to_csv(out / 'per_group.csv', index=False, float_format='%.4f')
    pr.to_csv(out / 'per_run.csv', index=False, float_format='%.4f')
    md = markdown(pg, pr)
    (out / 'summary.md').write_text(md)
    print(md)
    if a.docs:
        print(docs_table(pg, pr))


if __name__ == '__main__':
    main()
