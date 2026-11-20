#!/usr/bin/env python3
"""Unified CLI orchestrator for all ESM-MSR manuscript & supplementary analysis pipelines.

Usage examples:
    python -m analysis.run --list
    python -m analysis.run --fast
    python -m analysis.run --only benchmarks_external,tsuboyama
    python -m analysis.run --only p02 --split mega
    python -m analysis.run --skip epistasis_mechanisms --studies
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class StageSpec:
    code: str
    module: str
    stage: str
    description: str
    supports_split: bool = False


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        'p01', 'p01_benchmarks_external', 'benchmarks_external',
        'Task spines, core Spearman bootstrap, 12 external thermostability sets, DMS & Domainome',
    ),
    StageSpec(
        'p02', 'p02_tsuboyama', 'tsuboyama',
        'Tsuboyama / MegaScale rankings, NDCG@96, ESM3 scoring variants, and mega_splits comparison',
        supports_split=True,
    ),
    StageSpec(
        'p03', 'p03_proteingym', 'proteingym',
        'ProteinGym held-out DMS evaluations, property sigma sweep, and ours-vs-published audit',
    ),
    StageSpec(
        'p04', 'p04_mgnify_rocklin', 'mgnify_rocklin',
        'MGnify & Rocklin de novo stability benchmarks, homology strata, and beyond-global epistasis',
    ),
    StageSpec(
        'p05', 'p05_summary_main', 'summary_main',
        'Main-text Figure 3 composite, thermostability & ProteinGym category summaries, ESM-MSR vs ESM3dG',
    ),
    StageSpec(
        'p06', 'p06_epistasis_hierarchy', 'epistasis_hierarchy',
        'Epistasis hierarchy (val/test), multi vs additive ranking, masking ablations, and main composite',
        supports_split=True,
    ),
    StageSpec(
        'p07', 'p07_epistasis_mechanisms', 'epistasis_mechanisms',
        'Assay floor saturation, Tobit link, mutant-backbone & 3D proximity mechanisms, and pair heatmaps',
        supports_split=True,
    ),
    StageSpec(
        'p08', 'p08_calibration_length', 'calibration_length',
        'Affine RMSE calibration transfer and sequence-length performance analysis',
    ),
    StageSpec(
        'p09', 'p09_screening', 'screening',
        'Experimental screening yields, biophysical property gain, ProteinGym Activity, and gain-law composite',
        supports_split=True,
    ),
    StageSpec(
        'p10', 'p10_splits_and_leakage', 'splits_and_leakage',
        'Split construction, 4-part holdout filter, baseline leakage matrices, and mega clean-vs-homologous test',
    ),
    StageSpec(
        'p11', 'p11_ablations_and_heads', 'ablations_and_heads',
        '11 FINAL_* architecture/loss/masking/data ablations, head decompositions, and inference resources',
    ),
    StageSpec(
        'p12', 'p12_manifest', 'manifest',
        'Validate computed summaries against reference benchmark tables and verify table_figure_map.csv',
    ),
)

STUDIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        'faure_epistasis',
        ('analysis.studies.faure_epistasis.run', 'studies.faure_epistasis.run'),
        'GRB2-SH3 & PSD95-PDZ3 binding-vs-folding biophysical epistasis study',
    ),
    (
        'sigma_mechanisms',
        ('analysis.studies.sigma_mechanisms.run', 'studies.sigma_mechanisms.run'),
        'LoRA gating scale sigma=0 -> 1 mechanistic decomposition study',
    ),
)


def _import_first(candidates: tuple[str, ...]):
    last_exc: Exception | None = None
    for mod_name in candidates:
        try:
            return importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name in {mod_name, mod_name.split('.')[0]}:
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise ModuleNotFoundError(str(candidates))


def _match_token(spec: StageSpec, token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    num = spec.code[1:]  # '01' .. '12'
    return t in {
        spec.code.lower(),
        num,
        str(int(num)),
        spec.module.lower(),
        spec.stage.lower(),
    }


def _match_study(study_name: str, token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    sn = study_name.lower()
    return t in {'studies', sn, f'study:{sn}'}


def _flatten_tokens(raw: list[str] | str | None) -> list[str]:
    if not raw:
        return []
    items = [raw] if isinstance(raw, str) else list(raw)
    return [t.strip() for item in items for t in item.split(',') if t.strip()]


def _select_stages(only_arg: list[str] | str | None, skip_arg: list[str] | str | None) -> list[StageSpec]:
    selected = list(STAGES)
    tokens = _flatten_tokens(only_arg)
    if tokens:
        unknown = [
            t for t in tokens
            if not any(_match_token(s, t) for s in STAGES)
            and not any(_match_study(nm, t) for nm, _, _ in STUDIES)
        ]
        if unknown:
            valid = ', '.join([f'{s.code}:{s.stage}' for s in STAGES] + [f'study:{nm}' for nm, _, _ in STUDIES])
            raise SystemExit(f'Unknown --only stage(s): {unknown}. Valid targets: {valid}')
        selected = [s for s in STAGES if any(_match_token(s, t) for t in tokens)]
    skip_tokens = _flatten_tokens(skip_arg)
    if skip_tokens:
        selected = [s for s in selected if not any(_match_token(s, t) for t in skip_tokens)]
    return selected


def _select_studies(
    only_arg: list[str] | str | None,
    skip_arg: list[str] | str | None,
    studies_flag: bool,
) -> list[tuple[str, tuple[str, ...], str]]:
    tokens = _flatten_tokens(only_arg)
    selected: list[tuple[str, tuple[str, ...], str]] = []
    if studies_flag:
        selected = list(STUDIES)
    elif tokens:
        selected = [st for st in STUDIES if any(_match_study(st[0], t) for t in tokens)]
    skip_tokens = _flatten_tokens(skip_arg)
    if skip_tokens:
        selected = [st for st in selected if not any(_match_study(st[0], t) for t in skip_tokens)]
    return selected


def _print_list() -> int:
    print('Pipeline stages (execution order):')
    for s in STAGES:
        split_note = ' [--split capped|mega]' if s.supports_split else ''
        print(f'  {s.code}  {s.stage:<22} ({s.module}){split_note}')
        print(f'       {s.description}')
    print('\nOptional mechanistic studies (--studies):')
    for name, _, desc in STUDIES:
        print(f'  study:{name:<20} {desc}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='analysis.run',
        description='Unified runner for ESM-MSR analysis pipelines (p01..p12) and optional studies.',
    )
    ap.add_argument('--list', action='store_true', help='List all registered pipeline stages and exit')
    ap.add_argument('--only', nargs='+', default=None, help='Comma- or space-separated stage names/codes to run (e.g. p01,tsuboyama)')
    ap.add_argument('--skip', nargs='+', default=None, help='Comma- or space-separated stage names/codes to skip')
    ap.add_argument('--studies', action='store_true', help='Also run standalone mechanistic studies (faure_epistasis, sigma_mechanisms)')
    ap.add_argument('--split', choices=('capped', 'mega'), default='capped', help="Tsuboyama split for split-aware stages (default: 'capped')")
    ap.add_argument('--B', type=int, default=1000, help='Bootstrap replicates forwarded to every stage (default: 1000)')
    ap.add_argument('--fast', action='store_true', help='Fast smoke-test mode (sets B=100 across all stages)')
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        return _print_list()

    stages = _select_stages(args.only, args.skip)
    studies = _select_studies(args.only, args.skip, args.studies)

    eff_b = 100 if args.fast else int(args.B)
    timings: list[tuple[str, float, int]] = []
    t_total = time.perf_counter()

    for spec in stages:
        print(f'\n=== [{spec.code}] {spec.stage}: {spec.description} ===', flush=True)
        t0 = time.perf_counter()
        mod = _import_first((f'analysis.pipelines.{spec.module}', f'pipelines.{spec.module}'))
        stage_argv = ['--B', str(eff_b)]
        if args.fast:
            stage_argv.append('--fast')
        if spec.supports_split and args.split != 'capped':
            stage_argv.extend(['--split', args.split])
        rc = int(mod.main(stage_argv) or 0)
        dt = time.perf_counter() - t0
        timings.append((f'{spec.code}:{spec.stage}', dt, rc))
        print(f'--- [{spec.code}] {spec.stage} finished in {dt:.1f}s (exit={rc}) ---', flush=True)
        if rc != 0:
            return rc

    for name, mod_candidates, desc in studies:
        print(f'\n=== [study] {name}: {desc} ===', flush=True)
        t0 = time.perf_counter()
        mod = _import_first(mod_candidates)
        study_argv = ['--B', str(eff_b)] + (['--fast'] if args.fast else [])
        rc = int(mod.main(study_argv) or 0) if hasattr(mod, 'main') else 0
        dt = time.perf_counter() - t0
        timings.append((f'study:{name}', dt, rc))
        print(f'--- [study] {name} finished in {dt:.1f}s (exit={rc}) ---', flush=True)
        if rc != 0:
            return rc

    total_dt = time.perf_counter() - t_total
    print('\n=== Execution Summary ===', flush=True)
    for label, dt, rc in timings:
        print(f'  {label:<30} {dt:7.1f}s  rc={rc}', flush=True)
    print(f'  {"TOTAL":<30} {total_dt:7.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
