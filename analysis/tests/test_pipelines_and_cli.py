"""Integration tests for unified CLI, pipeline/study/audit imports, and 27-table manifest coverage."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ANALYSIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / 'src'), str(_ANALYSIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis import run as cli_run  # noqa: E402
from msr_analysis.paths import CANONICAL_RESULT_TOPS, REFERENCE, RESULTS, VALIDATION_VS_BENCHMARKS_CSV  # noqa: E402

PIPELINE_MODULES = (
    'p01_benchmarks_external',
    'p02_tsuboyama',
    'p03_proteingym',
    'p04_mgnify_rocklin',
    'p05_summary_main',
    'p06_epistasis_hierarchy',
    'p07_epistasis_mechanisms',
    'p08_calibration_length',
    'p09_screening',
    'p10_splits_and_leakage',
    'p11_ablations_and_heads',
    'p12_manifest',
)

STUDY_MODULES = (
    'analysis.studies.faure_epistasis',
    'analysis.studies.faure_epistasis.run',
    'analysis.studies.faure_epistasis.steps',
    'analysis.studies.sigma_mechanisms',
    'analysis.studies.sigma_mechanisms.run',
    'analysis.studies.sigma_mechanisms.metrics',
    'analysis.studies.sigma_mechanisms.figures',
)

AUDIT_MODULES = (
    'analysis.audits',
    'analysis.audits.baseline_bridge',
    'analysis.audits.baseline_bridge.inventory',
    'analysis.audits.homology_and_splits',
    'analysis.audits.homology_and_splits.lib',
    'analysis.audits.homology_and_splits.active_builders.tsuboyama_rule_flags',
    'analysis.audits.homology_and_splits.active_builders.set_overlap_matrix',
    'analysis.audits.homology_and_splits.active_builders.build_mega_tasks',
    'analysis.audits.homology_and_splits.active_builders.mega_split_provenance',
    'analysis.audits.homology_and_splits.active_builders.export_reference',
    'analysis.audits.homology_and_splits.oct2026_leak_audit.run_all',
)


def test_cli_list_and_selection_parsing(capsys):
    rc = cli_run.main(['--list'])
    assert rc == 0
    out = capsys.readouterr().out
    assert len(cli_run.STAGES) == 12
    for spec in cli_run.STAGES:
        assert spec.code in out
        assert spec.stage in out
        assert spec.module in out
    assert 'study:faure_epistasis' in out
    assert 'study:sigma_mechanisms' in out

    sel = cli_run._select_stages(['p01,tsuboyama', '12'], skip_arg=['p02'])
    assert [s.code for s in sel] == ['p01', 'p12']

    st_sel = cli_run._select_studies(['study:sigma_mechanisms'], skip_arg=None, studies_flag=False)
    assert [s[0] for s in st_sel] == ['sigma_mechanisms']
    assert cli_run._select_stages(['study:sigma_mechanisms'], skip_arg=None) == []

    st_all = cli_run._select_studies(None, skip_arg=['faure_epistasis'], studies_flag=True)
    assert [s[0] for s in st_all] == ['sigma_mechanisms']

    with pytest.raises(SystemExit):
        cli_run._select_stages(['nonexistent_stage'], skip_arg=None)


def test_run_all_sh_list_wrapper():
    script = _ANALYSIS_DIR / 'run_all.sh'
    assert script.exists()
    env = dict(os.environ, PYTHON=sys.executable)
    proc = subprocess.run(
        ['bash', str(script), '--list'],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'p01  benchmarks_external' in proc.stdout
    assert 'p12  manifest' in proc.stdout
    assert 'study:sigma_mechanisms' in proc.stdout


@pytest.mark.parametrize('mod_stem', PIPELINE_MODULES)
def test_clean_import_all_12_pipelines(mod_stem):
    mod = importlib.import_module(f'analysis.pipelines.{mod_stem}')
    assert isinstance(getattr(mod, 'STAGE', None), str) and mod.STAGE
    assert isinstance(getattr(mod, 'DESCRIPTION', None), str) and mod.DESCRIPTION
    assert callable(getattr(mod, 'main', None))


@pytest.mark.parametrize('mod_path', STUDY_MODULES + AUDIT_MODULES)
def test_clean_import_studies_and_audits(mod_path):
    mod = importlib.import_module(mod_path)
    assert mod is not None


def test_p12_manifest_table_coverage_without_disk_mutation(monkeypatch):
    from analysis.pipelines import p12_manifest as p12

    written: dict[str, pd.DataFrame] = {}

    def _capture_csv(df: pd.DataFrame, path, **kwargs):
        written[Path(path).name] = df.copy()
        return Path(path)

    monkeypatch.setattr(p12, 'atomic_to_csv', _capture_csv)

    idx = pd.read_csv(REFERENCE / 'benchmark_tables' / 'tables_index.csv', dtype=str)
    assert len(idx) == 27
    assert set(idx['table']) == set(p12.TABLE_FIGURES)

    map_df, inv_df, bad = p12.run_table_coverage_and_manifest()
    assert bad == [], f'Unmapped or missing benchmark table figures: {bad}'
    assert len(map_df) == 27
    assert (map_df['status'] == 'drawn').all()
    assert len(inv_df) > 100
    assert (inv_df['owner'] != 'other').all()
    assert set(inv_df['top_folder']) <= set(CANONICAL_RESULT_TOPS) | {'(root)'}
    assert {'table_figure_map.csv', 'manifest_outputs.csv'} == set(written)
    assert VALIDATION_VS_BENCHMARKS_CSV.exists()


def test_results_directory_reorganization_invariants():
    import re
    import stat

    # 1. 00_core retired completely; top-level folders strictly match CANONICAL_RESULT_TOPS
    assert not (RESULTS / '00_core').exists()
    assert list(RESULTS.rglob('*00_core*')) == []
    actual_tops = tuple(sorted(p.name for p in RESULTS.iterdir() if p.is_dir()))
    assert actual_tops == CANONICAL_RESULT_TOPS

    # 2. Zero legacy numbered folders inside 01_megascale/mega_splits
    mega_dirs = [p.name for p in (RESULTS / '01_megascale' / 'mega_splits').iterdir() if p.is_dir()]
    assert sorted(mega_dirs) == ['epistasis', 'gain_fit', 'provenance', 'screening']

    # 3. Zero ^fig(?:\d+[a-z]*)?_ prefixes outside 00_main_text_figs and 00_supplementary_figs
    fig_re = re.compile(r'^fig(?:\d+[a-z]*)?_')
    bad_fig_names = [
        str(p.relative_to(RESULTS))
        for p in RESULTS.rglob('*')
        if p.relative_to(RESULTS).parts[0] not in ('00_main_text_figs', '00_supplementary_figs')
        and fig_re.match(p.name)
    ]
    assert bad_fig_names == [], f'Unexpected fig-prefixed paths: {bad_fig_names}'

    # 4. 00_for_reviewers/reference contains ONLY reference items (zero leak_* or clean_means*)
    ref_files = sorted(p.name for p in (RESULTS / '00_for_reviewers' / 'reference').iterdir())
    assert ref_files == [
        'published_reference.pdf',
        'published_reference.png',
        'spurs_published_vs_retrained.csv',
        'spurs_published_vs_retrained.pdf',
        'spurs_published_vs_retrained.png',
        'validation_vs_benchmarks.csv',
    ]

    # 5. Required canonical subfolders & absorbed files exist
    for rel_dir in (
        '01_megascale/epistasis',
        '01_megascale/epistasis/pair_heatmaps',
        '01_megascale/gain_fit',
        '01_megascale/screening',
        '01_megascale/screening/property_gain',
        '02_thermostability/rmse',
        '02_thermostability/screening',
        '03_proteingym/screening',
        '04_domainome/screening',
        '07_screening_overview',
        '09_ablation',
        '10_inference_resources',
        '99_misc/dms3',
        '99_misc/dual_view',
        '99_misc/esm_msr_vs_esm3dg',
        '99_misc/faure_epistasis',
        '99_misc/rosetta_multi_vs_additive',
        '99_misc/sigma_mechanisms',
    ):
        assert (RESULTS / rel_dir).is_dir(), f'Missing canonical subfolder: {rel_dir}'
    for rel_file in (
        '00_supplementary_figs/pg_ours_vs_published.csv',
        '00_supplementary_figs/pg_ours_vs_published.pdf',
        '00_supplementary_figs/pg_ours_vs_published.png',
        '04_domainome/screening/domainome_screening_curves.pdf',
        '04_domainome/screening/domainome_screening_curves.png',
        '09_ablation/ablation_architecture_suite.pdf',
        '09_ablation/ablation_architecture_suite.png',
        '09_ablation/paired_vs_p0.csv',
    ):
        assert (RESULTS / rel_file).is_file(), f'Missing canonical file: {rel_file}'

    # 6. Zero 3-DMS leaks outside 99_misc/ and 00_for_reviewers/reference/
    dms_leak_re = re.compile(
        r'(GB1_Wu_2016|GRB2_HUMAN_Faure_2021_binding_domain|DLG4_HUMAN_Faure_2021_binding_domain|'
        r'GRB2_HUMAN_Faure_2021_abundance_domain|DLG4_HUMAN_Faure_2021_abundance_domain|'
        r'MYO_HUMAN_Kung_2025|ESTA_BACSU_Nutschel_2020_dTm|\bdms3\b|(?<!pg\|)\bdms_mean\b|'
        r'\bdms_multi_mean\b|DMS \(3\)|DLG4 bind\.|GRB2 bind\.)'
    )
    leaks = []
    for p in sorted(RESULTS.rglob('*')):
        rel = p.relative_to(RESULTS)
        if (
            rel.parts[0] == '99_misc'
            or str(rel).startswith('00_for_reviewers/reference/')
            or str(rel) in ('manifest_outputs.csv', 'table_figure_map.csv')
        ):
            continue
        if dms_leak_re.search(str(rel)):
            leaks.append(f'filename:{rel}')
        if p.is_file() and p.suffix.lower() in ('.csv', '.md', '.tex', '.txt'):
            m = dms_leak_re.search(p.read_text(encoding='utf-8', errors='ignore'))
            if m:
                leaks.append(f'{rel}:{m.group(0)}')
    assert leaks == [], f'3-DMS leaks found outside 99_misc / 00_for_reviewers/reference: {leaks}'

    mp_pg = pd.read_csv(RESULTS / '03_proteingym' / 'multi_property_index.csv')
    assert not mp_pg['protein'].isin(['DLG4_HUMAN', 'GRB2_HUMAN']).any()
    mp_dms = pd.read_csv(RESULTS / '99_misc' / 'dms3' / 'dms_multi_property_index.csv')
    assert set(mp_dms['protein']) == {'DLG4_HUMAN', 'GRB2_HUMAN'}

    # 7. All regular files in analysis/results/ have 0644 permissions
    bad_modes = [
        (str(p.relative_to(RESULTS)), oct(stat.S_IMODE(p.stat().st_mode)))
        for p in RESULTS.rglob('*')
        if p.is_file() and not p.is_symlink() and stat.S_IMODE(p.stat().st_mode) != 0o644
    ]
    assert bad_modes == [], f'Non-0644 files under results/: {bad_modes[:10]}'
