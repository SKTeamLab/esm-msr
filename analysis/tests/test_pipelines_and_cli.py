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
from msr_analysis.paths import REFERENCE  # noqa: E402

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
    assert {'table_figure_map.csv', 'manifest_outputs.csv'} == set(written)
