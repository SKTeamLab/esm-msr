"""Tests for atomic symlink-safe writers, immutable StyleSpec/MODELS, epistasis synthesis, heldout, and scoring."""
import dataclasses
import pickle
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
for _p in (str(_ANALYSIS_DIR), str(_ANALYSIS_DIR / 'scoring')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_predictions as C  # noqa: E402
import resources as RES  # noqa: E402
import score_runs as SR  # noqa: E402
from msr_analysis import data, heldout, models, paths, registry, style  # noqa: E402


def test_atomic_writers_replace_symlinks_without_modifying_targets(tmp_path):
    target_dir = tmp_path / 'authoritative'
    work_dir = tmp_path / 'worktree_cache'
    target_dir.mkdir()
    work_dir.mkdir()

    # 1. Pickle symlink safety
    orig_pkl = target_dir / 'frame.pkl'
    with open(orig_pkl, 'wb') as fh:
        pickle.dump({'orig': 123}, fh)
    orig_pkl_bytes = orig_pkl.read_bytes()
    sym_pkl = work_dir / 'frame.pkl'
    sym_pkl.symlink_to(orig_pkl)
    assert sym_pkl.is_symlink()

    paths.atomic_pickle_dump({'new': 456}, sym_pkl)
    assert not sym_pkl.is_symlink()
    assert orig_pkl.read_bytes() == orig_pkl_bytes
    with open(sym_pkl, 'rb') as fh:
        assert pickle.load(fh) == {'new': 456}

    # 2. CSV symlink safety
    orig_csv = target_dir / 'table.csv'
    orig_csv.write_text('a,b\n1,2\n')
    sym_csv = work_dir / 'table.csv'
    sym_csv.symlink_to(orig_csv)
    paths.atomic_to_csv(pd.DataFrame({'a': [9], 'b': [10]}), sym_csv, index=False)
    assert not sym_csv.is_symlink()
    assert orig_csv.read_text() == 'a,b\n1,2\n'
    assert pd.read_csv(sym_csv).iloc[0].to_dict() == {'a': 9, 'b': 10}

    # 3. Text & raw bytes symlink safety + 0644 mode enforcement
    orig_txt = target_dir / 'note.md'
    orig_txt.write_text('immutable target')
    sym_txt = work_dir / 'note.md'
    sym_txt.symlink_to(orig_txt)
    paths.atomic_write_text(sym_txt, 'worktree local write')
    assert not sym_txt.is_symlink()
    assert orig_txt.read_text() == 'immutable target'
    assert sym_txt.read_text() == 'worktree local write'

    raw_bin = work_dir / 'blob.bin'
    paths.atomic_write_bytes(raw_bin, b'abc123')
    for written_file in (sym_pkl, sym_csv, sym_txt, raw_bin):
        assert stat.S_IMODE(written_file.stat().st_mode) == 0o644


def test_canonical_results_routing_and_fig_prefix_stripping():
    expected_tops = (
        '00_for_reviewers',
        '00_main_text_figs',
        '00_supplementary_figs',
        '01_megascale',
        '02_thermostability',
        '03_proteingym',
        '04_domainome',
        '05_mgnify',
        '06_summary',
        '07_screening_overview',
        '08_sequence_length',
        '09_ablation',
        '10_inference_resources',
        '99_misc',
        'si_latex_tables',
    )
    assert paths.CANONICAL_RESULT_TOPS == expected_tops
    assert list(paths.CANONICAL_RESULT_TOPS) == sorted(paths.CANONICAL_RESULT_TOPS)

    assert paths.strip_fig_prefix('fig_best_ddg_vs_k.png') == 'best_ddg_vs_k.png'
    assert paths.strip_fig_prefix('fig5a_epistasis_levels_bars_test.pdf') == 'epistasis_levels_bars_test.pdf'
    assert paths.strip_fig_prefix('fig12bc_foo_bar') == 'foo_bar'
    assert paths.strip_fig_prefix('figure3.png') == 'figure3.png'
    assert paths.strip_fig_prefix('megascale_overview') == 'megascale_overview'

    assert paths.results_dir('megascale', split='capped', create=False) == paths.RESULTS / '01_megascale'
    assert paths.results_dir('megascale', split='mega', create=False) == paths.RESULTS / '01_megascale' / 'mega_splits'
    assert paths.results_dir('epistasis', split='capped', create=False) == paths.RESULTS / '01_megascale' / 'epistasis'
    assert paths.results_dir('epistasis', split='mega', create=False) == paths.RESULTS / '01_megascale' / 'mega_splits' / 'epistasis'
    assert paths.results_dir('pair_heatmaps', split='capped', create=False) == paths.RESULTS / '01_megascale' / 'epistasis' / 'pair_heatmaps'
    assert paths.results_dir('pair_heatmaps', split='mega', create=False) == paths.RESULTS / '01_megascale' / 'mega_splits' / 'epistasis' / 'pair_heatmaps'
    assert paths.results_dir('property_gain', split='capped', create=False) == paths.RESULTS / '01_megascale' / 'screening' / 'property_gain'
    assert paths.results_dir('property_gain', split='mega', create=False) == paths.RESULTS / '01_megascale' / 'mega_splits' / 'screening' / 'property_gain'
    assert paths.results_dir('reviewer_reference', create=False) == paths.RESULTS / '00_for_reviewers' / 'reference'
    assert paths.results_dir('domainome_screening', create=False) == paths.RESULTS / '04_domainome' / 'screening'
    assert paths.results_dir('dms3', create=False) == paths.RESULTS / '99_misc' / 'dms3'
    assert paths.VALIDATION_VS_BENCHMARKS_CSV == paths.RESULTS / '00_for_reviewers' / 'reference' / 'validation_vs_benchmarks.csv'
    assert paths.CORE_RHO_BOOTSTRAP_CSV == paths.CACHE / 'core_rho_bootstrap.csv'

    for retired in (
        '00_core',
        '01_summary',
        '03_tsuboyama',
        '04_dms_domainome',
        '05_proteingym',
        '06_epistasis',
        '07_rmse',
        '08_screening',
        '09_sequence_length',
        '10_leakage_and_reference',
        '12_pair_heatmaps',
        '16_screening_overview',
        '17_gain_fit',
        '18_mgnify',
        '19_rosetta_multi_vs_additive',
        '22_esm_msr_vs_esm3dg',
        '23_pg_ours_vs_published',
        '98_ablation',
        'studies/ablations',
    ):
        with pytest.raises(ValueError):
            paths.results_dir(retired, create=False)


def test_stylespec_and_models_registry_are_immutable():
    spec = registry.MODELS['msrp0:wt']
    assert spec['color'] == '#000000' and spec.color == '#000000'
    with pytest.raises(TypeError):
        spec['color'] = '#ffffff'  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.color = '#ffffff'  # type: ignore[misc]
    with pytest.raises(TypeError):
        registry.MODELS['injected'] = spec  # type: ignore[index]

    overridden = style.style_of('msrp0:wt', overrides={'color': '#123456', 'label': 'Custom'})
    assert overridden.color == '#123456' and overridden.label == 'Custom'
    assert registry.MODELS['msrp0:wt'].color == '#000000'


def test_all_roster_keys_registered_and_ablation_styles_non_conflicting():
    for m in models.ROSTER:
        assert m.name in registry.MODELS
        for attr in ('single', 'multi', 'add', 'epi'):
            key = getattr(m, attr)
            if key:
                assert key in registry.MODELS, f'{m.name}.{attr} = {key!r} missing from registry.MODELS'

    # Ablation styles must never use RING (#c0392b) and must reserve black diamond 'D' for FINAL_p0
    for fam, spec in models.ABLATION_STYLES.items():
        assert spec.color.lower() != '#c0392b'
        if fam != 'FINAL_p0':
            assert spec.marker != 'D'
        assert style.style_of(fam) == spec


def test_synthesize_missing_epi_and_sum_singles_dedup():
    raw_preds = {
        'pmpnn': np.array([[0.8, 0.9]]),
        'pmpnn+': np.array([[0.5, 0.7]]),
        'geoddg': np.array([[1.2, -0.4]]),
        'geoddg+': np.array([[1.0, -0.1]]),
        'rosetta': np.array([[-0.3, 0.2]]),
        'rosetta+': np.array([[-0.2, 0.0]]),
        'esm3dg': np.array([[0.4, 0.6]]),
        'esm3dg+': np.array([[0.3, 0.5]]),
        'ddgemb': np.array([[0.1, 0.2]]),
        'ddgemb+': np.array([[0.0, 0.1]]),
        'msrp0:comb': np.array([[1.5, -0.5]]),
        'msrp0:wt+': np.array([[1.1, -0.3]]),
    }
    syn = data.synthesize_missing_epi(raw_preds)
    for k in ('pmpnn:epi', 'geoddg:epi', 'rosetta:epi', 'esm3dg:epi', 'ddgemb:epi', 'msrp0:epi'):
        assert k in syn
    assert np.allclose(syn['pmpnn:epi'], [[0.3, 0.2]])
    assert np.allclose(syn['msrp0:epi'], [[0.4, -0.2]])

    # Duplicate single IDs in spine: sum_singles uses first occurrence deterministically
    sp = pd.DataFrame({
        'id': ['P_A1G', 'P_A1G', 'P_C2D', 'P_A1G:C2D'],
        'n_mut': [1, 1, 1, 2],
    })
    arr = np.array([1.0, 99.0, 2.5, 0.0])
    summed = data.sum_singles(sp, arr)
    assert summed[3] == pytest.approx(3.5)


def test_heldout_four_part_counts():
    expected = {'domainome': 86, 'proteingym': 41, 'mgnify': 27687, 'rocklin_denovo': 258}
    for ds, want_excl in expected.items():
        excl = heldout.excluded(ds)
        keep = heldout.keep_ids(ds)
        assert len(excl) == want_excl
        assert not (excl & keep)
        assert len(keep) > 0


def test_scoring_consolidation_and_collect_groups(tmp_path):
    pred_dir = tmp_path / 'pred'
    out_dir = tmp_path / 'scored'
    pred_dir.mkdir()
    pd.DataFrame({
        'mut_type': ['A1G', 'C2D', 'E3K', 'A1G:C2D'],
        'code': ['1abc'] * 4,
        'ddG': [-1.0, 0.2, 0.8, 1.4],
        'pred_col': [-0.9, 0.1, 0.9, 1.5],
    }).to_csv(pred_dir / 'external_s669.csv', index=False)

    summary_path, out_df = SR.score_baseline_dir(pred_dir, out_dir, 'test_run', 'pred_col')
    assert summary_path.exists()
    assert out_df.loc['external_s669', 'spearman_all'] == pytest.approx(1.0)
    assert out_df.loc['external_s669', 'n_singles'] == 3
    assert out_df.loc['external_s669', 'n_doubles'] == 1
    assert np.isfinite(out_df.loc['external_s669', 'ndcg@96'])

    # New collect_predictions groups ('final' and 'mega')
    assert C.final_file_dst('final_pgym_wt', 'FINAL_p0/s1/A4_HUMAN_Seuma_2022.csv') == 'final_pgym_wt/FINAL_p0/s1/A4_HUMAN_Seuma_2022.csv'
    assert C.final_file_dst('final_bench', 'slurm-12345.out') is None
    assert C.mega_tracked_dst('spurs_mega_s1', 'tsuboyama_test.csv') == 'baselines/retrained/mega_splits/tracked/spurs_mega_s1/tsuboyama_test.csv'
    assert C.mega_tracked_dst('me_mega_st1_s1', 'inputs/tmp.csv') is None
    assert RES.group_of('external_s669') == 'External'
    assert RES.group_of('dms_MYO_HUMAN_Kung_2025_display') == 'DMS'
