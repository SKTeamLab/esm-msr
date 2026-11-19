"""Tests for atomic symlink-safe writers, immutable StyleSpec/MODELS, epistasis synthesis, heldout, and scoring."""
import dataclasses
import pickle
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

    # 3. Text symlink safety
    orig_txt = target_dir / 'note.md'
    orig_txt.write_text('immutable target')
    sym_txt = work_dir / 'note.md'
    sym_txt.symlink_to(orig_txt)
    paths.atomic_write_text(sym_txt, 'worktree local write')
    assert not sym_txt.is_symlink()
    assert orig_txt.read_text() == 'immutable target'
    assert sym_txt.read_text() == 'worktree local write'


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
